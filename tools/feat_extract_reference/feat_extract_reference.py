#!/usr/bin/env python3
"""Pure-numpy reference of the firmware's MFCC + Δ + ΔΔ feature pipeline.

Mirrors `components/letter_recognizer/feat_extract.c` step-for-step. Bundled
under tools/ as the host-side companion to the on-device frontend so the
"host round-trip" check from Vikunja #52 (AC item 4) can be exercised
against a known WAV.

Usage:

    # Compute features for a 16 kHz mono PCM file, save (80, 20, 3) tensor.
    python3 tools/feat_extract_reference/feat_extract_reference.py \\
        --wav path/to/letter_a.wav \\
        --out features.npy

    # Self-test on a synthetic 1 kHz tone (no input WAV needed). Prints
    # shape + finite-check + post-norm range + the DC-coefficient row.
    python3 tools/feat_extract_reference/feat_extract_reference.py --selftest

    # Optional: int8-quantize using the model's published quant params
    # (scale=0.07303, zero_point=-8 from MODEL_INFO.txt).
    python3 tools/feat_extract_reference/feat_extract_reference.py \\
        --wav letter_a.wav --quantize --out int8_input.npy

The pipeline matches firmware exactly *except* the FFT backend: firmware
uses esp-dsp's Radix-2 SIMD FFT, this reference uses numpy.fft.rfft. Both
implement the same DFT — outputs differ only at numerical-precision noise
levels, dominated by the int8 quantization-step noise. The mel filterbank,
log, DCT-II, deltas, normalization, and quantization are byte-for-byte
faithful. The pre-emphasis-first-then-pad framing matches
`tf.signal.frame(frame_length=320, frame_step=160, pad_end=True)`.

To complete the round-trip on a fresh checkout:

    1. Install a TFLite runtime (one-time; not bundled to keep this tool
       dependency-light):
           pip install tflite-runtime
       (or `pip install tensorflow` for the larger but more portable runtime).

    2. Run this script with --invoke to load the .tflite, push the int8
       input through, and print the top-K letter prediction.

    3. On-device, log the same int8 input tensor for the same WAV (a
       "frontend dump" hook on the recognizer task) and compare. Top-1
       match is the round-trip pass criterion.

The model and norm-array dependencies are derived from the project root by
default; override with --model-tflite, --mean-npy, --std-npy.
"""
import argparse
import math
import os
import struct
import sys
import wave

import numpy as np


# Constants mirror components/spell_common/include/spell_config.h. They are
# duplicated here rather than imported so this script remains a self-
# contained reference. If the firmware constants change, update here too.
SAMPLE_RATE = 16000
INFERENCE_WINDOW_SAMPLES = 12800            # 800 ms @ 16 kHz
MEL_N_FFT = 512
MEL_HOP_LENGTH = 160                        # 10 ms hop @ 16 kHz
MEL_WIN_LENGTH = 320                        # 20 ms window @ 16 kHz
MEL_N_MELS = 40
MEL_FMIN = 50.0
MEL_FMAX = 7600.0
MFCC_N_COEFFS = 20
MFCC_PRE_EMPHASIS = 0.97
FEAT_N_CHANNELS = 3                         # static + Δ + ΔΔ
FEAT_N_FRAMES = 80                          # bumped 79 → 80 in #52
FEAT_NORM_LEN = MFCC_N_COEFFS * FEAT_N_CHANNELS  # 60

# Model handoff — MODEL_INFO.txt
MODEL_INPUT_SCALE = 0.07303
MODEL_INPUT_ZERO_POINT = -8

DEFAULT_TFLITE = "models/flat26_baseline_int8.tflite"
DEFAULT_MEAN_NPY = "models/norm_mean.npy"
DEFAULT_STD_NPY = "models/norm_std.npy"


# ─── WAV loading (stdlib only) ─────────────────────────────────────────────


def load_wav_int16(path: str) -> np.ndarray:
    """Load a mono 16-bit PCM WAV at SAMPLE_RATE. Returns int16 numpy array."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            sys.exit(f"{path}: expected mono WAV, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: expected 16-bit samples, got {8 * w.getsampwidth()}-bit")
        if w.getframerate() != SAMPLE_RATE:
            sys.exit(
                f"{path}: expected {SAMPLE_RATE} Hz, got {w.getframerate()} Hz "
                f"(re-export the WAV at the firmware's sample rate)"
            )
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2")


def pad_or_truncate_window(pcm: np.ndarray) -> np.ndarray:
    """Match the segmenter's center-pad / trailing-truncate behavior so
    short letter clips and longer-than-window clips both produce a
    INFERENCE_WINDOW_SAMPLES-length window."""
    out = np.zeros(INFERENCE_WINDOW_SAMPLES, dtype=np.int16)
    n = pcm.size
    if n <= INFERENCE_WINDOW_SAMPLES:
        pad_lead = (INFERENCE_WINDOW_SAMPLES - n) // 2
        out[pad_lead : pad_lead + n] = pcm
    else:
        out[:] = pcm[-INFERENCE_WINDOW_SAMPLES:]
    return out


# ─── Frontend ──────────────────────────────────────────────────────────────


def hann_window() -> np.ndarray:
    """Periodic Hann window — matches feat_extract.c's
    `0.5 * (1 - cos(2π·i / (WIN_LENGTH - 1)))` formulation."""
    i = np.arange(MEL_WIN_LENGTH, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (MEL_WIN_LENGTH - 1)))


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank() -> np.ndarray:
    """HTK-formulation mel filterbank. Output shape (N_MELS, N_FFT//2 + 1)."""
    n_fft_bins = MEL_N_FFT // 2 + 1
    fft_bin_hz = SAMPLE_RATE / MEL_N_FFT

    mel_min = hz_to_mel(np.array(MEL_FMIN))
    mel_max = hz_to_mel(np.array(MEL_FMAX))
    mel_points = np.linspace(mel_min, mel_max, MEL_N_MELS + 2)
    bin_points = mel_to_hz(mel_points) / fft_bin_hz

    fb = np.zeros((MEL_N_MELS, n_fft_bins), dtype=np.float64)
    for m in range(MEL_N_MELS):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(n_fft_bins):
            fk = float(k)
            if left <= fk <= center:
                fb[m, k] = (fk - left) / (center - left)
            elif center < fk <= right:
                fb[m, k] = (right - fk) / (right - center)
    return fb


def build_dct_matrix() -> np.ndarray:
    """Orthonormal DCT-II matrix, shape (MFCC_N_COEFFS, MEL_N_MELS)."""
    M = MEL_N_MELS
    scale_zero = math.sqrt(1.0 / M)
    scale_rest = math.sqrt(2.0 / M)
    D = np.empty((MFCC_N_COEFFS, M), dtype=np.float64)
    for k in range(MFCC_N_COEFFS):
        for m in range(M):
            if k == 0:
                D[k, m] = scale_zero
            else:
                D[k, m] = scale_rest * math.cos(math.pi * k * (m + 0.5) / M)
    return D


def compute_features(pcm_int16: np.ndarray) -> np.ndarray:
    """Mirror feat_extract.c::feat_extract_compute(). Returns a (FEAT_N_FRAMES,
    MFCC_N_COEFFS, FEAT_N_CHANNELS) float32 tensor in NHWC layout."""
    if pcm_int16.dtype != np.int16:
        sys.exit("compute_features: pcm must be int16 (firmware feeds int16)")
    if pcm_int16.size != INFERENCE_WINDOW_SAMPLES:
        sys.exit(
            f"compute_features: expected {INFERENCE_WINDOW_SAMPLES} samples, "
            f"got {pcm_int16.size} — pad/truncate first"
        )

    # Float32 in [-1, 1].
    pcm = pcm_int16.astype(np.float64) / 32768.0

    # On-the-fly pre-emphasis matches feat_extract.c. Equivalent to applying
    # y[n] = x[n] - 0.97·x[n-1] (with y[0] = x[0]) to the entire buffer first
    # then framing. We construct y once and frame it.
    y = np.empty_like(pcm)
    y[0] = pcm[0]
    y[1:] = pcm[1:] - MFCC_PRE_EMPHASIS * pcm[:-1]

    hann = hann_window()
    fb = build_mel_filterbank()
    D = build_dct_matrix()

    out = np.zeros((FEAT_N_FRAMES, MFCC_N_COEFFS, FEAT_N_CHANNELS), dtype=np.float64)

    floor_val = 1e-10

    for t in range(FEAT_N_FRAMES):
        start = t * MEL_HOP_LENGTH

        # Build the windowed frame. Matches the firmware's loop semantics:
        # in-bounds indices contribute (pre-emphasis already applied to y),
        # out-of-bounds indices write 0 — same as `pad_end=True` framing.
        frame = np.zeros(MEL_N_FFT, dtype=np.float64)
        end = min(start + MEL_WIN_LENGTH, INFERENCE_WINDOW_SAMPLES)
        if end > start:
            seg_len = end - start
            frame[:seg_len] = y[start:end] * hann[:seg_len]

        # Real FFT — magnitude bins 0..N//2.
        spec = np.fft.rfft(frame)
        power = (spec.real ** 2 + spec.imag ** 2)

        log_mel = np.log(np.maximum(fb @ power, floor_val))
        mfcc = D @ log_mel  # shape (MFCC_N_COEFFS,)

        out[t, :, 0] = mfcc

    # Δ MFCC: edge-replicated central difference on the static (channel 0)
    # values. Matches the firmware's exact 5-tap formula:
    #   Δ[t] = (c[t+1] - c[t-1] + 2·(c[t+2] - c[t-2])) / 10
    static_c = out[:, :, 0]
    for t in range(FEAT_N_FRAMES):
        tp1 = min(t + 1, FEAT_N_FRAMES - 1)
        tp2 = min(t + 2, FEAT_N_FRAMES - 1)
        tm1 = max(t - 1, 0)
        tm2 = max(t - 2, 0)
        out[t, :, 1] = (
            (static_c[tp1] - static_c[tm1])
            + 2.0 * (static_c[tp2] - static_c[tm2])
        ) / 10.0

    # ΔΔ — same formula on Δ.
    delta_c = out[:, :, 1]
    for t in range(FEAT_N_FRAMES):
        tp1 = min(t + 1, FEAT_N_FRAMES - 1)
        tp2 = min(t + 2, FEAT_N_FRAMES - 1)
        tm1 = max(t - 1, 0)
        tm2 = max(t - 2, 0)
        out[t, :, 2] = (
            (delta_c[tp1] - delta_c[tm1])
            + 2.0 * (delta_c[tp2] - delta_c[tm2])
        ) / 10.0

    return out.astype(np.float32)


def apply_normalization(features: np.ndarray, mean_npy: str, std_npy: str) -> np.ndarray:
    """Per-coefficient/channel `(features - mean) / std` broadcast over frames.
    Matches the loop in letter_recognizer.cpp::recognizer_task."""
    mean = np.load(mean_npy).astype(np.float32).reshape(MFCC_N_COEFFS, FEAT_N_CHANNELS)
    std = np.load(std_npy).astype(np.float32).reshape(MFCC_N_COEFFS, FEAT_N_CHANNELS)
    if not np.all(std > 0):
        sys.exit(f"{std_npy}: std contains non-positive values — invalid normalization")
    return (features - mean[np.newaxis, :, :]) / std[np.newaxis, :, :]


def quantize_int8(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """clip(round(features / scale + zero_point), -128, 127)."""
    q = np.round(features / scale).astype(np.int32) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


# ─── CLI ───────────────────────────────────────────────────────────────────


def selftest(args: argparse.Namespace) -> int:
    """Run the pipeline on a synthetic 1 kHz sine + brief silence tail.
    Confirms: shape, all-finite, sane post-norm range, frame-79 (the new
    last frame) actually produces non-zero MFCC[0] thanks to the trailing
    pre-emphasized samples (i.e. the 79→80 bump did get exercised)."""
    print("== feat_extract_reference self-test ==")

    # 600 ms of 1 kHz tone + 200 ms silence inside a 800 ms window.
    t = np.arange(int(0.6 * SAMPLE_RATE)) / SAMPLE_RATE
    tone = (16384 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    silence = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.int16)
    pcm = pad_or_truncate_window(np.concatenate([tone, silence]))
    print(f"  input pcm shape: {pcm.shape}, dtype {pcm.dtype}")
    print(f"  rms_dbfs: {20 * np.log10(np.sqrt(np.mean((pcm/32768.0)**2)) + 1e-12):.1f}")

    feats = compute_features(pcm)
    print(f"  features shape: {feats.shape} (want (80, 20, 3))")
    assert feats.shape == (FEAT_N_FRAMES, MFCC_N_COEFFS, FEAT_N_CHANNELS)
    assert np.all(np.isfinite(feats)), "non-finite feature value"
    print(f"  channel 0 MFCC (static) range: {feats[:,:,0].min():+.3f} .. {feats[:,:,0].max():+.3f}")
    print(f"  channel 1 (Δ) range:          {feats[:,:,1].min():+.3f} .. {feats[:,:,1].max():+.3f}")
    print(f"  channel 2 (ΔΔ) range:         {feats[:,:,2].min():+.3f} .. {feats[:,:,2].max():+.3f}")

    # Frame 79 is the new last frame from the 79→80 bump in #52. Verify it
    # actually contributes signal — the static MFCC[0] for frame 79 should
    # NOT be the silence floor's log of 1e-10 (= -23.0258).
    silence_floor = math.log(1e-10)
    f79_mfcc0 = feats[FEAT_N_FRAMES - 1, 0, 0]
    print(f"  frame {FEAT_N_FRAMES-1} static MFCC[0]: {f79_mfcc0:.3f}")
    if abs(f79_mfcc0 - silence_floor) < 1.0:
        print("  WARNING: frame 79 looks like silence floor — framing may be misaligned.")
    else:
        print("  OK: frame 79 carries audio signal (the #52 bump is exercised).")

    # Try normalization if .npy files are reachable.
    mean_npy = args.mean_npy or DEFAULT_MEAN_NPY
    std_npy = args.std_npy or DEFAULT_STD_NPY
    if os.path.exists(mean_npy) and os.path.exists(std_npy):
        norm = apply_normalization(feats, mean_npy, std_npy)
        print(
            f"  post-norm range: {norm.min():+.2f} .. {norm.max():+.2f} "
            f"(typical post-norm features sit roughly within ±10 — anything "
            f"absurdly larger suggests a mean/std mismatch)"
        )
    else:
        print(f"  (skip norm: missing {mean_npy} or {std_npy})")

    print("== self-test passed ==")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wav", help="input mono 16-bit 16 kHz WAV; required unless --selftest")
    p.add_argument("--out", help="output .npy path; defaults to stdout summary only")
    p.add_argument("--selftest", action="store_true", help="run the synthetic-input self-test and exit")
    p.add_argument("--no-norm", action="store_true", help="skip per-coef/channel normalization")
    p.add_argument("--quantize", action="store_true", help="emit int8 instead of float32 (uses --scale, --zero-point)")
    p.add_argument("--scale", type=float, default=MODEL_INPUT_SCALE, help="quantization scale (default = MODEL_INFO.txt's 0.07303)")
    p.add_argument("--zero-point", type=int, default=MODEL_INPUT_ZERO_POINT, help="quantization zero point (default -8)")
    p.add_argument("--mean-npy", default=DEFAULT_MEAN_NPY)
    p.add_argument("--std-npy", default=DEFAULT_STD_NPY)
    p.add_argument(
        "--model-tflite",
        default=DEFAULT_TFLITE,
        help="path to .tflite (currently used only by --invoke)",
    )
    p.add_argument(
        "--invoke",
        action="store_true",
        help="load the .tflite via tflite_runtime/tensorflow and print top-K (requires pip install tflite-runtime or tensorflow)",
    )
    return p.parse_args()


def maybe_invoke_model(int8_input: np.ndarray, model_tflite: str) -> int:
    """Load the .tflite and invoke. Top-K printed to stdout. Soft-fails if
    tflite_runtime / tensorflow are not installed — keeps the script
    dependency-light."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # type: ignore
        except ImportError:
            sys.exit(
                "--invoke requires tflite_runtime or tensorflow. Install one:\n"
                "    pip install tflite-runtime\n"
                "  (or `pip install tensorflow` for the larger runtime)"
            )

    interp = Interpreter(model_path=model_tflite)
    interp.allocate_tensors()
    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    # Validate the live model's quant params match the script's defaults; warn
    # if not (model handoff may have shifted).
    s, zp = in_details["quantization"]
    if abs(s - MODEL_INPUT_SCALE) > 1e-5 or zp != MODEL_INPUT_ZERO_POINT:
        print(
            f"  WARNING: live model input quant ({s}, {zp}) differs from "
            f"MODEL_INFO.txt ({MODEL_INPUT_SCALE}, {MODEL_INPUT_ZERO_POINT}) — "
            f"re-quantize with the live params"
        )

    interp.set_tensor(in_details["index"], int8_input.reshape(1, FEAT_N_FRAMES, MFCC_N_COEFFS, FEAT_N_CHANNELS))
    interp.invoke()
    raw = interp.get_tensor(out_details["index"]).reshape(-1)
    out_s, out_zp = out_details["quantization"]
    probs = (raw.astype(np.float32) - out_zp) * out_s
    if probs.min() < 0 or probs.max() > 1:
        # Logits — softmax in firmware-style.
        x = probs - probs.max()
        ex = np.exp(x)
        probs = ex / ex.sum()
    top5 = np.argsort(-probs)[:5]
    print("  top-5:", ", ".join(f"{chr(ord('A')+i)}={probs[i]:.3f}" for i in top5))
    return 0


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest(args)
    if not args.wav:
        sys.exit("--wav is required (or use --selftest). See --help.")
    if not os.path.exists(args.wav):
        sys.exit(f"{args.wav}: not found")

    pcm = load_wav_int16(args.wav)
    pcm = pad_or_truncate_window(pcm)
    feats = compute_features(pcm)
    if not args.no_norm:
        feats = apply_normalization(feats, args.mean_npy, args.std_npy)

    if args.quantize:
        out = quantize_int8(feats, args.scale, args.zero_point)
        print(f"int8 input: shape {out.shape}, range {out.min()}..{out.max()}")
    else:
        out = feats.astype(np.float32)
        print(f"features: shape {out.shape}, range {out.min():+.3f}..{out.max():+.3f}")

    if args.out:
        np.save(args.out, out)
        print(f"saved -> {args.out}")

    if args.invoke:
        if not args.quantize:
            # Quantize transparently for invoke.
            out = quantize_int8(feats, args.scale, args.zero_point)
        return maybe_invoke_model(out, args.model_tflite)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

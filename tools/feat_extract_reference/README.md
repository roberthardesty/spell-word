# feat_extract_reference

Pure-numpy reference of the firmware's MFCC + Δ + ΔΔ + per-coefficient/
channel-normalization pipeline. Mirrors
`components/letter_recognizer/feat_extract.c` step-for-step.

The firmware uses esp-dsp's Radix-2 SIMD FFT for performance; this script
uses `numpy.fft.rfft`. Both implement the same DFT and differ only at
numerical-precision noise levels — well below the int8 quantization step
once the output reaches the model. Every other math step (pre-emphasis
on-the-fly during frame loading, Hann windowing, HTK mel filterbank, log,
orthonormal DCT-II, central-difference Δ and ΔΔ, per-coef normalization,
int8 quantization with `MODEL_INFO.txt`'s `scale=0.07303 / zp=-8`) is
byte-faithful.

## What it's for

Vikunja #52's AC item 4 calls for a "host round-trip test" comparing a
known WAV through a Python pipeline against the firmware's `feat_extract`.
This script provides the **Python pipeline** half. Two ways to use it:

### 1. Self-test (no inputs needed)

```sh
make -C tools/feat_extract_reference selftest
```

Generates a synthetic 1 kHz tone + silence-tail, runs the full pipeline,
prints shape + sanity-check + per-channel range. Verifies the framing
math hits frame 79 (the new last frame from the 79 → 80 bump in #52).

### 2. Process a real WAV

```sh
python3 tools/feat_extract_reference/feat_extract_reference.py \
    --wav path/to/letter_a.wav \
    --out features.npy
```

Loads the WAV (must be mono 16-bit 16 kHz — re-export if it isn't),
runs MFCC + Δ + ΔΔ + per-coef normalization (using `models/norm_*.npy`),
saves the float32 (80, 20, 3) tensor to `--out`.

Add `--quantize` to emit the int8-quantized 4800-byte input tensor with
the model's quant params. Add `--invoke` to also load the .tflite and
print top-5 predictions — `--invoke` requires `pip install tflite-runtime`
(or `tensorflow`) which isn't bundled by default to keep the host tools
dependency-light.

## To complete the round-trip vs firmware

The Python side of the round-trip is local; the firmware side requires
on-device serial logging. Two options:

1. **Static-input round-trip.** Generate a fixed int16 PCM buffer (e.g.,
   the `--selftest` synthetic tone, written to `.wav`). Run this script
   with `--quantize --invoke` to get the host's top-1. On the recognizer
   task, add a temporary `ESP_LOGI("recognizer", "input[0..n]: …")` for
   the same fixed input, compare top-1. Pass criterion: same letter.

2. **Frontend dump.** Add a serial dump of `s_features[]` (or the post-
   normalization buffer) to the firmware. Capture a known WAV played back
   over a speaker or piped through I2S. Compare against this script's
   `.npy` output for the same WAV. Allow ≤ 0.5 dB-equivalent drift to
   account for the FFT backend difference; the int8 quantized inputs
   should match exactly.

Either approach is HITL-shaped (requires firmware on hardware). The
script alone validates the host-side math.

## Why a separate tool dir vs sibling of `gen_norm_arrays`

`gen_norm_arrays` produces compile-time artifacts (the embedded float
arrays). `feat_extract_reference` is a runtime check that exercises the
whole frontend math. Each tool has a distinct lifecycle — keep them
separated so future bench-comparison scripting lands here without
polluting the regenerator.

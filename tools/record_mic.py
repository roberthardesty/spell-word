#!/usr/bin/env python3
"""Capture a short clip from a Mac mic to a 16 kHz mono 16-bit WAV.

Output matches the format `tools/feat_extract_reference/feat_extract_reference.py`
requires (mono, 16-bit, 16 kHz), so the WAV plugs straight into the host
inference pipeline.

Quick use:

    .venv-hitl/bin/python tools/record_mic.py letter_a.wav

Options:

    --duration SECONDS    default 2.0 (enough for one letter + padding)
    --device   INDEX|NAME default = system default input
    --countdown N         seconds of beepless countdown before recording (default 2)
    --list-devices        print input devices and exit

Requires `sounddevice` in the active venv:

    .venv-hitl/bin/pip install sounddevice
"""
import argparse
import sys
import time
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000


def list_devices() -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  in_ch={d['max_input_channels']}  "
                  f"default_sr={int(d['default_samplerate'])}")
    return 0


def record(duration: float, device, countdown: int) -> np.ndarray:
    for n in range(countdown, 0, -1):
        print(f"  {n}...", flush=True)
        time.sleep(1.0)
    print("  GO (speak now)", flush=True)
    n_samples = int(duration * SAMPLE_RATE)
    audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1,
                   dtype="int16", device=device)
    sd.wait()
    return audio.reshape(-1)


def report_levels(pcm: np.ndarray) -> None:
    peak = int(np.max(np.abs(pcm)))
    rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2)))
    peak_dbfs = 20.0 * np.log10(peak / 32768.0 + 1e-12)
    rms_dbfs = 20.0 * np.log10(rms + 1e-12)
    print(f"  peak={peak}/32768 ({peak_dbfs:+.1f} dBFS)  rms={rms_dbfs:+.1f} dBFS")
    if peak >= 32767:
        print("  WARNING: clipping — back off the mic gain or move further away")
    elif peak_dbfs < -30:
        print("  WARNING: very quiet — move closer to the mic or check input gain")


def save_wav(path: str, pcm: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("output", nargs="?", help="output WAV path")
    p.add_argument("--duration", type=float, default=2.0,
                   help="recording length in seconds (default 2.0)")
    p.add_argument("--device", default=None,
                   help="input device index or substring match (default = system default)")
    p.add_argument("--countdown", type=int, default=2,
                   help="seconds of countdown before recording (default 2)")
    p.add_argument("--list-devices", action="store_true",
                   help="list available input devices and exit")
    args = p.parse_args()

    if args.list_devices:
        return list_devices()
    if not args.output:
        sys.exit("output path required (or use --list-devices). See --help.")

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    print(f"Recording {args.duration:.1f}s at {SAMPLE_RATE} Hz mono "
          f"(device={device or 'default'}).")
    pcm = record(args.duration, device, args.countdown)
    report_levels(pcm)
    save_wav(args.output, pcm)
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Host pipeline: WAV → letter prediction, no hardware

A faithful, host-side replay of the firmware's audio path. Lets you take a
recorded or synthetic WAV, run it through the same segmenter VAD and
feature pipeline the firmware uses, invoke the int8 TFLite model, and read
the top-K letter prediction — without flashing or wearing a board.

The pipeline is **two stages**, each a separate tool with its own build:

```
  recording.wav
       │
       ▼
  ┌──────────────────────────┐
  │ Stage 1: segmenter_replay │   tools/segmenter_replay/   (C)
  │                          │   links components/segmenter/segmenter_core.c
  │  VAD with onset/offset,  │   ── bit-identical to firmware modulo float
  │  preroll, EOW, early-    │      precision (Xtensa vs host).
  │  commit, force-split.    │
  └──────────────────────────┘
       │  --dump-letters DIR
       │  → DIR/letter_001.wav, letter_002.wav, ...
       │  (each WAV: 200 ms preroll + letter; variable length;
       │   force-split at SPELL_VAD_MAX_LETTER_MS = 800 ms.)
       ▼
  ┌──────────────────────────────┐
  │ Stage 2: feat_extract_reference │  tools/feat_extract_reference/  (Python)
  │                              │  numpy mirror of components/letter_recognizer/feat_extract.c
  │  Pad/truncate to 800 ms      │  + per-coef/channel norm from models/norm_*.npy
  │  → pre-emphasis → Hann →     │  + int8 quantize (scale, zero_point from
  │  rFFT → HTK mel → log →      │    MODEL_INFO.txt)
  │  DCT-II → Δ → ΔΔ → norm →    │  + optional --invoke for top-K via tf.lite.Interpreter
  │  int8 quantize → tflite.     │
  └──────────────────────────────┘
       │
       ▼
   top-K letters (host's verdict for this WAV)
```

Together: every byte of the on-device classifier path that doesn't require
the I2S mic or esp-dsp's SIMD FFT. The two FFT backends (numpy.fft.rfft on
host, dsps_fft2r_fc32 on device) implement the same DFT and differ only at
numerical-precision noise levels — well below the int8 quantization step.

## Prerequisites

### Python venv (Stage 2)

The script is pure-numpy plus an optional TFLite interpreter for `--invoke`.

```sh
# Use Python 3.11 or 3.12 — tensorflow doesn't have 3.14 wheels at time of writing.
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv-hitl
.venv-hitl/bin/pip install --upgrade pip
.venv-hitl/bin/pip install numpy tensorflow
```

`tflite-runtime` has **no macOS arm64 wheels**, so on Apple Silicon use the
full `tensorflow` package. The script imports `Interpreter` via
`tf.lite.Interpreter` attribute access — modern TF (≥ 2.16) dropped the
`from tensorflow.lite import Interpreter` submodule path.

For mic capture, also install `sounddevice` (see Stage 0 below).

### segmenter_replay binary (Stage 1)

Plain C, no IDF, no FreeRTOS. One command:

```sh
make -C tools/segmenter_replay
# → tools/segmenter_replay/segmenter_replay
```

### Self-tests (run once, after pulling)

Confirm the two halves haven't drifted before relying on them:

```sh
make -C tools/segmenter_replay test            # VAD unit tests
make -C tools/feat_extract_reference selftest  # feature pipeline math
```

A passing selftest on `feat_extract_reference` reports `frame 79 carries
audio signal (the #52 bump is exercised)` — that's the post-#52 sentinel.

## Stage 0: getting a WAV

Three sources:

1. **Live mic capture** — `tools/record_mic.py` writes a 16 kHz mono 16-bit
   WAV with a countdown:
   ```sh
   .venv-hitl/bin/python tools/record_mic.py letter_a.wav
   ```
   First run triggers the macOS mic-permission prompt for your terminal.
2. **TTS render** — any tool that produces a 16 kHz mono 16-bit WAV is
   fine. Out-of-distribution TTS voices may behave poorly; the model was
   trained on synthetic TTS but only a specific engine/voice set
   (see `models/MODEL_INFO.txt` and ADR-0004).
3. **On-device capture via `audio_dump_emit_b64()`** — pulls the exact PCM
   the firmware's segmenter saw, useful for chasing a host-vs-firmware
   divergence on a specific real utterance.

## End-to-end run

Given a recording with one or more spoken letters:

```sh
mkdir -p /tmp/letters
tools/segmenter_replay/segmenter_replay --dump-letters /tmp/letters recording.wav

# inspect what came out
ls /tmp/letters/
# letter_001.wav letter_002.wav ...

# classify each emitted utterance
for f in /tmp/letters/letter_*.wav; do
    echo "=== $f ==="
    .venv-hitl/bin/python tools/feat_extract_reference/feat_extract_reference.py \
        --wav "$f" --quantize --invoke
done
```

The `--invoke` flag loads `models/flat26_baseline_int8.tflite`, pushes the
int8 input through, and prints the top-5 letter predictions with
softmax-normalized probabilities.

## Stage 1: segmenter_replay (the VAD)

`tools/segmenter_replay/main.c` is a thin CLI wrapper around
`components/segmenter/segmenter_core.c`. The core is pure C99 (no IDF, no
FreeRTOS, no esp-dsp) and is the same translation unit that ships on the
device — so the algorithm is bit-identical modulo Xtensa-vs-host float
precision.

### What it emits

A `seg_event_t` per frame; `--dump-letters` writes one WAV per
`SEG_EVT_LETTER_EMITTED` (or `SEG_EVT_FORCE_SPLIT`). Each emitted WAV
contains:

- `~200 ms` of pre-onset audio (the preroll ring buffer), then
- the letter audio from detected onset to confirmed offset, accumulated
  frame-by-frame.

Total emission length is variable — typically ~400–700 ms for a single
letter, capped at ~1000 ms (preroll + `SPELL_VAD_MAX_LETTER_MS` = 800 ms)
via force-split when a single letter runs too long. End-of-word and
early-commit events are also reported but don't produce WAVs.

### Tuning flags (all override `spell_config.h` defaults)

```
--on DBFS            onset threshold     (default -38.0)
--off DBFS           offset threshold    (default -42.0)
--ema N              EMA alpha × 100     (default 35; smaller = smoother)
--off-frames N       sub-off frames to confirm offset (default 5)
--min-letter MS      reject letters shorter than this (default 150)
--max-letter MS      force-split letters longer than this (default 800)
--early-commit MS    inter-letter silence for early-commit window (default 500)
--eow MS             end-of-word silence (default 1200)
--preroll MS         pre-roll context    (default 200)
```

These match the `SPELL_VAD_*` defaults — tune here, then propagate to
`components/spell_common/include/spell_config.h` when bench-confirmed.

## Stage 2: feat_extract_reference (the classifier path)

`feat_extract_reference.py` mirrors `components/letter_recognizer/feat_extract.c`
step-for-step: on-the-fly pre-emphasis (α = 0.97), 320-sample periodic
Hann window, 512-pt rFFT, 40-band HTK mel filterbank (50 Hz – 7600 Hz),
log floor 1e-10, orthonormal DCT-II → 20 coefficients, central-difference
Δ and ΔΔ, per-coef/channel normalization from `models/norm_*.npy`, and
int8 quantization with `scale=0.07303, zero_point=-8` from
`models/MODEL_INFO.txt`.

### Flags

```
--wav PATH           input WAV (mono 16-bit 16 kHz)
--selftest           synthetic-input self-test (no WAV needed)
--out PATH           save features.npy / int8_input.npy
--no-norm            skip per-coef/channel normalization
--quantize           emit int8 instead of float32
--scale F            override quantization scale
--zero-point N       override quantization zero-point
--mean-npy PATH      override models/norm_mean.npy
--std-npy PATH       override models/norm_std.npy
--model-tflite PATH  override models/flat26_baseline_int8.tflite
--invoke             load .tflite and print top-5 (needs tensorflow)
```

The `--invoke` path also validates the live model's quantization params
against the script's defaults and warns if they've shifted — a cheap
guard against silent model-handoff drift.

## Truncation / framing — host vs firmware

Both stages target the same 800 ms × 80-frame inference window, but they
handle out-of-spec PCM lengths differently. Worth knowing when comparing
host and firmware top-1s.

| input length | host (`pad_or_truncate_window`)         | firmware (`feat_extract_compute`)            |
|--------------|------------------------------------------|----------------------------------------------|
| < 12800      | center-pad (zeros prepended **and** appended) | zeros appended only; first samples kept    |
| = 12800      | identity                                 | identity                                     |
| > 12800      | take **last** 12800 (leading dropped)    | read **first** ~12800; trailing dropped      |

For segmenter emissions ≤ 12800 samples (most letters) the host adds a
leading zero pad the firmware doesn't add — but `feat_extract`'s 200 ms
preroll-shaped onset and trailing zero padding minimize the divergence in
practice. For force-split emissions (~1000 ms = 16000 samples) the host
discards the preroll while the firmware discards trailing audio. If a
specific WAV produces a host/firmware top-1 mismatch and emission length
exceeds 12800 samples, this is the first suspect.

The script's `pad_or_truncate_window` is the place to change if you want
strict host-firmware parity; see `tools/feat_extract_reference/feat_extract_reference.py`
function of that name.

## Common failure modes

**`--invoke` exits with `Interpreter import` error.** macOS arm64 has no
`tflite-runtime` wheels. Install `tensorflow` instead. If `tensorflow` is
installed but import still fails, you're on Python 3.14 — drop to 3.11
or 3.12.

**Top-1 is consistently a different letter than what was spoken.**
First, check the WAV format: mono, 16-bit, 16 kHz. Then run
`segmenter_replay` with `--quiet` removed — confirm the letter is
actually detected (onset/offset events fire) and that the emitted WAV
length is sensible (~400–700 ms for a single letter). If segmentation
looks right but the prediction is way off, the input may be
out-of-distribution (different TTS voice than training, real-mic
acoustics gap — see ADR-0004) rather than a pipeline bug.

**Top-1 looks plausible but the int8 input range is unusually narrow**
(e.g., `int8 input: shape (80, 20, 3), range -21..24` for a clearly-spoken
letter, vs. the typical `-50..30` ballpark). Two likely causes:
1. `feat_extract` is processing trailing silence because the WAV was
   longer than 12800 samples and `pad_or_truncate_window` took the wrong
   end — re-crop to ≤ 800 ms with the letter centered, or feed through
   `segmenter_replay --dump-letters` first.
2. Normalization arrays drifted from the model. Verify with
   `make -C tools/gen_norm_arrays check`.

**`segmenter_replay` emits zero utterances from a clearly-spoken WAV.**
RMS is below `--on` threshold. Inspect with `--quiet` removed (it prints
per-frame state) or lower `--on` (default -38 dBFS).

## See also

- `tools/feat_extract_reference/README.md` — Stage 2 reference details
- `tools/segmenter_replay/Makefile` — Stage 1 build / run quick reference
- `tools/gen_norm_arrays/README.md` — regenerating the `feat_norm.c`
  arrays from `models/norm_*.npy`
- `docs/hitl/0001-feat-frontend-acceptance.md` — bench-mic procedure for
  the firmware-side acceptance; cross-references the host pipeline as
  the no-hardware alternative
- `docs/adr/0004-decoder-tuning-via-python-reference.md` — context for
  why the host reference exists and what acoustic-environment gap it
  doesn't close
- `models/MODEL_INFO.txt` — model handoff contract (quant params,
  normalization shape, frame count)

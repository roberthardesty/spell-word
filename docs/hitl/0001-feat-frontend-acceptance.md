# HITL acceptance: feature frontend alignment (Vikunja #52)

Status: ready for bench
Scope: Vikunja #52 acceptance criteria 4 (host round-trip) + 5 (alphabet recognition accuracy)
AFK code change: landed in commits `5692992` + `b51fedd` (2026-05-08)

## Why this acceptance exists

Commit `5692992` bumped `SPELL_FEAT_N_FRAMES` 79 → 80 and applied
per-coefficient/channel `(mfcc - mean) / std` normalization between
`feat_extract_compute()` and the int8 quantize loop in
`letter_recognizer.cpp::recognizer_task`. Both fixes correct silent
miscalibration: the old frontend left the trailing 60 input cells (frame
79) as stale int8-tensor noise, and the int8 quantization was calibrated
against post-norm features the firmware was never producing.

Both changes show up at letter-accuracy time, not at invoke time, so the
unit and host tests can't catch a regression here. The bench mic is the
verification path. This document captures the procedure so anyone with
hardware can run it.

The alternative — a fully host-side round-trip (#52 AC item 4) — is
documented in `tools/feat_extract_reference/README.md`. Either path
produces equivalent confidence; the bench-mic procedure below is the
faster of the two for the first pass since it requires no `pip install`
and no firmware-side dump hook.

## Prerequisites

1. **Board flashed with current `main`.** The patch must include both
   commits above. Sanity-check with `git log --oneline -3` — top should
   be `b51fedd` or later.

2. **Model + decoder partitions populated.** `decoder_init()` will log
   `partition is empty (erased) — awaiting OTA` for `confusion` and
   `dictionary` if not flashed; these don't gate frontend accuracy
   acceptance (the recognizer's top-K still emits) but they DO gate any
   spelling-level checks. For frontend acceptance alone, only the `model`
   partition (`models/flat26_baseline_int8.tflite`, written via
   `parttool.py write_partition --partition-name model …`) is required.

3. **Quiet recording environment.** ~30 dBA ambient or quieter. Loud
   rooms widen the per-utterance distribution; the segmenter's energy
   gate (`SPELL_ENERGY_GATE_DB = -45 dBFS`) and onset/offset hysteresis
   are tuned for normal-room TV-volume conditions but the bench check
   will be noisy in a busy office.

4. **Speaker held within ~15-20 cm of the mic.** ICS-43434 is omni; the
   training was on synthetic TTS with no room acoustics, so close-mic
   recording approximates the training distribution most closely. Per
   ADR-0004 the mic+room shift is the dominant accuracy gap; this
   acceptance measures the **frontend math is correct**, not that the
   model handles every acoustic environment.

5. **Serial monitor capturing.** The recognizer emits `recognizer:`
   prefix log lines. The pick-up-here.md guidance applies:

   ```sh
   : > /tmp/spell.log; nohup cat /dev/cu.usbmodem101 >> /tmp/spell.log & disown
   ```

   For interactive observation, `idf.py monitor` is fine but its UI
   makes log-grepping awkward; the `cat` capture is better for the
   alphabet-sweep procedure below.

## Procedure: alphabet-sweep top-1 accuracy (#52 AC item 5)

**Goal:** speak each of the 26 letters one at a time, record top-1
predictions, compute the accuracy rate.

1. Power-cycle the board so log lines start clean. Confirm the recognizer
   came up with the new geometry — look for these init lines:

   ```
   feat_extract: feat init: 512-pt FFT, 40 mels, 20 MFCC, 80 frames, 3 ch
   recognizer: model loaded from partition (… B)
   recognizer: interpreter ready — arena used … B (84% of alloc)
   recognizer:   input: (1,80,20,3) kTfLiteInt8, output dim0=26 …
   recognizer: recognizer task spawned on core 1 (model active)
   ```

   The `80 frames` and `(1,80,20,3)` are the post-#52 sentinels. If
   either still says `79` or `(1,79,20,3)`, the build is stale —
   recompile and reflash before continuing.

2. Wait for the segmenter's gate to settle (~1 s).

3. Speak A. Wait ~1 s. Speak B. Wait ~1 s. Continue through Z. Each
   utterance produces a log line of the form:

   ```
   recognizer: #N X=0.78 Y=0.10 Z=0.04 (Tms)
   ```

   where `N` is the cumulative inference count, `X/Y/Z` are the top-3
   letters with probabilities, and `T` is invoke time in milliseconds.
   The first 3 utterances + every 5th log this; for completeness, run
   the alphabet sweep twice so the second pass at #6, #11, #16, #21, #26
   captures every other letter. (Or temporarily change the sample rate
   in `recognizer_task` from `n_inf <= 3 || n_inf % 5 == 0` to log every
   utterance — bench-only patch, don't commit.)

4. Record per-letter results in a tally:

   | spoken | top-1 | top-1 prob | top-2 | notes |
   |--------|-------|------------|-------|-------|
   | A      |       |            |       |       |
   | B      |       |            |       |       |
   | …      |       |            |       |       |

5. Compute top-1 accuracy: `(# correct top-1) / 26`.

## Pass / fail criteria

- **Pass:** top-1 accuracy ≥ 70%, AND top-3 contains the spoken letter
  for ≥ 90% of utterances.

  The 97% accuracy in `MODEL_INFO.txt` is on synthetic TTS, not real
  mic audio (per ADR-0004 — the front-end distribution shift between
  Mac-recorded TTS and ICS-43434 + room acoustics is the dominant
  source of accuracy degradation). 70% top-1 + 90% top-3 indicates
  the frontend math is correct AND the model is mostly responding to
  the acoustic content, with the residual error attributable to the
  expected TTS-vs-mic distribution gap.

- **Fail (< 70% top-1 or < 90% top-3):** the frontend math is the
  primary suspect. Walk the failure-mode triage below.

- **Cargo-cult fail (top-1 looks plausible but is consistently shifted
  by one or two letters, e.g., A→B, B→C):** possible, but requires
  follow-up because it suggests a systematic offset (frame-count drift,
  or a confusion that's specific to mic acoustics). Note this in the
  Vikunja comment but do NOT block #52 close on it; track as a follow-up
  if it persists after #52's closure.

## Failure-mode triage

If top-1 accuracy is below the pass bar, the frontend math is the most
likely cause. Walk these in order:

### 1. Re-confirm the build picked up the changes

```sh
grep -n 'SPELL_FEAT_N_FRAMES' components/spell_common/include/spell_config.h
# expect: line ~112, value 80
grep -n 'spell_feat_norm_mean\|spell_feat_norm_std' \
    components/letter_recognizer/letter_recognizer.cpp
# expect: hit inside recognizer_task, between feat_extract_compute and the
# int8 quantize loop
```

If either is stale, you flashed a pre-#52 build. Rebuild and reflash.

### 2. Verify normalization arrays match the model handoff

```sh
make -C tools/gen_norm_arrays check
# Expected: round-trip check: OK (60 mean + 60 std floats match)
```

If this fails, the .npy files in `models/` and the embedded arrays in
`components/letter_recognizer/feat_norm.c` have drifted. Regenerate:

```sh
make -C tools/gen_norm_arrays gen
git diff components/letter_recognizer/feat_norm.{c,h}
```

If a diff appears, commit it as a #52 follow-up — the model handoff
shifted out from under the firmware.

### 3. Run the host-side feature pipeline reference

```sh
make -C tools/feat_extract_reference selftest
```

The self-test on a synthetic 1 kHz sine confirms the host-side pipeline
math hasn't drifted. If this fails (shape mismatch, frame-79 looks like
silence floor), the bug is in `feat_extract_reference.py` itself —
unlikely, since it self-tested at landing time, but worth ruling out.

### 4. Static-input round-trip (the AC item 4 second half)

The alphabet-sweep test fails when EITHER the firmware frontend OR the
acoustic environment is wrong. The host round-trip isolates frontend
math from acoustics by feeding both pipelines the same fixed PCM and
comparing the int8 input tensor.

**Setup (HITL-shaped, requires hardware):**

a. Generate a fixed-known WAV — e.g., from the `--selftest`'s synthetic
   1 kHz tone, or from a TTS render of "A" using the upstream training
   project's TTS pipeline, or from a quiet recording you've already
   classified as A on the host.

b. Add a temporary serial-dump hook to `recognizer_task` (do not
   commit — bench-only patch):

   ```cpp
   // After the int8 quantize loop, before Invoke():
   if (n_pre == 0) {
       ESP_LOGI(TAG, "int8_input dump (4800 B):");
       for (int i = 0; i < SPELL_FEAT_N_ELEMENTS; i += 32) {
           printf(" ");
           for (int j = 0; j < 32 && i+j < SPELL_FEAT_N_ELEMENTS; j++) {
               printf(" %4d", (int)input_data[i+j]);
           }
           printf("\n");
       }
   }
   ```

c. Pipe the WAV's PCM into the segmenter via the audio_dump back-fill
   hook (or for the simplest variant, swap the segmenter input for a
   ROM-baked PCM blob and force-emit one utterance event with that
   buffer).

d. On the host, install a TFLite runtime once:

   ```sh
   pip install tflite-runtime
   ```

   Then run:

   ```sh
   python3 tools/feat_extract_reference/feat_extract_reference.py \
       --wav /path/to/fixed.wav --quantize --invoke
   ```

   The host's `top-5: …` line is the comparison target.

e. Compare the host's int8 input tensor against the firmware-dumped
   int8 input tensor element-by-element. Allow ≤ 2 LSB drift per cell
   (the FFT backend difference accounts for this); anything larger is
   evidence of frontend math drift.

f. Compare the host's top-1 letter against the firmware's top-1 for
   the same WAV. Match → frontend math is correct, the alphabet-sweep
   accuracy gap is acoustics-only (track as a separate slice). Mismatch
   → frontend math is wrong somewhere; walk back through frame count,
   normalization, and pre-emphasis until they match.

### 5. Last-resort: regenerate from the model artifacts

Suspect the model handoff itself drifted (.tflite calibrated against a
different normalization than `models/norm_*.npy`):

```sh
parttool.py read_partition --partition-name model --output /tmp/model.tflite
diff /tmp/model.tflite models/flat26_baseline_int8.tflite
```

If they differ, the partition was flashed from a stale .tflite. Reflash
with the current `models/flat26_baseline_int8.tflite`. If they match,
the model handoff is consistent — rare to be the cause but cheap to
rule out.

## Recording the result

When the alphabet sweep produces a passing score, post a Vikunja comment
on #52 (already marked done) with the measurement summary:

```
HITL bench acceptance run 2026-MM-DD:
- top-1: NN/26 = NN%
- top-3: NN/26 = NN%
- failures (top-1): X spoken → Y predicted (recurrence: m runs of n)
- environment: <quiet office | TV background | …>
- mic distance: ~NN cm
- run-twice agreement: NN%

Conclusion: AC item 5 satisfied.
```

If the run failed, the comment captures the same data but with the
failure-mode triage outcome instead of "satisfied", and the task can be
re-opened (`done: false`) so the regression is tracked rather than
forgotten.

## What this acceptance does NOT cover

- **Real-mic decoder accuracy.** The decoder turns top-K letter events
  into a resolved word; that's a separate concern with its own
  acceptance (per ADR-0004's "QA fixture re-recorded on-device"
  approach). The alphabet-sweep above tests letter-level inference
  only.

- **W-recovery.** The W-trigger predicate (#28) is in place, but the
  full W-recovery integration (#30) holds the U emission and re-runs
  inference on the merged PCM. None of that fires unless #30 lands.
  The alphabet sweep should still produce a sensible top-K for a
  spoken "W" (likely top-1 = "U" with low confidence, top-2 or top-3
  picking up the W tail) — that's the trigger condition, not a failure.

- **Normalization-distribution shift over time.** If the model is
  retrained on new data, `models/norm_*.npy` and
  `components/letter_recognizer/feat_norm.c` need to be regenerated
  in lockstep with the new .tflite. The `make -C tools/gen_norm_arrays
  check` round-trip catches script drift but not training-distribution
  drift; only re-running the bench acceptance against the new model
  catches that.

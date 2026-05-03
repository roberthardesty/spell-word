# Spell-Word MVP Firmware — Implementation Plan

## 0. Scope and definition of done

**MVP success criterion.** A child holds the device near a book, presses the button, hears a confirmation tone, spells a word ("C... A... T"), pauses, and hears the device speak the word back through a speaker. End-to-end, on an ESP32-S3 DevKitC-1 with breadboarded mic + amp + speaker + button + switch. Battery and final enclosure are deferred.

**Explicit non-goals for MVP.**

- Custom PCB. Bring-up runs on the DevKitC-1 with breakouts.
- Battery operation, charge management, power optimization. Powered from USB.
- Final enclosure. Physical button + switch wired to the dev kit are sufficient.
- Training the DS-CNN. The trained, int8-quantized model and its empirical confusion matrix `M` are upstream inputs.
- 3,000-word corpus. Bring up with a 50–200 word smoke-test corpus; the production corpus build pipeline is in scope but the corpus content can be small.
- Field testing with children. Bench testing with adult-spelled words is sufficient to call MVP done; the Risks section flags this.

**Preconditions delivered by other workstreams.**

- `model.tflite` — int8-quantized DS-CNN, `[1, 79, 20, 3]` input (MFCC + Δ + ΔΔ), 26-class softmax (or logits).
- `confusion_matrix.bin` — 26×26 row-major float32, where `M[j, i] = P(true=i | top1=j)`. ~2.7 KB.
- `dictionary.bin` — N-word dictionary with per-word log-frequency. For MVP, 50–200 words is enough; the on-device data structures are sized for ≤3,000.
- `corpus.bin` — Opus-encoded word recordings + index, packed by the offline corpus build tool.

If any of these are not ready when firmware reaches the integration phase, Section 9 lists what to stub.

---

## 1. Architecture summary

A single binary, ESP-IDF v5.3.5 target `esp32s3`. The runtime has six long-lived FreeRTOS tasks plus an event-driven UI/state machine.

```
ICS-43434 mic ──I2S RX──┐
                        ▼
              ┌──────────────────┐
              │  capture_task    │  Core 0, prio 10
              │  256-frame reads │
              │  32→16 shift     │
              │  fan-out         │
              └────────┬─────────┘
                       │ StreamBuffer (PSRAM, ~20 KB)
                       ▼
              ┌──────────────────┐
              │  segmenter_task  │  Core 0, prio 8
              │  energy VAD      │  (frame-rate, low CPU)
              │  emits utterance │
              │  windows         │
              └────────┬─────────┘
                       │ Queue<utterance_t>
                       ▼
              ┌──────────────────┐
              │  inference_task  │  Core 1, prio 5
              │  feat extract    │  (mel → MFCC → Δ + ΔΔ)
              │  TFLM Invoke     │
              │  top-K + posts   │
              └────────┬─────────┘
                       │ esp_event: FW_EVENT_LETTER_TOP_K
                       ▼
              ┌──────────────────┐
              │  decoder_task    │  Core 1, prio 4
              │  posterior asm   │
              │  flat-list score │
              │  edit-distance   │
              │  abstention      │
              └────────┬─────────┘
                       │ esp_event: FW_EVENT_WORD_RESOLVED
                       ▼
              ┌──────────────────┐
              │  playback_task   │  Core 0, prio 6
              │  Opus decode     │
              │  I2S TX → amp    │
              └──────────────────┘

              ┌──────────────────┐
              │  ui_task         │  Core 0, prio 9
              │  button debounce │
              │  state machine   │
              │  tone playback   │
              └──────────────────┘
```

**State machine (owned by `ui_task`).**

```
IDLE      ──button press──► ARMED                      (enqueue confirmation tone)
ARMED     ──tone playback complete──► ARMED            (seg_active=true latched here)
ARMED     ──first letter onset──► SPELLING
SPELLING  ──early-commit predicate met──► RESOLVING    (see ADR-0001)
SPELLING  ──end-of-word silence (1.2 s)──► RESOLVING
SPELLING  ──button press──► IDLE                       (cancel, no tone)
SPELLING  ──timeout (10 s no audio)──► IDLE
RESOLVING ──word found──► PLAYING                      (resolved tone, then decode + I2S TX)
RESOLVING ──abstain──► IDLE                            (play error chirp)
PLAYING   ──playback complete──► IDLE
PLAYING   ──button press──► ARMED                      (cancel + re-arm; mute amp BEFORE I2S disable)
ARMED     ──button press──► (ignored)
RESOLVING ──button press──► (ignored)
```

`seg_active` is gated to false until the confirmation tone finishes playing — the segmenter would otherwise hear its own tone via mic feedback and may detect a spurious onset that bleeds into the user's first letter. The cost is that the pre-roll ring is empty when the segmenter activates (`seg_reset` runs every inactive frame), so the very first letter has no leading silence in its 800 ms window. Tolerable for MVP; see §12.F.

Capture continues to flow into the StreamBuffer at all times; the segmenter only emits utterances while the state machine is in `ARMED` (post-tone) or `SPELLING`. This is simpler than gating the I2S driver and avoids click/pop artifacts on resume.

**Component layout** (mirrors the ESP32-S3 doc; new components italicized):

```
components/
  fw_common/        # fw_config.h, error helpers, event base decls
  audio_capture/    # I2S RX driver, fan-out, subscribers
  letter_classifier/
    feat_extract.[ch] # pre-emphasis, mel filterbank, MFCC (DCT-II), Δ, ΔΔ
    inference.cpp   # TFLM, energy gate, top-K event
  segmenter/        # NEW: energy VAD, utterance windows
  decoder/          # NEW: posterior assembly, flat-list scoring,
                    #      edit-distance, abstention, dictionary
  audio_playback/   # NEW: Opus decode, I2S TX, tone synthesis
  ui/               # NEW: button debounce, state machine
  corpus/           # NEW: corpus partition reader, word_id → bytes
main/
  app_main.c        # init ordering, event registration
```

---

## 2. Hardware bring-up (Phase 0)

The DevKitC-1 doesn't include a microphone or speaker; bring-up means wiring breakouts to specific GPIOs and recording the pinout in `fw_config.h` so it's the single source of truth.

**Parts list for the breadboard.**

| Function | Part | Notes |
|---|---|---|
| Microphone | ICS-43434 breakout (Adafruit / similar) | 24-bit I2S, SEL→GND for LEFT slot |
| Audio amp + DAC | MAX98357A breakout | I2S in, mono speaker out, ~3W class D |
| Speaker | 8 Ω 1–3 W, ~28–40 mm | Validate intelligibility; this is a flagged risk |
| Button | Momentary tactile | Active-low to GPIO with internal pull-up |
| Switch | SPST slide | Hard power gate from USB 5 V; firmware doesn't see it for MVP |

**Pin assignments (proposed; tune to dev board headers).** Add to `fw_config.h`:

```c
// I2S RX (mic) — already in ESP32-S3 doc
#define FW_I2S_RX_PORT        I2S_NUM_0
#define FW_I2S_RX_SD_PIN      4
#define FW_I2S_RX_SCK_PIN     5
#define FW_I2S_RX_WS_PIN      6

// I2S TX (amp/speaker) — pick free pins on the DevKitC-1
#define FW_I2S_TX_PORT        I2S_NUM_1
#define FW_I2S_TX_SD_PIN      7   // DIN
#define FW_I2S_TX_SCK_PIN     15  // BCLK
#define FW_I2S_TX_WS_PIN      16  // LRC

// Controls
#define FW_BUTTON_GPIO        9   // active-low, internal pull-up
#define FW_AMP_SHUTDOWN_GPIO  10  // optional: pull MAX98357A SD low to mute
                                  // when not playing, reduces hiss
```

**Bring-up checklist before writing any firmware logic.**

1. Flash a hello-world ESP-IDF app, confirm USB-serial monitor.
2. Wire mic, run a 10-second I2S RX dump to a fixed buffer, scp it off, view in Audacity. Confirm clean PCM, expected level around –30 to –40 dBFS for normal speech.
3. Wire amp + speaker, play a 1-second 1 kHz sine wave generated in firmware. Confirm audible, not distorted, no DC offset on idle.
4. Wire button, log press/release events via GPIO ISR. Confirm clean edges (debounce in software).
5. Save a "rig photo" and the pin assignments to the repo so future-you doesn't rewire the board.

This is a half-day to one-day phase. It looks like padding but unblocks every subsequent phase, and skipping it is the most reliable way to lose two days chasing a swapped wire later.

---

## 3. Implementation phases

The phases are sequenced so each phase has a testable demo at the end. The intent is that you can stop at the end of any phase and have a coherent, working slice of the system.

### Phase 1 — Audio capture pipeline (~2 days)

**Goal.** I2S RX → 16 kHz mono 16-bit PCM → fanned out to one or more PSRAM StreamBuffer subscribers. Verified at 32 kB/s with no dropouts.

This is essentially the ESP32-S3 doc Sections 1–3 transcribed into source. There is no novel work here.

**Deliverables.**

- `components/audio_capture/` with `audio_capture_init`, `audio_capture_subscribe`, `audio_capture_start`, `audio_capture_read`.
- 32→16 conversion via `>> 16`.
- Per-subscriber overrun counters logged once per second.
- A `debug_dump` subscriber that, when `CONFIG_FW_DEBUG_AUDIO_DUMP=y`, writes the last N seconds of PCM to a circular buffer accessible over a serial command. Invaluable for every later phase.

**Acceptance test.** Boot device, observe in serial: `audio_cap: 32000 B/s, 0 overruns` for 60 seconds. Capture a known signal (whistle, finger snap) and confirm it appears in `debug_dump`.

### Phase 2 — Log-mel front-end and DS-CNN inference (~3 days)

**Goal.** Given a 9,600-sample window in PSRAM, produce 26 softmax probabilities. Top-K event posted on `FW_INFERENCE_EVENT`.

Again, the ESP32-S3 doc Sections 5–7 are the reference. The only deviations from that doc:

- The inference task's input is *not yet* a continuous 800 ms stream from the StreamBuffer. For Phase 2 the input is a fixed-length window passed in via a queue. This decouples the front-end from segmentation and lets us bring up inference with canned audio.
- A debug command (`infer_replay <wav_path>` over serial) replays a pre-recorded WAV through the inference path, so the model can be validated without needing the segmenter to work first.

**Deliverables.**

- `components/letter_classifier/feat_extract.[ch]` — pre-emphasis on-the-fly during framing, mel filterbank → log → DCT-II for MFCC, then Δ and ΔΔ across time. Output as `[n_frames, n_mfcc, 3]` flat NHWC. Internal-SRAM/PSRAM allocation strategy carried over from the EARS log_mel:
  - FFT buffer in internal SRAM (`MALLOC_CAP_INTERNAL`).
  - Mel filterbank in PSRAM.
  - Sparse `mel_bound_t` array for the inner-loop optimization.
  - DCT matrix (n_mfcc × n_mels = 20×40 floats ≈ 3.2 KB) in internal SRAM.
  - `floor_val = 1e-10` in the log compression.
- `components/letter_classifier/inference.cpp`:
  - `MicroMutableOpResolver<9>` with the exact op set.
  - Model load from custom data partition `"model"` (subtype 0x80), with fallback "no model present" log line so a fresh device boots cleanly.
  - 512 KB tensor arena in PSRAM, log actual usage after `AllocateTensors`.
  - Per-tensor int8 quantize/dequantize using tensor metadata.
  - Top-K = 5 (the Tech Description default; the ESP32-S3 doc shows 3, but the decoder benefits from more candidates and the cost is trivial).
  - RMS energy gate at –45 dBFS, threshold in `fw_config.h`.
  - Hot-reload via `inference_reload_model()` flag check at top of loop.
- A small canned test set: 5–10 WAVs of clearly-spelled letters in `test/letters/`, each labeled. The replay command exercises the inference path against these.

**Acceptance test.**

1. `infer_replay test/letters/B.wav` produces top-1 = `B` with probability > 0.5 (assuming reasonable training).
2. `infer_replay test/letters/silence.wav` is rejected by the energy gate; no event posted.
3. End-to-end window timing logged: `feat ~50 ms, invoke ~250 ms` (extrapolated from EARS log-mel; the DCT and delta passes add ~5 ms total). Remeasure on actual model.
4. Hot-reload test: write a new model to the partition via `esptool`, send `infer_reload`, verify next invoke uses the new tensors.

### Phase 3 — Energy-VAD letter segmentation (~3 days)

**Goal.** A continuous PCM stream is consumed and emits discrete utterance windows, one per spelled letter. End-of-word is signaled by a sustained silence event.

This is the highest-risk component, per the Tech Description. The MVP implementation is conservative: better to over-segment and let the decoder handle spurious utterances than to merge two letters.

**Algorithm.**

The segmenter consumes 16 ms frames (256 samples) from the audio_capture StreamBuffer and maintains state:

```c
typedef enum {
    SEG_IDLE,            // waiting for onset
    SEG_IN_LETTER,       // currently capturing
    SEG_POST_LETTER,     // brief silence after letter; might be inter-letter gap or end-of-word
} seg_state_t;
```

Per-frame RMS energy is computed (cheap; no FFT). Smoothed via short EMA (~3 frames) to suppress single-frame transients. Onset/offset thresholds use hysteresis: enter `IN_LETTER` when smoothed energy crosses `FW_VAD_ON_DBFS = -38 dBFS`, leave when it drops below `FW_VAD_OFF_DBFS = -42 dBFS` for at least `FW_VAD_OFF_FRAMES = 5` frames (~80 ms).

**Tunable parameters (all in `fw_config.h`).**

```c
#define FW_VAD_ON_DBFS               -38.0f   // letter onset threshold
#define FW_VAD_OFF_DBFS              -42.0f   // letter offset threshold
#define FW_VAD_MIN_LETTER_MS         150      // reject sub-150 ms blips
#define FW_VAD_MIN_GAP_MS            80       // merge if next onset within 80 ms
#define FW_VAD_END_OF_WORD_MS        1200     // silence → word complete
#define FW_VAD_MAX_LETTER_MS         800      // force split if a "letter" exceeds this
```

**Utterance window construction.** When a letter offset is confirmed, the segmenter emits a `utterance_t` containing the PCM window padded to `FW_INFERENCE_WINDOW_SAMPLES = 12800` (800 ms):

- Center the detected speech in the window when possible.
- If the letter is shorter than 800 ms, zero-pad before and after symmetrically.
- If longer than 800 ms (rare; `FW_VAD_MAX_LETTER_MS` should prevent this), take the trailing 800 ms slice (preserves the offset, which carries the most discriminative information for stop consonants).

This shape is critical: the DS-CNN was trained on 800 ms windows, so the segmenter's output must match that distribution. Pre-roll is preserved by maintaining a small ring buffer (~200 ms = 6.4 KB) of the most recent samples, so the segmenter can emit a window that includes the moment *before* the detected onset.

**Deliverables.**

- `components/segmenter/segmenter.[ch]` with init/start and an output `QueueHandle_t` of `utterance_t` records.
- The segmenter is gated by a `seg_active` flag set by `ui_task`. When inactive, frames are read and discarded (state stays `IDLE`).
- An end-of-word event posted to `esp_event` when sustained silence is detected during `SEG_POST_LETTER`.
- Debug logging: per-utterance, log onset frame, duration, RMS dBFS, and frame counts.

**Acceptance test.**

- Speak "C-A-T" with clear pauses near the mic: segmenter emits 3 utterances and one EOW event within ~5 s.
- Speak quickly with run-together letters: segmenter may merge or split incorrectly — log this as known behavior, validate that the decoder's edit-distance handling can recover.
- Sustained silence before any letter: no utterances emitted, state stays `IDLE`.
- Coughing / non-speech transient: ideally emits one spurious utterance which the decoder's "other" bucket and edit-distance penalty will absorb.

### Phase 4 — Press-to-arm UX and tone playback (~1.5 days)

**Goal.** Button press flips the state machine into `ARMED`, plays a confirmation tone, and starts the segmenter. Tone playback is the same I2S TX path used by Phase 6's word playback, so we build it now and reuse the plumbing later.

**Tone synthesis.** Three short tones, generated in firmware (no flash audio):

| Event | Tone | Duration |
|---|---|---|
| Confirm (armed) | 880 Hz sine | 120 ms |
| Resolved (about to speak word) | 1320 Hz sine | 80 ms |
| Error (abstain) | 200 Hz square + noise | 250 ms |

Pre-compute one cycle of each waveform at boot, scale to int16, and stream into the I2S TX channel from RAM. No decoder needed. This validates the I2S TX path before we add Opus complexity.

**Deliverables.**

- `components/audio_playback/audio_playback.[ch]` with `playback_init`, `playback_play_tone(tone_id)`, `playback_play_word(word_id)`, `playback_cancel()`, and a queue-based playback worker. The worker is the only writer to the I2S TX channel — anything that wants to play audio enqueues it, never directly drives I2S. `playback_cancel()` is **cooperative**: sets a flag, the worker checks it between Opus frames (Phase 6) and between I2S writes; on cancel, the worker pulls `SPELL_AMP_SHUTDOWN_GPIO` low *first*, then disables the I2S TX channel, then frees the in-flight PCM buffer. Order matters — flushing I2S while the amp is still enabled produces a hard click.
- `components/ui/ui.[ch]` with button ISR + debounce (FreeRTOS timer-based, 30 ms), state machine implemented as a switch over `enum ui_state_t`, and `ui_event_t` queue.
- The state machine sets `seg_active=true` only on the `playback_complete` event for the confirmation tone, not on the button press itself. This costs a ~120 ms "deaf window" after the press; the alternative (activating the segmenter immediately) lets the device's own tone bleed into the first letter via mic feedback.
- Button presses in non-IDLE states: in `ARMED` and `RESOLVING` the press is ignored (those states are <150 ms; debounce filters most accidental ones anyway). In `PLAYING` the press triggers `playback_cancel()` followed by an immediate transition to `ARMED` with a fresh confirmation tone — the cancel-and-re-arm UX, not just cancel-to-IDLE.
- App-level `esp_event` registrations in `app_main.c` so the state machine sees: button press, EOW, **early-commit window** (Phase 5), top-K, word resolved/abstain, playback complete.

**Acceptance test.**

- Press button → confirm tone plays → `seg_active=true` only after tone completes (verify via log).
- Speak nothing for 10 s → timeout → `seg_active=false`, return to idle.
- Press during `SPELLING` → cancel, return to idle, no tone.
- Press during `PLAYING` → playback stops within ~10 ms, no audible click, fresh confirmation tone follows, segmenter ready for new spelling.

### Phase 5 — Confusion-aware word resolution (~3 days)

**Goal.** Given a sequence of top-K events ending in EOW, output a word ID or abstain. This is the Tech Description's core algorithmic content (Sections "Confusion-Aware Word Resolution" and "Length-Mismatch Robustness").

**Data structures loaded at boot.**

- `M`: 26 × 26 float32 from `confusion_matrix.bin` partition (~2.7 KB). Loaded into PSRAM. Each row sums to 1.0; assert this at load time.
- `dictionary`: array of `{word_id, length, letters[16], log_freq}`. For MVP, 50–200 entries; production, ~3,000. Sorted by length so the scorer can iterate only same-length entries first, then ±1 and ±2 for edit-distance. Total size at 3,000 words: < 100 KB.

**Per-utterance posterior (`p_i(c)`).**

```
top1 = top_k[0].letter_index
For each candidate letter c:
    if c is in top_k:
        net_prob[c] = top_k[k].probability     // renormalized over K letters + "other"
    else:
        net_prob[c] = remaining_mass / 21      // uniform over the rest
mat_prob[c] = M[top1, c]                       // confusion-matrix row
posterior[c] = (1 - alpha) * net_prob[c] + alpha * mat_prob[c]
```

`alpha` is in `fw_config.h`, default `0.15`. Tunable.

**Flat-list scoring.**

For each candidate word `w` of matching length:

```
score(w) = sum over i of log(posterior_i(w[i])) + lambda * log_freq(w)
```

`lambda` default `0.5`. Both `alpha` and `lambda` should be tuned on a small validation set; don't burn engineering time tuning until end-to-end works.

**Edit-distance decoding.**

For each candidate word `w` of length `len(utterances) ± 2`:

- Run a banded Needleman-Wunsch DP, where the substitution score for aligning utterance `i` to word letter `w[j]` is `log(posterior_i(w[j]))`.
- Insertion penalty (extra utterance, e.g. cough): `FW_EDIT_INS_PENALTY` (default `-2.0`).
- Deletion penalty (missed letter): `FW_EDIT_DEL_PENALTY` (default `-2.5`).
- Score is max alignment score + `lambda * log_freq(w)`.

With ≤2 edits, ~3,000 words, ~5 letters average, this runs in well under 10 ms. No optimization needed.

**Abstention.**

```
if best.score - runner_up.score < FW_DECISION_MARGIN:
    abstain
```

Default margin: `2.0` (in log-prob space; tunable). Better to abstain than say the wrong word.

**Aggressive early-commit.** See ADR-0001 for the architectural decision. The decoder evaluates a three-clause predicate after every top-K event and on receipt of a new `SPELL_EVENT_EARLY_COMMIT_WINDOW` segmenter event (fired once at `SPELL_EARLY_MIN_SILENCE_MS = 500 ms` post-letter silence). If the predicate passes, the decoder commits early and posts `WORD_RESOLVED` without waiting for full EOW.

```
score(best)        - score(runner_up_same_length) > SPELL_EARLY_MARGIN_RUNNERUP   (default 3.0)
AND  score(best)   - score(best_longer_candidate) > SPELL_EARLY_MARGIN_LONGER     (default 2.0)
AND  silence_since_last_letter_ms                 ≥ SPELL_EARLY_MIN_SILENCE_MS    (default 500)
```

The third clause is load-bearing: without a silence floor, a valid prefix of a longer word (BAN inside BANK) wins on 0-edit-vs-1-edit-deletion because `SPELL_EDIT_DEL_PENALTY = -2.5` exceeds typical log-frequency separation, and the decoder systematically truncates users mid-word. The second clause (margin over best longer candidate) is what makes the predicate aware of words the user might still be spelling.

The decoder logs every predicate evaluation with the per-clause result for QA visibility:

```
decoder: early-commit eval @ letter 3: best=BAT score=-2.1 ru=BAR score=-3.4
         margin_ru=1.3 (need 3.0 ✗) margin_longer=BARS=-3.8 → 1.7 (need 2.0 ✗)
         silence=520ms (need 500 ✓) → wait
```

A new letter onset cancels any pending early-commit window; the segmenter automatically re-fires the event after the next letter's silence reaches the threshold. The race between the in-flight inference for the most recent letter and the early-commit-window event is bounded: at 500 ms silence floor and ~30–40 ms inference latency, the top-K event always lands at the decoder before the window event fires. If `SPELL_EARLY_MIN_SILENCE_MS` is later tuned below ~50 ms, the decoder will need to wait for inference drain explicitly.

**New segmenter event.** `SPELL_EVENT_EARLY_COMMIT_WINDOW = 3` (sibling to `SPELL_EVENT_END_OF_WORD`). Empty payload — the decoder has all the per-letter context already. Implementation: add `early_commit_frames` to `seg_config_t`, add a `SEG_EVT_EARLY_COMMIT_WINDOW` enum value to the core, fire it once when post-letter frame count crosses the threshold during `SEG_POST_LETTER`. Update `tools/segmenter_replay/test_segmenter_core.c` to cover the new event.

**Deliverables.**

- `components/decoder/decoder.[ch]` with init (loads `M` and dictionary), `decoder_on_topk(event)`, `decoder_on_eow()`, and emits `FW_EVENT_WORD_RESOLVED { word_id }` or `FW_EVENT_WORD_ABSTAIN`.
- A bounded internal buffer of `MAX_LETTERS_PER_WORD = 16` top-K events. Overflow → abstain (an unreasonably long spelling).
- A `decode_replay <letter_sequence>` serial command that takes a comma-separated list of (letter, prob, ...) tuples and runs the decoder, for unit testing without audio.
- Unit test fixtures: at least 10 hand-crafted top-K sequences with expected resolved word, including:
  - Clean spelling: `[B@0.9, A@0.85, T@0.92]` → `BAT`.
  - One wrong top-1 (D instead of B): the dictionary + confusion matrix should still pin `BAT` over `DAT` (which isn't a word).
  - Spurious extra letter (cough): `[B, X@0.3, A, T]` → `BAT` via 1-edit deletion.
  - Genuinely ambiguous: `[?, ?, ?]` with flat distributions → abstain.

**Acceptance test.** All decoder unit tests pass, and the live system, given the canned WAVs from Phase 2 spelled into a known word, resolves it correctly.

### Phase 6 — Audio corpus, Opus decode, word playback (~4 days)

**Goal.** Given a `word_id`, decode its Opus payload from the corpus partition and play it through the I2S TX path.

This is the largest phase by elapsed time, mostly because of the corpus build pipeline and Opus decoder integration.

**Corpus build pipeline (offline, runs on dev machine).**

A standalone Python tool: `tools/build_corpus.py`.

```
input:  recordings/{word}.wav  (mono, 16 kHz or higher, may have silence)
        words.csv  (word, word_id, optional: speaker, gain_offset)
output: corpus.bin  (custom-format flat blob)
        corpus_index.bin  (word_id → byte_offset, byte_length)
```

Per-word steps:

1. Trim leading/trailing silence (–40 dBFS gate, 50 ms tail).
2. Loudness-normalize to a fixed LUFS target (e.g. ITU-R BS.1770 –20 LUFS).
3. Resample to 16 kHz mono if needed.
4. Encode with `opusenc --bitrate 24 --comp 10 --framesize 60`.
5. Append the encoded bytes to `corpus.bin`, record offset/length in the index.

Average per-clip size at 24 kbit/s: ~3 KB for a one-second word. 3,000 words × 3 KB ≈ 9 MB — *exceeds* the 5 MB budget in the Tech Description, so for production tune down: 16 kbit/s + careful trimming gets to ~2 KB/clip and 6 MB total, or use an even narrower band. For MVP (50–200 words) the budget is irrelevant; aim for clear playback first, optimize encoding later.

The corpus + index are flashed to two custom data partitions (`"corpus"`, `"corpus_idx"`) at build time via `partitions.csv`.

**On-device Opus decoder.**

Use libopus (`opus-1.4`) compiled as an ESP-IDF component, or [esp32-opus-decoder](https://github.com/espressif/esp-adf-libs) which is already pre-built for the platform. ADF brings the same dependency baggage flagged in the ESP32-S3 doc, so prefer building libopus directly:

- `components/libopus/` — vendored upstream sources, IDF `CMakeLists.txt`.
- `OPUS_FIXED_POINT` and `OPUS_ARM_PRESUME_AARCH64_NEON_INTR=0` (we're on Xtensa). The Xtensa optimizations in upstream Opus are limited; expect the decoder to run mostly in C. Decode of a 1-second 16 kHz mono Opus frame is ~30–80 ms on this MCU at 240 MHz. Decode-then-play is fine for MVP; streaming decode-and-play is a future optimization.

**Playback path.**

```c
playback_play_word(word_id) {
    (offset, length) = corpus_index_lookup(word_id);
    pcm_buf = malloc_psram(MAX_PCM_PER_WORD);  // ~32 KB for 1 s @ 16 kHz
    opus_decode_partition_range(offset, length, pcm_buf);
    i2s_channel_write(s_tx_chan, pcm_buf, pcm_bytes, ...);
    free(pcm_buf);
}
```

The MAX98357A enable pin (`FW_AMP_SHUTDOWN_GPIO`) is asserted high a few ms before playback and lowered ~20 ms after, to suppress the "click" at idle.

**Deliverables.**

- `tools/build_corpus.py` plus a 50-word smoke-test corpus (use any clear recording — yourself, a TTS engine, whatever — for bring-up).
- `components/libopus/` (vendored).
- `components/corpus/corpus.[ch]` — `corpus_init`, `corpus_lookup(word_id) → {offset, length}`, `corpus_decode(word_id, pcm_buf) → length`.
- `components/audio_playback/` extended with `playback_play_word(word_id)`.
- I2S TX channel setup mirroring the I2S RX setup from Phase 1, on `I2S_NUM_1` to keep the channels independent.
- Wiring in `ui_task`: on `FW_EVENT_WORD_RESOLVED`, play the resolved tone, then enqueue `playback_play_word`; on completion, return to `IDLE`.

**Acceptance test.**

- `play_word CAT` over serial → CAT plays cleanly through the speaker.
- End-to-end: button → "C-A-T" → cat audio plays. This is the MVP demo.

### Phase 7 — Integration, tuning, demo (~2–3 days)

**Goal.** Iterate end-to-end on the bench until it works reliably for an adult speaker spelling 20 in-vocabulary words at conversational distance.

**Tuning sweeps (in priority order).**

1. **VAD thresholds** (`FW_VAD_ON_DBFS`, `FW_VAD_OFF_DBFS`, `FW_VAD_END_OF_WORD_MS`). Most bring-up issues will be segmentation issues. Use `debug_dump` to capture the audio of a failed spelling and replay it offline against different VAD parameters before re-flashing.
2. **Energy gate** (`FW_ENERGY_GATE_DB`). Should match the segmenter's onset threshold or be slightly more permissive — letters that segmented but failed energy gate are a wasted opportunity.
3. **Decoder mixing weight `alpha` and frequency weight `lambda`**. Sweep on a small validation set (recorded spellings of 20 known words). The Tech Description guides: `alpha` should be small.
4. **Edit penalties and abstention margin**. Tune so that 1-edit recovery beats 0-edit decode into a wrong word, but not 0-edit decode into the right one. This is the most subtle calibration; don't over-tune without a real validation set.

**Demo script.** A 30-second video of: power on → press → spell "CAT" → cat plays → press → spell "DOG" → dog plays → press → spell garbage → error chirp. This is the sign-off artifact for MVP.

---

## 4. Source-tree skeleton

Useful as a starting commit so the directory structure is locked in before code lands.

```
spell-word-fw/
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults
├── main/
│   ├── CMakeLists.txt
│   └── app_main.c
├── components/
│   ├── fw_common/
│   │   ├── CMakeLists.txt
│   │   └── include/
│   │       ├── fw_config.h
│   │       └── fw_events.h
│   ├── audio_capture/
│   │   ├── CMakeLists.txt
│   │   ├── audio_capture.c
│   │   └── include/audio_capture.h
│   ├── letter_classifier/
│   │   ├── CMakeLists.txt
│   │   ├── inference.cpp
│   │   ├── feat_extract.c
│   │   └── include/{inference.h, feat_extract.h}
│   ├── segmenter/
│   ├── decoder/
│   ├── audio_playback/
│   ├── corpus/
│   ├── ui/
│   └── libopus/
├── tools/
│   ├── build_corpus.py
│   └── make_partitions.py
├── test/
│   ├── letters/        # canned WAVs for inference replay
│   └── words/          # canned audio of full spellings
└── recordings/         # raw word audio for corpus build
```

`partitions.csv` (relevant entries):

```
# Name,    Type, SubType,   Offset,   Size
nvs,      data, nvs,        0x9000,   0x6000
phy_init, data, phy,        0xf000,   0x1000
factory,  app,  factory,    0x10000,  0x200000
model,    data, 0x80,       ,         0x40000  # 256 KB
matrix,   data, 0x81,       ,         0x4000   # 16 KB (3 KB used)
dict,     data, 0x82,       ,         0x20000  # 128 KB
corpus,   data, 0x83,       ,         0x800000 # 8 MB on N16R8, tighten on N8R8
corpus_idx, data, 0x84,     ,         0x10000  # 64 KB
```

---

## 5. `fw_config.h` — single source of truth

Every tunable referenced above lives in one header. This matters more than it sounds: drift between firmware constants and training-pipeline constants silently degrades model accuracy. The training tool should ingest this header (or a generated JSON sidecar) and use the same values.

Required entries (consolidated):

```c
// Audio format
#define FW_SAMPLE_RATE              16000
#define FW_BITS_PER_SAMPLE          16
#define FW_CHANNELS                 1
#define FW_BYTES_PER_SAMPLE         2

// Capture
#define FW_CAPTURE_CHUNK_FRAMES     256
#define FW_CAPTURE_MAX_SUBS         4

// Inference window — 800 ms, matches training pipeline's max_duration_ms
#define FW_INFERENCE_WINDOW_SAMPLES 12800
#define FW_INFERENCE_WINDOW_BYTES   25600
#define FW_INFERENCE_SUB_BYTES      26112

// Mel filterbank stage
#define FW_MEL_N_FFT                512
#define FW_MEL_HOP_LENGTH           160       // 10 ms
#define FW_MEL_WIN_LENGTH           320       // 20 ms
#define FW_MEL_N_MELS               40
#define FW_MEL_FMIN                 50.0f
#define FW_MEL_FMAX                 7600.0f

// MFCC + Δ + ΔΔ stage (DS-CNN expects this; not raw log-mel)
#define FW_MFCC_N_COEFFS            20
#define FW_MFCC_PRE_EMPHASIS        0.97f
#define FW_FEAT_N_CHANNELS          3         // MFCC + Δ + ΔΔ
#define FW_FEAT_N_FRAMES            79        // floor((12800 - 320) / 160) + 1
                                              // — TF-style center=False framing
#define FW_FEAT_N_ELEMENTS          (79 * 20 * 3)  // 4 740 floats per window

// Energy gate
#define FW_ENERGY_GATE_DB           -45.0f

// VAD
#define FW_VAD_ON_DBFS              -38.0f
#define FW_VAD_OFF_DBFS             -42.0f
#define FW_VAD_OFF_FRAMES           5
#define FW_VAD_MIN_LETTER_MS        150
#define FW_VAD_MIN_GAP_MS           80
#define FW_VAD_END_OF_WORD_MS       1200
#define FW_VAD_MAX_LETTER_MS        800

// Inference
#define FW_LETTER_TOP_K             5
#define FW_TENSOR_ARENA_SIZE        (512 * 1024)
#define FW_MODEL_PARTITION_LABEL    "model"
#define FW_INFERENCE_TASK_STACK     8192    // 8 KB; heavy buffers in PSRAM heap

// sdkconfig.defaults must include:
//   CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n
// Without this, a ~250 ms TFLM Invoke() on Core 1 trips the task watchdog.
// (Lesson carried over from the EARS POC.)

// Decoder
#define FW_DECODER_ALPHA            0.15f
#define FW_DECODER_LAMBDA           0.5f
#define FW_EDIT_INS_PENALTY         -2.0f
#define FW_EDIT_DEL_PENALTY         -2.5f
#define FW_DECISION_MARGIN          2.0f
#define FW_MAX_LETTERS_PER_WORD     16

// Aggressive early-commit (ADR-0001)
#define FW_EARLY_MIN_SILENCE_MS     500     // post-letter silence floor for early commit
#define FW_EARLY_MARGIN_RUNNERUP    3.0f    // margin over same-length runner-up
#define FW_EARLY_MARGIN_LONGER      2.0f    // margin over best longer candidate

// UI
#define FW_BUTTON_DEBOUNCE_MS       30
#define FW_ARM_TIMEOUT_MS           10000

// Pins (see Section 2)
// ...
```

---

## 6. Test strategy

**Unit tests** (esp-idf `unity` framework, run on-device):

- `decoder_test`: hand-crafted top-K sequences, expected resolutions. Covers clean, top-1-wrong, edit-distance, abstention paths.
- `feat_extract_test`: feed a known sinusoid, assert MFCC channel-0 energy lands in the expected DCT bins; assert Δ and ΔΔ are zero on a stationary signal and non-zero on a transient.
- `vad_test`: synthesized PCM with controlled energy envelopes, assert correct utterance count and durations. (Already implemented host-side in `tools/segmenter_replay/test_segmenter_core.c`.)

**Integration tests** (replay-driven, no live mic):

- `infer_replay <wav>`: WAV → feat extract → invoke → top-K. Compare to expected.
- `pipeline_replay <wav>`: WAV → audio_capture (synthetic source) → segmenter → inference → decoder → resolved word ID. Compare to expected. This is the highest-value test and should run on every CI build.

**Live tests** (manual, end-of-phase):

- 20-word fixed-list bench test at the end of Phase 7. Record per-word: success / wrong word / abstain. Target: ≥ 80% correct on adult speakers in a quiet room. Below that, the gating issue is almost certainly VAD or model accuracy, not the decoder.

**No CI for now.** The build is straightforward enough that local `idf.py build flash monitor` is fine for MVP. Add CI when the project survives MVP.

**Decoder hyperparameter tuning** (see ADR-0004). The C decoder is a port of the Python reference from the model-training project. The tuning loop is:

1. Record the QA set on Mac (16 kHz mono WAV per word, multiple speakers if available).
2. Run each WAV through the Python pipeline → top-K events per utterance + ground-truth word ID.
3. Sweep decoder parameters (`SPELL_DECODER_ALPHA`, `SPELL_DECODER_LAMBDA`, `SPELL_EDIT_INS_PENALTY`, `SPELL_EDIT_DEL_PENALTY`, `SPELL_DECISION_MARGIN`, `SPELL_EARLY_MARGIN_RUNNERUP`, `SPELL_EARLY_MARGIN_LONGER`) in Python against the fixture.
4. Bake tuned values into `spell_config.h`.
5. Maintain a host-side `tools/decoder_replay/` C build that consumes the same fixture format as the Python tool, so the C port can be regression-tested against the Python reference (same input → same resolved word, same scores within float tolerance).

A re-tuning pass is expected once device-mic recordings exist, because the front-end distribution shifts (Mac mic frequency response ≠ ICS-43434, room acoustics, distance to mouth). Keep the QA set's word list and script consistent across both recording rounds so the tuned-against-Mac and tuned-against-device parameter sets are directly comparable.

---

## 7. Risks and mitigations

The Tech Description's Risks section is correctly scoped; here are the MVP-specific framings.

| Risk | MVP impact | Mitigation |
|---|---|---|
| **VAD over- or under-segments** | Most likely failure mode of MVP. Children running letters together is hard. | Test with adult speakers first (slower, more deliberate). Tune VAD on recorded audio (use `debug_dump`). Edit-distance decoding absorbs ±2 errors. |
| **Model accuracy degrades on real-world audio** | The training reports 95% top-1; expect substantially worse on the dev rig. | Tolerable for MVP — the dictionary + confusion matrix are the safety net. If top-1 accuracy is below 50%, the project's foundation is broken and segmentation/decoder work is wasted; sanity-check this at end of Phase 2. |
| **Speaker too quiet or distorted** | Word is unrecognizable to the child even when correct. | Validate at end of Phase 0 with a 1 kHz tone. If MAX98357A + 8 Ω 1 W speaker is too quiet, swap in a 3 W speaker; the amp can drive it. |
| **Opus decode too slow / clicks during playback** | Word audio sounds bad. | Decode entire word into PSRAM before starting I2S TX (no streaming). 1-second clip = ~32 KB PCM in PSRAM, decoded in <100 ms — easily fits the user's expectation of <500 ms latency. |
| **PSRAM contention between tensor arena, mel filterbank, audio buffers** | Inference slowdown or playback dropouts. | Profile after Phase 6 integration. The arena (~250 KB used) + mel filterbank (40 KB) + corpus PCM buffer (32 KB) + StreamBuffer (~20 KB) is comfortably below 8 MB. |
| **OOV words** | Child spells a word not in the 200-word smoke corpus. | MVP: abstain. Production: same — the Tech Description's UX call is that abstention is the correct failure mode. |
| **Children's speech vs. adult speech** | The DS-CNN is presumed trained on children's audio; if not, accuracy will collapse with kid testers. | Out of MVP scope: bench-test with adult voices only. Children's-speech accuracy is a pre-deployment milestone. |

---

## 8. Timeline estimate

These are *implementation* estimates assuming one full-time engineer with ESP-IDF familiarity, working with the preconditions delivered, **and reusing the EARS POC firmware** (see `EARS-Reuse-Assessment.md`). Add 50% if learning ESP-IDF or TFLite Micro from scratch.

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Hardware bring-up | 1 day | 1 d |
| 1 — Audio capture pipeline (lift from EARS) | 0.5 day | 1.5 d |
| 2 — Log-mel + DS-CNN inference (port from EARS) | 1.5 days | 3 d |
| 3 — Energy-VAD segmentation | 3 days | 6 d |
| 4 — Press-to-arm UX + tones | 1.5 days | 7.5 d |
| 5 — Confusion-aware decoder | 3 days | 10.5 d |
| 6 — Corpus + Opus + word playback | 4 days | 14.5 d |
| 7 — Integration and tuning | 2 days | 16.5 d |

**~3.5 weeks** to MVP demo with EARS reuse (down from ~4 weeks net-new). The largest single risk to this estimate is Phase 3 (VAD) if real-world audio behaves badly; budget a 2-day overrun.

The EARS POC also de-risks the parts of the input pipeline that were "could surprise us" — TFLM on ESP32-S3, esp-dsp FFT performance, PSRAM allocation strategy, I2S timing. Remaining risk is concentrated in the segmenter, decoder, and corpus/playback path, each testable in isolation.

---

## 9. Stubs if preconditions slip

If the upstream model / matrix / dictionary / corpus aren't ready when needed, the MVP can still progress with these stubs.

- **No DS-CNN model.** Phase 2 can be validated with a TFLite "Speech Commands" 12-class model (Google's open dataset). It won't classify letters, but it exercises feat extract + TFLM + quantize/dequantize + top-K. Note such a stub model expects a different input shape, so this requires temporarily reverting `spell_config.h` to whatever the stub model wants (or switching to a stub model that happens to want `[1, 79, 20, 3]`). Substitute the real model when ready.
- **No confusion matrix.** Set `alpha = 0` in the decoder. Posterior reduces to the network's softmax. Slightly worse accuracy on confusable pairs; otherwise functional.
- **No dictionary.** Use a hand-typed 20-word dictionary and uniform `log_freq`. Sufficient for bench testing.
- **No corpus.** Use TTS-generated WAVs (any local TTS engine, espeak, even macOS `say -o`) as a placeholder, run them through `build_corpus.py`. Quality is poor but the pipeline is exercised.

---

## 10. Open questions to revisit before v1

These are explicitly out of MVP scope but should be on a v1 watch-list:

- Final part selection: N16R8 vs. N8R8 (depends on final corpus size + firmware footprint).
- Battery, charge management IC, power profile, expected runtime.
- Custom PCB layout, including amp shutdown wiring and mic placement relative to enclosure ports.
- Top-K size tuning (5 is the Tech Description's default; revisit after empirical data).
- Children's-speech data collection and model fine-tuning.
- OOV behavior: abstain vs. closest-in-vocab (the Tech Description names this as an explicit UX call).
- OTA mechanism for the model partition (constrained by ADR-0003: production is fully offline; any OTA path is a deliberate exception via test-build or USB).
- Loudness, mic placement, and SNR validation under realistic noise (TV in the background, sibling chatter).

---

## 11. Implementation status

What's currently committed to `/Users/roberthardesty/spell-word/`. Phases not listed here are pending.

### Phase 1 — Audio capture pipeline ✅

Lifted from the EARS POC with prefix-only renaming.

- `components/audio_capture/audio_capture.{c,h}` — I2S RX (ICS-43434, 32-bit LEFT slot), 32→16 conversion via `>> 16`, three-phase init lifecycle (init / subscribe / start), N-subscriber fan-out with PSRAM-backed StreamBuffers, non-blocking sends with per-subscriber overrun counts, capture task pinned to Core 0 priority 10. Logs `32000 B/s` once per second for verification.

### Phase 2 — Feature front-end + DS-CNN inference ✅

The mel filterbank stage was lifted from EARS; the front-end was then **rebuilt as a full MFCC + Δ + ΔΔ pipeline** after the training-project `config.yaml` revealed the model expects a 3-channel feature tensor, not raw log-mel. Section "Front-end correction" below describes what changed and why.

- `components/letter_classifier/feat_extract.c` — pre-emphasis (`y[n] = x[n] − 0.97·x[n−1]`) applied on-the-fly during framing, 512-pt esp-dsp FFT, 40-band mel filterbank with sparse bounds, log compression, **DCT-II orthonormal** to 20 MFCC coefficients per frame, 79 frames per 800 ms window, then time-derivatives (Δ and ΔΔ) computed with edge-replicated boundary handling and stacked as channels 1 and 2. Output tensor shape is `[79, 20, 3]` flat row-major NHWC. FFT buffer in internal SRAM, mel filterbank in PSRAM, DCT matrix in internal SRAM (~3.2 KB).
- `components/letter_classifier/inference.cpp` — `MicroMutableOpResolver<9>` (drops `Logistic`, keeps `Softmax`), model load from custom partition with all-`0xFF` empty-partition fallback, 512 KB tensor arena in PSRAM with `arena_used_bytes()` logging, per-tensor int8 quantize/dequantize using runtime scale + zero-point, hot-reload via flag check at top of loop, runtime detection of in-graph softmax vs. raw logits, output-shape sanity check (must be 26). Stat counters refactored to `std::atomic<uint32_t>` to silence `-Wvolatile` and to actually be safe across tasks.
- **Architectural change vs. EARS:** inference no longer subscribes to `audio_capture`. The segmenter pushes utterance buffers via `inference_submit_utterance(int16_t *pcm, size_t n_samples)`; ownership of the PSRAM buffer transfers on success and the inference task `heap_caps_free`s after processing.

#### Front-end correction (logged for posterity)

The first cut of the firmware computed raw log-mel features (40 coefficients per frame, single channel, 58 frames, 600 ms window) — derived from the ESP32-S3-DS-CNN technical document I started from. When the training project's `config.yaml` was shared, the actual model frontend turned out to be substantially different:

| Parameter | First cut | Per training config |
|---|---|---|
| Feature type | log-mel, 1 channel | MFCC + Δ + ΔΔ, 3 channels |
| Coefficients per frame | 40 (raw mel) | 20 (after DCT) |
| Window length | 25 ms (400 samples) | 20 ms (320 samples) |
| Frequency bounds | 0–8000 Hz | 50–7600 Hz |
| Inference window | 600 ms (9 600 samples) | 800 ms (12 800 samples) |
| Frames per window | 58 | 79 |
| Pre-emphasis | none | 0.97 |
| Implied input shape | `[1, 58, 40, 1]` | `[1, 79, 20, 3]` |

The fix replaced `log_mel.{c,h}` with `feat_extract.{c,h}`, updated the constants in `spell_config.h`, and adjusted `inference.cpp`'s tensor sizes. The deprecated `log_mel.{c,h}` files are kept in the tree only to preserve git history; they're no longer compiled. Two known mismatches remain that the user should verify against the training pipeline:

- **Mel scale formula.** The firmware uses HTK's `2595 · log10(1 + f/700)`. librosa defaults to the Slaney piecewise scale (different curve below 1 kHz). Either matches a common training pipeline; pick whichever the trainer used.
- **Frame count.** 79 assumes TF-style `center=False` framing. librosa's `center=True` produces 81. Verify against the model's actual `input_details[0]['shape']`.

### Phase 3 — Energy-VAD segmentation ✅

The full algorithm plus host-side tuning tools.

- `components/segmenter/segmenter_core.{c,h}` — pure C99, zero IDF dependencies, host-portable. EMA-smoothed RMS in dBFS, onset/offset hysteresis with N-frame offset confirmation, pre-roll ring buffer (~200 ms), min-letter rejection, max-letter force-split (residual segment continues capturing without re-arming pre-roll), end-of-word distinct from inter-letter silence (only fires from IDLE while `word_in_flight` is set), peak-dBFS tracking. Caller-allocated buffers — the core owns no memory.
- `components/segmenter/segmenter.c` — IDF wrapper. Owns the PSRAM buffers, runs the FreeRTOS task on Core 0 priority 8, dispatches utterance windows to `inference_submit_utterance` with center-pad (or trailing-slice for over-length letters) into the 800 ms inference window, posts `SPELL_EVENT_END_OF_WORD` events.
- `components/audio_dump/audio_dump.{c,h}` — continuous 10-second PSRAM ring driven by a low-priority drain task. `audio_dump_emit_b64(seconds)` freezes the drain briefly, base64-encodes the most recent slice (RFC 4648 alphabet, 76-char wrap), emits a framed payload over UART. `audio_dump_init(10)` is wired into `app_main.c`; the trigger is intentionally not yet bound — the UI state machine in Phase 4 will own it.

### Phase 3 tooling — host-side VAD tuning ✅

The iteration loop the plan called for.

- `tools/segmenter_replay/main.c` — host-build CLI that links `segmenter_core.c` directly. Reads a 16 kHz mono WAV, runs the VAD with overrideable thresholds, prints a per-event timeline, optionally writes each emitted utterance to `letter_NNN.wav`. Build with `make`; run with `./segmenter_replay [--on -36 --off -40 ...] input.wav`.
- `tools/segmenter_replay/wav.{c,h}` — minimal WAV reader/writer (16-bit mono 16 kHz only, rejects anything else).
- `tools/segmenter_replay/test_segmenter_core.c` — eight unit tests covering silence, sub-onset audio, single clean letter, three letters + EOW, sub-min-letter blip rejection, EOW gating on `word_in_flight`, max-letter force-split, invalid-config rejection. **All eight pass.** Run with `make test`.
- `tools/serial_to_wav.py` — Python receiver. Captures the framed audio_dump payload from a serial port (or stdin), tolerates interleaved `ESP_LOG` lines between base64 lines, reconstructs a 16 kHz mono WAV. End-to-end round-trip validated byte-exact.

### Project scaffold ✅

- `CMakeLists.txt` (top-level), `partitions.csv` (custom subtypes 0x80 model, 0x81 confusion, 0x82 dictionary, 0x83 corpus, 0x84 corpus_idx), `sdkconfig.defaults` (octal PSRAM, 16 MB flash, **`CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n`**, 240 MHz CPU).
- `components/spell_common/include/spell_config.h` — every tunable in one header per plan §5. Pin assignments, audio format, capture geometry, mel parameters, energy gate, VAD thresholds, inference task stack, decoder weights, UI timing.
- `components/spell_common/include/spell_events.h` — event base + payload types for what's currently wired up. Phase 4/5 declarations land alongside their producer components, intentionally.
- `main/app_main.c` — NVS init with auto-erase fallback, boot banner with chip rev / flash / PSRAM, factory reset (BOOT held 5 s at boot erases NVS), default `esp_event` loop, top-K + EOW handlers wired for bring-up logging, audio pipeline init in correct order.

### Verified end-to-end (host-side)

- Synthesized three 250 ms tone bursts with 400 ms gaps, ran through `segmenter_replay` → 3 letters emitted at ~340 ms duration each, peak ~−15 dBFS, 1 EOW event after 1500 ms trailing silence. Matches the algorithm's expected behavior.
- Synthesized a base64-framed UART log with 1500 known PCM samples, ran through `serial_to_wav.py` → byte-exact round-trip recovered.

### What remains

| Phase | Status |
|---|---|
| 0 — Hardware bring-up | **Pending — your end.** No firmware to write; ribbon-cabling the dev kit and confirming I2S RX/TX on a scope. |
| 4 — Press-to-arm UX + tones | Not started. Includes I2S TX path, tone synthesis, button debounce, state machine, `segmenter_set_active()` integration. |
| 5 — Confusion-aware decoder | Not started. |
| 6 — Corpus + Opus + word playback | Not started. |
| 7 — Integration and tuning | Not started. |

---

## 12. Things you need to do or verify

The firmware is software-complete through Phase 3 and the tooling chain is closed, but several preconditions need to be met on your end before a board can boot, classify a letter, and report a top-K event. Group A is hardware bring-up; Group B is the build and runtime smoke test; Group C is integrating the trained model; Group D is calibration. Group E is sanity checks I made assumptions about that you should confirm.

### A. Hardware bring-up (Phase 0 of the plan)

- [ ] Wire ICS-43434 breakout to the dev kit per `spell_config.h`: `SD→GPIO 4`, `SCK→GPIO 5`, `WS→GPIO 6`, `SEL→GND`, `VDD→3V3`, `GND→GND`. Confirm `SEL→GND` specifically — if it's tied high, the mic drives the RIGHT slot and the firmware will see silence.
- [ ] Wire MAX98357A breakout: `DIN→GPIO 7`, `BCLK→GPIO 15`, `LRC→GPIO 16`, `GAIN→GND` (default 9 dB) or float (12 dB), `SD→GPIO 10` (so firmware can mute the amp at idle), `VIN→5V` (USB rail), `GND→GND`, plus 8 Ω speaker on the screw terminals.
- [ ] Wire momentary push-button between `GPIO 9` and `GND`. Internal pull-up will hold the line high; press pulls it low. (The Phase 4 button code expects active-low.)
- [ ] Take a "rig photo" and commit the pin assignments next to the dev kit. The single most reliable way to lose two days is to rewire the board and forget which pin was which.
- [ ] Confirm the dev kit is N16R8 (16 MB flash, 8 MB octal PSRAM). N8R8 also works but `partitions.csv` will need the corpus partition shrunk before the corpus is loaded.

### B. Build and runtime smoke test

- [x] Install ESP-IDF v5.3.5 (the version the EARS POC was built against) per Espressif's docs. Confirm `idf.py --version` reports 5.3.5.
- [x] From `/Users/roberthardesty/spell-word/`, run `idf.py set-target esp32s3` then `idf.py build`. The build should succeed.
- [x] If `idf.py build` complains about a missing `esp-tflite-micro` or `esp-dsp` component: install via `idf.py add-dependency espressif/esp-tflite-micro` and `idf.py add-dependency espressif/esp-dsp`. (Both are referenced from `letter_classifier/CMakeLists.txt` REQUIRES.)
- [x] Flash and monitor: `idf.py flash monitor`. Expected initial log:
  - `spell:  Spell-Word firmware booting`
  - `spell:    psram: octal, configured via Kconfig` (warning if absent)
  - `audio_capture: I2S configured: 16000 Hz, 32-bit LEFT slot...`
  - `feat_extract: feat init: 512-pt FFT, 40 mels, 20 MFCC, 79 frames, 3 ch`
  - `inference: model partition is empty (erased) — awaiting OTA` (expected — no model flashed yet)
  - `segmenter: ready: on=-38.0 off=-42.0 off_frames=5 ...`
  - `audio_dump: ready: 10 s ring (320 KB PSRAM)`
  - `audio_cap: capture task started with 2 subscriber` (segmenter + audio_dump)
  - `audio_cap: 32000 B/s` (every second)
- [x] Speak letters into the mic. Confirm the segmenter logs onset/offset events (`letter #1: NNNN samples, NNN ms, peak -X.X dBFS`). The "no model loaded" warning is expected at this stage.
- [x] Build and run the host tools: `cd tools/segmenter_replay && make && make test`. Expect 8/8 tests pass.

### C. Model integration

The training-project `config.yaml` has been incorporated; the firmware front-end now produces MFCC + Δ + ΔΔ at the dimensions the model expects. The remaining checks are about confirming the integration matches.

- [ ] Confirm the model's input shape is `[1, 79, 20, 3]` (NHWC, time × MFCC × channels). Quick check:
  ```
  python3 -c "import tensorflow as tf; i = tf.lite.Interpreter('model.tflite'); i.allocate_tensors(); print(i.get_input_details()[0]['shape'], i.get_input_details()[0]['dtype'])"
  ```
  - If the time dimension is `81` instead of `79`, the training pipeline is using librosa's `center=True` framing. Bump `SPELL_FEAT_N_FRAMES` to 81 in `spell_config.h` and add a 2-frame symmetric pre/post zero-pad in `feat_extract.c` to match.
  - If the channel dimension is something other than `3`, the trainer probably isn't stacking Δ + ΔΔ — confirm and adjust.
  - If the layout is `[1, 3, 79, 20]` (NCHW), TFLite Micro doesn't support that natively; either re-export NHWC or transpose in firmware before quantize.
- [ ] Confirm the op set actually used by the exported model is a subset of `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `AveragePool2D`, `Mean`, `Reshape`, `Softmax`, `Quantize`, `Dequantize`. If the model uses additional ops, `inference.cpp`'s `MicroMutableOpResolver<9>` template parameter and `init_op_resolver()` need to grow.
- [ ] **Mel scale formula.** The firmware uses HTK's `2595·log10(1 + f/700)`. Confirm the training pipeline matches. librosa's `mfcc()` defaults to `htk=False` (Slaney piecewise scale, different curve below 1 kHz) — if your trainer is librosa with default settings, swap `hz_to_mel` / `mel_to_hz` in `feat_extract.c` for the Slaney variant. tensorflow's `linear_to_mel_weight_matrix` also uses HTK, so TF-based pipelines match by default.
- [ ] **DCT type and norm.** The firmware uses orthonormal DCT-II (matches `scipy.fftpack.dct(..., norm='ortho')` and librosa default). If your trainer uses a different DCT norm (e.g. unscaled / type-III), the MFCC values will be off by a constant factor and accuracy will collapse. Easiest sanity check: feed a known WAV through both training and firmware and compare MFCC[0..19] for one frame.
- [ ] **Lifter.** The firmware does NOT apply a cepstral lifter. librosa default is `lifter=0` (no lifter), HTK default is `lifter=22`. If your trainer applies a lifter (e.g. `librosa.feature.mfcc(..., lifter=22)`), apply the same multiplicative factor `1 + (L/2)*sin(π*k/L)` per-coefficient in `feat_extract.c` after the DCT.
- [ ] **Delta formula.** The firmware computes `Δc[t] = (c[t+1] - c[t-1] + 2·(c[t+2] - c[t-2])) / 10` with edge-replicated boundaries. librosa uses `librosa.feature.delta(..., width=9, mode='interp', order=N)` by default — width=9 means N=4 (using ±4 frames), and `mode='interp'` does polynomial interpolation at boundaries. If accuracy is off after model flash, this is a likely culprit; align the formulas.
- [ ] Decide model output format: in-graph `Softmax` (probabilities in [0,1] summing to ~1.0) or raw logits. Either works — the firmware auto-detects — but the in-graph variant runs slightly faster on int8 with esp-nn. The exporter's `representative_dataset` step controls this.
- [ ] Flash the model to the `model` partition: `parttool.py --port /dev/cu.usbmodem101 write_partition --partition-name model --input model.tflite`. After this, the inference task should switch from `no model loaded — inference disabled` to per-utterance `TOPK A=0.92 B=0.04 ...` lines.
- [ ] Confirm the empirical confusion matrix `M` exists (26×26 row-major float32 = 2,704 bytes) and the row-sums are 1.0 (each row is a conditional distribution over true letters). This becomes a precondition for Phase 5 decoder; not needed for Phases 1–3 to function.
- [ ] Confirm the dictionary file format. The plan assumes `{word_id (uint16), length (uint8), letters[16] (chars), log_freq (float32)}` records. If your training/data pipeline produces a different format, either match it in the firmware decoder or write a converter. Same: not blocking until Phase 5.

### D. Calibration (do this with real recordings, before tuning anything)

- [ ] Record QA-set words on the host (Mac mic, 16 kHz mono WAV) per ADR-0004. The on-device `audio_dump` path is shelved; device-mic recordings happen later, once the test-rig build (§12.F) exists. For VAD calibration of the on-device segmenter, the same Mac-recorded WAVs are fed to `tools/segmenter_replay/` — frequency response differences between the Mac mic and ICS-43434 mostly do not affect energy-based VAD, so threshold tuning transfers reasonably.
- [ ] Run each recording through `tools/segmenter_replay/segmenter_replay`. Confirm the letter counts match what you actually spelled. If they don't, this is where the iteration loop pays off — tweak `--on` / `--off` / `--ema` / `--min-letter` / `--eow` and re-run until segmentation matches reality.
- [ ] When the parameters look right, copy them into `components/spell_common/include/spell_config.h` (`SPELL_VAD_*`), rebuild, reflash. The `VAD_EMA_ALPHA_X100` constant currently lives in `segmenter.c` rather than `spell_config.h` — promote it to the header if you tune it.
- [ ] After model flash, confirm the energy gate threshold (`SPELL_ENERGY_GATE_DB = -45`) matches whatever the training-pipeline curation tool used to filter training segments. Drift here silently degrades on-device accuracy.

### E. Assumptions I made — please confirm

- [ ] **Microphone.** I assumed ICS-43434 throughout. If you're using a different I2S MEMS mic (INMP441, SPH0645, etc.), the I2S configuration in `audio_capture.c` is correct as long as it's a 24-bit-in-32-bit-slot LEFT-channel device. INMP441 specifically is pin-compatible. Other mics may need slot config changes.
- [ ] **Amp + speaker.** The pin map assumes MAX98357A. The Phase 4 work will need this confirmed before I write the I2S TX driver.
- [ ] **Mic→speaker pin pool.** I picked GPIO 7/15/16 for I2S TX without checking the DevKitC-1 pin diagram. Some pins are reserved for strapping or USB-JTAG. Confirm GPIO 7, 15, 16, and 10 are free on your specific dev kit. The DevKitC-1-N16R8V1 silkscreen marks reserved pins; if any of these conflict, pick alternates and update `spell_config.h` — no other code changes needed.
- [ ] **Model partition size.** `partitions.csv` reserves 256 KB for the model. The training reports a ~60 KB int8 model; 256 KB leaves comfortable OTA headroom. If the actual model is bigger, bump the partition.
- [ ] **PSRAM presence.** The firmware will refuse to run usefully without PSRAM enabled. The boot banner logs `psram: DISABLED — bringup needs it for subscriber buffers + tensor arena` if the SPIRAM kconfig isn't set. The `sdkconfig.defaults` we ship enables it; just confirm the warning doesn't fire.
- [ ] **Mic SEL→GND.** The single most common breakout-wiring mistake. Many breakouts ship with SEL floating, which can latch either way.

### F. Carry-forward notes (not blocking, but worth filing)

- [ ] **Min inter-letter gap merging** (~80 ms) is documented in the plan but not implemented in `segmenter_core.c`. The off-frames hysteresis approximates it. Revisit if real-world testing produces pathological splits.
- [ ] **Highest-energy slice for over-length letters.** The wrapper currently takes the trailing 800 ms when a captured letter exceeds the inference window. Plan calls for highest-energy slice. Trivial to upgrade in `dispatch_letter()` if needed.
- [ ] **No `console` component.** The plan mentions `infer_replay <wav>` and `decode_replay` serial commands. They're not implemented; substitute the host-side `segmenter_replay` for VAD work, and revisit the on-device console story when Phase 5 needs `decode_replay`.
- [ ] **`device_config.c` from EARS was not ported** — Spell-Word has nothing yet to persist. Add it back when Phase 4 introduces tunables (volume, etc.) worth keeping across reboots.
- [ ] **Empty pre-roll on first letter of a spelling attempt.** `seg_reset` runs every inactive frame, so the pre-roll ring is empty at the moment `seg_active` flips true (post-confirmation-tone). The first letter therefore has no leading silence in its 800 ms inference window. Tolerable for MVP — adult bench testers reliably leave 100+ ms before the first letter onset, which the segmenter's onset detection captures from real-time frames. Revisit if accuracy on the first letter is materially worse than subsequent letters.
- [ ] **OTA mechanism for model and corpus partitions.** Production firmware is fully offline (ADR-0003), so an OTA path requires either a USB-attached host (`parttool.py`, current MVP approach), an SD-card slot (hardware change), or a build-flagged "update mode" that temporarily enables WiFi for partition fetch. v1 design question; not a blocker for MVP.
- [ ] **Test rig.** Real-device QA — once we move past Mac-recorded WAVs — is intended to use a separate Kconfig-gated build that captures audio + inference + decoder + segmenter state to PSRAM and uploads via WiFi for offline analysis. This replaces the `audio_dump` component's role; `audio_dump` itself is shelved for now (no firmware-side trigger bound). The test-rig design is post-MVP; for now, host-recorded WAVs and the host-side replay tools are the entire QA story.
- [ ] **`audio_dump_emit_b64()` has no firmware-side trigger** and is not expected to gain one. Treat it as a dormant component; the eventual replacement is the test-rig build above.

---

## 13. Architectural decisions

The decisions below are recorded as ADRs because they're hard to reverse, surprising without context, or deviate deliberately from the obvious path. See `docs/adr/` for the full text. Domain language used throughout this plan — utterance, letter, EOW, early-commit window, demo set vs QA set — is defined in the root `CONTEXT.md`.

- **[ADR-0001](docs/adr/0001-aggressive-early-commit-decoder.md)** — Aggressive early-commit decoder. Three-clause predicate replaces 1.2 s full-EOW wait; silence-floor clause defeats prefix-truncation.
- **[ADR-0002](docs/adr/0002-confusion-matrix-with-tts-trained-model.md)** — Confusion matrix `M` retained at α=0.15 despite TTS-only training. Phonetic universals transfer; M is the only outside-top-K recovery mechanism.
- **[ADR-0003](docs/adr/0003-fully-offline-production-firmware.md)** — Production firmware is fully offline. All telemetry/test instrumentation lives behind a Kconfig flag in a separate build target.
- **[ADR-0004](docs/adr/0004-decoder-tuning-via-python-reference.md)** — Decoder tuning happens in the Python reference against host-recorded WAVs. The C decoder is a port validated against shared fixtures.
- **[ADR-0005](docs/adr/0005-w-detection-via-inference-results.md)** — W is detected via inference-result patterns (high-confidence U after two low-confidence utterances → re-run inference on merged PCM), not via segmenter-level energy heuristics. Adds latency only when a W candidate is in flight.

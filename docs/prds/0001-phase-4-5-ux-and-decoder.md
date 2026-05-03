# PRD: Phase 4–5 — Press-to-arm UX, early-commit decoder, W detection

Status: needs-triage

## Problem Statement

The current firmware can capture audio, segment letter-utterances, and run inference to produce top-K letter predictions, but a child cannot use the device end-to-end. There is no way to start a spelling attempt, no audible feedback that the device heard the press, no resolution of letters into words, and no spoken-word output. Two specific problems would make even a working pipeline unusable: a 1.2-second silence wait at the end of every spelling feels like the device is broken to a child; and the letter "W" is uniquely three syllables (DUB-uhl-YOO), so the energy-VAD splits it into three short fragments that the model doesn't recognize on their own — each fragment lands in the decoder as a low-confidence guess unrelated to W.

## Solution

Three new pure-C deep modules and one segmenter extension, integrated into a state machine that runs the full press-to-spell-to-speak loop.

The decoder turns a sequence of top-K events into a resolved word via dictionary scoring with confusion-matrix-aware posteriors and bounded edit-distance, and commits *aggressively* — as soon as a three-clause predicate (margin over runner-up, margin over best longer candidate, post-letter silence floor) is satisfied — instead of waiting for full end-of-word silence.

The W detector watches the stream of top-K events post-inference. When it sees a high-confidence "U" preceded by two low-confidence utterances, it re-runs inference on the concatenated PCM of all three and, if the result is a confident "W," emits a retract-and-replace signal that tells the decoder to swap those three letters for one W in its in-flight word buffer. This costs nothing on normal spelling and adds ~40 ms (one extra inference cycle) only when a W candidate is in flight.

The UX module owns button input, the IDLE/ARMED/SPELLING/RESOLVING/PLAYING state machine, audible tone feedback (confirmation, resolved, error chirp), the I2S TX path, and a cancel-and-re-arm interruption flow during playback.

## User Stories

1. As a **child**, I want to press the button and hear a confirmation tone, so that I know the device is listening.
2. As a **child**, I want the device to start listening for letters only after the confirmation tone finishes, so that the device's own tone doesn't get confused with my first letter.
3. As a **child**, I want to spell a word letter by letter with brief pauses, so that the device recognizes each letter individually.
4. As a **child**, I want the device to play back the word I spelled within a fraction of a second of finishing, so that I don't think the device is broken.
5. As a **child**, I want the device to recognize "W" as one letter even though it's pronounced "double-you," so that I can spell words containing W as easily as other letters.
6. As a **child**, I want to press the button while the device is saying the wrong word, so that I can stop it and try spelling again immediately.
7. As a **child**, I want the device to play an error chirp when it doesn't recognize my word, so that I know to try again instead of waiting silently.
8. As a **child**, I want the device to ignore me when I'm not actively spelling, so that ambient sound doesn't trigger random word playback.
9. As a **child**, I want a brief "got it" tone before the device says my word, so that I know my spelling was understood and the playback is intentional.
10. As an **adult demo participant**, I want the device to recognize words spelled at conversational speed, so that the demo doesn't require artificial slow-spelling.
11. As an **adult demo participant**, I want a press during playback to clearly stop the device and re-arm it, so that I can recover quickly from a misrecognition.
12. As a **firmware engineer**, I want decoder hyperparameters tunable from `spell_config.h`, so that I can sweep them in the Python reference and bake the tuned values into firmware.
13. As a **firmware engineer**, I want every early-commit predicate evaluation logged with per-clause results, so that when a spelling fails I can see which clause prevented or caused commit.
14. As a **firmware engineer**, I want the decoder, UI state machine, and W-detection logic testable host-side, so that I can iterate without flashing a board.
15. As a **firmware engineer**, I want the W-detector trigger thresholds (U probability, low-confidence ceiling, W-confirmation probability) tunable, so that I can adapt to model behavior on real audio.
16. As a **firmware engineer**, I want the W detector to log every trigger evaluation and re-inference outcome, so that I can debug both false negatives (W missed) and false positives (real U treated as W tail).
17. As a **firmware engineer**, I want the segmenter to emit an early-commit-window event distinct from end-of-word, so that the decoder can evaluate its predicate at a defined moment without polling.
18. As a **firmware engineer**, I want the decoder to abstain rather than guess when scores are too close, so that the device errs on the side of saying nothing wrong.
19. As a **firmware engineer**, I want playback cancellation to be cooperative (not preemptive), so that I can guarantee the amp is muted before I2S TX is disabled and avoid clicks.
20. As a **firmware engineer**, I want the state machine encoded as a pure transition table, so that I can validate every state-event combination without simulating hardware.
21. As a **firmware engineer**, I want the decoder to load the confusion matrix and dictionary from dedicated partitions, so that they can be updated independently of the firmware binary.
22. As a **firmware engineer**, I want the W detector bypassable via Kconfig, so that I can A/B test "with vs. without" detection during QA.
23. As a **firmware engineer**, I want a host-side `decoder_replay` tool, so that I can sweep decoder parameters against fixtures in seconds rather than minutes.
24. As a **firmware engineer**, I want a host-side `w_detector_replay` tool that operates on top-K event streams, so that I can validate trigger logic without recording or re-running real audio.
25. As a **firmware engineer**, I want each spelling attempt to log a single summary line — input letters → resolved word + score margin + commit reason (early vs full-EOW) + W-detection events — so that I can scan a session log quickly.
26. As a **firmware engineer**, I want the inference component to retain the last N utterance PCM buffers in PSRAM, so that the W detector can re-run inference on the merged audio without changing the segmenter contract.
27. As a **firmware engineer**, I want the decoder to support a retract-and-replace operation on its in-flight word buffer, so that the W detector can correct three letters into one without race conditions against early commit.

## Implementation Decisions

### Modules

- **`components/decoder/`** — owns "letters → word" end-to-end. Inputs: top-K events, retract-and-replace events (new), early-commit-window events, EOW events. Outputs: `WORD_RESOLVED { word_id, commit_reason }` or `WORD_ABSTAIN`. Internally split into `decoder_core` (pure C99, host-portable) and a thin IDF wrapper for partition loading and `esp_event` plumbing.
- **`components/spell_ui/`** — owns "press → speak" end-to-end. Includes button + debounce, the IDLE/ARMED/SPELLING/RESOLVING/PLAYING state machine, tone synthesis at boot, the I2S TX channel, the playback queue worker, amp-shutdown gating, and cooperative cancel. Internally split into `ui_core` (pure transition table, host-testable) and an IDF wrapper that materializes actions.
- **`components/w_detector/`** — owns the "recognize W as a single letter" capability end-to-end. Subscribes to top-K events; when the trigger pattern fires, re-runs inference on concatenated PCM of the last three utterances and, on a confident W, emits a retract-and-replace event to the decoder. Internally split into `w_detector_core` (pure C99 — trigger logic on top-K-event metadata, no PCM, no inference) and an IDF wrapper that owns the PCM ring access, re-inference dispatch, and event publication.
- **`components/segmenter/`** (extended) — `seg_core` gains `SEG_EVT_EARLY_COMMIT_WINDOW`, fired exactly once per inter-letter silence cycle when the post-letter frame count crosses the early-commit threshold. Latch resets on next letter onset. The IDF wrapper publishes `SPELL_EVENT_EARLY_COMMIT_WINDOW`.
- **`components/letter_classifier/`** (extended) — retains the last N=4 utterance PCM buffers in a PSRAM ring (3 in detection window + 1 currently being processed) instead of freeing immediately after inference. Exposes `inference_get_recent_pcm(int back_index)` for the W detector. The ring is sized at `4 × SPELL_INFERENCE_WINDOW_BYTES ≈ 100 KB`.

### Public interface shapes

- **Decoder** subscribes to `SPELL_INFERENCE_EVENT::LETTER_TOP_K`, `SPELL_W_DETECTOR_EVENT::RETRACT_AND_REPLACE`, `SPELL_SEGMENTER_EVENT::EARLY_COMMIT_WINDOW`, `SPELL_SEGMENTER_EVENT::END_OF_WORD`. Publishes `SPELL_DECODER_EVENT::WORD_RESOLVED` and `WORD_ABSTAIN`.
- **W detector** subscribes to `SPELL_INFERENCE_EVENT::LETTER_TOP_K`. When trigger fires, calls `inference_run_synchronous(merged_pcm)` (or its async equivalent with a callback) and, on confident W, posts `SPELL_W_DETECTOR_EVENT::RETRACT_AND_REPLACE { retract_count: 3, replacement: top_k_event }`. Latency: zero when no trigger, ~40 ms when triggered.
- **UI** subscribes to button events, `SPELL_DECODER_EVENT::WORD_RESOLVED` / `WORD_ABSTAIN`, and `SPELL_PLAYBACK_EVENT::COMPLETE`. Drives `segmenter_set_active`, `playback_play_tone`, `playback_play_word` (Phase 6 stub), `playback_cancel`.

### Aggressive early-commit predicate

After every top-K event and every early-commit-window event, the decoder evaluates:

```
margin_runnerup     = score(best) - score(runner_up_same_length)
margin_longer       = score(best) - score(best_candidate_longer_by_1_or_2)
silence_since_last  = ms since last letter offset

commit_early IFF
    margin_runnerup     > SPELL_EARLY_MARGIN_RUNNERUP   (default 3.0)
  AND margin_longer     > SPELL_EARLY_MARGIN_LONGER     (default 2.0)
  AND silence_since_last ≥ SPELL_EARLY_MIN_SILENCE_MS   (default 500)
```

The third clause is load-bearing — without it, prefix-truncation (BAN inside BANK) systematically wins on 0-edit-vs-1-edit-deletion. On full-EOW, the standard `SPELL_DECISION_MARGIN` abstention test applies.

The 500 ms silence floor also serves as a natural barrier against the W-detection race: re-inference (~40 ms) finishes well before the floor, so a W retract-and-replace lands at the decoder before any early-commit evaluation that would have used the un-corrected sequence.

### Posterior assembly

```
remaining_mass = 1.0 - Σ top_k.probability
For each letter c in 0..25:
    if c in top_k:    net_prob[c] = top_k_probability_for[c]
    else:             net_prob[c] = remaining_mass / 21
posterior[c] = (1 - α) · net_prob[c] + α · M[top1, c]
```

α = `SPELL_DECODER_ALPHA` (default 0.15, retained per ADR-0002).

### W-detection trigger

Examined on each new top-K event arriving at the W detector. Define:

- `current_top1` = top-K[0].letter_index of the new event
- `current_prob` = top-K[0].probability of the new event
- `prev_top1_prob[k]` = top-K[0].probability of the event k positions before the new event

Trigger:

```
fire_w_detection IFF
    current_top1 == 'U'
  AND current_prob          ≥ SPELL_W_DETECT_U_THRESHOLD          (default 0.50)
  AND prev_top1_prob[1]     <  SPELL_W_DETECT_LOW_CONF_THRESHOLD  (default 0.40)
  AND prev_top1_prob[2]     <  SPELL_W_DETECT_LOW_CONF_THRESHOLD  (default 0.40)
  AND we have PCM for prev[1] and prev[2] in the inference ring
```

On fire: concatenate the three PCM buffers (with the inter-utterance gaps preserved as silence — total ≤ `3 × SPELL_INFERENCE_WINDOW_SAMPLES`, post-trim to the trailing 800 ms slice if over-length, since the model expects a fixed 800 ms window), submit for re-inference. On result:

```
confirm_w IFF
    new_top_k[0].letter_index == 'W'
  AND new_top_k[0].probability ≥ SPELL_W_DETECT_CONFIRM_THRESHOLD (default 0.70)
```

If confirmed: emit `SPELL_W_DETECTOR_EVENT::RETRACT_AND_REPLACE { retract_count: 3, replacement: new_top_k_event }`. Otherwise: do nothing (the original three top-K events stay in the decoder's buffer; if the result is plausibly a real word containing those three letters, edit-distance recovery still gets a chance).

### Decoder retract-and-replace

The decoder supports `decoder_on_retract_and_replace(count, replacement)`:

- Pops the most recent `count` top-K events from its in-flight word buffer
- Pushes the `replacement` event
- Re-evaluates its margin / posterior caches as if the sequence were always `[..., replacement]`
- Does not commit early as part of this operation; commit decisions still flow through the normal predicate path on the next event.

Race: a retract event arriving at the decoder *after* the decoder has already committed (early-commit fired before re-inference completed) is logged and dropped. The 500 ms silence floor makes this race vanishingly unlikely in practice but the safety check is cheap.

### Confusion matrix and dictionary loading

Both load at decoder init from custom data partitions (`matrix` / subtype 0x81; `dict` / subtype 0x82). At load: `M` row sums validated to 1.0 ± 1e-3; dictionary records validated as well-formed. Empty/erased partitions fail loudly. Mirrors `inference.cpp`'s empty-model handling.

### Configuration additions to `spell_config.h`

- Decoder: `SPELL_EARLY_MIN_SILENCE_MS`, `SPELL_EARLY_MARGIN_RUNNERUP`, `SPELL_EARLY_MARGIN_LONGER` (already added per plan §5).
- W detector: `SPELL_W_DETECT_U_THRESHOLD`, `SPELL_W_DETECT_LOW_CONF_THRESHOLD`, `SPELL_W_DETECT_CONFIRM_THRESHOLD`, `SPELL_W_DETECT_PCM_RING_DEPTH` (default 4).
- UI: tone IDs and frequencies; `SPELL_BUTTON_DEBOUNCE_MS` and `SPELL_ARM_TIMEOUT_MS` already present.

### Event base additions to `spell_events.h`

- `SPELL_DECODER_EVENT` base with `WORD_RESOLVED` and `WORD_ABSTAIN` IDs and payloads.
- `SPELL_PLAYBACK_EVENT` base with `COMPLETE` ID.
- `SPELL_UI_EVENT` base for state-transition events.
- `SPELL_W_DETECTOR_EVENT` base with `RETRACT_AND_REPLACE` ID and payload (retract_count + replacement top-K event).
- `SPELL_EVENT_EARLY_COMMIT_WINDOW = 3` added to the existing `SPELL_SEGMENTER_EVENT` enum.

### Cancel ordering contract

`playback_cancel()` MUST: (1) pull `SPELL_AMP_SHUTDOWN_GPIO` low; (2) set the cooperative-cancel flag; (3) wait for the worker to acknowledge; (4) disable the I2S TX channel; (5) free in-flight PCM. Reversing (1) and (4) produces an audible click.

### Dataflow

```
mic → audio_capture → segmenter → inference ──top-K──┬──► decoder ──► spell_ui ──► speaker
                                       │             │       ▲
                                       │             ▼       │ retract-and-replace
                                       │         w_detector ─┘
                                       │             │
                                       │             ▼ (on trigger)
                                       └──► re-inference ──► top-K
                                            (synchronous, ~40 ms)

                                  early-commit-window event ─► decoder
                                  end-of-word event         ─► decoder
                                  button press              ─► spell_ui
```

## Testing Decisions

A good test in this codebase exercises the *external behavior* of a `_core` module — input data in, output data out. Internal scratchpads, EMA buffers, DP scratch arrays are not directly poked. Prior art: `tools/segmenter_replay/test_segmenter_core.c`, eight cases all passing.

### Modules with host-side unit tests

- **`decoder_core`** via `tools/decoder_replay/test_decoder_core.c`. Cases: clean spelling → exact resolution; one wrong top-1 with confusion-pair recovery; prefix-truncation (BAN/BANK) with each predicate clause failing in isolation; 1-edit insertion; 1-edit deletion; abstain on flat distribution; `MAX_LETTERS_PER_WORD` overflow → abstain; per-clause early-commit predicate coverage; **retract-and-replace** with the replaced sequence shorter than the buffer (count=3, buffer length 5) and longer (count=3, buffer length 3); retract-and-replace arriving after early-commit (logged-and-dropped path).
- **`w_detector_core`** via `tools/w_detector_replay/test_w_detector_core.c`. Pure-C trigger logic, operating on synthetic top-K event sequences (no PCM, no real inference). Cases: three high-confidence non-W letters → no trigger; two low-conf + high-conf U → trigger fires; two high-conf + high-conf U → no trigger; two low-conf + high-conf non-U → no trigger; two low-conf + low-conf U → no trigger; boundary cases at each threshold; not-enough-history (only 1 prior event) → no trigger; consecutive triggers (W-W back to back) handled correctly.
- **`ui_core`** via host transition tests. Cases: every (state, event) pair → assert (next_state, action_list). Specific scenarios: cancel + re-arm from PLAYING; button-during-ARMED ignored; arm-timeout from ARMED → IDLE; EOW from SPELLING → RESOLVING; early-commit-window with predicate-met → RESOLVING; abstain from RESOLVING → IDLE with error chirp.
- **`seg_core`** extended (existing test file). New cases: early-commit-window fires once per inter-letter gap; latch resets on next onset; window event ordering is always before EOW; window does not fire if min-silence not reached.

### Modules NOT unit-tested (intentionally)

- IDF wrappers around each `_core`. IDF/hardware-coupled. Tested via on-device acceptance.
- The W detector's *re-inference path* (synchronous TFLM invocation on merged PCM). Trigger logic is host-tested; the actual re-inference invocation is on-device acceptance.
- The state machine's debounce timer interaction, the I2S TX driver, partition loaders, the inference PCM ring management.

### New tooling

- **`tools/decoder_replay/`** — host C build linking `decoder_core` directly. CLI param overrides for every tunable. Reads a fixture file (CSV: `word_id, [(letter_idx, prob)] per utterance, optional retract_after_event_n`) emitted by the Python reference. Prints resolved word + score margin + per-evaluation log.
- **`tools/w_detector_replay/`** — host C build linking `w_detector_core` directly. Reads a CSV of synthetic top-K events (`event_n, top1_letter, top1_prob, ...`), prints trigger decisions per event. `make test` runs unit tests.

## Out of Scope

- **Phase 6 word audio playback.** `playback_play_word(word_id)` is a stub returning `ESP_ERR_NOT_SUPPORTED`. Opus decode, corpus partition reader, full per-word PCM decode all happen in Phase 6.
- **Phase 7 hyperparameter tuning.** This PRD ships the *mechanism* for tuning, not the *tuned values*. W-detection thresholds in particular need empirical tuning against real recordings; defaults are best-guess.
- **OTA mechanism for partitions.** Production firmware is fully offline (ADR-0003); MVP uses USB `parttool.py`.
- **Test rig (PSRAM telemetry + WiFi upload).** Separate Kconfig-gated build, post-MVP.
- **Children's-speech model fine-tuning.** Out of MVP scope per ADR-0002.
- **Multi-syllable handling for letters other than W.** Only W is ambiguously multi-syllable in spoken English; the detector is W-specific by design (no general "multi-syllable" abstraction).
- **Detection of W via energy / segmentation features.** Considered and rejected in favor of inference-result-based detection: re-running the model on merged PCM is more reliable than energy-pattern heuristics and adds latency only when a W candidate is in flight, not on every utterance.
- **Fixture format alignment with the Python reference.** The C decoder consumes a fixture *equivalent* to Python's output; the precise on-disk format is left to the implementation pass.

## Further Notes

- **ADR cross-references.** Aggressive early-commit: [ADR-0001](../adr/0001-aggressive-early-commit-decoder.md). Confusion matrix retained at α=0.15: [ADR-0002](../adr/0002-confusion-matrix-with-tts-trained-model.md). Fully-offline-production: [ADR-0003](../adr/0003-fully-offline-production-firmware.md). Python-reference tuning: [ADR-0004](../adr/0004-decoder-tuning-via-python-reference.md). W detection via inference results: [ADR-0005](../adr/0005-w-detection-via-inference-results.md).
- **Domain glossary.** Domain language (utterance, letter, EOW, early-commit window, demo set vs QA set, top-K, etc.) is in `CONTEXT.md`. The "W detector" / "trigger" / "retract-and-replace" terminology is W-specific and confined to this feature; not added to the global glossary.
- **Sequencing.** Recommended implementation order: (a) extend `seg_core` with the early-commit-window event + tests; (b) extend `letter_classifier` with the PCM ring; (c) build `w_detector_core` with synthetic-event tests; (d) build `decoder_core` (including retract-and-replace) with hand-crafted fixture tests; (e) build `ui_core` transition table with tests; (f) integrate IDF wrappers; (g) bench-test on hardware once Phase 0 mic + speaker bring-up is complete.
- **Latency budget.** Worst-case user-perceived latency from last-letter-offset to word-playback-start under aggressive early-commit, full pipeline, no W in word: early-commit silence floor (500 ms) + predicate eval (~1 ms) + resolved tone (80 ms) + Opus decode (Phase 6, ~30–80 ms) + I2S TX warmup (~10 ms) ≈ 620–680 ms. With W in word: +40 ms re-inference, only if the U arrives early enough that re-inference completes before the silence floor expires (which is the common case). Status-quo full-EOW path: ~1300–1400 ms. Aggressive path saves ~700 ms typical.
- **No issue tracker.** This PRD is filed locally because the repo has no GitHub remote. When a tracker is configured, this document migrates with the `needs-triage` label preserved.

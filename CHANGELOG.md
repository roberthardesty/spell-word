# Changelog

Notable changes per branch/commit. Newest first. Format loosely follows Keep a Changelog; entries reference Vikunja task IDs (project: Spell-Word) where applicable.

## Unreleased

### Added

- **Segmenter early-commit-window event** (Vikunja #19, PRD §"Modules"/§"Sequencing" step (c)). `seg_core` now fires `SEG_EVT_EARLY_COMMIT_WINDOW` exactly once per inter-letter silence cycle when the post-letter frame count crosses a new `early_commit_frames` config (defaulted to `SPELL_EARLY_MIN_SILENCE_MS = 500 ms` in the IDF wrapper). Latch resets on next letter onset. `seg_init` validates `0 < early_commit_frames < eow_frames` so the window event is always observed before EOW in the same silence stretch. The IDF wrapper republishes `SPELL_SEGMENTER_EVENT::EARLY_COMMIT_WINDOW`. Four new host tests (`tools/segmenter_replay/test_segmenter_core.c`) cover fires-once-per-gap, latch-resets-on-onset, ordering-before-EOW, no-fire-below-min-silence; all 12 cases pass. Unblocks the decoder early-commit predicate (Vikunja #21).

### Changed

- **Rename `letter_classifier` → `letter_recognizer`** (Vikunja #41). The deepened module owns audio-to-letter-evidence end-to-end (per ADR-0005), not just classification. `git mv` preserved history on all 6 files. Public API renamed: `inference_init` → `letter_recognizer_init`, `inference_submit_utterance` → `letter_recognizer_submit_utterance`, plus `_reload_model` and the four `_get_*` stat getters. Internal signal-pipeline helpers (`load_model_from_partition`, `setup_interpreter`, `top_k_select`, `window_rms_dbfs`, the future `inference_run`) retained `inference_*` names per the PRD carve-out. Internal `inference_task` renamed to `recognizer_task`; logging TAG changed to `recognizer`.
- **Unified letter event vocabulary** (Vikunja #40, ADR-0006). Removed `SPELL_INFERENCE_EVENT::LETTER_TOP_K` and `SPELL_W_DETECTOR_EVENT::RETRACT_AND_REPLACE`. Added `SPELL_RECOGNIZER_EVENT::LETTER_RECOGNIZED` with payload `{ spell_letter_top_k_t top_k; uint8_t retract_count; }`. The decoder's letter-event handler will be one code path with no race against early commit. Emitter currently posts `retract_count = 0`; W-recovery (Vikunja #30) will be the only path that sets it to 2.
- `spell_letter_top_k_t` introduced as the per-utterance result type (`candidates[K]` plus `invoke_ms` for diagnostic). Replaces `spell_letter_topk_event_t`.

### Added

- ADR-0006 (`docs/adr/0006-unified-letter-recognized-event.md`) formalizes the unified letter event vocabulary.

### Removed

- `SPELL_INFERENCE_EVENT` event base + `SPELL_EVENT_LETTER_TOP_K` ID + `spell_letter_topk_event_t` payload.
- `SPELL_W_DETECTOR_EVENT` event base + `SPELL_EVENT_RETRACT_AND_REPLACE` ID + `spell_retract_and_replace_event_t` payload.

## 0.0.1 — Initial commit

- Phase 0–3 audio pipeline: `audio_capture` (I2S RX), `segmenter` (energy VAD with onset/offset hysteresis, EOW), `letter_classifier` (MFCC + TFLM DS-CNN, top-K event posting). `app_main` Day-3 bring-up scaffold logs top-K events and EOW. No decoder, UI state machine, button, tones, or speaker output yet.

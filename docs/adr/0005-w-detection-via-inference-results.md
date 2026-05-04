# W detection via inference results, not segmenter timing

W is the only letter pronounced with multiple syllables in English ("DUB-uhl-YOO"), so the energy-based VAD reliably splits it into three short utterances that each land at the model as low-confidence guesses unrelated to W. The **letter recognizer** owns the recovery: it keeps the last 4 utterance PCM buffers (~100 KB PSRAM) and a short top-K history (top1 letter + confidence for the last 2 utterances). When a high-confidence U arrives after two low-confidence utterances, the recognizer holds the U emission, re-runs inference on the concatenated PCM of all three (~40 ms), and then emits a single `LETTER_RECOGNIZED` event: either `retract_count = 0` with the original U top-K (rerun did not confirm W), or `retract_count = 2` with the merged W top-K (rerun confirmed W with confidence > `SPELL_W_DETECT_CONFIRM_THRESHOLD`). The U is never emitted as an independent event. Cost: 0 ms on normal spelling, ~40 ms only when a W candidate is in flight.

## Considered options

- **Segmenter-level energy coalescing** (hold short utterances briefly, merge if a quick follow-up arrives, dispatch as one). Rejected: adds latency to every utterance, requires tuning timing thresholds that overlap with real inter-letter pauses, and the merge decision uses energy heuristics that approximate W's pattern instead of measuring it directly.
- **Train the model to recognize partial-W fragments.** Rejected as out of MVP scope — would require regenerating the TTS training corpus with W fragments and is upstream work, not firmware.
- **Decoder-only fix via edit-distance.** Rejected: with three garbage low-confidence letters from a split W, edit-distance recovery is forced to find a real word matching the pattern, which fails systematically for W-containing words against a 200-word smoke corpus.
- **Separate `w_detector` component watching the recognizer's event stream and emitting a retract-and-replace event after the fact (the original draft of this ADR).** Rejected: splits one cycle (PCM in → letter evidence out) across two components for no data-flow reason — only the recognizer has the PCM ring and the interpreter, both of which the recovery cycle needs. The "emit U then retract" wire model also briefly puts the decoder in an inconsistent state that races against early-commit, even if rarely.

## Consequences

- The recognizer holds the U emission for ~40 ms when the trigger fires; the decoder never sees an inconsistent (U-then-retract) state. The early-commit race is structurally absent — 40 ms ≪ 500 ms silence floor (ADR-0001).
- `retract_count = 2`, not 3: only the two prior low-confidence emissions are retracted; the U is never published.
- PCM ring + trigger logic + merge-and-rerun all live inside the letter recognizer; `w_detector_core` survives as a sub-module for host testing the trigger predicate against synthetic top-K sequences.
- Three tunables (`SPELL_W_DETECT_U_THRESHOLD`, `SPELL_W_DETECT_LOW_CONF_THRESHOLD`, `SPELL_W_DETECT_CONFIRM_THRESHOLD`); defaults are best-guess and need empirical tuning against real recordings.
- The detector is W-specific by design — no abstraction for "multi-syllable letters" because W is the only one in English.

# W detection via inference results, not segmenter timing

W is the only letter pronounced with multiple syllables in English ("DUB-uhl-YOO"), so the energy-based VAD reliably splits it into three short utterances that each land at the model as low-confidence guesses unrelated to W. We detect this case post-inference: the W detector watches the top-K event stream, fires when a high-confidence U follows two low-confidence utterances, re-runs inference on the concatenated PCM of all three, and — on a confident W — emits a retract-and-replace event that swaps those three letters for one W in the decoder's in-flight buffer. The PCM ring needed for re-inference lives in the `letter_classifier` component (last 4 utterance buffers in PSRAM); the decoder gains a retract-and-replace handler. Cost: 0 ms on normal spelling, ~40 ms (one extra inference cycle) only when a W candidate is in flight.

## Considered options

- **Segmenter-level energy coalescing** (hold short utterances briefly, merge if a quick follow-up arrives, dispatch as one). Rejected: adds latency to every utterance, requires tuning timing thresholds that overlap with real inter-letter pauses, and the merge decision uses energy heuristics that approximate W's pattern instead of measuring it directly.
- **Train the model to recognize partial-W fragments.** Rejected as out of MVP scope — would require regenerating the TTS training corpus with W fragments and is upstream work, not firmware.
- **Decoder-only fix via edit-distance.** Rejected: with three garbage low-confidence letters from a split W, edit-distance recovery is forced to find a real word matching the pattern, which fails systematically for W-containing words against a 200-word smoke corpus.

## Consequences

- The `letter_classifier` no longer frees an utterance's PCM buffer immediately after inference; it keeps the last N=4 in a PSRAM ring (~100 KB).
- The decoder grows a retract-and-replace path; a retract event arriving after the decoder has already committed early is logged and dropped (made vanishingly rare by the 500 ms early-commit silence floor).
- Three new tunables (`SPELL_W_DETECT_U_THRESHOLD`, `SPELL_W_DETECT_LOW_CONF_THRESHOLD`, `SPELL_W_DETECT_CONFIRM_THRESHOLD`); defaults are best-guess and need empirical tuning against real recordings.
- The detector is W-specific by design — no abstraction for "multi-syllable letters" because W is the only one in English.

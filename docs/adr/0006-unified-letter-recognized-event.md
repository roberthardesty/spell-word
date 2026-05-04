# Unified `LETTER_RECOGNIZED` event with optional retract count

The letter recognizer emits one event type — `LETTER_RECOGNIZED { top_k, retract_count }` — rather than two distinct events for the normal case and the W-recovery case. The decoder always handles letter events uniformly: drop `retract_count` entries from its in-flight buffer, append `top_k`. `retract_count = 0` is the normal case; `retract_count = 2` fires only when W-recovery confirms a multi-utterance W (per ADR-0005). The `retract_count` field is generalizable: a future letter-level recovery mechanism needing different arity can use the same event without growing the event vocabulary.

## Considered options

- **Two distinct event types: `LETTER_TOP_K` (normal) and `RETRACT_AND_REPLACE` (W).** This is what `spell_events.h` originally declared. Rejected: the decoder grows two code paths to keep its in-flight buffer consistent, with the W path being a special case requiring separate test coverage. Both events do the same fundamental thing — adjust the in-flight letter buffer — and unifying them concentrates the buffer-mutation logic on the decoder side.
- **Hybrid: always emit `LETTER_TOP_K`, plus an optional `RETRACT` event when needed.** Rejected: re-introduces the inconsistent-state window (decoder sees a letter, then has to undo it) that ADR-0005's variant 2 was specifically chosen to avoid.

## Consequences

- The previously declared events `LETTER_TOP_K` and `RETRACT_AND_REPLACE` in `spell_events.h` are replaced by `LETTER_RECOGNIZED { top_k, retract_count }`.
- The decoder has one letter-event handler, not two; its in-flight-buffer mutation is a single code path.
- Future letter-level recovery mechanisms (none currently planned beyond W) reuse the same event with different `retract_count` values rather than growing the event vocabulary.
- The recognizer's `LETTER_RECOGNIZED` emission may be delayed by ~40 ms when the W-trigger predicate fires; this is structurally inside the 500 ms early-commit silence floor (ADR-0001).

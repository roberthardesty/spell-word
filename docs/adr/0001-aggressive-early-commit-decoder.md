# Aggressive early-commit decoder

The decoder commits as soon as a three-clause predicate is satisfied — score margin over same-length runner-up, score margin over best longer candidate, and post-letter silence ≥ `SPELL_EARLY_MIN_SILENCE_MS` (default 500 ms) — rather than waiting for the full end-of-word silence (`SPELL_VAD_END_OF_WORD_MS`, default 1200 ms). The silence-floor clause is load-bearing: without it, a valid prefix of a longer word (BAN inside BANK) wins on 0-edit-vs-1-edit-deletion because `SPELL_EDIT_DEL_PENALTY = -2.5` exceeds typical log-frequency separation, and the decoder would systematically truncate users mid-word. Implemented via a new `SPELL_EVENT_EARLY_COMMIT_WINDOW` segmenter event; the decoder evaluates the predicate on receipt of that event and on full EOW.

## Considered options

- **Status quo full-EOW wait.** Rejected: 1.2 s of silence is brittle UX for the kid users we're optimizing for under our (b) robustness bar.
- **Margin-over-runner-up alone (no silence floor).** Rejected: prefix-truncation failure mode above.
- **Asymmetric EOW (shorter when more letters).** Rejected: ad-hoc heuristic that masks segmentation bugs and tells you nothing about why latency is what it is.

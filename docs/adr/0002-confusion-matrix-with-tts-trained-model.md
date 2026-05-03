# Confusion matrix retained at α=0.15 despite TTS-only training

The DS-CNN is trained on TTS synthetic audio, so the empirical confusion matrix `M` derived from held-out TTS captures phonetic confusion universals (B/D/G rhyme; F/S/X hiss) but not speaker-variance confusions present in real human speech. We ship `M` anyway with default `SPELL_DECODER_ALPHA = 0.15` because the phonetic universals transfer across audio distributions and `M` is the only mechanism that recovers letters outside top-K (where the network's softmax effectively assigns near-zero mass and the only signal is `M[top1, ·]`). α is tunable via `spell_config.h`; expect to A/B test α ∈ {0, 0.05, 0.15} against real-audio QA fixtures once they exist, and revisit this ADR if the TTS-derived matrix actively hurts on real audio.

## Considered options

- **Skip M entirely (α=0, drop the partition).** Rejected: removes the only recovery path for outside-top-K cases and forfeits free phonetic-universal correction.
- **Build the wiring but ship α=0 by default.** Rejected as middle ground: same implementation cost as shipping at α=0.15, but defers the upside to a tuning pass that may never happen if no one re-enables it.

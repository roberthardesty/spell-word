# Decoder tuning uses the Python reference against host-recorded WAVs

The C decoder's hyperparameters (`SPELL_DECODER_ALPHA`, `SPELL_DECODER_LAMBDA`, edit penalties, decision margin, early-commit margins) require fast-iteration sweeps against a representative QA fixture, which can't be done with manual mic sessions on the device. The Python pipeline from the model-training project is treated as the reference implementation: Mac-recorded WAVs → Python feat-extract → Python int8 inference → top-K → Python decoder. All hyperparameter sweeps run there. The C decoder is implemented as a port of the Python reference and validated against shared fixtures (same input → identical resolved word and score). Re-tuning is expected when moving from Mac mic to ICS-43434 + room acoustics, because the front-end distribution shifts; the QA set is therefore designed to be re-recorded on-device cheaply (consistent word list, consistent script).

## Considered options

- **Build host-side C inference (TFLM + feat-extract host build).** Rejected: ~2–3 days of work to refactor `feat_extract.c` for a host FFT shim and cross-compile TFLM, when an existing Python pipeline already does the same job.
- **Defer all tuning to live device sessions in Phase 7.** Rejected: too slow a loop to actually hit the (b) robustness bar; you converge on "good enough" defaults rather than tuning to data.
- **Hand-label top-K events without running inference.** Rejected as unrealistic — humans can't predict softmax distributions.

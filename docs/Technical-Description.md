### Spell-Word Device Firmware

Embedded firmware for a battery-powered handheld reading aid that performs on-device classification of spelled letters and plays back the corresponding word from a fixed audio corpus.

**Hardware Platform**

- MCU: ESP32-S3, either N16R8 (16 MB flash / 8 MB PSRAM) or N8R8 (8 MB flash / 8 MB PSRAM)
- Audio input: I2S MEMS microphone
- Audio output: I2S DAC/amplifier driving a small speaker
- Controls: SPST on/off switch, momentary push-to-talk button
- Power: Single-cell Li-ion/LiPo with onboard charge management
- Enclosure: Clip-on housing designed to attach to a book spine or page edge

**Audio Capture & Preprocessing**

The button UX is press-once-to-arm: a single press wakes the device into capture mode and emits a brief confirmation tone. The user then spells the word with short pauses between letters, and end-of-word is detected by sustained silence (~1.0–1.5 s). Press-and-hold is rejected as a UX option. A child reading from a book is already using both hands to hold the book and turn pages, and cannot reliably maintain a button press while spelling.

In capture mode, the firmware streams 16 kHz mono audio into a ring buffer and segments it with energy-based VAD: frame-level energy marks letter onsets and offsets, with a minimum letter duration (~150 ms) to reject transient noise and a minimum inter-letter gap (~80 ms) to avoid splitting single letters. Each segmented utterance is converted to log-mel features for inference.

Letter segmentation is the highest-risk part of the front-end pipeline. Children pause irregularly, run letters together, restart, and add filler sounds. The segmenter is intentionally conservative. better to over-segment and let the decoder handle spurious utterances (see Length-Mismatch Robustness) than to merge two letters into one window the network cannot recognize.

**Inference**

A pre-trained, int8-quantized Depthwise Separable CNN (DS-CNN) classifies each utterance across the 26-letter alphabet. DS-CNN is well-suited to this MCU class. small parameter count, low MAC count per inference, and good accuracy on short keyword-spotting tasks. Quantization is required for both performance and memory; ESP-NN and TensorFlow Lite Micro (or ESP-DL) provide the int8 runtime.

For each utterance, the inference stage outputs the top-K letter predictions with their softmax probabilities (K = 5 is the default. large enough to capture the true letter on near-miss errors, small enough that the long tail is discarded as noise). The probability mass below top-K is collapsed into a single "other" bucket so the downstream scorer can still account for it as a small uniform-over-the-rest prior rather than treating those letters as exactly zero.

**Confusion-Aware Word Resolution**

Spoken-letter recognition has well-documented systematic errors driven by acoustic similarity. The English alphabet clusters into groups whose members rhyme and share phonetic structure:

- E-set: B, C, D, E, G, P, T, V, Z (all end in /iː/). the largest and most error-prone group
- A-set: A, J, K (and sometimes H)
- Sibilants: F, S, X
- Nasals: M, N
- /aɪ/ pair: I, Y
- Mostly distinctive: L, O, Q, R, U, W

The DS-CNN's own softmax already captures most of this. When the network is unsure between B and D it will assign nontrivial probability to both, and the top-K preserves that. The role of an explicit confusion model is narrower: to correct for systematic miscalibration (the network is confidently wrong in patterned ways) and to recover the rare case where the true letter falls outside top-K.

1. **Empirical confusion matrix.** During model evaluation, a 26×26 matrix M is computed on held-out children's speech, where M[j, i] = P(true = i | predicted top-1 = j). Row j is the empirical distribution over true letters conditioned on the network's most-confident prediction being j. Storage: ~2.7 KB. Note the direction. This is the inverse of the network's behavior and is the conditional we actually need at inference time, not P(predicted | true).

2. **Posterior assembly.** For each utterance, the per-letter posterior p_i(c) is a convex combination of the network's top-K softmax (renormalized over the K letters and the "other" bucket) and row M[top1_i, ·]. The mixing weight α controls how much the system defers to the network's own uncertainty vs. the empirical confusion pattern; α is tuned on a validation set and is expected to be small, since the network's softmax already encodes most of the relevant uncertainty and the matrix is a correction term, not the primary signal.

3. **Flat-list scoring.** With ~3,000 words averaging ~5 letters, the dictionary is small enough to score exhaustively. For each candidate word w whose length matches the utterance count, score(w) = Σ_i log p_i(w_i) + λ · log freq(w), where the second term is a Zipfian frequency prior. This is trivial to implement, easy to debug, and runs in well under a millisecond. A trie with beam search would be the textbook data structure but is overengineering at this corpus size. the constants matter more than the asymptotics.

4. **Decision and abstention.** The argmax word is played back only if its score margin over the runner-up exceeds a threshold; otherwise the device emits a short error chirp. Saying the wrong word to a child learning to read actively teaches them wrong, which is worse than saying nothing. abstention is the correct failure mode.

The dictionary itself is doing most of the heavy lifting: it acts as an error-correcting code over the letter sequence. Even when several top-1 predictions are wrong, the constraint that the sequence must spell a real common word almost always pins down the intended target. A noisy "B-A-T" with low confidence on the first letter is naturally resolved against the small set of common three-letter words ending in -AT, weighted by per-position posteriors and word frequency.

**Length-Mismatch Robustness**

The segmenter will sometimes emit too many or too few utterances. a doubled letter, a swallowed pause, a coughed frame, an "um." The flat-list scorer is extended with bounded-edit-distance decoding: for each candidate word, its score is the maximum over alignments of summed log-posteriors minus a per-edit penalty, computed by standard DP. With ≤2 edits permitted, ~3,000 words, and ~5 letters average, this stays well within the compute budget. The penalty is tuned so that a real one-edit recovery beats a zero-edit decode into a wrong word but does not beat a clean zero-edit decode into the right one.

**Audio Playback**

The 3,000-word pronunciation corpus lives in flash with a target footprint of ≤5 MB, leaving ample room for firmware and resources on either N8R8 (8 MB flash) or N16R8 (16 MB flash). Achieving this budget requires aggressive but well-understood audio compression: low-bitrate Opus at 16 kHz mono, careful trimming of leading and trailing silence on each clip, and per-clip loudness normalization done at corpus build time so the decoder doesn't need to gain-stage at runtime. Average per-clip size lands in the low-single-digit kilobytes.

The corpus is built offline as part of the firmware image: source recordings are normalized, trimmed, encoded, and packed into a flat indexed blob keyed by word ID. At runtime, the matched word ID looks up its byte range, the encoded payload is streamed through the decoder, and PCM is pushed out the I2S DAC. Either the N8R8 or N16R8 is viable on flash budget alone; final part selection is driven by firmware size and headroom for future corpus expansion rather than by the audio corpus itself.

**Risks**

- **Training data.** The DS-CNN must be trained or fine-tuned on children's speech. Children differ from adults in pitch, articulation, and prosody, and an off-the-shelf adult-trained letter classifier will degrade substantially. Sourcing a sufficiently large and demographically diverse children's-speech corpus is a precondition for the project.
- **Out-of-vocabulary words.** A 3,000-word dictionary covers most running text in graded readers but misses many specific nouns common in children's books (animal names, character names, place names). The device's behavior on OOV. abstain vs. play closest in-vocab word. is an explicit UX call.
- **Acoustic environment.** The microphone sits ~30 cm from the child's mouth in a typical reading posture, often in a noisy room. SNR will be substantially worse than typical keyword-spotting benchmarks; the model, segmenter, and matrix M must all be calibrated against representative real-world recordings, not clean studio audio.
- **Speaker intelligibility.** A small embedded speaker may not produce sufficient volume or clarity for the played-back word to be useful, especially for early readers in noisy environments. Speaker and amplifier selection should be validated early.
- **Decoder cost.** Low-bitrate Opus decoding on the ESP32-S3 is feasible but non-trivial. Decode-time CPU and PSRAM use should be benchmarked early to confirm playback is gapless and does not collide with capture-mode buffering.

**Open Questions**

- Children's-speech dataset: source, licensing, size, demographic coverage
- Top-K size (default K = 5 is a starting point, not a tuned value)
- Confusion-matrix mixing weight α and abstention threshold
- Edit-distance penalty calibration
- Power profile and expected runtime per charge
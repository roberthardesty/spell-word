# Spell-Word Firmware

Embedded firmware for a handheld reading aid: a child presses a button, spells a word letter by letter near the device, and the device speaks the resolved word back through a speaker.

## Language

### Capture and segmentation

**Utterance**:
A single segmented audio span believed to contain one spelled letter, packaged into the 800 ms inference window before being passed to the classifier.
_Avoid_: "letter audio", "letter clip", "sample"

**Pre-roll**:
The ~200 ms ring buffer of audio kept before a detected onset, so the inference window includes the moments leading into the spoken letter.

**End-of-word (EOW)**:
A sustained-silence event from the segmenter (default `SPELL_VAD_END_OF_WORD_MS` = 1200 ms) signaling that the user has finished spelling and the decoder must commit or abstain.

**Early-commit window**:
A shorter-silence event from the segmenter (default `SPELL_EARLY_MIN_SILENCE_MS` = 500 ms) signaling that the decoder *may* commit, conditional on its three-clause predicate. Distinct from EOW; not a commit on its own.

### Classification and decoding

**Letter**:
A position in the alphabet (A–Z), referred to by index 0–25. The symbol, not the audio.

**Top-K**:
The K most probable letters (with their softmax probabilities) for one utterance. Default K=5. Probability mass below top-K is treated as a uniform-over-21 prior in the decoder.

**Confusion matrix (M)**:
The 26×26 matrix where `M[j, i] = P(true=i | top1=j)`, computed upstream from held-out evaluation data. Mixed into the per-letter posterior with weight α (`SPELL_DECODER_ALPHA`).

**Word**:
A multi-letter target spelled by the user. Resolved against the dictionary by the decoder.

**Resolved word**:
The decoder's chosen `word_id` (or abstain) for one spelling attempt.

**Spelling attempt**:
One IDLE → ARMED → SPELLING → (PLAYING | abstain) → IDLE cycle. One press, one outcome.

### Dictionaries and corpus

**Demo set**:
A curated ~30-word dictionary used for scripted demos. Hand-picked to avoid confusion-pair pitfalls.

**QA set**:
A representative ~100–200-word dictionary used for tuning and robustness validation. Reflects realistic word distribution.

**Corpus**:
Per-word audio recordings (Opus-encoded) packed into a flash partition, keyed by `word_id`, decoded on-device for playback.

### Builds

**Production firmware**:
The shipped device-facing build. No network interfaces, no telemetry, no upload paths.

**Test build**:
A separate Kconfig-gated build that adds telemetry (PSRAM-resident audio capture, WiFi upload of utterance + inference + decoder + segmenter state). Not shipped.

## Relationships

- An **utterance** produces one **top-K** event after inference.
- A sequence of **utterances** plus an **EOW** (or an **early-commit window** with predicate met) produces one **resolved word**.
- A **resolved word** maps via the corpus index to a byte range in the **corpus**, which is decoded and played.
- The **demo set** is a curated subset of the **QA set**'s shape (not necessarily a literal subset of words).

## Example dialogue

> **Dev:** "If the user spells 'B-A-N-K', does the decoder ever commit on 'B-A-N'?"
> **Architect:** "Only if the early-commit predicate's third clause is satisfied — 500 ms of silence after the 'N'. The first two clauses (margin over same-length runner-up, margin over best longer candidate) will both push toward BAN over BANK because BANK costs a deletion penalty. The silence floor is what defeats prefix-truncation."

## Flagged ambiguities

- "Letter" originally referred ambiguously to the audio span and the alphabet symbol. Resolved: **utterance** is the audio span, **letter** is the symbol (index 0–25).
- "Dictionary" was used for both the curated demo set and the representative QA set during planning. Resolved: those are distinct dictionaries with different purposes; the demo set ships in the corpus partition for MVP demos, the QA set lives in `tools/` for tuning sweeps.

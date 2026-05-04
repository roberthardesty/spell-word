# Pick up here

You are landing into the Spell-Word firmware repo (ESP32-S3 handheld reading aid) mid-implementation of Phase 4–5. Read this first.

## Orient

- **Domain glossary**: `CONTEXT.md`. Use the terms there (utterance, letter, top-K, EOW, early-commit window, letter recognizer, etc.) — they are load-bearing across the design docs.
- **Active PRD**: `docs/prds/0001-phase-4-5-ux-and-decoder.md`. The §"Sequencing" subsection in "Further Notes" is the canonical implementation order.
- **ADRs**: `docs/adr/0001`–`0006`. ADR-0005 (W-recovery inside the recognizer) and ADR-0006 (unified `LETTER_RECOGNIZED` event) are the most recent and newly load-bearing.
- **Issue tracker**: Vikunja, project ID 3 ("Spell-Word"). 22 open tasks at last count, all `needs-triage`. Use the `vikunja` skill to scan/update.
- **Recent work**: see `CHANGELOG.md` and `git log`. The last commit (`6594f51`) closed Vikunja #40 + #41.

## What is done

- **Phases 0–3**: audio capture (I2S RX), segmenter (energy VAD, EOW), first-pass letter recognizer (MFCC + TFLM DS-CNN → top-K).
- **Naming + event vocabulary**: `letter_classifier` is now `letter_recognizer`; the wire vocabulary is the unified `SPELL_RECOGNIZER_EVENT::LETTER_RECOGNIZED { top_k, retract_count }`. Decoder + W-recovery code can now be written against the final shape.
- **Build**: `idf.py build` is clean (`spell_word.bin` 0x61150 bytes, 81% app partition free).

## What is NOT done

- **Phase 4** (UX): no button, debounce, state machine, tones, I2S TX, or playback path. `components/spell_ui/` exists with stubs only.
- **Phase 5** (decoder + W-recovery): no decoder, no PCM ring, no W detector, no early-commit predicate. `components/decoder/` does not exist yet.
- **TFLM arena precursor (Vikunja #39)**: arena still in PSRAM with a 512 KB allocation; the new 51 KB model has not been loaded into the `model` partition; invoke-time on hardware has not been re-measured. ADR-0001's 500 ms early-commit floor and ADR-0005's ~40 ms W-recovery latency claims both depend on this landing.
- **Phase 6** (word audio playback): out of scope for the active PRD. `playback_play_word` is a stub returning `ESP_ERR_NOT_SUPPORTED`.

## Where to start

Pick from these unblocked tasks. They are all independent code-only AFK slices except where noted.

| Vikunja | Title | Type | Notes |
|---|---|---|---|
| **#19** | `seg_core` early-commit-window event + tests | AFK | Pure extension to `segmenter_core` — fire `SEG_EVT_EARLY_COMMIT_WINDOW` once per inter-letter silence cycle. No dependencies on other open work. PRD sequencing step (c). Probably the cleanest next slice. |
| **#27** | letter_recognizer PCM ring (last 4 utterances in PSRAM, internal) | AFK | Just unblocked by the rename. Internal-only — no public access function. PRD sequencing step (d). |
| **#28** | `w_detector_core` trigger logic + host tests | AFK | Pure C99 sub-module of letter_recognizer. Just unblocked by the rename + event migration. Trigger predicate operates on top-K event metadata only — no PCM. PRD sequencing step (e). |
| **#20** | `decoder_core` full-EOW resolution + unified retract-aware letter handler | AFK | Just unblocked by event migration. Pure C99, host-portable. PRD sequencing step (g). Largest of the four. |
| #39 | TFLM arena → SRAM precursor | AFK with HW dep | Code change is small (resize arena, swap heap cap, log invoke time), but the acceptance criterion "invoke time in 30–40 ms range" requires hardware. Defer until you have board access OR if you only intend to ship the code change and let hardware verification land separately, file a follow-up. |
| #21, #22, #24, #30, #25, #26, #32 | Decoder predicate, decoder IDF wrapper, replay tool, W-recovery integration, UI state machine, cancel ordering, hardware bench | mixed | Each is blocked by one of the above; pick them up only after their dependencies. Read the task `Blocked by` field on Vikunja before starting. |

**Suggested next slice**: `#19` (seg_core early-commit-window). Smallest, independent, the segmenter has an existing host test rig at `tools/segmenter_replay/` that you can extend rather than build from scratch, and landing it unblocks the decoder's early-commit predicate (#21).

## Working norms in this repo

- Pure-C `_core` modules are host-tested via `tools/<module>_replay/` with input-data-in / output-data-out tests. IDF wrappers around each `_core` are NOT unit-tested; they get on-device acceptance only. Prior art: `tools/segmenter_replay/test_segmenter_core.c`, eight cases passing.
- Decoder hyperparameters (margins, silence floors, α, W-detect thresholds) live in `components/spell_common/include/spell_config.h` so they sweep cleanly in the Python reference and bake into firmware.
- Event bases declared in `spell_events.h`, defined in the producing component's `.c/.cpp`. Producing component owns the `ESP_EVENT_DEFINE_BASE`.
- Use `git mv` for renames so history is preserved.
- The Vikunja `Spell-Word` project labels track work-type (`AFK`, `Human`/HITL, `research`) and triage state (`needs-triage`). Progress is **not** tracked via labels — treat them as descriptive metadata only.
- This repo has no GitHub remote, so no PRs. Land directly on `main` with descriptive commit messages.

## Tool inventory

- `idf.py build` works (ESP-IDF v5.3.5 at `~/esp/esp-idf/`). Source `~/esp/esp-idf/export.sh` first.
- `vikunja` skill is configured (env vars set) — use it for task scans + closes.
- No GitHub remote; no `gh` operations expected.

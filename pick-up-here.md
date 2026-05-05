# Pick up here

You are landing into the Spell-Word firmware repo (ESP32-S3 handheld reading aid) mid-implementation of Phase 4–5. Read this first.

## Orient

- **Domain glossary**: `CONTEXT.md`. Use the terms there (utterance, letter, top-K, EOW, early-commit window, letter recognizer, etc.) — they are load-bearing across the design docs.
- **Active PRD**: `docs/prds/0001-phase-4-5-ux-and-decoder.md`. The §"Sequencing" subsection in "Further Notes" is the canonical implementation order.
- **ADRs**: `docs/adr/0001`–`0006`. ADR-0001 (aggressive early-commit), ADR-0005 (W-recovery inside the recognizer), and ADR-0006 (unified `LETTER_RECOGNIZED` event) are the most load-bearing for ongoing slices.
- **Issue tracker**: Vikunja, project ID 3 ("Spell-Word"). Use the `vikunja` skill to scan/update. The `Blocked by` field is now reliable as of 2026-05-04 — earlier stale dependencies on `#27` and `#28` were cleaned up.
- **Recent work**: see `CHANGELOG.md` and `git log`. Most recent slices, newest first: Vikunja #28 (`w_detector_core` trigger predicate), #21 (decoder early-commit predicate), #20 (decoder_core full-EOW + retract-aware letter handler), #19 (segmenter early-commit-window event), #41 (recognizer rename + unified `LETTER_RECOGNIZED` event).

## What is done

- **Phases 0–3**: audio capture (I2S RX), segmenter (energy VAD, EOW), first-pass letter recognizer (MFCC + TFLM DS-CNN → top-K).
- **Naming + event vocabulary**: `letter_classifier` is now `letter_recognizer`; the wire vocabulary is the unified `SPELL_RECOGNIZER_EVENT::LETTER_RECOGNIZED { top_k, retract_count }`. Decoder + W-recovery code can now be written against the final shape.
- **Segmenter early-commit-window event** (#19): `seg_core` fires `SEG_EVT_EARLY_COMMIT_WINDOW` once per inter-letter silence cycle, IDF wrapper republishes `SPELL_SEGMENTER_EVENT::EARLY_COMMIT_WINDOW`. 12 host tests passing in `tools/segmenter_replay/`.
- **Decoder core** (#20): `components/decoder/decoder_core.{c,h}` — pure C99, host-portable. Unified retract-aware letter handler + full-EOW resolution (posterior assembly with `(1−α)·net_prob + α·M[top1,·]` blend; bounded-edit dictionary scoring; abstain on margin).
- **Decoder early-commit predicate** (#21): `dec_try_early_commit()` evaluates the three-clause aggressive early-commit (margin over same-length runner-up, margin over best length-(N+1)/(N+2) deletion-aligned competitor, post-letter silence floor). 12 decoder host tests passing in `tools/decoder_replay/test_decoder_core.c` (7 full-EOW + 5 early-commit).
- **W-recovery trigger predicate** (#28): `components/letter_recognizer/w_detector_core.{c,h}` — pure-C99 sub-module of the recognizer (per ADR-0005). Four-condition trigger over a sliding window of the last 2 prior top-1 probabilities; per-condition diagnostic flags for US-16 logging. 10 host tests passing in `tools/w_detector_replay/test_w_detector_core.c`.
- **Build**: `idf.py build` clean (`spell_word.bin` 0x611b0 bytes, 81% app partition free).

## What is NOT done

- **Phase 4** (UX): no button, debounce, state machine, tones, I2S TX, or playback path. `components/spell_ui/` exists with stubs only.
- **Phase 5** (decoder + W-recovery), remaining work:
  - **Decoder IDF wrapper + partition loaders** (#22) — wires `decoder_core` to `esp_event` and loads the confusion matrix / dictionary from custom partitions. Now unblocked (#21 done).
  - **Letter recognizer PCM ring** (#27) — internal 4-deep PSRAM ring of the last utterance buffers (~100 KB), no public access function (per ADR-0005).
  - **W-recovery integration** (#30) — wires the trigger + PCM ring + merged-rerun into the recognizer task; emits `LETTER_RECOGNIZED { retract_count = 2 }` on confirmed W. Blocked on #22, #27, #39 (#28 now done).
  - **`decoder_replay` CLI** (#24) — host fixture-replay tool sharing `decoder_core` with the unit tests. Now unblocked (#21 done).
  - **`w_detector_replay` CLI** (#29) — host fixture-replay tool sharing `w_detector_core` with the unit tests. Now unblocked (#28 done).
- **TFLM arena precursor** (#39): arena still in PSRAM with a 512 KB allocation; the new 51 KB model has not been loaded into the `model` partition; invoke-time on hardware has not been re-measured. ADR-0001's 500 ms early-commit floor and ADR-0005's ~40 ms W-recovery latency claims both depend on this landing.
- **Phase 6** (word audio playback): out of scope for the active PRD. `playback_play_word` is a stub returning `ESP_ERR_NOT_SUPPORTED`.

## Where to start

Pick from these unblocked tasks. They are all independent code-only AFK slices except where noted.

| Vikunja | Title | Type | Notes |
|---|---|---|---|
| **#22** | Decoder IDF wrapper + matrix/dict partition loading | AFK | Newly unblocked by #21. Wires `decoder_core` (full-EOW + early-commit) to `esp_event` and loads confusion matrix + dictionary from custom partitions. Touches `partitions.csv`. PRD sequencing step (h). |
| **#24** | `decoder_replay` host CLI + fixture format | AFK | Newly unblocked by #21. CSV fixture replay sharing `decoder_core` with the unit tests; all `spell_config.h` tunables overridable by CLI flag. Pairs with #22 — same `decoder_core` linkage. |
| **#27** | letter_recognizer PCM ring (last 4 utterances in PSRAM, internal) | AFK | Internal-only — no public access function. PRD sequencing step (d). |
| **#29** | `w_detector_replay` host CLI + fixture format | AFK | Newly unblocked by #28. CSV fixture replay sharing `w_detector_core` with the unit tests. Smallest follow-on to #28. |
| #39 | TFLM arena → SRAM precursor | AFK with HW dep | Code change is small (resize arena, swap heap cap, log invoke time), but the acceptance criterion "invoke time in 30–40 ms range" requires hardware. Defer until you have board access OR if you only intend to ship the code change and let hardware verification land separately, file a follow-up. |
| #30, #25, #26, #32 | W-recovery integration, UI state machine, cancel ordering, hardware bench | mixed | Each is blocked by one of the above; pick them up only after their dependencies. The `Blocked by` field on Vikunja is now accurate after the 2026-05-04 cleanup. |

**Suggested next slice**: `#22` (decoder IDF wrapper + partition loaders). Tightest continuation of the decoder critical path you now have full context for — `decoder_core`'s full-EOW resolver, early-commit predicate, and the unified `LETTER_RECOGNIZED` event are all in place; this just wires them to `esp_event` and the model/dictionary partitions. Unblocks `#30` (W-recovery integration). Alternatives: `#27` (PCM ring — also feeds #30) or `#24` / `#29` (host CLIs, smaller).

## Working norms in this repo

- Pure-C `_core` modules are host-tested via `tools/<module>_replay/` with input-data-in / output-data-out tests. IDF wrappers around each `_core` are NOT unit-tested; they get on-device acceptance only. Three prior-art examples: `tools/segmenter_replay/test_segmenter_core.c` (12 cases passing), `tools/decoder_replay/test_decoder_core.c` (12 cases passing), and `tools/w_detector_replay/test_w_detector_core.c` (10 cases passing).
- Per-module `tools/<x>_replay/` directories own a `.gitignore` excluding `*.o` and the test/CLI binaries (host-build artifacts, platform-specific). The component-side `*.o` is excluded from the root `.gitignore`. Don't commit build artifacts.
- Decoder hyperparameters (margins, silence floors, α, W-detect thresholds) live in `components/spell_common/include/spell_config.h` so they sweep cleanly in the Python reference and bake into firmware.
- Event bases declared in `spell_events.h`, defined in the producing component's `.c/.cpp`. Producing component owns the `ESP_EVENT_DEFINE_BASE`.
- Use `git mv` for renames so history is preserved.
- The Vikunja `Spell-Word` project labels track work-type (`AFK`, `Human`/HITL, `research`) and triage state (`needs-triage`). Progress is **not** tracked via labels — treat them as descriptive metadata only. Mark tasks done via `done: true` on the task itself.
- This repo has no GitHub remote, so no PRs. Land directly on `main` with descriptive commit messages.
- Spurious clangd diagnostics (`-mlongcalls` unknown, `'../hal.h' file not found`, etc.) come from the cached Xtensa toolchain flags and are emitted for every existing IDF file. Ignore them; rely on `idf.py build` and the host `make test` for actual validation.

## Tool inventory

- `idf.py build` works (ESP-IDF v5.3.5 at `~/esp/esp-idf/`). Source `~/esp/esp-idf/export.sh` first.
- `vikunja` skill is configured (env vars set) — use it for task scans + closes. After landing a slice, mark the corresponding Vikunja task done with `vk update /tasks/<id> -` body `{"done": true}`.
- No GitHub remote; no `gh` operations expected.

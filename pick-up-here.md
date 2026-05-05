# Pick up here

You are landing into the Spell-Word firmware repo (ESP32-S3 handheld reading aid) mid-implementation of Phase 4–5. Read this first.

## Orient

- **Domain glossary**: `CONTEXT.md`. Use the terms there (utterance, letter, top-K, EOW, early-commit window, letter recognizer, etc.) — they are load-bearing across the design docs.
- **Active PRD**: `docs/prds/0001-phase-4-5-ux-and-decoder.md`. The §"Sequencing" subsection in "Further Notes" is the canonical implementation order.
- **ADRs**: `docs/adr/0001`–`0006`. ADR-0001 (aggressive early-commit), ADR-0005 (W-recovery inside the recognizer), and ADR-0006 (unified `LETTER_RECOGNIZED` event) are the most load-bearing for ongoing slices.
- **Issue tracker**: Vikunja, project ID 3 ("Spell-Word"). Use the `vikunja` skill to scan/update. The `Blocked by` field is now reliable as of 2026-05-04 — earlier stale dependencies on `#27` and `#28` were cleaned up.
- **Recent work**: see `CHANGELOG.md` and `git log`. Most recent slices, newest first: Vikunja #27 (letter_recognizer internal PCM ring), #22 (decoder IDF wrapper + matrix/dict partition loaders), #29 (`w_detector_replay` CLI + CSV fixture format), #28 (`w_detector_core` trigger predicate), #21 (decoder early-commit predicate), #20 (decoder_core full-EOW + retract-aware letter handler), #19 (segmenter early-commit-window event), #41 (recognizer rename + unified `LETTER_RECOGNIZED` event).

## What is done

- **Phases 0–3**: audio capture (I2S RX), segmenter (energy VAD, EOW), first-pass letter recognizer (MFCC + TFLM DS-CNN → top-K).
- **Naming + event vocabulary**: `letter_classifier` is now `letter_recognizer`; the wire vocabulary is the unified `SPELL_RECOGNIZER_EVENT::LETTER_RECOGNIZED { top_k, retract_count }`. Decoder + W-recovery code can now be written against the final shape.
- **Segmenter early-commit-window event** (#19): `seg_core` fires `SEG_EVT_EARLY_COMMIT_WINDOW` once per inter-letter silence cycle, IDF wrapper republishes `SPELL_SEGMENTER_EVENT::EARLY_COMMIT_WINDOW`. 12 host tests passing in `tools/segmenter_replay/`.
- **Decoder core** (#20): `components/decoder/decoder_core.{c,h}` — pure C99, host-portable. Unified retract-aware letter handler + full-EOW resolution (posterior assembly with `(1−α)·net_prob + α·M[top1,·]` blend; bounded-edit dictionary scoring; abstain on margin).
- **Decoder early-commit predicate** (#21): `dec_try_early_commit()` evaluates the three-clause aggressive early-commit (margin over same-length runner-up, margin over best length-(N+1)/(N+2) deletion-aligned competitor, post-letter silence floor). 12 decoder host tests passing in `tools/decoder_replay/test_decoder_core.c` (7 full-EOW + 5 early-commit).
- **W-recovery trigger predicate** (#28): `components/letter_recognizer/w_detector_core.{c,h}` — pure-C99 sub-module of the recognizer (per ADR-0005). Four-condition trigger over a sliding window of the last 2 prior top-1 probabilities; per-condition diagnostic flags for US-16 logging. 10 host tests passing in `tools/w_detector_replay/test_w_detector_core.c`.
- **W-detector host replay CLI** (#29): `tools/w_detector_replay/main.c` builds a `w_detector_replay` binary that ingests a CSV of synthetic top-K events and prints the trigger decision + per-condition flags per row. Both thresholds (`SPELL_W_DETECT_U_THRESHOLD`, `SPELL_W_DETECT_LOW_CONF_THRESHOLD`) overridable via CLI flag. Same `w_detector_core.c` linkage as the unit-test rig.
- **Decoder IDF wrapper + partition loaders** (#22): `components/decoder/decoder.{c,h}` glues `decoder_core` to `esp_event` (subscribes to `LETTER_RECOGNIZED`, `EARLY_COMMIT_WINDOW`, `END_OF_WORD`; publishes `WORD_RESOLVED` / `WORD_ABSTAIN`) and loads the confusion matrix (subtype 0x81, label `confusion`) + dictionary (subtype 0x82, label `dictionary`) from custom data partitions. Strict 1.0 ± 1e-3 row-sum validation; `SPDC` magic + record-format validation for the dictionary; empty/erased partition fails init loudly with a clear log line (intentional divergence from `letter_recognizer.cpp`'s soft-fail since the matrix and dictionary are load-bearing for any decoding). `decoder_init()` is called from `app_main.c` and soft-fails the boot continuation when partitions are erased so dev boards still spell into the void. Dictionary binary format documented at the head of `decoder.c`. Handlers run on the default event-loop task — no separate task.
- **Letter recognizer PCM ring** (#27): `components/letter_recognizer/letter_recognizer.cpp` now retains the last 4 inference-eligible utterance windows in a single contiguous 100 KB PSRAM block (`SPELL_W_DETECT_PCM_RING_DEPTH × SPELL_INFERENCE_WINDOW_BYTES`). Allocation happens once at `letter_recognizer_init()`; failure returns `ESP_ERR_NO_MEM` and aborts boot via the existing `ESP_ERROR_CHECK` site in `app_main`. `pcm_ring_insert()` is invoked only on the inference-success path (after `esp_event_post(LETTER_RECOGNIZED)`), copying up to `SPELL_INFERENCE_WINDOW_SAMPLES` int16 samples into the head slot, advancing the head modulo depth, and saturating the count at depth. Ring state (`s_pcm_ring_storage`, `s_pcm_ring_n_samples`, `s_pcm_ring_head`, `s_pcm_ring_count`) is file-scope `static` — no public access function exposed (per ADR-0005, W-recovery is the sole consumer and runs inside the same recognizer task). `idf.py build` clean (`spell_word.bin` 0x63230 bytes, 81% free); 12/12 decoder + 10/10 w_detector host tests still pass.
- **Build**: `idf.py build` clean (`spell_word.bin` 0x63230 bytes, 81% app partition free).

## What is NOT done

- **Phase 4** (UX): no button, debounce, state machine, tones, I2S TX, or playback path. `components/spell_ui/` exists with stubs only.
- **Phase 5** (decoder + W-recovery), remaining work:
  - **W-recovery integration** (#30) — wires the trigger + PCM ring + merged-rerun into the recognizer task; emits `LETTER_RECOGNIZED { retract_count = 2 }` on confirmed W. Now blocked only on #39 (#22, #27, and #28 done) — the AFK gate is gone, only the hardware-dependent TFLM arena precursor remains.
  - **`decoder_replay` CLI** (#24) — host fixture-replay tool sharing `decoder_core` with the unit tests. Unblocked since #21 landed.
  - **Per-spelling summary log line** (#31) — one greppable digest per `WORD_RESOLVED` / `WORD_ABSTAIN`. Now blocked only on #30.
  - **Dictionary partition population**: the `confusion` and `dictionary` partitions are wired up in firmware but still erased on dev boards. `decoder_init()` logs a warning and the boot continues; spelling is silent until both are flashed. The dictionary binary format is documented at the head of `components/decoder/decoder.c` (16 B header `'SPDC' | version=1 | n_words | reserved=0` + packed `word_id, length, letters[length]` records). No host tooling exists yet to generate either blob.
- **TFLM arena precursor** (#39): arena still in PSRAM with a 512 KB allocation; the new 51 KB model has not been loaded into the `model` partition; invoke-time on hardware has not been re-measured. ADR-0001's 500 ms early-commit floor and ADR-0005's ~40 ms W-recovery latency claims both depend on this landing.
- **Phase 6** (word audio playback): out of scope for the active PRD. `playback_play_word` is a stub returning `ESP_ERR_NOT_SUPPORTED`.

## Where to start

Pick from these unblocked tasks. They are all independent code-only AFK slices except where noted.

| Vikunja | Title | Type | Notes |
|---|---|---|---|
| **#24** | `decoder_replay` host CLI + fixture format | AFK | Unblocked since #21. CSV fixture replay sharing `decoder_core` with the unit tests; all `spell_config.h` tunables overridable by CLI flag. Mirror the `tools/w_detector_replay/main.c` pattern. |
| #39 | TFLM arena → SRAM precursor | AFK with HW dep | Code change is small (resize arena, swap heap cap, log invoke time), but the acceptance criterion "invoke time in 30–40 ms range" requires hardware. Defer until you have board access OR if you only intend to ship the code change and let hardware verification land separately, file a follow-up. |
| #25 | spell_ui IDF wrapper (button + tones + I2S TX + state machine) | HITL | Unblocked by #22 + #23. Hardware bench test required. |
| #30, #26, #31, #32 | W-recovery integration, cancel ordering, per-spelling summary, hardware bench | mixed | Each is blocked by one of the above; pick them up only after their dependencies. The `Blocked by` field on Vikunja is now accurate after the 2026-05-04 cleanup. |

**Suggested next slice**: `#24` (decoder host CLI). Pure host-build AFK slice — mirror `tools/w_detector_replay/main.c` for the CLI shape, share `decoder_core.{c,h}` directly with `tools/decoder_replay/test_decoder_core.c`, expose every `spell_config.h` decoder tunable (margins, α, silence floor, edit penalties) as `--<name>` overrides, and add a sample fixture under `fixtures/`. #30 is now blocked only on the hardware-dependent #39, so the next AFK code slice belongs in tooling. Alternatives: `#39` (small code change, hardware-bound acceptance — file a follow-up if you ship without bench access) or `#25` (UI state-machine IDF wrapper, HITL — needs hardware).

## Working norms in this repo

- Pure-C `_core` modules are host-tested via `tools/<module>_replay/` with input-data-in / output-data-out tests. IDF wrappers around each `_core` are NOT unit-tested; they get on-device acceptance only. Three prior-art examples for `_core` tests: `tools/segmenter_replay/test_segmenter_core.c` (12 cases passing), `tools/decoder_replay/test_decoder_core.c` (12 cases passing), and `tools/w_detector_replay/test_w_detector_core.c` (10 cases passing). Three prior-art IDF wrappers to mirror: `components/segmenter/segmenter.c` (subscribes to audio_capture, owns a task, publishes EOW + early-commit-window), `components/letter_recognizer/letter_recognizer.cpp` (partition load with erased-flash detection, owns a task, internal PCM ring with init-time PSRAM allocation and `static`-only access), and `components/decoder/decoder.c` (event-loop handlers only, partition load with strict validation, fail-loud init). Two prior-art replay CLIs to mirror when adding new ones: `tools/segmenter_replay/main.c` (WAV in → seg events out) and `tools/w_detector_replay/main.c` (CSV in → trigger eval out, with `--<tunable>` overrides + sample fixture).
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

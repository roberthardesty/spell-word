# Pick up here

You are landing into the Spell-Word firmware repo (ESP32-S3 handheld reading aid) mid-implementation of Phase 4–5. Read this first.

## Orient

- **Domain glossary**: `CONTEXT.md`. Use the terms there (utterance, letter, top-K, EOW, early-commit window, letter recognizer, etc.) — they are load-bearing across the design docs.
- **Active PRD**: `docs/prds/0001-phase-4-5-ux-and-decoder.md`. The §"Sequencing" subsection in "Further Notes" is the canonical implementation order.
- **ADRs**: `docs/adr/0001`–`0006`. ADR-0001 (aggressive early-commit), ADR-0005 (W-recovery inside the recognizer), and ADR-0006 (unified `LETTER_RECOGNIZED` event) are the most load-bearing for ongoing slices.
- **Issue tracker**: Vikunja, project ID 3 ("Spell-Word"). Use the `vikunja` skill to scan/update. The `Blocked by` field is now reliable as of 2026-05-04 — earlier stale dependencies on `#27` and `#28` were cleaned up.
- **Recent work**: see `CHANGELOG.md` and `git log`. **Uncommitted on disk (2026-05-07, follow-up to #39, scoped under Vikunja #53)**: per-Conv2D-instance `MicroProfiler` hook in `letter_recognizer.cpp` + on-device baseline captured. The per-instance numbers update the working hypothesis inside the **#53 cold-start guide** at the bottom of this doc — read that before next steps. `git diff components/letter_recognizer/letter_recognizer.cpp` to see the change; either commit it (recommended — it's bench equipment for every subsequent #53 experiment) or revert before doing anything else. Most recent committed slices, newest first: Vikunja #39 (TFLM tensor arena + model in internal SRAM, 118 KiB arena, per-op profiling, `flat26_baseline_int8` bench measurement → surfaced #53 Conv2D bottleneck), #24 (`decoder_replay` host CLI + fixture format), #27 (letter_recognizer internal PCM ring), #22 (decoder IDF wrapper + matrix/dict partition loaders), #29 (`w_detector_replay` CLI + CSV fixture format), #28 (`w_detector_core` trigger predicate), #21 (decoder early-commit predicate), #20 (decoder_core full-EOW + retract-aware letter handler), #19 (segmenter early-commit-window event), #41 (recognizer rename + unified `LETTER_RECOGNIZED` event).

## What is done

- **Phases 0–3**: audio capture (I2S RX), segmenter (energy VAD, EOW), first-pass letter recognizer (MFCC + TFLM DS-CNN → top-K).
- **Naming + event vocabulary**: `letter_classifier` is now `letter_recognizer`; the wire vocabulary is the unified `SPELL_RECOGNIZER_EVENT::LETTER_RECOGNIZED { top_k, retract_count }`. Decoder + W-recovery code can now be written against the final shape.
- **Segmenter early-commit-window event** (#19): `seg_core` fires `SEG_EVT_EARLY_COMMIT_WINDOW` once per inter-letter silence cycle, IDF wrapper republishes `SPELL_SEGMENTER_EVENT::EARLY_COMMIT_WINDOW`. 12 host tests passing in `tools/segmenter_replay/`.
- **Decoder core** (#20): `components/decoder/decoder_core.{c,h}` — pure C99, host-portable. Unified retract-aware letter handler + full-EOW resolution (posterior assembly with `(1−α)·net_prob + α·M[top1,·]` blend; bounded-edit dictionary scoring; abstain on margin).
- **Decoder early-commit predicate** (#21): `dec_try_early_commit()` evaluates the three-clause aggressive early-commit (margin over same-length runner-up, margin over best length-(N+1)/(N+2) deletion-aligned competitor, post-letter silence floor). 12 decoder host tests passing in `tools/decoder_replay/test_decoder_core.c` (7 full-EOW + 5 early-commit).
- **W-recovery trigger predicate** (#28): `components/letter_recognizer/w_detector_core.{c,h}` — pure-C99 sub-module of the recognizer (per ADR-0005). Four-condition trigger over a sliding window of the last 2 prior top-1 probabilities; per-condition diagnostic flags for US-16 logging. 10 host tests passing in `tools/w_detector_replay/test_w_detector_core.c`.
- **W-detector host replay CLI** (#29): `tools/w_detector_replay/main.c` builds a `w_detector_replay` binary that ingests a CSV of synthetic top-K events and prints the trigger decision + per-condition flags per row. Both thresholds (`SPELL_W_DETECT_U_THRESHOLD`, `SPELL_W_DETECT_LOW_CONF_THRESHOLD`) overridable via CLI flag. Same `w_detector_core.c` linkage as the unit-test rig.
- **Decoder IDF wrapper + partition loaders** (#22): `components/decoder/decoder.{c,h}` glues `decoder_core` to `esp_event` (subscribes to `LETTER_RECOGNIZED`, `EARLY_COMMIT_WINDOW`, `END_OF_WORD`; publishes `WORD_RESOLVED` / `WORD_ABSTAIN`) and loads the confusion matrix (subtype 0x81, label `confusion`) + dictionary (subtype 0x82, label `dictionary`) from custom data partitions. Strict 1.0 ± 1e-3 row-sum validation; `SPDC` magic + record-format validation for the dictionary; empty/erased partition fails init loudly with a clear log line (intentional divergence from `letter_recognizer.cpp`'s soft-fail since the matrix and dictionary are load-bearing for any decoding). `decoder_init()` is called from `app_main.c` and soft-fails the boot continuation when partitions are erased so dev boards still spell into the void. Dictionary binary format documented at the head of `decoder.c`. Handlers run on the default event-loop task — no separate task.
- **`decoder_replay` host CLI + fixture format** (#24): `tools/decoder_replay/main.c` builds a `decoder_replay` binary alongside the existing `test_decoder_core` rig (both link the same `decoder_core.{c,h}`). Tagged-row fixture format: `L` carries 5×(letter, prob) + `retract_count` and feeds `dec_on_letter` then `dec_try_early_commit(silence_floor=false)`; `W` simulates `EARLY_COMMIT_WINDOW` (auto-resets state on commit); `E` calls `dec_resolve_full_eow` and resets; `R` resets without resolving; implicit EOW at end-of-file. Dictionary via `--dict` (`<word_id> <LETTERS>` per line); confusion matrix via `--matrix` or defaults to neutral diag=0.6 / off=0.4/25. Every `dec_config_t` tunable is exposed as a CLI flag — `--alpha`, `--decision-margin`, `--ins-penalty`, `--del-penalty`, `--early-margin-runnerup`, `--early-margin-longer`. Per-event lines log all top-K candidates + per-clause early-commit flags + computed margins. Bundled `fixtures/sample.csv` + `fixtures/sample_dict.txt` exercise both clean CAT (early-commit) and the W-recovery `retract=2` → WAT path. `make test` runs all 12 unit cases (still passing). `idf.py build` unchanged (`spell_word.bin` 0x63230 bytes, 81% free — host-only).
- **Letter recognizer PCM ring** (#27): `components/letter_recognizer/letter_recognizer.cpp` now retains the last 4 inference-eligible utterance windows in a single contiguous 100 KB PSRAM block (`SPELL_W_DETECT_PCM_RING_DEPTH × SPELL_INFERENCE_WINDOW_BYTES`). Allocation happens once at `letter_recognizer_init()`; failure returns `ESP_ERR_NO_MEM` and aborts boot via the existing `ESP_ERROR_CHECK` site in `app_main`. `pcm_ring_insert()` is invoked only on the inference-success path (after `esp_event_post(LETTER_RECOGNIZED)`), copying up to `SPELL_INFERENCE_WINDOW_SAMPLES` int16 samples into the head slot, advancing the head modulo depth, and saturating the count at depth. Ring state (`s_pcm_ring_storage`, `s_pcm_ring_n_samples`, `s_pcm_ring_head`, `s_pcm_ring_count`) is file-scope `static` — no public access function exposed (per ADR-0005, W-recovery is the sole consumer and runs inside the same recognizer task). `idf.py build` clean (`spell_word.bin` 0x63230 bytes, 81% free); 12/12 decoder + 10/10 w_detector host tests still pass.
- **TFLM tensor arena + model in internal SRAM, with bench measurement** (#39): `components/letter_recognizer/letter_recognizer.cpp` now allocates BOTH the tensor arena and the model flatbuffer from `MALLOC_CAP_INTERNAL`. Arena size is **118 KiB** (`SPELL_TENSOR_ARENA_SIZE = 118 * 1024`), set at ≤ 1.2 × the measured `arena_used_bytes` of 100,924 B (84% utilization). Model alloc tries an internal-SRAM cap of `SPELL_MODEL_INTERNAL_BUF_SIZE` (64 KiB — covers the 53 KiB `flat26_baseline_int8.tflite` + headroom) and falls back to PSRAM with a loud warning on memory pressure. Pre-/post-arena heap diagnostic logs free + largest-block on every recognizer init so future re-sizing can be done from real numbers. Per-op timing breakdown (`per-op us: conv= … dwconv= … fc= … pool= … smax= … add= … mul= …`) logs alongside the existing `timing:` line for the first 5 utterances + every 20th. **Bench measurements (2026-05-07, ESP32-S3 v0.2 @ 240 MHz, esp-tflite-micro 1.3.5 + esp-nn 1.2.3)**: pre-arena internal heap 354,359 B free / 258,048 B largest block; post-arena 167,983 B free / 73,728 B largest; arena_used 100,924 / 120,832 B (84%); USB-Serial-JTAG stays alive after flash. Total invoke = **532 ms** wall-clock, broken down as `conv=495628µs dwconv=18155µs fc=111µs pool=0µs smax=242µs add=0µs mul=0µs`. **Conv2D dominates at 93% of invoke time** — the original AC of "30-40 ms invoke" was based on extrapolation from the smaller EARS-POC sigmoid model and is unreachable with this Conv2D-heavy `flat26` architecture without further work; that work is now scoped under Vikunja **#53** — see the **#53 cold-start guide** at the bottom of this doc. The original investigation direction ("re-enable 3×3 optimized path / retrain to depthwise-separable") was based on a misread of the on-device model arch — the model is already canonical DS-CNN-S with zero full 3×3 Conv2D layers, so the 3×3 path is dead code on this architecture. #53 now blocks #30 (W-recovery latency claim).
- **Build**: `idf.py build` clean (`spell_word.bin` 0x63230 bytes, 81% app partition free).

## What is NOT done

- **Phase 4** (UX): no button, debounce, state machine, tones, I2S TX, or playback path. `components/spell_ui/` exists with stubs only.
- **Phase 5** (decoder + W-recovery), remaining work:
  - **Conv2D bottleneck on `flat26_baseline_int8`** (#53, surfaced from #39 bench on 2026-05-07): with arena and model both in internal SRAM, total invoke = 532 ms with **495 ms (93%) in Conv2D ops**. ADR-0001's 500 ms early-commit silence floor is **broken** at this invoke time (recognizer fires AFTER the floor expires, so `dec_try_early_commit` never sees the LETTER_RECOGNIZED event in time); ADR-0005's ~40 ms W-recovery latency claim is off by ~13×. The 2026-05-07 follow-up bench (per-Conv2D-instance, uncommitted on disk) splits the 495 ms aggregate into: **op 0 (wide first 4×10, im2col path) = 179 ms (34%)**, **each 1×1 PW = ~79 ms (60% total across ops 2/4/6/8)**, depthwise = 18 ms total (already fast — leave alone), MEAN = 18 ms. Op 0 is **5× slower per MAC** than 1×1 PW; the prior working hypothesis "1×1 PW dominates" is *partially* correct but op 0 is the single biggest hit. **All paths (including the asm-backed 1×1 PW) run 18-220× off realistic int8 SIMD throughput**, and MEAN at 18 ms for 25 K trivial reductions is a global-slowness canary, so the live question is now "is SIMD actually engaging?", not "which Conv2D kernel to rewrite". Investigation continues in the **#53 cold-start guide** at the bottom of this doc — step 1 (per-instance timing) is done; pick up at step 3 (verify SIMD engagement). The previous session's 3×3-opt experiment in `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c` (+ unguarded dispatcher in `esp_nn_conv_esp32s3.c`) is **dead code** on this architecture and carries a latent requantize bit-exactness bug — the 2026-05-07 per-instance bench confirms the gate never fires (op 0 routes to `esp_nn_conv_s8_im2col_s3`, ops 2/4/6/8 to `esp_nn_conv_s8_mult8_1x1_esp32s3.S`). ADR-0001 + ADR-0005 should be updated to reflect achievable latency once measured. **#53 now blocks #30**.
  - **W-recovery integration** (#30) — wires the trigger + PCM ring + merged-rerun into the recognizer task; emits `LETTER_RECOGNIZED { retract_count = 2 }` on confirmed W. Was blocked on #39 (now done); now blocked on **#53** (the ~40 ms W-recovery latency claim is invalid until Conv2D speed is addressed).
  - **Per-spelling summary log line** (#31) — one greppable digest per `WORD_RESOLVED` / `WORD_ABSTAIN`. Blocked only on #30.
  - **Feature frontend mismatches for `flat26_baseline_int8`** (#52, blocked by #39 — now unblocked): firmware's MFCC frontend produces 79 frames per 800 ms window (`SPELL_FEAT_N_FRAMES = 79`); the model expects 80 (input shape `[1, 80, 20, 3]`). Firmware also skips the per-coefficient/channel `(mfcc - mean) / std` normalization that the int8 quantization was calibrated against. Both surface only at letter-accuracy time, not at invoke time, so they don't gate #39 — but they DO block any real-letter recognition end-to-end. Norm arrays sit at `models/norm_mean.npy` + `models/norm_std.npy` (60 floats each).
  - **Dictionary partition population**: the `confusion` and `dictionary` partitions are wired up in firmware but still erased on dev boards. `decoder_init()` logs a warning and the boot continues; spelling is silent until both are flashed. The dictionary binary format is documented at the head of `components/decoder/decoder.c` (16 B header `'SPDC' | version=1 | n_words | reserved=0` + packed `word_id, length, letters[length]` records). No host tooling exists yet to generate either blob.
- **Phase 6** (word audio playback): out of scope for the active PRD. `playback_play_word` is a stub returning `ESP_ERR_NOT_SUPPORTED`.

## Where to start

Pick from these unblocked tasks.

| Vikunja | Title | Type | Notes |
|---|---|---|---|
| #53 | Conv2D bottleneck investigation | AFK + HITL | **Read the `#53 cold-start guide` at the bottom of this doc before touching anything.** Per-Conv2D-instance timing is wired in (uncommitted on disk in `letter_recognizer.cpp`; commit before next experiment). The 2026-05-07 baseline: op 0 = 179 ms, each 1×1 PW = ~79 ms, MEAN = 18 ms, depthwise = 18 ms total. Op 0 is 5× slower per MAC than 1×1 PW; everything is 18-220× off realistic SIMD peak. **The earlier 3×3-opt direction is still dead code on this model arch.** Next step is cold-start guide step 3 (verify SIMD engagement) — check arena/model alignment, cycle-count one 1×1 PW asm call to separate "SIMD stalls" from "SIMD doesn't engage". |
| #52 | Feature frontend: 79→80 frames + per-coef normalization | AFK + HITL | Unblocked now that #39 is done. Pure code change for the frame count + norm arrays; HITL accuracy verification needs a working invoke (which the current 532 ms invoke can technically do, just slowly). |
| #25 | spell_ui IDF wrapper (button + tones + I2S TX + state machine) | HITL | Unblocked by #22 + #23. Hardware bench test required. Independent of the Conv2D investigation. |
| #30, #26, #31, #32 | W-recovery integration, cancel ordering, per-spelling summary, hardware bench | mixed | Now blocked on **#53** rather than #39 (the latency claims that gated these are still unmet until the Conv2D issue resolves). |

**Suggested next slice**: pick whichever is most actionable for your context. **#53** is the highest-leverage (it unblocks every Phase 5 hardware-acceptance task) and lets you directly experiment on the device that's currently flashed. **#52** is parallel, smaller, and gets letter accuracy unstuck independently. **#25** needs no model timing at all and is a clean slice if you just want to make UI progress.

## Working norms in this repo

- Pure-C `_core` modules are host-tested via `tools/<module>_replay/` with input-data-in / output-data-out tests. IDF wrappers around each `_core` are NOT unit-tested; they get on-device acceptance only. Three prior-art examples for `_core` tests: `tools/segmenter_replay/test_segmenter_core.c` (12 cases passing), `tools/decoder_replay/test_decoder_core.c` (12 cases passing), and `tools/w_detector_replay/test_w_detector_core.c` (10 cases passing). Three prior-art IDF wrappers to mirror: `components/segmenter/segmenter.c` (subscribes to audio_capture, owns a task, publishes EOW + early-commit-window), `components/letter_recognizer/letter_recognizer.cpp` (partition load with erased-flash detection, owns a task, internal PCM ring with init-time PSRAM allocation and `static`-only access), and `components/decoder/decoder.c` (event-loop handlers only, partition load with strict validation, fail-loud init). Three prior-art replay CLIs to mirror when adding new ones: `tools/segmenter_replay/main.c` (WAV in → seg events out), `tools/w_detector_replay/main.c` (CSV in → trigger eval out, with `--<tunable>` overrides + sample fixture), and `tools/decoder_replay/main.c` (tagged-row CSV + `--dict` + optional `--matrix` + every `dec_config_t` tunable as `--<name>`).
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

## #53 cold-start guide: Conv2D bottleneck investigation

Read this before editing any conv kernel. The original #53 investigation lines were premised on a model arch the on-device flatbuffer doesn't actually have, and a previous session left a partial 3×3-opt experiment on disk that does not exercise on this model. Don't repeat that path.

### Ground truth: what the on-device model actually is

Verified by parsing `models/flat26_baseline_int8.tflite` directly + cross-checking the training config at `~/alphabet-R/main/configs/config.yaml`. This is canonical **DS-CNN-S** (Hello Edge / ARM ML-KWS-for-MCU), with the only divergence being 3-channel input (static + Δ + ΔΔ MFCC) instead of 1-channel:

| op | builtin | filter | spatial | MACs |
|----|---------|--------|---------|------|
| 00 | CONV_2D | 4×10, in_ch=3, out=64, stride (2,2) | 80×20 → 40×10 | ~770K |
| 01 | DEPTHWISE_CONV_2D | 3×3, 64ch | 40×10 → 40×10 | 230K |
| 02 | CONV_2D | 1×1, 64→64 (pointwise) | 40×10 | 1.64M |
| 03 | DEPTHWISE_CONV_2D | 3×3, 64ch | 40×10 → 40×10 | 230K |
| 04 | CONV_2D | 1×1, 64→64 | 40×10 | 1.64M |
| 05 | DEPTHWISE_CONV_2D | 3×3, 64ch | 40×10 → 40×10 | 230K |
| 06 | CONV_2D | 1×1, 64→64 | 40×10 | 1.64M |
| 07 | DEPTHWISE_CONV_2D | 3×3, 64ch | 40×10 → 40×10 | 230K |
| 08 | CONV_2D | 1×1, 64→64 | 40×10 | 1.64M |
| 09 | MEAN | over (40,10) → 64 | | trivial |
| 10 | FULLY_CONNECTED | 64 → 26 | | 1.7K |
| 11 | SOFTMAX | 26 | | trivial |

**There are zero full 3×3 Conv2D ops.** The 3×3 work is in DepthwiseConv2D, which routes through a different esp-nn dispatcher (`esp_nn_depthwise_conv_s8_*` in the depthwise files). The full Conv2Ds are either 4×10 (op 0) or 1×1 (ops 2/4/6/8).

### Where the 495 ms goes (measured 2026-05-07, per-instance)

A small custom `MicroProfilerInterface` is now wired into `letter_recognizer.cpp:setup_interpreter` and prints one event per kernel-instance alongside the existing `timing:` and `per-op us:` lines (cadence: first 5 invokes + every 20th). Storage is 64 events × 16 B = 1 KiB BSS — the canonical `tflite::MicroProfiler` is 128 KiB BSS and won't fit alongside the 118 KiB arena in internal SRAM. Tags are stable string literals from `EnumNameBuiltinOperator` (see `micro_interpreter_graph.cc`), so pointer storage is safe. **Status: uncommitted in this checkout** — `git diff components/letter_recognizer/letter_recognizer.cpp` to see, commit before running any new experiment so the baseline tag is reproducible.

Captured `per-instance us:` from inference #21 (sums to 532,778 µs ≈ matches 532 ms wall-clock):

| op | builtin | filter / shape | µs | % | per-MAC eff | esp-nn path |
|----|---------|----------------|-----|---|-------------|-------------|
| 0  | CONV_2D            | 4×10, in_ch=3, stride (2,2)  | 179,385 | **33.7%** | 0.233 µs/MAC | `esp_nn_conv_s8_im2col_s3` (C, in `esp_nn_conv_esp32s3.c`) |
| 1,3,5,7 | DEPTHWISE_CONV_2D | 3×3, 64 ch              | ~4,500–4,670 ea | 3.4% total | 0.020 µs/MAC | `esp_nn_depthwise_conv_s8_*` (S3 dispatcher) |
| 2,4,6,8 | CONV_2D       | 1×1, 64→64 (PW)              | ~79,054–79,214 ea | **59.4% total** | 0.048 µs/MAC | `esp_nn_conv_s8_mult8_1x1_esp32s3.S` (asm fast path) |
| 9  | MEAN               | over (40,10) → 64            | 18,168 | 3.4% | — | reference (no esp-nn MEAN) |
| 10 | FULLY_CONNECTED    | 64 → 26                      | 119 | 0.0% | — | — |
| 11 | SOFTMAX            | 26                           | 277 | 0.1% | — | — |

**Hypothesis update vs the prior working hypothesis ("1×1 PW dominates"):**

- Partially correct. The four 1×1 PWs total 316 ms (60%). But **op 0 alone is 179 ms (34%) — the single biggest op**, larger than any individual 1×1 PW.
- Op 0 is **5× slower per MAC** than the 1×1 PW asm. Op 0 routes through `esp_nn_conv_s8_im2col_s3` (the C im2col path for small in_ch — gated on `filter_row_size < 16 && window_len >= 16`; for op 0 that's row=12, window=120). 1×1 PW routes through `esp_nn_conv_s8_mult8_1x1_esp32s3.S` (asm; gated on `channels % 8 == 0`, satisfied by 64).
- **Both paths are 18-220× off realistic int8 SIMD throughput on S3** (~3-6 MAC/cycle effective × 240 MHz ≈ 0.7-1.4 GMAC/sec; we measured 4-50 MMAC/sec across paths).
- DepthwiseConv2D is the *fastest* per-MAC (50 MMAC/sec), even though it's "the slow side" of a typical depthwise-separable. PW being slower than DW is backwards for a working SIMD pipeline. Combined with **MEAN at 18 ms for 25 K trivial reductions** (≥ 100× slower than even scalar should produce), this points at a **system-level** issue, not a Conv2D-kernel-design issue. The next investigative move is *not* "rewrite the 1×1 asm" — it's "find why every path runs at single-digit MMAC/sec".

### What's been done in this checkout (2026-05-07 follow-up to #39)

- Added `LetterProfiler` + `s_profiler` to `components/letter_recognizer/letter_recognizer.cpp` (subclasses `tflite::MicroProfilerInterface`, 1 KiB BSS, 64-event ring; reset before each `Invoke()`).
- Passed `&s_profiler` into the `tflite::MicroInterpreter` constructor (`setup_interpreter`).
- Added `per-instance us:` log line right after the existing `per-op us:` aggregate, behind the same first-5/every-20th gate.
- Build: `idf.py build` clean (`spell_word.bin` 0x63e30 / 80% free; +3 KiB vs the 0x63230 / 81% baseline from #39).
- Bench: flashed the device at `/dev/cu.usbmodem101`, captured serial via `cat /dev/cu.usbmodem101 > /tmp/spell.log` (HITL: user spoke letters), pulled the per-instance line at inference #21. The numbers above are from that capture; `/tmp/spell.log` is not preserved across reboots, but a fresh capture is straightforward.
- Verified `.S` esp-nn kernels DO build (`build/esp-idf/espressif__esp-nn/.../esp_nn_conv_s8_mult8_1x1_esp32s3.S.obj`, `esp_nn_dot_s8_esp32s3.S.obj`, etc. all present).
- Verified `CONFIG_NN_OPTIMIZED=y`, `CONFIG_IDF_TARGET_ESP32S3=y`, `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ=240`, `# CONFIG_NN_SKIP_NUDGE is not set`, `-DESP_NN` is passed to the esp-tflite-micro component. So the SIMD path *should* be selected; whatever's slow is happening *inside* it.

Three things NOT done that the next agent might want to start with:
- The per-instance instrumentation is **uncommitted**. Recommended commit message: `feat(letter_recognizer): per-Conv2D-instance MicroProfiler hook (Vikunja #53)`.
- Boot-time diagnostic logging of `(uintptr_t)s_tensor_arena & 15`, `(uintptr_t)s_model_buf & 15`, and the input-tensor data pointer alignment — would take seconds to add and rules out one whole hypothesis cheaply (see step 2 below — first probe).
- A Vikunja note. The per-instance work is scoped under #53 but #53 itself is still open until the latency target is hit.

### What's already on disk (do not repeat)

A previous session edited two files in the gitignored `managed_components/` tree:

1. `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c` — replaced the broken qup-pipelined inline asm with an im2col-once-per-pixel + cached aligned filter + `esp_nn_dot_s8_aligned_esp32s3` dot-product kernel.
2. `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` — added a 3×3-opt scratch-size case to `esp_nn_get_conv_scratch_size_esp32s3` and unguarded the dispatcher gate around `esp_nn_conv_s8_3x3_opt`.

Why this didn't help and shouldn't be revived as-is:

- **The path never fires on this model.** Eligibility is `filter_wd==3 && filter_ht==3 && in_ch>=16 && in_ch%16==0 && pad==0`. Op 0 has filter_wd=4 (no match); ops 2/4/6/8 are 1×1 (caught by the 1×1 path before reaching the 3×3 check); DepthwiseConv2D goes to a different dispatcher entirely.
- **Latent correctness bug if it ever did fire.** Line 124 of `esp_nn_conv_s8_3x3_opt_esp32s3.c` calls `esp_nn_multiply_by_quantized_mult_fast` directly. With `CONFIG_NN_SKIP_NUDGE` unset (verified in `sdkconfig`: `# CONFIG_NN_SKIP_NUDGE is not set`), the rest of the codebase's `esp_nn_requantize` macro (in `managed_components/espressif__esp-nn/src/common/common_functions.h:233-237`) resolves to the **bit-exact** `esp_nn_multiply_by_quantized_mult`. Mixing fast and bit-exact across layers shifts int8 outputs and degrades accuracy.
- **Not in git.** `managed_components/` is gitignored at `.gitignore:5`. The 3×3-opt code only exists on disk in this checkout.
- **Dispatcher addition itself is safe** — the gate correctly falls through to the existing `esp_nn_conv_s8_im2col_s3` path for op 0 and to the 1×1 path for ops 2/4/6/8. So the current model behaves identically to before, just with dead code present.

Recommended action: either **delete** the 3×3-opt path additions outright (cleanest), or **fix the requantize call** (replace with the `esp_nn_requantize` macro) and leave for a future architecture, but do NOT spend more time pursuing 3×3-Conv2D speedups for this model.

### Investigation order (cheapest → most invasive)

**Step 1 — done (2026-05-07).** Per-Conv2D-instance timing wired in; baseline measured. See "Where the 495 ms goes" above.

**Step 2 (now the live step) — Verify SIMD is actually engaging.** Skip rewriting any conv kernel until this is answered. Three cheap probes, ordered cheapest first:

- **Log alignment at boot.** Add `ESP_LOGI(TAG, "align: arena=%zu model=%zu input=%zu", (size_t)((uintptr_t)s_tensor_arena & 15), (size_t)((uintptr_t)s_model_buf & 15), (size_t)((uintptr_t)s_input->data.int8 & 15))` after `AllocateTensors()` in `setup_interpreter`. esp-nn's S3 asm paths assume 16-byte alignment; if the input tensor pointer (which moves around inside the arena based on the memory planner) lands on an odd offset, the asm has to either bail to a slow path or do unaligned scalar fixups. `MALLOC_CAP_INTERNAL` heap *should* hand out 16-aligned blocks, but the input tensor's offset inside the arena is determined by the greedy memory planner, not the heap — worth verifying.
- **Cycle-count one 1×1 PW asm call.** Wrap `esp_nn_conv_s8_mult8_1x1_esp32s3` (in `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` near line 445) with `XT_RSR_CCOUNT()` (or `esp_cpu_get_cycle_count()` from `esp_cpu.h`). For one of the 1×1 PWs we measured 79 ms ≈ 19 M cycles at 240 MHz; for 1.64 M MACs that's ~12 cycles/MAC. If the asm is genuinely running int8 vector MACs, expected is sub-cycle/MAC. If we're at 12 cyc/MAC, the asm is running but stalling — most likely on cache misses, on filter loads (the per-OC inner loop), or on slow `esp_nn_multiply_by_quantized_mult` requantize. If we're at >100 cyc/MAC, it's not vectorizing at all.
- **Disassemble one of the .S obj files** to confirm the EE.* (Xtensa AI extension) instructions are actually emitted: `xtensa-esp32s3-elf-objdump -d build/esp-idf/espressif__esp-nn/CMakeFiles/__idf_espressif__esp-nn.dir/src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S.obj | grep -E 'ee\.|EE\.'`. If it's empty, the toolchain is lowering them to scalar — most likely a missing `-mtext-section-literals` / Xtensa core flag in the assembler invocation. If it's populated, SIMD ops are present in the binary; the question becomes whether they execute or get serialized by the LX7 pipeline.

**Step 3 — Read the 1×1 asm path.** Only if step 2 says SIMD ops are emitted but still slow. Files:
- `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S` — the path we exercise. Header claims "Processes 8 spatial positions simultaneously via QACC lanes" — verify against the actual asm.
- `managed_components/espressif__esp-nn/src/common/esp_nn_dot_s8_esp32s3.S` — shared aligned/unaligned dot product (used by the im2col path that op 0 takes). Op 0's 4.3 MMAC/sec strongly suggests this is also unhealthy.

If the 1×1 asm is firing but slow, look for per-OC reload patterns. The pointwise op is `output[h,w,oc] = Σ_ic input[h,w,ic] * filter[oc,ic]` — re-fetching the input every OC is a waste; processing the entire input once and accumulating into all 64 OC outputs would be the optimization. The asm header *claims* this already, but the wall-clock disagrees.

**Step 4 — Upgrade `esp-nn` and/or `esp-tflite-micro`.** Currently on esp-tflite-micro 1.3.5 + esp-nn 1.2.3 (per `dependencies.lock`). Newer versions may have fixes for known LX7 codegen / scheduling regressions. Cheap delta to try after step 2 if alignment + cycle counts look healthy — bump versions in `main/idf_component.yml`, `idf.py reconfigure`, rebuild, re-bench against the per-instance line. Important: a version bump will rewrite `managed_components/`, which will discard the prior session's dead 3×3-opt edits documented below — that's *fine* (those edits are dead code anyway), but if you've added cycle-count instrumentation in step 2 it will also be discarded; either commit the diff against `managed_components/` to a side branch first or re-apply afterwards.

**Step 5 — Last-resort training-side levers** (`~/alphabet-R/main/configs/config.yaml`):
   - Drop Δ/ΔΔ (`use_delta: false, use_delta2: false`) → first conv shrinks from in_ch=3 to in_ch=1, ~770K → ~256K MACs. Doesn't address the 1×1 PW dominance, but recovers ~10-15 ms in op 0.
   - Shrink to 32 filters everywhere → halves PW compute, the dominant term. Requires retrain + re-export.
   - Both reversible; both kick the can if SIMD itself is broken. Don't pivot architecture without finishing step 2 first.

### Files to read before starting

- `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` — main dispatcher; `esp_nn_conv_s8_esp32s3` at line ~402 routes to specialized paths.
- `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S` — likely the actual bottleneck for this model.
- `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_1x1_esp32s3.c` — the 1×1 C fallback (not taken here).
- `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.S` — general asm path (also not taken here, since op 0 falls into im2col).
- `managed_components/espressif__esp-nn/src/common/esp_nn_dot_s8_esp32s3.S` — shared aligned/unaligned dot-product kernels used by the im2col path.
- `managed_components/espressif__esp-nn/src/common/common_functions.h` — requantize macros, alignment helpers, the `CONFIG_NN_SKIP_NUDGE` toggle.
- `components/letter_recognizer/letter_recognizer.cpp` — `LetterProfiler` class (subclass of `tflite::MicroProfilerInterface`), `s_profiler` static instance, and the `per-instance us:` log line live here (uncommitted in this checkout). The existing `extern long long conv_total_time, ...` accumulators alongside it are esp-nn-side globals; they remain useful as a sanity cross-check (sum of `CONV_2D` events ≈ `conv_total_time`).
- `components/spell_common/include/spell_config.h:160-194` — arena sizing rationale (118 KiB, 84% utilization at baseline), historical bench notes (256/192/144 KiB experiments), and the relationship between arena placement (internal SRAM) and observed invoke time.
- `~/alphabet-R/main/configs/config.yaml` — training-side ground truth for the model arch; only relevant if pivoting to retrain.

### Tools

- `idf.py build` (after `source ~/esp/esp-idf/export.sh`).
- `idf.py -p /dev/cu.usbmodem101 flash` flashes a connected board (port confirmed working 2026-05-07).
- Quick serial capture without entering the IDF monitor's interactive UI: `: > /tmp/spell.log; cat /dev/cu.usbmodem101 > /tmp/spell.log &` then speak letters; tail/grep the log. The recognizer logs `timing:`, `per-op us:`, and the new `per-instance us:` lines for the first 5 utterances + every 20th, so the first per-instance snapshot lands at inference #21 (the cadence keys off `n_pre`, the pre-increment count). Note: `cat` started after flash misses the first 5 utterances — either accept that and wait for #21, or restart the cat *before* the device finishes booting.
- HITL: hold the device, speak letters, observe.
- No host benchmark exists for esp-nn kernels — measurement is on-device.

### Targets to hit

- **ADR-0001 (500 ms early-commit silence floor):** unlocks if total invoke ≤ ~150 ms (recognizer must produce LETTER_RECOGNIZED before the floor expires, with retract-aware paths still useful in the remaining window).
- **ADR-0005 (~40 ms W-recovery re-inference):** unlocks if total invoke ≤ ~50 ms.
- **Stretch goal:** Espressif's micro_speech reaches ~5 ms on S3; DS-CNN-S at our footprint should be reachable in 10-30 ms with SIMD properly engaged. **30 ms total invoke** is a reasonable target for #53.

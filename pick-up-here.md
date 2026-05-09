# Pick up here

You are landing into the Spell-Word firmware repo (ESP32-S3 handheld reading aid) mid-implementation of Phase 4–5. Read this first.

## Orient

- **Domain glossary**: `CONTEXT.md`. Use the terms there (utterance, letter, top-K, EOW, early-commit window, letter recognizer, etc.) — they are load-bearing across the design docs.
- **Active PRD**: `docs/prds/0001-phase-4-5-ux-and-decoder.md`. The §"Sequencing" subsection in "Further Notes" is the canonical implementation order.
- **ADRs**: `docs/adr/0001`–`0006`. ADR-0001 (aggressive early-commit), ADR-0005 (W-recovery inside the recognizer), and ADR-0006 (unified `LETTER_RECOGNIZED` event) are the most load-bearing for ongoing slices.
- **Issue tracker**: Vikunja, project ID 3 ("Spell-Word"). Use the `vikunja` skill to scan/update. The `Blocked by` field is now reliable as of 2026-05-04 — earlier stale dependencies on `#27` and `#28` were cleaned up.
- **Recent work**: see `CHANGELOG.md` and `git log`. **#52 LANDED on 2026-05-08** — feature frontend now matches the model: `SPELL_FEAT_N_FRAMES = 80` (was 79) and per-coefficient/channel `(mfcc − mean) / std` normalization is applied between `feat_extract_compute()` and the int8 quantize loop in `letter_recognizer.cpp`. Norm arrays are generated from `models/norm_*.npy` by `tools/gen_norm_arrays/gen_norm_arrays.py` and live as `static const float[60]` in `components/letter_recognizer/feat_norm.{c,h}` (checked in so IDF build needs no Python). Bench-mic HITL acceptance is the only remaining gate. **#53 RESOLVED on 2026-05-07** — the Conv2D bottleneck was a one-line dispatcher-guard bug in `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` that mis-routed every TFLM Conv2D to the C reference kernel; total invoke dropped 532 ms → 84 ms cold / 76 ms warm (6.3×). The fix lives in `docs/patches/53-esp-nn-conv-dispatcher-guard.patch` and is **auto-applied at CMake configure time** by `cmake/apply_patches.cmake` — no manual reapply step. See `docs/patches/README.md` for the architecture and the diagnosis playbook for an `[patches] … APPLY FAILED` configure-time error. See the **#53 cold-start guide** at the bottom of this doc for the full investigation log + post-fix per-op numbers. Most recent committed slices, newest first: Vikunja #52 (frontend: 79→80 frames + per-coef normalization, `gen_norm_arrays` host tool, generated `feat_norm.{c,h}`), #53 (esp-nn dispatcher-guard fix, auto-applied via `cmake/apply_patches.cmake` — see `docs/patches/53-*.patch` + `docs/patches/README.md`), #39 (TFLM tensor arena + model in internal SRAM, 118 KiB arena, per-op profiling, `flat26_baseline_int8` bench measurement → surfaced #53 Conv2D bottleneck), #24 (`decoder_replay` host CLI + fixture format), #27 (letter_recognizer internal PCM ring), #22 (decoder IDF wrapper + matrix/dict partition loaders), #29 (`w_detector_replay` CLI + CSV fixture format), #28 (`w_detector_core` trigger predicate), #21 (decoder early-commit predicate), #20 (decoder_core full-EOW + retract-aware letter handler), #19 (segmenter early-commit-window event), #41 (recognizer rename + unified `LETTER_RECOGNIZED` event).

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
- **Feature frontend matches model** (#52, landed 2026-05-08): `SPELL_FEAT_N_FRAMES` bumped 79 → 80 in `components/spell_common/include/spell_config.h` so the firmware writes all 4800 input cells (was 4740, leaving the trailing frame as random int8-tensor stale state) and the model's expected `[1, 80, 20, 3]` shape lines up byte-for-byte with `tf.signal.frame(frame_length=320, frame_step=160, pad_end=True)` semantics. The existing `else` branch in `feat_extract.c`'s framing loop already zero-fills any out-of-bounds index, so frame 79 (samples [12640, 12960)) gets zero pad at [12800, 12960) without source changes — exactly matching the training pipeline's pre-emphasize-then-pad flow. `letter_recognizer.cpp::recognizer_task` now applies per-coefficient/channel normalization `s_features[i] = (s_features[i] - mean[i % 60]) / std[i % 60]` between `feat_extract_compute()` and the int8 quantize loop; the int8 quant params (input scale=0.07303, zero_point=-8) were calibrated against this post-norm distribution so the previous skip was leaving every input cell mean-shifted and std-unscaled. Norm arrays generated from `models/norm_mean.npy` + `models/norm_std.npy` (both `(1, 1, 20, 3)` float32, 60 floats each) by `tools/gen_norm_arrays/gen_norm_arrays.py` and emitted to `components/letter_recognizer/feat_norm.{c,h}` as two `static const float[60]` arrays plus extern decls + a `_Static_assert(SPELL_FEAT_NORM_LEN == 60, …)` guard against a future channel/coef change in `spell_config.h`. The generated `.c/.h` are checked in so IDF build needs no Python; re-run `make -C tools/gen_norm_arrays gen` (or `… check` for the round-trip self-test, atol=1e-7) only when the model handoff changes. The host-side companion `tools/feat_extract_reference/feat_extract_reference.py` is a pure-numpy reference of the entire firmware MFCC + Δ + ΔΔ + normalization pipeline (mirrors `feat_extract.c` step-for-step modulo the FFT backend — `numpy.fft.rfft` vs esp-dsp Radix-2; both compute the same DFT) and provides the Python half of the AC item 4 host round-trip; `make -C tools/feat_extract_reference selftest` verifies shape + sanity + that frame 79 actually carries signal post-bump. Build clean (`spell_word.bin` 0x64050 bytes, 80% app partition free); 12+10+12+14 host tests still pass; both `gen_norm_arrays --check` and `feat_extract_reference --selftest` pass. **HITL acceptance is the only remaining gate**: speak a clean alphabet sequence into the bench mic, confirm top-1 hits at MODEL_INFO's promised rate (97% on synthetic TTS; expect lower on real-mic acoustics per ADR-0004). The full host-vs-firmware round-trip comparison (Python feature tensor vs firmware-dumped feature tensor for the same WAV) requires either a temporary serial-dump hook in the recognizer task or a `pip install tflite-runtime` to drive `feat_extract_reference --invoke` against the same fixed PCM the firmware processes — both are documented inline in `tools/feat_extract_reference/README.md`'s "complete the round-trip" section.
- **TFLM tensor arena + model in internal SRAM, with bench measurement** (#39): `components/letter_recognizer/letter_recognizer.cpp` now allocates BOTH the tensor arena and the model flatbuffer from `MALLOC_CAP_INTERNAL`. Arena size is **118 KiB** (`SPELL_TENSOR_ARENA_SIZE = 118 * 1024`), set at ≤ 1.2 × the measured `arena_used_bytes` of 100,924 B (84% utilization). Model alloc tries an internal-SRAM cap of `SPELL_MODEL_INTERNAL_BUF_SIZE` (64 KiB — covers the 53 KiB `flat26_baseline_int8.tflite` + headroom) and falls back to PSRAM with a loud warning on memory pressure. Pre-/post-arena heap diagnostic logs free + largest-block on every recognizer init so future re-sizing can be done from real numbers. Per-op timing breakdown (`per-op us: conv= … dwconv= … fc= … pool= … smax= … add= … mul= …`) logs alongside the existing `timing:` line for the first 5 utterances + every 20th. **Bench measurements (2026-05-07, ESP32-S3 v0.2 @ 240 MHz, esp-tflite-micro 1.3.5 + esp-nn 1.2.3)**: pre-arena internal heap 354,359 B free / 258,048 B largest block; post-arena 167,983 B free / 73,728 B largest; arena_used 100,924 / 120,832 B (84%); USB-Serial-JTAG stays alive after flash. Original measurement was 532 ms invoke (later traced to #53 dispatcher bug — see below). With the #53 fix in place, total invoke = **84 ms cold / 76 ms warm**, broken down (warm) as `conv=39538µs dwconv=18190µs fc=115µs pool=0µs smax=244µs add=0µs mul=0µs`.
- **esp-nn dispatcher-guard fix** (#53, RESOLVED 2026-05-07): root cause for the 532 ms baseline was a one-line guard in `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` that read `if (channels != filter_dims->channels) { fall back to ansi; return; }`. The TFLM glue at `managed_components/espressif__esp-tflite-micro/.../conv.cc:241` sets `filter_dims.channels = 0` by convention ("derive from input"); the guard misread that 0 as "different from input channels" and shunted **every** Conv2D from TFLM to `esp_nn_conv_s8_ansi` (pure-C reference), making all SIMD paths unreachable. DepthwiseConv was unaffected because it goes through a different dispatcher (`esp_nn_depthwise_conv_esp32s3.c`) without this guard — that's why it was the only fast op all along. Fix changes the guard to `if (filter_dims->channels != 0 && channels != filter_dims->channels)`. Patch lives at `docs/patches/53-esp-nn-conv-dispatcher-guard.patch` and is **auto-applied at CMake configure time** by `cmake/apply_patches.cmake` (see `docs/patches/README.md` for architecture). Validated by per-Conv2D-instance bench: op 0 = 179 ms → 20 ms (8.9×), each 1×1 PW = ~79 ms → ~5 ms (16×). Cycle-count probe at the 1×1 PW asm call site post-fix: 0.71 cyc/MAC ≈ 340 MMAC/sec int8 SIMD (was unreachable / probe never fired pre-fix). ADR-0001 (500 ms early-commit silence floor) is now unbroken; ADR-0005's "~40 ms re-inference" claim is still off by ~2× but absolute latency is fine for W-recovery (single rerun ≈ 76 ms; full path ≈ 152 ms). Both ADRs to be updated to reflect achieved numbers in the next #53-followup slice.
- **Build**: `idf.py build` clean (`spell_word.bin` 0x63e20 bytes, 80% app partition free).

## What is NOT done

- **Phase 4** (UX): no button, debounce, state machine, tones, I2S TX, or playback path. `components/spell_ui/` exists with stubs only.
- **Phase 5** (decoder + W-recovery), remaining work:
  - **ADR-0001 + ADR-0005 latency-claim updates** (#53 follow-up): #53 itself is RESOLVED (invoke 532 ms → 84 ms cold / 76 ms warm via the esp-nn dispatcher-guard fix; see "What is done"). ADR-0001's 500 ms early-commit silence floor is now satisfied. ADR-0005's "~40 ms W-recovery re-inference" claim is still off by ~2× (single rerun ≈ 76 ms; full W-recovery path ≈ 152 ms), but absolute latency is fine for the user-perceived target. Both ADRs to be revised with measured numbers; the work is small (docs only) and unblocking for #30. Stretch sub-30 ms invoke is *not* required for any acceptance criterion; if pursued, the dominant remaining costs are MEAN (18 ms, no esp-nn implementation, falls through to TFLM reference) and op 0 (20 ms, im2col-overhead-dominated).
  - **W-recovery integration** (#30) — wires the trigger + PCM ring + merged-rerun into the recognizer task; emits `LETTER_RECOGNIZED { retract_count = 2 }` on confirmed W. Was blocked on #39 → #53; **#53 is resolved**, so #30 is unblocked. Pair with the ADR-0005 update above.
  - **Per-spelling summary log line** (#31) — one greppable digest per `WORD_RESOLVED` / `WORD_ABSTAIN`. Blocked only on #30.
  - **#52 follow-up: bench-side firmware feature-tensor dump** (the second half of AC item 4 that's still HITL-shaped): the Python half of the host round-trip landed in `tools/feat_extract_reference/` and self-tests cleanly; the firmware-output side still needs a way to expose `s_features[]` (or the post-normalization buffer) to host comparison. Two acceptable paths documented in `tools/feat_extract_reference/README.md`: (1) a temporary `ESP_LOGI` dump of the int8 input tensor for a fixed-known-PCM input, paired with `feat_extract_reference --quantize` host-side and a top-1 match check after `pip install tflite-runtime` lets the host invoke the .tflite; (2) a more permanent serial dump that uploads the float32 feature tensor for any WAV the bench plays back, compared to the host's `--out features.npy`. Path (1) is the smaller change. Either depends on hardware access, so it travels with #52's HITL acceptance step.
  - **Dictionary partition population**: the `confusion` and `dictionary` partitions are wired up in firmware but still erased on dev boards. `decoder_init()` logs a warning and the boot continues; spelling is silent until both are flashed. The dictionary binary format is documented at the head of `components/decoder/decoder.c` (16 B header `'SPDC' | version=1 | n_words | reserved=0` + packed `word_id, length, letters[length]` records). No host tooling exists yet to generate either blob.
- **Phase 6** (word audio playback): out of scope for the active PRD. `playback_play_word` is a stub returning `ESP_ERR_NOT_SUPPORTED`.

## Where to start

Pick from these unblocked tasks.

| Vikunja | Title | Type | Notes |
|---|---|---|---|
| #30 | W-recovery integration | AFK + HITL | Unblocked now that #53 is resolved (single rerun ≈ 76 ms; full W-recovery path ≈ 152 ms — well within user-perceived budget). Pair with an ADR-0005 update to reflect measured numbers. |
| #25 | spell_ui IDF wrapper (button + tones + I2S TX + state machine) | HITL | Unblocked by #22 + #23. Hardware bench test required. Independent of the recognizer. |
| #26, #31, #32 | Cancel ordering, per-spelling summary, hardware bench | mixed | All previously gated on #53; now unblocked, queued behind #30. |

**Suggested next slice**: **#30** (W-recovery integration) is the highest-leverage move now that #52 has landed end-to-end recognizer accuracy. #30 closes out the W-recovery flow that's been waiting on #39, #53, and #52, and pairs cleanly with the ADR-0001/ADR-0005 latency-number updates and the #52 follow-up host round-trip test. **#25** is a clean independent slice for UI progress that doesn't touch the recognizer at all. The HITL acceptance for #52 itself (speaking the alphabet into the bench mic, observing top-1 accuracy) should be done before #30 to confirm the recognizer's outputs are calibrated, otherwise debugging W-recovery becomes a moving target.

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

## #53 post-mortem: Conv2D bottleneck (RESOLVED 2026-05-07)

#53 is closed; this section is preserved as a reference for the next time anyone touches the esp-nn dispatch path or hits a "SIMD looks faster than reality" diagnostic problem.

### The bug

`managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` had a grouped-conv guard at the top of `esp_nn_conv_s8_esp32s3`:

```c
if (channels != filter_dims->channels) {
    esp_nn_conv_s8_ansi(...);  // pure-C reference kernel
    return;
}
```

The TFLM glue at `managed_components/espressif__esp-tflite-micro/.../conv.cc:241` sets `filter_dims.channels = 0` by convention ("derive from input"). The guard misread that 0 as "different from input channels" and shunted **every** Conv2D from TFLM to the C reference, making all SIMD paths unreachable. DepthwiseConv was unaffected because it goes through a different dispatcher (`esp_nn_depthwise_conv_esp32s3.c`) — that's why DW was the only fast op all along (50 MMAC/sec vs single-digit MMAC/sec for everything else).

The guard was almost certainly added by an earlier session (same comment style as the dead 3×3-opt code in the same file). Upstream esp-nn 1.2.3 may not have had it.

**Fix**: `if (filter_dims->channels != 0 && channels != filter_dims->channels)`. Lives in `docs/patches/53-esp-nn-conv-dispatcher-guard.patch` and is auto-applied at CMake configure time by `cmake/apply_patches.cmake`. If `idf.py reconfigure` ever shows `[patches] … APPLY FAILED`, see the diagnosis playbook in `docs/patches/README.md` — most likely cause is an upstream esp-nn version bump that shifted the diff context.

### How it was diagnosed

Three probes, only one of which fingered the slowdown:

| probe | what it tested | result |
|---|---|---|
| objdump 1×1 PW `.S.obj` for `ee.*` | toolchain lowering SIMD to scalar? | 125 EE instructions including `ee.vsmulas.s16.qacc.ld.incp` MAC ops — toolchain was fine |
| `align: arena/model/input` log after `AllocateTensors()` | misaligned tensors causing asm scalar fixups? | arena=&15=0, input=&15=0, model=&15=12 (model offset doesn't matter once SIMD path is unreachable anyway) |
| cycle-count probe wrapping the 1×1 PW asm call | how many cyc/MAC inside the asm? | **probe never fired** — the call site was unreachable |

The third probe failing to fire was the smoking gun: the dispatcher returned before reaching the 1×1 PW branch. Tracing back from the dispatcher entry point in the disassembly led directly to the `beq a8, a5` → `call8 esp_nn_conv_s8_ansi` early-exit, which traced back to the grouped-conv guard.

### Numbers

| | baseline (532 ms invoke) | post-fix (84 ms cold / 76 ms warm) | speedup |
|---|---|---|---|
| op 0 (CONV_2D 4×10, ic=3, oc=64) | 179,393 µs | 20,053 µs | 8.9× |
| op 2/4/6/8 (PW1×1, ic=oc=64) | ~79,000 µs ea | ~4,900 µs ea | 16× |
| ops 1/3/5/7 (DW 3×3) | ~4,700 µs ea | ~4,700 µs ea | unchanged ✓ |
| op 9 (MEAN) | 18,177 µs | 18,177 µs | unchanged (no esp-nn MEAN) |

Probe-2 cyc/MAC on the 1×1 PW asm post-fix: **0.71** (≈340 MMAC/sec int8 SIMD at 240 MHz). That's healthy throughput.

### What's still on disk (do not repeat)

A previous session also left a partial 3×3-opt experiment in two files in `managed_components/`:

1. `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c` — replacement im2col-once-per-pixel kernel.
2. `managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c` — 3×3-opt scratch-size case + unguarded dispatcher gate.

The path **never fires on this model** (op 0 has filter_wd=4, ops 2/4/6/8 are 1×1, DW goes to a different dispatcher), and the kernel has a latent requantize bit-exactness bug (calls `esp_nn_multiply_by_quantized_mult_fast` directly while the rest of the codebase resolves `esp_nn_requantize` to the bit-exact form when `CONFIG_NN_SKIP_NUDGE` is unset). Recommended action if you ever touch this file: delete the 3×3-opt additions outright. The dispatcher gate addition itself is harmless (correctly falls through), so leaving it in place is also fine.

### Reference: the on-device model

Verified against the .tflite directly + training config at `~/alphabet-R/main/configs/config.yaml`. Canonical **DS-CNN-S** (Hello Edge / ARM ML-KWS-for-MCU) with 3-channel input (static + Δ + ΔΔ MFCC):

| op | builtin | filter | spatial | MACs | esp-nn path (post-fix) |
|----|---------|--------|---------|------|-----|
| 00 | CONV_2D | 4×10, in_ch=3, oc=64, stride (2,2) | 80×20 → 40×10 | ~770K | `esp_nn_conv_s8_im2col_s3` |
| 01,03,05,07 | DEPTHWISE_CONV_2D | 3×3, 64ch | 40×10 | 230K ea | `esp_nn_depthwise_conv_s8_*` |
| 02,04,06,08 | CONV_2D | 1×1, 64→64 (PW) | 40×10 | 1.64M ea | `esp_nn_conv_s8_mult8_1x1_esp32s3.S` |
| 09 | MEAN | over (40,10) → 64 | | trivial | TFLM reference (no esp-nn MEAN) |
| 10 | FULLY_CONNECTED | 64 → 26 | | 1.7K | — |
| 11 | SOFTMAX | 26 | | trivial | — |

There are zero full 3×3 Conv2D ops. The 3×3 work is all in DepthwiseConv2D.

### Tools

- `idf.py build` (after `source ~/esp/esp-idf/export.sh`).
- `idf.py -p /dev/cu.usbmodem101 flash` flashes a connected board.
- Quick serial capture without entering the IDF monitor's interactive UI: `: > /tmp/spell.log; nohup cat /dev/cu.usbmodem101 >> /tmp/spell.log & disown`. The recognizer logs `timing:`, `per-op us:`, and `per-instance us:` lines for the first 5 utterances + every 20th.
- HITL: hold the device, speak letters, observe.
- No host benchmark exists for esp-nn kernels — measurement is on-device.

### If invoke time regresses

The patch is auto-applied at CMake configure time, so the first thing to verify is whether it's still in place. Two checks, in order:

1. **Did the configure step apply it?** Look at the last `idf.py reconfigure` (or first `idf.py build` after a `fullclean`) output. You should see one of these on the `-- [patches]` line:

   ```
   -- [patches] esp-nn #53 dispatcher-guard: already applied (marker found in target).
   -- [patches] esp-nn #53 dispatcher-guard: applied successfully.
   ```

   If you see `APPLY FAILED` instead, the upstream esp-nn version probably shifted — see the diagnosis playbook in `docs/patches/README.md`.

2. **Does the file actually have the marker?**

   ```sh
   grep -A1 'Vikunja #53' managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c | head -3
   # Expected: a comment mentioning Vikunja #53 followed by the guard
   #          `if (filter_dims->channels != 0 && channels != filter_dims->channels)`
   ```

   If the marker is missing despite a "successfully applied" log, the patch is structurally broken (no-op hunk, header diff). Inspect both the patch file and the target by hand.

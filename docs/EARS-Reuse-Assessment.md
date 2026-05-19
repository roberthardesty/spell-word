# EARS Firmware → Spell-Word: Reuse Assessment

The EARS bark-detection POC covers most of the input half of the Spell-Word pipeline (capture → log-mel → TFLM inference) and a sizable chunk of the boilerplate (NVS, init sequencing, factory reset). The output half (segmenter, decoder, playback) is net-new development. Net effect: **roughly 4 days off the 20-day MVP estimate**, and — more valuably — the highest-uncertainty firmware bits are de-risked because they're already running on hardware.

---

## 1. File-by-file reuse table

| File | Reuse class | Changes needed | Effort to port |
|---|---|---|---|
| `audio_capture.c` | **Lift-and-shift** | Rename `EARS_*` → project prefix. Pin assignments come from `fw_config.h`. | <1 hour |
| `log_mel.c` | **Lift-and-shift** | Same rename. Frame count changes 98 → 58, but that's a constant in the header — no code changes. | <1 hour |
| `inference.cpp` | **Adapt** | TFLM scaffolding (model load, partition, hot-reload, quantize/dequantize, arena, op resolver) is reusable as-is. The **I/O loop** and **output handling** rewrite. See §3. | 1–1.5 days |
| `device_config.c` | **Reusable as a pattern** | Drop the EARS-specific keys (HA webhook). Add Spell-Word keys later if needed (volume, abstention threshold, language). For MVP, may not be needed at all. | 1 hour, or skip |
| `app_main.c` | **Reusable as a skeleton** | NVS init, boot banner, factory reset, status LED, init ordering all carry over. WiFi / BLE provisioning / uploader sections are dropped; UI / playback / segmenter / decoder are added. | <1 hour to template |

---

## 2. Direct lift-and-shift candidates (~95% identical)

### `audio_capture.c`

The entire I2S RX → 32→16 conversion → PSRAM-backed StreamBuffer fan-out is exactly what Spell-Word needs. Sub-detail worth noting from the actual code:

- `dma_desc_num × dma_frame_num = 6 × 240` gives ~90 ms of DMA buffering — confirms the comment in the ESP32-S3 doc.
- `xStreamBufferCreateStatic` with `trigger_level=1` so receivers wake on any data.
- Fan-out uses `xStreamBufferSend` with timeout `0` — slow consumers count overruns but never stall capture. This is the right pattern for Spell-Word; the segmenter is fast (per-frame RMS) so it should never overrun, but the architecture protects it anyway.
- Capture pinned to Core 0 at priority 10 — keep as-is.

**Action.** Copy `audio_capture.c` and `audio_capture.h` into `components/audio_capture/`, run `sed 's/EARS_/SPELL_/g'`, build.

### `log_mel.c`

Identical math, identical optimizations. The internal-SRAM/PSRAM split (FFT buf + Hann + bounds in SRAM, filterbank in PSRAM) is exactly what the ESP32-S3 doc prescribes. The sparse `mel_bound_t` optimization is implemented.

The only thing that changes for Spell-Word is the input shape constant (`MEL_N_FRAMES = 58` instead of `98`), and that's just a header value the function reads from. The mel filterbank itself is independent of frame count.

**Action.** Copy `log_mel.c` and `log_mel.h` into `components/letter_classifier/`, rename. No code changes.

---

## 3. `inference.cpp` — the largest adaptation

This is the most valuable file in the dump because it gets the TFLM-on-ESP32-S3 plumbing right, but it's also the one with the most Spell-Word-specific changes. Two categories of edits.

### 3a. Direct reuse, no changes

- `init_op_resolver()` structure (only the op list changes — drop `AddLogistic`, keep `AddSoftmax`).
- `load_model_from_partition()` — including the all-`0xFF` empty-partition check. Already supports the "no model flashed yet" case cleanly.
- `setup_interpreter()` — tensor arena allocation in PSRAM, `AllocateTensors`, `arena_used_bytes()` logging.
- Per-tensor int8 quantize/dequantize using `s_input->params.scale` and `zero_point`. Both Float32 and Int8 input paths are handled; keep both for debugging convenience.
- Hot-reload pattern (`s_reload_requested` flag checked at top of loop).
- Energy gate: `10 * log10(mean_sq)` formulation, log-throttled debug output. Same threshold convention. Match the firmware threshold to the training pipeline's curation threshold — flagged as an integration check during Phase 7.
- WDT handling via `CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n` in `sdkconfig`. **This is a detail that didn't make it into the MVP plan and should be added** — without it, a multi-hundred-millisecond `Invoke()` will trip the watchdog on Core 1.
- Stats logging cadence (first 5, then every 10th/20th).

### 3b. Rewrites for Spell-Word

| EARS pattern | Spell-Word change |
|---|---|
| Op resolver `<10>` includes `AddLogistic` | Drop `Logistic`, drop the count to `<9>` (or `<10>` if leaving headroom). The plan currently says `<9>` — verify against your exported model. |
| Inference task reads its own 1-second window from StreamBuffer continuously | Inference task waits on a `QueueHandle_t` of `utterance_t` records pushed by the segmenter. The StreamBuffer subscriber is *moved to the segmenter* — inference no longer subscribes directly. |
| 1-second window (16 000 samples) | 600 ms window (9 600 samples) — already padded by the segmenter so the `log_mel_compute` call uses the constant. |
| Output is a single sigmoid value, compared to `EARS_BARK_THRESHOLD` | Output is a 26-element distribution; run `top_k_select`, post `FW_LETTER_TOP_K` event. No firmware-side threshold. |
| Event payload: `ears_bark_event_t { confidence, invoke_ms }` | Event payload: `fw_letter_topk_event_t { top_k[K], invoke_ms }`. |

**The architectural inversion is the only non-trivial change.** In EARS, the inference task pulls audio. In Spell-Word, the segmenter pulls audio and pushes utterances to inference. This means:

- The audio_capture subscriber registration moves from `inference_init()` to `segmenter_init()`.
- `inference_init()` creates a queue (`xQueueCreate` with `utterance_t` items) and exposes a `inference_submit_utterance()` API.
- The inference task's main loop becomes: `xQueueReceive(...) → energy gate → log-mel → invoke → top-K → post event`.

The energy gate stays in the inference task (not the segmenter) — the segmenter's onset threshold is more permissive than the inference gate, by design. A letter that segmented but failed the gate is logged but no event is posted.

**Action.** New file `components/letter_classifier/inference.cpp`, structurally derived from the EARS file. Replace the StreamBuffer read loop with a queue receive. Replace the sigmoid-threshold output with `top_k_select` + event post. Approximately 60% of the original lines survive.

---

## 4. `device_config.c` — pattern, not content

The EARS file persists a device name and an HA webhook URL in NVS. Spell-Word doesn't have either of those for MVP — there's no networking, no per-device naming. The *pattern* (NVS open with auto-erase fallback, namespaced keys, default-on-first-boot, separate read/write helpers) is good and will save time when persistent config is needed: volume preference, abstention threshold tuning between sessions, child profile id.

**Action for MVP.** Skip. Add it back in a later phase if any tunable is worth persisting across reboots. The `MVP-Implementation-Plan.md` doesn't currently call for any persistent config.

---

## 5. `app_main.c` — keep the skeleton, drop two-thirds of it

EARS' `app_main.c` is a clean template:

```
init_nvs()              → keep
log_boot_banner()       → keep, useful for debugging
check_factory_reset()   → keep, repurpose: hold button at boot to reset NVS
                          (or trigger model reload, if useful)
device_config_init()    → drop for MVP
wifi_manager_init()     → drop entirely (no networking)
wifi_manager_start()    → drop
audio_capture_init()    → keep
inference_init()        → keep, structurally similar
uploader_init()         → drop entirely (no networking)
audio_capture_start()   → keep
blink_forever()         → keep as a minimal status LED until the UI state
                          machine takes over the LED in Phase 4
```

Plus the new components needed for Spell-Word:

```
segmenter_init()        → between inference_init and audio_capture_start
decoder_init()          → after model is up
audio_playback_init()   → before ui_init (UI plays tones via playback)
corpus_init()           → before audio_playback (playback reads from corpus)
ui_init()               → last; subscribes to events, drives state machine
```

The init-ordering rule from EARS — "all subscribers must register between `audio_capture_init()` and `audio_capture_start()`" — carries over verbatim. With the segmenter being the only subscriber for MVP (instead of inference + uploader in EARS), the rule is easier to satisfy.

The factory-reset pattern (BOOT button held 5 s at boot) is a nice freebie for Spell-Word: if you OTA a bad model and the device starts misbehaving, hold BOOT at power-on to fall back to a known-good state. Cheap to keep in.

**Action.** Use EARS `app_main.c` as the literal starting template. Comment out (don't delete) the WiFi/uploader sections so the diff is reviewable. Add the new init calls as they come online.

---

## 6. Things the EARS code reveals that the plan should incorporate

Reading the actual code (not just the design doc) surfaces a handful of real-world details that should land in the implementation:

1. **`CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n` in `sdkconfig.defaults`.** The 250 ms `Invoke()` will trip the task watchdog on Core 1 IDLE without this. Documented in `inference.cpp` line 226 comment, not in the design doc.
2. **8 KB stack for the inference task is the right number.** EARS uses 8 KB (with all heavy buffers in heap). The MVP plan can lock this in.
3. **Per-tensor scale/zero_point read from `s_input->params` at runtime.** Don't hardcode — different model exports will have different quant params, and reading them lets you swap models via OTA without firmware changes.
4. **Both Float32 and Int8 input paths handled in the quantize step.** Useful for debugging — bring up with a float model, swap to int8 once accuracy is validated.
5. **`s_resolver_initialized` static flag.** Guards against double-registration if `setup_interpreter()` is called via hot-reload; the EARS code already handles this correctly.
6. **`ESP_LOG_LEVEL` for stats.** EARS logs every 10th-20th inference at INFO level and gates the energy-gate-skip log at DEBUG. The MVP plan should match this (especially the gate-skip log, which floods the console during quiet periods if left at INFO).
7. **The energy-gate threshold and the segmenter's onset threshold should be coupled.** EARS uses `EARS_ENERGY_GATE_DB = -45 dBFS` to match the training-pipeline curation threshold. For Spell-Word, the segmenter onset threshold (`FW_VAD_ON_DBFS = -38`) is *more permissive* than the inference gate (`-45`), so segmented letters that are too quiet get logged-and-skipped rather than running through the model. The plan already says this; the EARS code is the proof point.

I'll fold (1)–(2) into the MVP plan as concrete additions.

---

## 7. Revised effort estimate

Original 20-day estimate, with EARS reuse applied:

| Phase | Original | With EARS reuse | Notes |
|---|---|---|---|
| 0 — Hardware bring-up | 1 d | **1 d** | No change. New TX/button hardware still needs validation. |
| 1 — Audio capture pipeline | 2 d | **0.5 d** | Lift `audio_capture.c`, rename, verify 32 kB/s. |
| 2 — Log-mel + DS-CNN inference | 3 d | **1.5 d** | Lift `log_mel.c` directly; adapt `inference.cpp` (queue-driven I/O, top-K output, op resolver delta). |
| 3 — Energy-VAD segmentation | 3 d | **3 d** | Net-new. No EARS analog. |
| 4 — Press-to-arm UX + tones | 1.5 d | **1.5 d** | Net-new. The button-debounce pattern from EARS factory-reset is loosely reusable. |
| 5 — Confusion-aware decoder | 3 d | **3 d** | Net-new. |
| 6 — Corpus + Opus + word playback | 4 d | **4 d** | Net-new. |
| 7 — Integration and tuning | 2.5 d | **2 d** | Slightly less integration risk because the input half is already proven. |
| **Total** | **20 d** | **~16.5 d** | ~3.5-day savings, ~17% reduction. |

The savings are real but modest because the heavy lift in MVP is the net-new output half, not the input half. The bigger win is in **risk reduction**: the parts of Spell-Word that were "could surprise us" (TFLM on ESP32-S3, esp-dsp FFT performance, PSRAM allocation strategy, I2S timing) are now "we have working code that does this." The remaining risk is concentrated in the segmenter, the decoder, and the corpus/playback path — each of which is testable in isolation.

---

## 8. Recommended ordering for the port

A specific sequence for the first three days, before continuing into the original plan's Phase 3+:

**Day 1 — Skeleton and capture.**

- Create the project tree from §4 of the MVP plan.
- Drop in EARS `audio_capture.c` + `.h` renamed.
- Adapt `app_main.c` minimally: NVS, banner, factory reset, `audio_capture_init`/`start`, blink. No subscribers yet.
- Verify "32000 B/s, 0 overruns" in the serial log.
- Verify the `debug_dump` subscriber path (add if not in EARS — the EARS uploader serves a similar role but is too heavy for MVP).

**Day 2 — Log-mel and TFLM.**

- Drop in EARS `log_mel.c` + `.h` renamed, frame count 58.
- Adapt EARS `inference.cpp`: keep `setup_interpreter`, `load_model_from_partition`, energy gate, quantize/dequantize, hot-reload. Replace the StreamBuffer-read loop with a stub that runs inference on a fixed test buffer (so we can validate end-to-end without the segmenter yet).
- Replace sigmoid output with `top_k_select` + event post.
- Add `infer_replay <wav>` serial command (use `console` component).
- Validate top-K output against canned single-letter WAVs.

**Day 3 — Convert to queue-driven inference and add stub segmenter.**

- Add `inference_submit_utterance(int16_t *pcm, size_t n_samples)` API.
- Convert the inference task to `xQueueReceive`-driven.
- Write a placeholder segmenter that splits on simple energy threshold without onset/offset hysteresis. Just enough to feed inference end-to-end.
- Verify: speak letters into the mic, see top-K events.

By end of Day 3, the input half is fully ported and adapted, ready for the real VAD work in Phase 3.

---

/**
 * @file letter_recognizer.cpp
 * @brief Letter recognizer task driving SPELL_EVENT_LETTER_RECOGNIZED.
 *
 * Adapted from the EARS POC's inference.cpp. The TFLM scaffolding (model
 * partition load, hot-reload, tensor arena in PSRAM, per-tensor int8
 * quantize/dequantize, energy gate) carries over essentially unchanged.
 * Three things change for Spell-Word:
 *
 *   1. I/O inversion. EARS subscribed to audio_capture and pulled
 *      continuous 1-second windows. Here, the recognizer task waits on an
 *      utterance queue fed by the segmenter, and the segmenter owns the
 *      audio_capture subscription. The arrangement allows VAD-aligned
 *      windows instead of fixed-rate ones.
 *
 *   2. Output. EARS' DS-CNN produced a single sigmoid value compared to a
 *      threshold. Spell-Word's DS-CNN produces a 26-class softmax over
 *      A–Z; we run a top-K partial sort and post the top SPELL_LETTER_TOP_K
 *      candidates as an event payload. There is no firmware-side
 *      confidence threshold — even low-confidence top-1 results carry
 *      information when paired with their alternates.
 *
 *   3. Op resolver. AddLogistic (sigmoid) is dropped; AddSoftmax stays.
 *      Template parameter shrinks from <10> to <9>.
 *
 * Operational details that DO carry over verbatim, called out so they
 * survive review:
 *
 *   - Tensor arena lives in PSRAM. The activations of any non-trivial
 *     DS-CNN exceed available contiguous internal SRAM. Size with slack;
 *     log arena_used_bytes() after AllocateTensors.
 *   - WDT for Core 1 IDLE is disabled in sdkconfig.defaults
 *     (CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n). Without it the ~250 ms
 *     Invoke() trips the watchdog.
 *   - Both Float32 and Int8 input paths are handled. Useful for bringing
 *     up a float-precision model first, swapping to int8 after accuracy
 *     is validated.
 *   - The energy gate threshold MUST match the value used by the training
 *     pipeline's segment-curation tool (SPELL_ENERGY_GATE_DB).
 */

#include "letter_recognizer.h"
#include "feat_extract.h"
#include "spell_config.h"
#include "spell_events.h"

#include <atomic>
#include <cstring>
#include <cmath>
#include <cinttypes>

#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// NOTE: esp-nn SIMD kernels (Conv2D, DepthwiseConv2D, etc.) are linked
// transitively through esp-tflite-micro — no explicit #include needed.

static const char *TAG = "recognizer";

// ── Event base definition ─────────────────────────────────────────────
ESP_EVENT_DEFINE_BASE(SPELL_RECOGNIZER_EVENT);

// ── Utterance queue (segmenter → recognizer) ──────────────────────────
//
// Ownership: producer (segmenter) heap_caps_mallocs the pcm buffer in
// PSRAM and submits via letter_recognizer_submit_utterance(). On successful
// enqueue, ownership transfers to the recognizer task, which heap_caps_frees
// after processing. On failed enqueue (queue full, pre-init), the caller
// retains ownership.
//
typedef struct {
    int16_t *pcm;
    size_t   n_samples;
} utterance_t;

static QueueHandle_t s_utterance_q = nullptr;

// ── TFLite Micro state ───────────────────────────────────────────────
static uint8_t *s_model_buf     = nullptr;
static size_t   s_model_size    = 0;
static uint8_t *s_tensor_arena  = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input    = nullptr;
static TfLiteTensor *s_output   = nullptr;
static bool s_model_loaded      = false;
static std::atomic<bool> s_reload_requested{false};

// Op resolver — DS-CNN with 26-class softmax. AddLogistic (binary sigmoid
// from EARS) is dropped; AddSoftmax stays.
static tflite::MicroMutableOpResolver<9> s_resolver;
static bool s_resolver_initialized = false;

// ── Feature workspace (MFCC + Δ + ΔΔ, NHWC) ──────────────────────────
// Shape [SPELL_FEAT_N_FRAMES][SPELL_MFCC_N_COEFFS][SPELL_FEAT_N_CHANNELS],
// = SPELL_FEAT_N_ELEMENTS floats. PSRAM-resident.
static float *s_features = nullptr;

// ── Task + stats ──────────────────────────────────────────────────────
//
// Stat counters are written from the recognizer task and the segmenter task
// (s_drop_count, via letter_recognizer_submit_utterance) and read from any
// task via the public getters. std::atomic<uint32_t> with relaxed semantics
// gives us correct cross-task increments without forcing memory fences;
// volatile alone wouldn't prevent torn reads on a 32-bit MCU bus.
static TaskHandle_t            s_task                  = nullptr;
static std::atomic<uint32_t>   s_inference_count       {0};
static std::atomic<uint32_t>   s_gate_skip_count       {0};
static std::atomic<uint32_t>   s_drop_count            {0};  // queue-full drops
static std::atomic<uint32_t>   s_invalid_drop_count    {0};  // bad inputs etc.

// ── PCM ring (internal, for W-recovery in #30) ───────────────────────
//
// Retains the last SPELL_W_DETECT_PCM_RING_DEPTH inference-eligible utterance
// PCM windows. The W-recovery cycle (ADR-0005, integrated in #30) reads the
// ring directly to concatenate the 3 utterances ending at the current top-1
// 'U' and re-run inference on the merged window.
//
// Storage is one contiguous PSRAM block sized at
// SPELL_W_DETECT_PCM_RING_DEPTH × SPELL_INFERENCE_WINDOW_BYTES (≈100 KB).
// Each slot is a fixed-size SPELL_INFERENCE_WINDOW_SAMPLES window. We copy
// the inbound utterance into the slot rather than swapping pointers because
// the segmenter retains its allocation pattern and the ring lifetime is
// independent of any individual inbound buffer.
//
// The ring is INTERNAL to the recognizer — no public access function is
// exposed (per ADR-0005, W-recovery is also internal to the recognizer, so
// there is no other consumer).
static int16_t  *s_pcm_ring_storage   = nullptr;
static size_t    s_pcm_ring_n_samples[SPELL_W_DETECT_PCM_RING_DEPTH] = {0};
static unsigned  s_pcm_ring_head      = 0;   // next-write index
static unsigned  s_pcm_ring_count     = 0;   // grows 0 → DEPTH then stays

// Insert a window into the ring at the head, evicting the oldest entry.
// Called only on the inference-success path so the ring tracks utterances
// that produced a LETTER_RECOGNIZED event. Buffers shorter than the window
// are right-padded with zero (the recognizer's input is already zero-padded
// to SPELL_INFERENCE_WINDOW_SAMPLES by the segmenter, so this is a guard).
static void pcm_ring_insert(const int16_t *pcm, size_t n_samples)
{
    if (!s_pcm_ring_storage) return;

    size_t cap = SPELL_INFERENCE_WINDOW_SAMPLES;
    size_t n   = (n_samples > cap) ? cap : n_samples;

    int16_t *slot = s_pcm_ring_storage + (size_t)s_pcm_ring_head * cap;
    memcpy(slot, pcm, n * sizeof(int16_t));
    if (n < cap) {
        memset(slot + n, 0, (cap - n) * sizeof(int16_t));
    }
    s_pcm_ring_n_samples[s_pcm_ring_head] = n;
    s_pcm_ring_head = (s_pcm_ring_head + 1u) % SPELL_W_DETECT_PCM_RING_DEPTH;
    if (s_pcm_ring_count < SPELL_W_DETECT_PCM_RING_DEPTH) {
        s_pcm_ring_count++;
    }
}

// =====================================================================
// Op resolver
// =====================================================================

static bool init_op_resolver()
{
    if (s_resolver_initialized) return true;

    if (s_resolver.AddConv2D()          != kTfLiteOk) return false;
    if (s_resolver.AddDepthwiseConv2D() != kTfLiteOk) return false;
    if (s_resolver.AddFullyConnected()  != kTfLiteOk) return false;
    if (s_resolver.AddAveragePool2D()   != kTfLiteOk) return false;
    if (s_resolver.AddMean()            != kTfLiteOk) return false;
    if (s_resolver.AddReshape()         != kTfLiteOk) return false;
    if (s_resolver.AddSoftmax()         != kTfLiteOk) return false;
    if (s_resolver.AddQuantize()        != kTfLiteOk) return false;
    if (s_resolver.AddDequantize()      != kTfLiteOk) return false;

    s_resolver_initialized = true;
    return true;
}

// =====================================================================
// Model loading from flash partition
// =====================================================================

static bool load_model_from_partition()
{
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        static_cast<esp_partition_subtype_t>(0x80),
        SPELL_MODEL_PARTITION_LABEL);

    if (!part) {
        ESP_LOGW(TAG, "model partition '%s' not found",
                 SPELL_MODEL_PARTITION_LABEL);
        return false;
    }

    // Check the partition isn't erased (all 0xFF). This handles fresh
    // boards before any model has been flashed/OTA'd.
    uint8_t magic[4];
    esp_partition_read(part, 0, magic, 4);
    if (magic[0] == 0xFF && magic[1] == 0xFF &&
        magic[2] == 0xFF && magic[3] == 0xFF) {
        ESP_LOGW(TAG, "model partition is empty (erased) — awaiting OTA");
        return false;
    }

    // Allocate (or reuse) model buffer in PSRAM.
    if (!s_model_buf) {
        s_model_buf = static_cast<uint8_t *>(
            heap_caps_malloc(part->size, MALLOC_CAP_SPIRAM));
        if (!s_model_buf) {
            ESP_LOGE(TAG, "PSRAM alloc for model buffer failed (%" PRIu32 " B)",
                     part->size);
            return false;
        }
    }

    esp_err_t err = esp_partition_read(part, 0, s_model_buf, part->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "partition read failed: %s", esp_err_to_name(err));
        return false;
    }
    s_model_size = part->size;

    // Validate flatbuffer.
    const tflite::Model *model = tflite::GetModel(s_model_buf);
    if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "invalid TFLite model (version mismatch or corrupt)");
        return false;
    }

    ESP_LOGI(TAG, "model loaded from partition (%" PRIu32 " B)", part->size);
    return true;
}

// =====================================================================
// Interpreter setup
// =====================================================================

static bool setup_interpreter()
{
    const tflite::Model *model = tflite::GetModel(s_model_buf);

    if (!s_tensor_arena) {
        s_tensor_arena = static_cast<uint8_t *>(
            heap_caps_malloc(SPELL_TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM));
        if (!s_tensor_arena) {
            ESP_LOGE(TAG, "PSRAM alloc for tensor arena failed (%d B)",
                     SPELL_TENSOR_ARENA_SIZE);
            return false;
        }
    }

    // Clean up previous interpreter (hot-reload path).
    if (s_interpreter) {
        delete s_interpreter;
        s_interpreter = nullptr;
    }

    if (!init_op_resolver()) {
        ESP_LOGE(TAG, "op resolver init failed — bump MicroMutableOpResolver<N>");
        return false;
    }

    s_interpreter = new tflite::MicroInterpreter(
        model, s_resolver, s_tensor_arena, SPELL_TENSOR_ARENA_SIZE);

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed — arena too small?");
        delete s_interpreter;
        s_interpreter = nullptr;
        return false;
    }

    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    size_t arena_used = s_interpreter->arena_used_bytes();
    ESP_LOGI(TAG, "interpreter ready — arena used %u / %d B (%.0f%% of alloc)",
             (unsigned)arena_used, SPELL_TENSOR_ARENA_SIZE,
             100.0f * arena_used / SPELL_TENSOR_ARENA_SIZE);
    ESP_LOGI(TAG, "  input: (%d,%d,%d,%d) %s, output dim0=%d %s",
             s_input->dims->data[0], s_input->dims->data[1],
             s_input->dims->data[2], s_input->dims->data[3],
             TfLiteTypeGetName(s_input->type),
             s_output->dims->data[s_output->dims->size - 1],
             TfLiteTypeGetName(s_output->type));

    // Sanity-check the output: we expect 26 elements (A-Z).
    int out_last_dim = s_output->dims->data[s_output->dims->size - 1];
    if (out_last_dim != 26) {
        ESP_LOGE(TAG, "model output last-dim is %d, expected 26 — wrong model?",
                 out_last_dim);
        return false;
    }

    return true;
}

// =====================================================================
// Top-K partial sort (insertion-sort over 26 elements)
// =====================================================================

static void top_k_select(const float *probs,
                         int n,
                         spell_letter_candidate_t *out,
                         int k)
{
    for (int i = 0; i < k; i++) {
        out[i].letter_index = 255;
        out[i].probability  = -1.0f;
    }
    for (int i = 0; i < n; i++) {
        if (probs[i] <= out[k - 1].probability) continue;
        int j = k - 1;
        while (j > 0 && out[j - 1].probability < probs[i]) {
            out[j] = out[j - 1];
            j--;
        }
        out[j].letter_index = (uint8_t)i;
        out[j].probability  = probs[i];
    }
}

// =====================================================================
// Energy gate (in dBFS)
// =====================================================================
//
// Skips inference when the window is below the training-pipeline curation
// threshold. Feeding the model audio quieter than it saw during training
// produces out-of-distribution garbage. Also saves substantial CPU on
// silent windows during arming pauses.
//
static float window_rms_dbfs(const int16_t *pcm, int n_samples)
{
    double sum_sq = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double s = static_cast<double>(pcm[i]) / 32768.0;
        sum_sq += s * s;
    }
    if (sum_sq < 1e-20) return -100.0f;
    return 10.0f * log10f(static_cast<float>(sum_sq / n_samples));
    // Note: 10*log10(rms²) == 20*log10(rms) — matches the training
    // pipeline's compute_rms_db().
}

// =====================================================================
// Inference task
// =====================================================================

static void recognizer_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "recognizer_task running on core %d", xPortGetCoreID());

    // WDT for Core 1 IDLE is disabled via sdkconfig
    // (CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU1=n) so Invoke() can run for
    // hundreds of milliseconds without tripping the watchdog.

    while (true) {
        // ── Pull next utterance from the segmenter ───────────────────
        utterance_t utt;
        if (xQueueReceive(s_utterance_q, &utt, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (!utt.pcm || utt.n_samples == 0) {
            s_invalid_drop_count.fetch_add(1, std::memory_order_relaxed);
            if (utt.pcm) heap_caps_free(utt.pcm);
            continue;
        }

        // ── Check for hot-reload request ─────────────────────────────
        if (s_reload_requested.exchange(false, std::memory_order_relaxed)) {
            ESP_LOGI(TAG, "hot-reloading model from flash...");
            if (load_model_from_partition() && setup_interpreter()) {
                s_model_loaded = true;
                ESP_LOGI(TAG, "model reloaded successfully");
            } else {
                ESP_LOGE(TAG, "model reload failed — keeping previous");
            }
        }

        // ── Energy gate ──────────────────────────────────────────────
        // Skip inference on silent / low-energy windows. The segmenter's
        // onset threshold is more permissive than this gate, so a letter
        // that segmented but failed the gate is logged and dropped.
        float rms_db = window_rms_dbfs(utt.pcm, (int)utt.n_samples);
        if (rms_db < SPELL_ENERGY_GATE_DB) {
            uint32_t n_skips = s_gate_skip_count.fetch_add(
                                   1, std::memory_order_relaxed) + 1;
            if (n_skips <= 3 || n_skips % 30 == 0) {
                ESP_LOGD(TAG, "energy gate: %.1f dBFS < %.1f — skipping "
                              "(#%" PRIu32 " skips)",
                         rms_db, SPELL_ENERGY_GATE_DB, n_skips);
            }
            heap_caps_free(utt.pcm);
            continue;
        }

        // ── No model yet? Drop and loop. ─────────────────────────────
        if (!s_model_loaded || !s_interpreter) {
            static int no_model_log_counter = 0;
            if (no_model_log_counter++ % 10 == 0) {
                ESP_LOGW(TAG, "no model loaded — inference disabled "
                              "(awaiting OTA or flash)");
            }
            heap_caps_free(utt.pcm);
            continue;
        }

        // ── Compute MFCC + Δ + ΔΔ feature tensor ─────────────────────
        int64_t t0 = esp_timer_get_time();
        feat_extract_compute(utt.pcm, (int)utt.n_samples, s_features);
        int64_t dt_feat_us = esp_timer_get_time() - t0;

        // ── Quantize features → int8 input tensor ────────────────────
        int64_t t1 = esp_timer_get_time();
        if (s_input->type == kTfLiteInt8) {
            float scale = s_input->params.scale;
            int32_t zero_point = s_input->params.zero_point;
            int8_t *input_data = s_input->data.int8;

            for (int i = 0; i < SPELL_FEAT_N_ELEMENTS; i++) {
                int32_t q = static_cast<int32_t>(
                    roundf(s_features[i] / scale)) + zero_point;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                input_data[i] = static_cast<int8_t>(q);
            }
        } else if (s_input->type == kTfLiteFloat32) {
            // Float model — bring-up / debugging path.
            memcpy(s_input->data.f, s_features,
                   SPELL_FEAT_N_ELEMENTS * sizeof(float));
        }
        int64_t dt_quant_us = esp_timer_get_time() - t1;

        // ── TFLite Invoke ────────────────────────────────────────────
        int64_t t2 = esp_timer_get_time();
        TfLiteStatus status = s_interpreter->Invoke();
        int64_t dt_invoke_us = esp_timer_get_time() - t2;

        int dt_feat_ms   = (int)(dt_feat_us / 1000);
        int dt_quant_ms  = (int)(dt_quant_us / 1000);
        int dt_invoke_ms = (int)(dt_invoke_us / 1000);

        // Timing breakdown — log first 5, then every 20th. Reads the
        // pre-increment count (the increment is below at top-K dispatch).
        uint32_t n_pre = s_inference_count.load(std::memory_order_relaxed);
        if (n_pre < 5 || n_pre % 20 == 0) {
            ESP_LOGI(TAG, "timing: feat=%dms quant=%dms invoke=%dms total=%dms",
                     dt_feat_ms, dt_quant_ms, dt_invoke_ms,
                     dt_feat_ms + dt_quant_ms + dt_invoke_ms);
        }

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke() failed");
            heap_caps_free(utt.pcm);
            continue;
        }

        // ── Read & dequantize 26-class output ────────────────────────
        float probs[26];
        if (s_output->type == kTfLiteInt8) {
            float scale = s_output->params.scale;
            int32_t zero_point = s_output->params.zero_point;
            for (int i = 0; i < 26; i++) {
                probs[i] = (s_output->data.int8[i] - zero_point) * scale;
            }
        } else if (s_output->type == kTfLiteFloat32) {
            for (int i = 0; i < 26; i++) probs[i] = s_output->data.f[i];
        } else {
            ESP_LOGE(TAG, "unexpected output tensor type %s",
                     TfLiteTypeGetName(s_output->type));
            heap_caps_free(utt.pcm);
            continue;
        }

        // If the model emits raw logits (no in-graph Softmax), apply a
        // numerically-stable softmax in firmware. Detect logits by checking
        // for negative values or values > 1 — true probabilities live in
        // [0, 1].
        bool looks_like_logits = false;
        for (int i = 0; i < 26; i++) {
            if (probs[i] < 0.0f || probs[i] > 1.0f) { looks_like_logits = true; break; }
        }
        if (looks_like_logits) {
            float max_logit = probs[0];
            for (int i = 1; i < 26; i++) {
                if (probs[i] > max_logit) max_logit = probs[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < 26; i++) {
                probs[i] = expf(probs[i] - max_logit);
                sum += probs[i];
            }
            for (int i = 0; i < 26; i++) probs[i] /= sum;
        }

        // ── Top-K selection + event post ─────────────────────────────
        // retract_count is always 0 here; the W-recovery cycle (ADR-0005)
        // is the only path that sets a non-zero value, and it lands in a
        // later slice (#30) by interposing on this same emission point.
        spell_letter_recognized_event_t evt;
        top_k_select(probs, 26, evt.top_k.candidates, SPELL_LETTER_TOP_K);
        evt.top_k.invoke_ms = (uint32_t)dt_invoke_ms;
        evt.retract_count   = 0;

        uint32_t n_inf = s_inference_count.fetch_add(
                              1, std::memory_order_relaxed) + 1;

        // First three results + every 5th thereafter at INFO. The state
        // machine + decoder log their own digest events anyway, so this
        // is mostly for low-level bring-up debugging.
        if (n_inf <= 3 || n_inf % 5 == 0) {
            ESP_LOGI(TAG, "#%" PRIu32 " %c=%.2f %c=%.2f %c=%.2f (%dms)",
                     n_inf,
                     'A' + evt.top_k.candidates[0].letter_index,
                     evt.top_k.candidates[0].probability,
                     'A' + evt.top_k.candidates[1].letter_index,
                     evt.top_k.candidates[1].probability,
                     'A' + evt.top_k.candidates[2].letter_index,
                     evt.top_k.candidates[2].probability,
                     dt_invoke_ms);
        }

        esp_event_post(SPELL_RECOGNIZER_EVENT, SPELL_EVENT_LETTER_RECOGNIZED,
                       &evt, sizeof(evt), 0);

        // Retain a copy in the PCM ring for W-recovery (#30). The original
        // inbound buffer is freed below — the ring owns its own storage.
        pcm_ring_insert(utt.pcm, utt.n_samples);

        heap_caps_free(utt.pcm);
    }
}

// =====================================================================
// Public API
// =====================================================================

esp_err_t letter_recognizer_init(void)
{
    if (s_utterance_q) {
        ESP_LOGW(TAG, "letter_recognizer_init() called twice — ignoring");
        return ESP_OK;
    }

    // ── Feature frontend (MFCC + Δ + ΔΔ) ─────────────────────────────
    esp_err_t ret = feat_extract_init();
    if (ret != ESP_OK) return ret;

    // Feature output buffer — [79, 20, 3] floats ≈ 19 KB.
    s_features = static_cast<float *>(heap_caps_malloc(
        SPELL_FEAT_N_ELEMENTS * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!s_features) {
        ESP_LOGE(TAG, "PSRAM alloc failed for feature buffer");
        return ESP_ERR_NO_MEM;
    }

    // ── PCM ring storage (W-recovery, #30) ───────────────────────────
    // 4 × 25,600 B = 102,400 B contiguous in PSRAM. Fail loudly here —
    // ADR-0005's W-recovery is load-bearing for accurate W spelling.
    {
        const size_t ring_bytes =
            (size_t)SPELL_W_DETECT_PCM_RING_DEPTH * SPELL_INFERENCE_WINDOW_BYTES;
        s_pcm_ring_storage = static_cast<int16_t *>(heap_caps_malloc(
            ring_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (!s_pcm_ring_storage) {
            ESP_LOGE(TAG, "PSRAM alloc for PCM ring failed (%u B, depth=%d)",
                     (unsigned)ring_bytes, SPELL_W_DETECT_PCM_RING_DEPTH);
            return ESP_ERR_NO_MEM;
        }
        memset(s_pcm_ring_storage, 0, ring_bytes);
        ESP_LOGI(TAG, "PCM ring allocated: %d slots × %d B = %u B PSRAM",
                 SPELL_W_DETECT_PCM_RING_DEPTH,
                 SPELL_INFERENCE_WINDOW_BYTES,
                 (unsigned)ring_bytes);
    }

    // ── Load model (non-fatal if empty — awaiting OTA) ───────────────
    if (load_model_from_partition()) {
        if (setup_interpreter()) {
            s_model_loaded = true;
            ESP_LOGI(TAG, "letter recognizer ready (top-K=%d)",
                     SPELL_LETTER_TOP_K);
        } else {
            ESP_LOGW(TAG, "interpreter setup failed — recognition disabled");
        }
    } else {
        ESP_LOGW(TAG, "no model in flash — recognition disabled until OTA");
    }

    // ── Utterance queue (segmenter → recognizer) ─────────────────────
    s_utterance_q = xQueueCreate(SPELL_INFERENCE_QUEUE_DEPTH,
                                 sizeof(utterance_t));
    if (!s_utterance_q) {
        ESP_LOGE(TAG, "xQueueCreate failed");
        return ESP_ERR_NO_MEM;
    }

    // ── Spawn recognizer task on Core 1 ──────────────────────────────
    BaseType_t r = xTaskCreatePinnedToCore(
        recognizer_task,
        "recognizer",
        SPELL_INFERENCE_TASK_STACK,
        nullptr,
        5,          // priority — below capture (10), above decoder (4)
        &s_task,
        1);         // Core 1

    if (r != pdPASS) {
        ESP_LOGE(TAG, "xTaskCreatePinnedToCore failed (rc=%d)", r);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "recognizer task spawned on core 1 (model %s)",
             s_model_loaded ? "active" : "pending OTA");
    return ESP_OK;
}

esp_err_t letter_recognizer_submit_utterance(int16_t *pcm, size_t n_samples)
{
    if (!s_utterance_q) return ESP_ERR_INVALID_STATE;
    if (!pcm || n_samples == 0) return ESP_ERR_INVALID_ARG;

    utterance_t u = { pcm, n_samples };
    if (xQueueSend(s_utterance_q, &u, 0) != pdTRUE) {
        uint32_t n_drops = s_drop_count.fetch_add(
                               1, std::memory_order_relaxed) + 1;
        if (n_drops <= 3 || n_drops % 10 == 0) {
            ESP_LOGW(TAG, "utterance queue full — dropping (#%" PRIu32 ")",
                     n_drops);
        }
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t letter_recognizer_reload_model(void)
{
    s_reload_requested.store(true, std::memory_order_relaxed);
    ESP_LOGI(TAG, "model reload requested — will apply on next utterance");
    return ESP_OK;
}

uint32_t letter_recognizer_get_count(void) {
    return s_inference_count.load(std::memory_order_relaxed);
}
uint32_t letter_recognizer_get_gate_skip_count(void) {
    return s_gate_skip_count.load(std::memory_order_relaxed);
}
uint32_t letter_recognizer_get_drop_count(void) {
    return s_drop_count.load(std::memory_order_relaxed);
}
uint32_t letter_recognizer_get_invalid_drop_count(void) {
    return s_invalid_drop_count.load(std::memory_order_relaxed);
}

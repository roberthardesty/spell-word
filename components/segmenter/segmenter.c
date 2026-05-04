/**
 * @file segmenter.c
 * @brief IDF wrapper around segmenter_core — task, audio_capture
 *        subscription, PSRAM buffer ownership, event publishing.
 *
 * The wrapper does no VAD logic of its own; everything is delegated to
 * segmenter_core, which is unit-tested host-side via tools/segmenter_replay/.
 *
 * Buffer ownership:
 *
 *   - Pre-roll ring buffer (~6.4 KB, 200 ms @ 16 kHz): allocated once in
 *     PSRAM during init; lifetime equals the firmware run.
 *   - Accumulator (sized for pre-roll + max-letter): allocated once in PSRAM.
 *   - Per-utterance PSRAM buffer for the recognizer: heap_caps_malloc'd at
 *     emit time, ownership transfers to the recognizer task on successful
 *     submit (it heap_caps_frees after processing). On submit failure, freed
 *     here.
 */

#include "segmenter.h"
#include "segmenter_core.h"
#include "spell_config.h"
#include "spell_events.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "audio_capture.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "letter_recognizer.h"

static const char *TAG = "segmenter";

ESP_EVENT_DEFINE_BASE(SPELL_SEGMENTER_EVENT);

// Frame size in samples — must equal capture chunk size so the StreamBuffer
// wakes us once per chunk and we process exactly one VAD frame per read.
#define VAD_FRAME_SAMPLES   SPELL_CAPTURE_CHUNK_FRAMES
#define VAD_FRAME_BYTES     (VAD_FRAME_SAMPLES * (int)sizeof(int16_t))

// Pre-roll capacity, in samples. ~200 ms @ 16 kHz for context.
#define VAD_PREROLL_SAMPLES SPELL_VAD_PREROLL_SAMPLES   // 3200

// Accumulator capacity. Has to fit pre-roll + max-letter + one trailing
// off-confirmation frame. We over-allocate slightly for safety.
#define VAD_ACCUM_CAPACITY \
    (VAD_PREROLL_SAMPLES + \
     (SPELL_VAD_MAX_LETTER_MS * SPELL_SAMPLE_RATE / 1000) + \
     SPELL_VAD_OFF_FRAMES * VAD_FRAME_SAMPLES + 256)

// Convert a millisecond config value to whole frames. Always rounds down to
// match the "consecutive frames" counter semantics in the core.
#define MS_TO_FRAMES(ms) \
    (((ms) * SPELL_SAMPLE_RATE) / (1000 * VAD_FRAME_SAMPLES))

// EMA smoothing — 35 ≈ 0.35 alpha, ~3-frame time constant.
#define VAD_EMA_ALPHA_X100  35

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

static audio_capture_sub_t s_sub      = NULL;
static TaskHandle_t        s_task     = NULL;
static atomic_bool         s_active   = ATOMIC_VAR_INIT(true);

static int16_t            *s_preroll  = NULL;  // PSRAM
static int16_t            *s_accum    = NULL;  // PSRAM
static seg_state_t         s_seg;              // BSS — no buffers inside

static volatile uint32_t   s_alloc_fail = 0;

// ---------------------------------------------------------------------------
// Emit a letter event by handing PCM to the inference task.
//
// The core's accum buffer is reused on the next frame, so we MUST copy the
// samples out before returning. We allocate a fresh PSRAM buffer of the
// inference-window size, center-pad the captured samples into it, and
// submit. If submit fails (queue full), we free.
// ---------------------------------------------------------------------------
static void dispatch_letter(const seg_event_t *evt, bool is_force_split)
{
    int16_t *out = heap_caps_malloc(SPELL_INFERENCE_WINDOW_BYTES,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!out) {
        s_alloc_fail++;
        ESP_LOGW(TAG, "PSRAM alloc for utterance failed (drop #%" PRIu32 ")",
                 (uint32_t)s_alloc_fail);
        return;
    }

    // Center-pad the captured samples into a 600 ms window.
    // If the captured letter (incl. pre-roll) exceeds 600 ms, take the
    // highest-energy window-sized slice. For now the simple heuristic is
    // "take the trailing 600 ms" — this preserves the offset, which carries
    // the most discriminative information for stop consonants. The proper
    // highest-energy slice can replace this if accuracy demands it.
    int captured = evt->n_samples;
    int win      = SPELL_INFERENCE_WINDOW_SAMPLES;

    memset(out, 0, SPELL_INFERENCE_WINDOW_BYTES);

    if (captured <= win) {
        int pad_lead = (win - captured) / 2;
        memcpy(out + pad_lead, evt->pcm, captured * sizeof(int16_t));
    } else {
        int start = captured - win;   // trailing slice
        memcpy(out, &evt->pcm[start], win * sizeof(int16_t));
    }

    esp_err_t err = letter_recognizer_submit_utterance(out, win);
    if (err != ESP_OK) {
        heap_caps_free(out);
        return;
    }

    ESP_LOGI(TAG, "%s%u: %d samples, %d ms, peak %.1f dBFS",
             is_force_split ? "letter (force-split) #" : "letter #",
             (unsigned)s_seg.total_letters_emitted,
             evt->n_samples, evt->duration_ms, evt->peak_dbfs);
}

// ---------------------------------------------------------------------------
// Segmenter task
// ---------------------------------------------------------------------------
static void segmenter_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "segmenter_task running on core %d", xPortGetCoreID());

    int16_t frame[VAD_FRAME_SAMPLES];

    while (true) {
        size_t got = audio_capture_read(s_sub, frame, VAD_FRAME_BYTES,
                                        portMAX_DELAY);
        if (got != VAD_FRAME_BYTES) {
            // Partial read — rare; just discard.
            continue;
        }

        if (!atomic_load(&s_active)) {
            // Inactive: drain frames so capture never overruns; reset the
            // state machine so a future activation starts clean.
            seg_reset(&s_seg);
            continue;
        }

        seg_event_t evt = seg_process_frame(&s_seg, frame, VAD_FRAME_SAMPLES);

        switch (evt.kind) {
        case SEG_EVT_NONE:
            break;

        case SEG_EVT_LETTER_EMITTED:
            dispatch_letter(&evt, /*is_force_split=*/false);
            break;

        case SEG_EVT_FORCE_SPLIT:
            dispatch_letter(&evt, /*is_force_split=*/true);
            break;

        case SEG_EVT_LETTER_REJECTED_SHORT:
            ESP_LOGD(TAG, "rejected short blip: %d ms, peak %.1f dBFS",
                     evt.duration_ms, evt.peak_dbfs);
            break;

        case SEG_EVT_END_OF_WORD: {
            spell_eow_event_t eow = {
                .letter_count = evt.letters_in_word,
            };
            ESP_LOGI(TAG, "end-of-word #%" PRIu32 " (letters=%" PRIu32 ")",
                     (uint32_t)s_seg.total_words_completed,
                     evt.letters_in_word);
            esp_event_post(SPELL_SEGMENTER_EVENT, SPELL_EVENT_END_OF_WORD,
                           &eow, sizeof(eow), 0);
            break;
        }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

esp_err_t segmenter_init(void)
{
    if (s_sub) {
        ESP_LOGW(TAG, "segmenter_init() called twice — ignoring");
        return ESP_OK;
    }

    // Allocate VAD buffers in PSRAM. The state struct itself is small
    // and lives in BSS.
    s_preroll = heap_caps_malloc(VAD_PREROLL_SAMPLES * sizeof(int16_t),
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_accum   = heap_caps_malloc(VAD_ACCUM_CAPACITY  * sizeof(int16_t),
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_preroll || !s_accum) {
        ESP_LOGE(TAG, "PSRAM alloc failed (preroll=%p accum=%p)",
                 s_preroll, s_accum);
        if (s_preroll) heap_caps_free(s_preroll);
        if (s_accum)   heap_caps_free(s_accum);
        s_preroll = s_accum = NULL;
        return ESP_ERR_NO_MEM;
    }

    seg_config_t cfg = {
        .frame_samples       = VAD_FRAME_SAMPLES,
        .sample_rate         = SPELL_SAMPLE_RATE,
        .on_dbfs             = SPELL_VAD_ON_DBFS,
        .off_dbfs            = SPELL_VAD_OFF_DBFS,
        .ema_alpha_x100      = VAD_EMA_ALPHA_X100,
        .off_frames          = SPELL_VAD_OFF_FRAMES,
        .min_letter_frames   = MS_TO_FRAMES(SPELL_VAD_MIN_LETTER_MS),
        .max_letter_frames   = MS_TO_FRAMES(SPELL_VAD_MAX_LETTER_MS),
        .eow_frames          = MS_TO_FRAMES(SPELL_VAD_END_OF_WORD_MS),
        .preroll             = s_preroll,
        .preroll_samples     = VAD_PREROLL_SAMPLES,
        .accum               = s_accum,
        .accum_capacity      = VAD_ACCUM_CAPACITY,
    };
    if (seg_init(&s_seg, &cfg) != 0) {
        ESP_LOGE(TAG, "seg_init() rejected configuration");
        return ESP_ERR_INVALID_ARG;
    }

    s_sub = audio_capture_subscribe(SPELL_SEGMENTER_SUB_BYTES, "segmenter");
    if (!s_sub) {
        ESP_LOGE(TAG, "audio_capture_subscribe() failed");
        return ESP_FAIL;
    }

    BaseType_t r = xTaskCreatePinnedToCore(
        segmenter_task,
        "segmenter",
        4096,
        NULL,
        8,
        &s_task,
        0);
    if (r != pdPASS) {
        ESP_LOGE(TAG, "xTaskCreatePinnedToCore failed (rc=%d)", r);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "ready: on=%.1f off=%.1f off_frames=%d "
                  "min_letter=%d max_letter=%d eow=%d (in frames)",
             cfg.on_dbfs, cfg.off_dbfs, cfg.off_frames,
             cfg.min_letter_frames, cfg.max_letter_frames, cfg.eow_frames);
    return ESP_OK;
}

void segmenter_set_active(bool active)
{
    atomic_store(&s_active, active);
    ESP_LOGI(TAG, "active=%d", active ? 1 : 0);
}

bool segmenter_is_active(void)
{
    return atomic_load(&s_active);
}

uint32_t segmenter_get_letter_count(void)     { return s_seg.total_letters_emitted; }
uint32_t segmenter_get_word_count(void)       { return s_seg.total_words_completed; }
uint32_t segmenter_get_alloc_fail_count(void) { return s_alloc_fail; }

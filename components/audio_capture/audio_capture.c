/**
 * @file audio_capture.c
 * @brief I2S RX, 32→16-bit conversion, fan-out to N PSRAM-backed subscribers.
 *
 * Lifted from the EARS POC firmware with prefix-only renaming. See the
 * EARS-Reuse-Assessment.md document for the discussion of why this code is
 * reused unchanged and which design decisions carry over.
 *
 * Intentionally ADF-free: drives the I2S new-driver (driver/i2s_std.h)
 * directly, because ADF's i2s_stream / raw_stream pull in board / codec /
 * speech components that don't apply to a single-mic capture device, and
 * introduce an esp-dsp version conflict.
 */

#include "audio_capture.h"

#include <string.h>

#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"

#include "spell_config.h"

static const char *TAG = "audio_capture";

// ---------------------------------------------------------------------------
// Subscriber table
// ---------------------------------------------------------------------------
struct audio_capture_sub {
    StreamBufferHandle_t    stream;
    StaticStreamBuffer_t    stream_struct;  // small metadata, kept in internal RAM
    uint8_t                *storage;        // PSRAM — owned by this sub
    size_t                  capacity;
    uint32_t                overrun_count;
    char                    label[16];
    bool                    active;
};

static struct audio_capture_sub s_subs[SPELL_CAPTURE_MAX_SUBS];
static size_t                   s_sub_count = 0;

// ---------------------------------------------------------------------------
// I2S + task state
// ---------------------------------------------------------------------------
static i2s_chan_handle_t  s_rx_chan      = NULL;
static TaskHandle_t       s_capture_task = NULL;
static bool               s_inited       = false;
static bool               s_started      = false;

static void capture_task(void *arg);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
esp_err_t audio_capture_init(void)
{
    if (s_inited) {
        ESP_LOGW(TAG, "audio_capture_init() called twice — ignoring");
        return ESP_OK;
    }

    // --- I2S channel ------------------------------------------------------
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        SPELL_I2S_RX_PORT, I2S_ROLE_MASTER);
    // dma_desc_num × dma_frame_num defaults (6 × 240) give ~90 ms of
    // buffering — fine for our 16 ms read chunks.
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan),
                        TAG, "i2s_new_channel");

    // --- Slot + clock config ---------------------------------------------
    // ICS-43434 emits 24-bit two's-complement samples left-justified in a
    // 32-bit slot. Read at 32-bit; we shift >> 16 in the capture task to
    // get the top 16 bits.
    //
    // SEL pin on our board is tied to GND → mic drives data into the LEFT
    // slot only. MONO + slot_mask=LEFT captures exactly those samples.
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SPELL_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = SPELL_I2S_RX_SCK_PIN,
            .ws   = SPELL_I2S_RX_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din  = SPELL_I2S_RX_SD_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx_chan, &std_cfg),
                        TAG, "i2s_channel_init_std_mode");

    s_inited = true;

    ESP_LOGI(TAG, "I2S configured: %d Hz, 32-bit LEFT slot (ICS-43434), "
                  "expected PCM rate %d B/s",
             SPELL_SAMPLE_RATE,
             SPELL_SAMPLE_RATE * SPELL_BYTES_PER_SAMPLE);
    return ESP_OK;
}

audio_capture_sub_t audio_capture_subscribe(size_t byte_capacity,
                                            const char *label)
{
    if (!s_inited) {
        ESP_LOGE(TAG, "subscribe() before init()");
        return NULL;
    }
    if (s_started) {
        ESP_LOGE(TAG, "subscribe('%s') after start() — not supported",
                 label ? label : "?");
        return NULL;
    }
    if (byte_capacity == 0) {
        ESP_LOGE(TAG, "subscribe('%s'): byte_capacity must be > 0",
                 label ? label : "?");
        return NULL;
    }
    if (s_sub_count >= SPELL_CAPTURE_MAX_SUBS) {
        ESP_LOGE(TAG, "subscribe('%s'): MAX_SUBS (%d) reached",
                 label ? label : "?", SPELL_CAPTURE_MAX_SUBS);
        return NULL;
    }

    struct audio_capture_sub *sub = &s_subs[s_sub_count];
    memset(sub, 0, sizeof(*sub));

    // StreamBuffer storage in PSRAM. SPIRAM|8BIT is correct for byte-oriented
    // buffers — PSRAM doesn't support unaligned internal accesses but
    // xStreamBufferSend/Receive memcpy through it as bytes.
    sub->storage = heap_caps_malloc(byte_capacity,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!sub->storage) {
        ESP_LOGE(TAG, "subscribe('%s'): PSRAM alloc %u B failed",
                 label ? label : "?", (unsigned)byte_capacity);
        return NULL;
    }

    // trigger level = 1: wake xStreamBufferReceive as soon as any data is
    // available. Consumers set their own "wait for N bytes" logic via the
    // max_bytes argument + portMAX_DELAY.
    sub->stream = xStreamBufferCreateStatic(byte_capacity,
                                            1,
                                            sub->storage,
                                            &sub->stream_struct);
    if (!sub->stream) {
        heap_caps_free(sub->storage);
        sub->storage = NULL;
        ESP_LOGE(TAG, "subscribe('%s'): xStreamBufferCreateStatic failed",
                 label ? label : "?");
        return NULL;
    }

    sub->capacity      = byte_capacity;
    sub->overrun_count = 0;
    if (label) {
        strncpy(sub->label, label, sizeof(sub->label) - 1);
        sub->label[sizeof(sub->label) - 1] = '\0';
    } else {
        snprintf(sub->label, sizeof(sub->label), "sub%u",
                 (unsigned)s_sub_count);
    }
    sub->active = true;
    s_sub_count++;

    ESP_LOGI(TAG, "subscriber[%u]='%s' registered (%u B in PSRAM)",
             (unsigned)(s_sub_count - 1), sub->label, (unsigned)byte_capacity);
    return sub;
}

esp_err_t audio_capture_start(void)
{
    if (!s_inited) {
        ESP_LOGE(TAG, "start() before init()");
        return ESP_ERR_INVALID_STATE;
    }
    if (s_started) {
        ESP_LOGW(TAG, "start() called twice — ignoring");
        return ESP_OK;
    }
    if (s_sub_count == 0) {
        ESP_LOGW(TAG, "start() with zero subscribers — capture will discard "
                      "all samples");
    }

    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_rx_chan), TAG, "i2s_channel_enable");

    // Pin capture to Core 0. Inference runs on Core 1; this matches the
    // EARS topology and keeps the latency-sensitive I2S reads isolated
    // from the multi-hundred-millisecond TFLM Invokes.
    BaseType_t r = xTaskCreatePinnedToCore(capture_task,
                                           "audio_cap",
                                           4096,       // stack
                                           NULL,
                                           10,         // priority
                                           &s_capture_task,
                                           0);         // core
    if (r != pdPASS) {
        ESP_LOGE(TAG, "xTaskCreatePinnedToCore failed (rc=%d)", r);
        i2s_channel_disable(s_rx_chan);
        return ESP_FAIL;
    }

    s_started = true;
    ESP_LOGI(TAG, "capture task started with %u subscriber%s",
             (unsigned)s_sub_count, s_sub_count == 1 ? "" : "s");
    return ESP_OK;
}

size_t audio_capture_read(audio_capture_sub_t sub,
                          void *dst,
                          size_t max_bytes,
                          TickType_t timeout)
{
    if (!sub || !sub->active || !dst || max_bytes == 0) return 0;
    return xStreamBufferReceive(sub->stream, dst, max_bytes, timeout);
}

uint32_t audio_capture_sub_overrun_count(audio_capture_sub_t sub)
{
    if (!sub) return 0;
    return sub->overrun_count;
}

// ---------------------------------------------------------------------------
// Capture task
// ---------------------------------------------------------------------------
//
// Reads CHUNK_FRAMES of 32-bit I2S samples per iteration, converts to int16
// via >> 16, fans out to every active subscriber, and logs bytes/sec once
// per second for verification.
//
// All subscriber sends are NON-BLOCKING: a slow consumer increments its own
// overrun_count but never delays the capture loop or other consumers.
//
static void capture_task(void *arg)
{
    (void)arg;

    int32_t i2s_buf[SPELL_CAPTURE_CHUNK_FRAMES];
    int16_t pcm_buf[SPELL_CAPTURE_CHUNK_FRAMES];

    size_t   bytes_this_second = 0;
    int64_t  t_last_rate_log   = esp_timer_get_time();

    ESP_LOGI(TAG, "capture_task running on core %d", xPortGetCoreID());

    while (true) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(s_rx_chan,
                                         i2s_buf,
                                         sizeof(i2s_buf),
                                         &bytes_read,
                                         portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "i2s_channel_read: %s", esp_err_to_name(err));
            continue;
        }

        const size_t n_frames = bytes_read / sizeof(int32_t);
        if (n_frames == 0) continue;

        // 32 → 16 bit: keep the top 16 bits of the 24-bit sample.
        for (size_t i = 0; i < n_frames; i++) {
            pcm_buf[i] = (int16_t)(i2s_buf[i] >> 16);
        }

        const size_t pcm_bytes = n_frames * sizeof(int16_t);

        // Fan out to every subscriber. Non-blocking; slow consumers count
        // overruns but never stall capture.
        for (size_t i = 0; i < s_sub_count; i++) {
            struct audio_capture_sub *sub = &s_subs[i];
            if (!sub->active) continue;
            size_t sent = xStreamBufferSend(sub->stream,
                                            pcm_buf,
                                            pcm_bytes,
                                            0 /* no wait */);
            if (sent < pcm_bytes) {
                sub->overrun_count++;
            }
        }

        bytes_this_second += pcm_bytes;

        // Log data rate once a second — exit criterion for Phase 1
        // ("steady 32 000 B/s, 0 overruns").
        const int64_t t_now = esp_timer_get_time();
        if (t_now - t_last_rate_log >= 1000000LL) {
            ESP_LOGI(TAG, "%u B/s", (unsigned)bytes_this_second);
            bytes_this_second = 0;
            t_last_rate_log   = t_now;
        }
    }
}

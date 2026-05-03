/**
 * @file audio_dump.c
 * @brief Continuous PSRAM ring of recent PCM, with a base64 UART dump path.
 *
 * Architecture:
 *
 *   audio_capture_subscribe(N seconds × 32 KB) "audio_dump"
 *     ↓
 *   drain_task (Core 0, prio 4 — below capture/segmenter/inference)
 *     ↓ reads chunks from StreamBuffer, writes into PSRAM ring (overwriting)
 *     ↓ atomic "frozen" flag pauses drain during emit
 *
 *   audio_dump_emit_b64(seconds):
 *     1. set frozen = true
 *     2. snapshot ring write_pos
 *     3. base64-encode the slice [write_pos - seconds*rate .. write_pos)
 *     4. printf the framed payload
 *     5. clear frozen
 */

#include "audio_dump.h"
#include "spell_config.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "audio_capture.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "audio_dump";

// PSRAM ring buffer (int16 samples).
static int16_t          *s_ring         = NULL;
static size_t            s_ring_samples = 0;        // capacity in samples
static volatile size_t   s_ring_wpos    = 0;        // next write index
static volatile size_t   s_ring_filled  = 0;        // total samples written (capped)

static audio_capture_sub_t s_sub        = NULL;
static TaskHandle_t        s_task       = NULL;
static atomic_bool         s_frozen     = ATOMIC_VAR_INIT(false);
static volatile uint32_t   s_drop_while_frozen = 0;

// Drain chunk size. We read this many samples at a time from the
// StreamBuffer; the ring's wpos advances by exactly this much.
#define DRAIN_CHUNK_SAMPLES   SPELL_CAPTURE_CHUNK_FRAMES
#define DRAIN_CHUNK_BYTES     (DRAIN_CHUNK_SAMPLES * (int)sizeof(int16_t))

// =============================================================================
// Base64 encoder — RFC 4648 standard alphabet, line-wrapped at 76 chars.
// =============================================================================

static const char b64_alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Emit @p len bytes as base64 to stdout, wrapping every 76 chars onto a
/// new line. Padding ('=') is emitted as required.
static void b64_emit(const uint8_t *data, size_t len)
{
    char line[80];
    int  col = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t triple = ((uint32_t)data[i]) << 16;
        if (i + 1 < len) triple |= ((uint32_t)data[i + 1]) << 8;
        if (i + 2 < len) triple |= ((uint32_t)data[i + 2]);

        line[col++] = b64_alphabet[(triple >> 18) & 0x3F];
        line[col++] = b64_alphabet[(triple >> 12) & 0x3F];
        line[col++] = (i + 1 < len) ? b64_alphabet[(triple >> 6) & 0x3F] : '=';
        line[col++] = (i + 2 < len) ? b64_alphabet[(triple)      & 0x3F] : '=';

        if (col >= 76) {
            line[col] = '\0';
            puts(line);
            col = 0;
        }
    }
    if (col > 0) {
        line[col] = '\0';
        puts(line);
    }
}

// =============================================================================
// Drain task
// =============================================================================

static void drain_task(void *arg)
{
    (void)arg;
    int16_t buf[DRAIN_CHUNK_SAMPLES];

    ESP_LOGI(TAG, "drain_task running on core %d", xPortGetCoreID());

    while (true) {
        if (atomic_load(&s_frozen)) {
            // Sleep briefly while a dump is in progress. The audio_capture
            // StreamBuffer keeps backing up; on resume we'll either drain
            // it fast or it'll start dropping samples (overrun_count ticks).
            vTaskDelay(pdMS_TO_TICKS(20));
            s_drop_while_frozen++;
            continue;
        }

        size_t got = audio_capture_read(s_sub, buf, DRAIN_CHUNK_BYTES,
                                        portMAX_DELAY);
        if (got == 0) continue;

        size_t n = got / sizeof(int16_t);
        size_t wpos = s_ring_wpos;
        for (size_t i = 0; i < n; i++) {
            s_ring[wpos] = buf[i];
            wpos++;
            if (wpos >= s_ring_samples) wpos = 0;
        }
        s_ring_wpos = wpos;
        s_ring_filled += n;
        if (s_ring_filled > s_ring_samples) s_ring_filled = s_ring_samples;
    }
}

// =============================================================================
// Public API
// =============================================================================

esp_err_t audio_dump_init(int buffer_seconds)
{
    if (s_ring) {
        ESP_LOGW(TAG, "audio_dump_init() called twice — ignoring");
        return ESP_OK;
    }
    if (buffer_seconds <= 0) return ESP_ERR_INVALID_ARG;

    s_ring_samples = (size_t)buffer_seconds * SPELL_SAMPLE_RATE;
    s_ring = heap_caps_malloc(s_ring_samples * sizeof(int16_t),
                              MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_ring) {
        ESP_LOGE(TAG, "PSRAM alloc for ring (%u samples = %u KB) failed",
                 (unsigned)s_ring_samples,
                 (unsigned)(s_ring_samples * sizeof(int16_t) / 1024));
        return ESP_ERR_NO_MEM;
    }
    memset(s_ring, 0, s_ring_samples * sizeof(int16_t));

    // Subscribe with a generous slack — drain runs at low priority so the
    // StreamBuffer needs to absorb a few hundred ms of jitter under load.
    size_t sub_bytes = SPELL_SAMPLE_RATE * sizeof(int16_t) / 2;  // 0.5 s
    s_sub = audio_capture_subscribe(sub_bytes, "audio_dump");
    if (!s_sub) {
        ESP_LOGE(TAG, "audio_capture_subscribe() failed");
        heap_caps_free(s_ring);
        s_ring = NULL;
        return ESP_FAIL;
    }

    BaseType_t r = xTaskCreatePinnedToCore(
        drain_task,
        "audio_dump",
        3072,
        NULL,
        4,          // below capture/segmenter/inference; not latency-critical
        &s_task,
        0);
    if (r != pdPASS) {
        ESP_LOGE(TAG, "xTaskCreatePinnedToCore failed (rc=%d)", r);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "ready: %d s ring (%u KB PSRAM)",
             buffer_seconds,
             (unsigned)(s_ring_samples * sizeof(int16_t) / 1024));
    return ESP_OK;
}

esp_err_t audio_dump_emit_b64(int seconds)
{
    if (!s_ring) return ESP_ERR_INVALID_STATE;

    // ── Freeze the drain so the snapshot is consistent ───────────────
    atomic_store(&s_frozen, true);
    // Brief grace period so any in-flight drain iteration finishes.
    vTaskDelay(pdMS_TO_TICKS(5));

    size_t filled = s_ring_filled;
    size_t wpos   = s_ring_wpos;

    size_t want_samples;
    if (seconds < 0) {
        want_samples = filled;
    } else {
        want_samples = (size_t)seconds * SPELL_SAMPLE_RATE;
        if (want_samples > filled) want_samples = filled;
    }

    if (want_samples == 0) {
        ESP_LOGW(TAG, "ring is empty — nothing to dump");
        atomic_store(&s_frozen, false);
        return ESP_ERR_INVALID_STATE;
    }

    // Compute the start index of the slice we want. wpos is the next-write
    // position; the most recent sample is at (wpos - 1) mod ring_samples.
    // The slice we want is the last want_samples ending at wpos-1.
    size_t start = (wpos + s_ring_samples - want_samples) % s_ring_samples;

    // ── Frame begin ──────────────────────────────────────────────────
    printf("\r\n===AUDIO_DUMP_BEGIN n_samples=%u rate=%d===\r\n",
           (unsigned)want_samples, SPELL_SAMPLE_RATE);

    // Encode in two contiguous chunks if the slice wraps the ring boundary.
    // Each chunk is base64'd separately; the receiver concatenates the
    // decoded bytes — the line wrapping doesn't affect this.
    if (start + want_samples <= s_ring_samples) {
        b64_emit((const uint8_t *)&s_ring[start],
                 want_samples * sizeof(int16_t));
    } else {
        size_t first = s_ring_samples - start;
        b64_emit((const uint8_t *)&s_ring[start],
                 first * sizeof(int16_t));
        b64_emit((const uint8_t *)&s_ring[0],
                 (want_samples - first) * sizeof(int16_t));
    }

    printf("===AUDIO_DUMP_END===\r\n");

    // ── Resume drain ─────────────────────────────────────────────────
    atomic_store(&s_frozen, false);

    ESP_LOGI(TAG, "dumped %u samples (%.2f s); drained %" PRIu32
                  " freeze ticks during emit",
             (unsigned)want_samples, (double)want_samples / SPELL_SAMPLE_RATE,
             (uint32_t)s_drop_while_frozen);
    s_drop_while_frozen = 0;
    return ESP_OK;
}

int audio_dump_buffered_seconds(void)
{
    if (!s_ring) return 0;
    return (int)(s_ring_filled / SPELL_SAMPLE_RATE);
}

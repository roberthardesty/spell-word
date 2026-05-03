/**
 * @file audio_capture.h
 * @brief I2S microphone capture with PSRAM-backed StreamBuffer fan-out.
 *
 * Three-phase init lifecycle (enforced by the API):
 *
 *   1. audio_capture_init()         — configure I2S, zero subscriber table.
 *   2. audio_capture_subscribe(..)  — one call per consumer; returns a handle.
 *   3. audio_capture_start()        — enable I2S, spawn capture task.
 *
 * Subscribers registered after start() return NULL. The capture task never
 * mutates the subscriber list, so there is no locking in the hot loop.
 *
 * Per-subscriber sends are non-blocking — a slow consumer increments its own
 * overrun_count, but cannot stall the capture task or starve other consumers.
 *
 * Ported from the EARS POC firmware. Architecture unchanged; only the
 * EARS_* prefix → SPELL_* and the I2S RX pin numbers change between projects.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle returned by audio_capture_subscribe().
typedef struct audio_capture_sub *audio_capture_sub_t;

/**
 * Configure the I2S RX driver and zero the subscriber table.
 * Must be called exactly once, before any subscribe() call.
 */
esp_err_t audio_capture_init(void);

/**
 * Allocate a PSRAM-backed StreamBuffer for a consumer.
 *
 * @param byte_capacity  Buffer size in bytes (sized by the consumer to its
 *                       window length plus a chunk of slack).
 * @param label          Short human-readable name, used in log lines.
 *                       Truncated to 15 chars; may be NULL.
 * @return  Subscriber handle, or NULL if init() not yet called, start()
 *          already called, allocation failed, or the table is full.
 */
audio_capture_sub_t audio_capture_subscribe(size_t byte_capacity,
                                            const char *label);

/**
 * Enable I2S and spawn the capture task on Core 0 at priority 10.
 * After this call, no further subscribers may be added.
 */
esp_err_t audio_capture_start(void);

/**
 * Read up to max_bytes of PCM from a subscriber.
 *
 * Wraps xStreamBufferReceive — same blocking semantics: if timeout==
 * portMAX_DELAY, blocks until data is available; if timeout==0, returns
 * immediately with whatever is available (possibly 0). Returns the number
 * of bytes copied into dst.
 */
size_t audio_capture_read(audio_capture_sub_t sub,
                          void *dst,
                          size_t max_bytes,
                          TickType_t timeout);

/**
 * @return Total number of times the capture task tried to write into this
 *         subscriber's buffer while it was full. Useful for logging /
 *         monitoring per-consumer health.
 */
uint32_t audio_capture_sub_overrun_count(audio_capture_sub_t sub);

#ifdef __cplusplus
}  // extern "C"
#endif

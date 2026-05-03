/**
 * @file inference.h
 * @brief DS-CNN spoken-letter classifier — public API.
 *
 * Driven by the segmenter: the segmenter submits utterance windows via
 * inference_submit_utterance(); the inference task pulls them off a queue,
 * runs the energy gate, log-mel + TFLM forward pass, and posts the top-K
 * letter candidates as SPELL_EVENT_LETTER_TOP_K on SPELL_INFERENCE_EVENT.
 *
 * This is the architectural inversion from the EARS POC: there, the
 * inference task subscribed to audio_capture and pulled 1-second windows
 * directly. Spell-Word needs discrete utterances aligned to letter
 * boundaries, so the segmenter owns the StreamBuffer and pushes work into
 * the inference task instead.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "spell_events.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the log-mel front-end, load the model from the "model" partition
 * (best-effort — boots cleanly if the partition is empty, awaiting OTA), and
 * spawn the inference task on Core 1.
 */
esp_err_t inference_init(void);

/**
 * Submit a 600 ms PCM window for classification.
 *
 * Ownership transfer: on success, the inference task takes ownership of
 * @p pcm and will free it (heap_caps_free) after processing. The pointer
 * MUST point to a buffer allocated with heap_caps_malloc — that is, it
 * must be safe to free via heap_caps_free.
 *
 * On failure (queue full or pre-init), the caller retains ownership and
 * must free the buffer itself.
 *
 * @param pcm        PSRAM-allocated buffer, length = n_samples × int16_t.
 * @param n_samples  Should equal SPELL_INFERENCE_WINDOW_SAMPLES; shorter
 *                   buffers are accepted and zero-padded by the feature
 *                   extractor's framer.
 * @return ESP_OK on enqueue, ESP_ERR_NO_MEM if queue full,
 *         ESP_ERR_INVALID_STATE if init not yet called.
 */
esp_err_t inference_submit_utterance(int16_t *pcm, size_t n_samples);

/**
 * Request a model hot-reload. The next time the inference task wakes (i.e.
 * the next utterance arrives), it will re-read the model partition and
 * rebuild the interpreter. Useful after an OTA model update.
 */
esp_err_t inference_reload_model(void);

// --- Stats / introspection (best-effort, lock-free reads) -------------
uint32_t inference_get_count(void);
uint32_t inference_get_gate_skip_count(void);
uint32_t inference_get_drop_count(void);   ///< queue-full drops
uint32_t inference_get_invalid_drop_count(void);  ///< dropped due to validation

#ifdef __cplusplus
}  // extern "C"
#endif

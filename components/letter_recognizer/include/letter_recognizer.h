/**
 * @file letter_recognizer.h
 * @brief Letter recognizer — audio-to-letter-evidence for one letter position.
 *
 * Driven by the segmenter: the segmenter submits utterance windows via
 * letter_recognizer_submit_utterance(); the recognizer task pulls them off a
 * queue, runs the energy gate, MFCC + TFLM forward pass, and posts the
 * top-K letter candidates as SPELL_EVENT_LETTER_RECOGNIZED on
 * SPELL_RECOGNIZER_EVENT.
 *
 * This is the architectural inversion from the EARS POC: there, the
 * inference task subscribed to audio_capture and pulled 1-second windows
 * directly. Spell-Word needs discrete utterances aligned to letter
 * boundaries, so the segmenter owns the StreamBuffer and pushes work into
 * the recognizer task instead.
 *
 * Phase 4 will deepen this module to own a PCM ring + the W-recovery cycle
 * (ADR-0005); for now it emits LETTER_RECOGNIZED with retract_count = 0
 * on every utterance.
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
 * Initialize the MFCC front-end, load the model from the "model" partition
 * (best-effort — boots cleanly if the partition is empty, awaiting OTA), and
 * spawn the recognizer task on Core 1.
 */
esp_err_t letter_recognizer_init(void);

/**
 * Submit a 600 ms PCM window for classification.
 *
 * Ownership transfer: on success, the recognizer task takes ownership of
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
esp_err_t letter_recognizer_submit_utterance(int16_t *pcm, size_t n_samples);

/**
 * Request a model hot-reload. The next time the recognizer task wakes (i.e.
 * the next utterance arrives), it will re-read the model partition and
 * rebuild the interpreter. Useful after an OTA model update.
 */
esp_err_t letter_recognizer_reload_model(void);

// --- Stats / introspection (best-effort, lock-free reads) -------------
uint32_t letter_recognizer_get_count(void);
uint32_t letter_recognizer_get_gate_skip_count(void);
uint32_t letter_recognizer_get_drop_count(void);   ///< queue-full drops
uint32_t letter_recognizer_get_invalid_drop_count(void);  ///< dropped due to validation

#ifdef __cplusplus
}  // extern "C"
#endif

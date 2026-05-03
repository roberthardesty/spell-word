/**
 * @file segmenter.h
 * @brief Energy-VAD letter segmenter — public API.
 *
 * Subscribes to audio_capture for a stream of int16 PCM frames, runs a
 * frame-rate energy VAD, emits per-letter utterance windows to the
 * letter_classifier's queue, and posts SPELL_EVENT_END_OF_WORD on
 * sustained silence.
 *
 * Bring-up state (Day 3): basic onset/offset hysteresis only. No pre-roll
 * ring buffer, no min-letter / max-letter enforcement, no inter-letter
 * gap merging — those land in Phase 3 of the MVP plan.
 *
 * Lifecycle (mirrors audio_capture):
 *
 *   1. segmenter_init()   — allocates buffers, subscribes to audio_capture.
 *      MUST run between audio_capture_init() and audio_capture_start().
 *   2. The segmenter task runs autonomously after audio_capture_start().
 *
 * UI state-machine integration: segmenter_set_active(true) tells the
 * segmenter to actually emit utterances; in inactive mode it consumes and
 * discards frames so the StreamBuffer never overruns. Default state is
 * active for bring-up (i.e., spelling "into the void" produces top-K
 * events on the serial console).
 */

#pragma once

#include <stdbool.h>
#include "esp_err.h"
#include "spell_events.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate buffers, subscribe to audio_capture, spawn the segmenter task.
 * Must be called between audio_capture_init() and audio_capture_start().
 */
esp_err_t segmenter_init(void);

/**
 * Enable or disable utterance emission.
 *
 * When inactive, the segmenter still drains the StreamBuffer (so capture
 * never overruns), but discards frames without VAD/emission.
 *
 * Default after init: ACTIVE — useful for bring-up. The UI state machine
 * (Phase 4) will toggle this to gate spelling sessions.
 */
void segmenter_set_active(bool active);

bool segmenter_is_active(void);

// --- Stats ------------------------------------------------------------
uint32_t segmenter_get_letter_count(void);  ///< total utterances emitted
uint32_t segmenter_get_word_count(void);    ///< total EOW events posted
uint32_t segmenter_get_alloc_fail_count(void); ///< PSRAM exhaustion drops

#ifdef __cplusplus
}  // extern "C"
#endif

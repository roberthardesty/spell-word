/**
 * @file playback.h
 * @brief Playback API for spell_ui — tones now, words in Phase 6.
 *
 * The state machine consumes this API as opaque action data; the IDF wrapper
 * for ui_core materialises the calls. `playback_play_word` is a Phase-6 stub
 * today: see PRD docs/prds/0001-phase-4-5-ux-and-decoder.md §"Out of Scope".
 */

#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Play one of the SPELL_TONE_* tones synthesised at boot. (Phase 5.)
esp_err_t playback_play_tone(uint32_t tone_id);

/// Cancel any in-flight playback cooperatively. Order is load-bearing — see
/// PRD 0001 §"Cancel ordering contract". (Phase 5.)
esp_err_t playback_cancel(void);

/// Play the resolved word out of the corpus partition.
///
/// Phase-6 stub: always returns ESP_ERR_NOT_SUPPORTED. The real implementation
/// (Opus decode + corpus partition reader + I2S TX feed) ships in Phase 6.
esp_err_t playback_play_word(uint32_t word_id);

#ifdef __cplusplus
}  // extern "C"
#endif

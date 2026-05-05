/**
 * @file decoder.h
 * @brief IDF wrapper around decoder_core — public init + stat getters.
 *
 * Loads the confusion matrix (subtype 0x81, label SPELL_MATRIX_PARTITION_LABEL)
 * and dictionary (subtype 0x82, label SPELL_DICT_PARTITION_LABEL) from custom
 * data partitions, validates them, then subscribes to:
 *
 *   SPELL_RECOGNIZER_EVENT::LETTER_RECOGNIZED  → dec_on_letter + early-commit
 *                                                eval (silence_floor = false)
 *   SPELL_SEGMENTER_EVENT::EARLY_COMMIT_WINDOW → early-commit eval
 *                                                (silence_floor = true)
 *   SPELL_SEGMENTER_EVENT::END_OF_WORD         → dec_resolve_full_eow
 *
 * Publishes:
 *
 *   SPELL_DECODER_EVENT::WORD_RESOLVED  payload spell_word_resolved_event_t
 *   SPELL_DECODER_EVENT::WORD_ABSTAIN   empty payload
 *
 * Init is fail-loud per the PRD acceptance criterion: an empty/erased matrix
 * or dictionary partition, an out-of-tolerance row sum, or a malformed
 * dictionary record returns a non-OK esp_err_t with a clear ESP_LOGE line.
 * Unlike letter_recognizer (which gracefully degrades when the model is
 * missing because it can be OTA'd later), the decoder cannot do anything
 * useful without these tables, so partial init is not a meaningful state.
 *
 * Handlers run on the default event loop task — no separate task is spawned.
 * Per-event work is bounded (worst case: scoring the dictionary), and the
 * handler latency budget is ms-level, well below the human-pace inter-letter
 * gap. Caller must have created the default event loop before init.
 */

#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load both partitions, initialise decoder_core state, register handlers.
 * Must be called after esp_event_loop_create_default(). Returns:
 *   ESP_OK                — ready
 *   ESP_ERR_NOT_FOUND     — required partition missing from partitions.csv
 *   ESP_ERR_INVALID_STATE — partition exists but is empty/erased (0xFF magic)
 *   ESP_ERR_INVALID_SIZE  — partition size doesn't match expected layout
 *   ESP_ERR_INVALID_CRC   — matrix row sum or dict record format invalid
 *   ESP_ERR_NO_MEM        — PSRAM allocation for tables/handlers failed
 *   ESP_FAIL              — esp_event_handler_register failed
 */
esp_err_t decoder_init(void);

/// How many WORD_RESOLVED events the decoder has published since init.
uint32_t decoder_get_resolved_count(void);
/// How many WORD_ABSTAIN events the decoder has published since init.
uint32_t decoder_get_abstain_count(void);
/// Subset of resolved_count that committed via the early-commit predicate
/// (the rest committed at full EOW).
uint32_t decoder_get_early_commit_count(void);

#ifdef __cplusplus
}  // extern "C"
#endif

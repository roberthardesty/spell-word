/**
 * @file decoder.c
 * @brief IDF wrapper around decoder_core — partition loaders + event glue.
 *
 * Two custom data partitions are owned by this wrapper:
 *
 *   confusion (subtype 0x81)  26x26 float32, row-major, exact size 2704 B.
 *                             Each row P(true | top1) sums to 1.0 ± 1e-3.
 *
 *   dictionary (subtype 0x82) Header (16 B):
 *                               char    magic[4]  = 'S','P','D','C'
 *                               uint32  version   = 1
 *                               uint32  n_words
 *                               uint32  reserved  (pad to 16, must be 0)
 *                             Records (n_words of them, packed):
 *                               uint32  word_id
 *                               uint8   length    in [1, SPELL_MAX_LETTERS_PER_WORD]
 *                               uint8   letters[length]   each in [0, 25]
 *                             Records are tightly packed; the loader walks
 *                             the byte stream and bails out on any malformed
 *                             field or if the byte count would exceed the
 *                             partition.
 *
 * On any validation failure (missing partition, erased flash, bad row sum,
 * malformed record) init returns non-OK with a clear ESP_LOGE line — per
 * PRD acceptance criterion (#22). The decoder can't do anything useful
 * without these tables, so partial init is not a meaningful state.
 *
 * Handlers run on the default event loop task. Each LETTER_RECOGNIZED event
 * triggers (a) dec_on_letter and (b) an early-commit eval at silence_floor=false;
 * EARLY_COMMIT_WINDOW re-evaluates with silence_floor=true; END_OF_WORD calls
 * dec_resolve_full_eow. Any RESOLVED outcome publishes WORD_RESOLVED and
 * resets the in-flight buffer; ABSTAIN at full EOW publishes WORD_ABSTAIN.
 */

#include "decoder.h"
#include "decoder_core.h"
#include "spell_config.h"
#include "spell_events.h"

#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"

static const char *TAG = "decoder";

// Define the event base declared in spell_events.h. Producing component
// owns the DEFINE_BASE per the repo working norms.
ESP_EVENT_DEFINE_BASE(SPELL_DECODER_EVENT);

// Tolerance for confusion-matrix row sums. Strict 1e-3 per the spec — the
// looser 0.5..1.5 sanity check inside dec_init() exists for host tests with
// coarse synthesized matrices and is intentionally non-load-bearing here.
#define DECODER_ROW_SUM_TOL  1.0e-3f

// Magic bytes at the head of the dictionary partition.
static const uint8_t DICT_MAGIC[4] = { 'S', 'P', 'D', 'C' };
#define DICT_HEADER_BYTES   16
#define DICT_FORMAT_VERSION 1u

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

// Caller-owned tables for decoder_core. Allocated in PSRAM at init time;
// lifetimes equal the firmware run.
static dec_confusion_t *s_confusion = NULL;
static dec_word_t      *s_words     = NULL;
static dec_dict_t       s_dict;            // {.words, .n_words}, points at s_words
static dec_state_t      s_state;           // in-flight buffer + dict/confusion ptrs
static bool             s_ready    = false;

static atomic_uint_least32_t s_resolved_count      = ATOMIC_VAR_INIT(0);
static atomic_uint_least32_t s_abstain_count       = ATOMIC_VAR_INIT(0);
static atomic_uint_least32_t s_early_commit_count  = ATOMIC_VAR_INIT(0);

// ---------------------------------------------------------------------------
// Event publishing
// ---------------------------------------------------------------------------

static void publish_resolved(uint32_t word_id, float score_margin,
                             spell_commit_reason_t reason)
{
    spell_word_resolved_event_t evt = {
        .word_id       = word_id,
        .commit_reason = reason,
        .score_margin  = score_margin,
    };
    atomic_fetch_add(&s_resolved_count, 1);
    if (reason == SPELL_COMMIT_REASON_EARLY) {
        atomic_fetch_add(&s_early_commit_count, 1);
    }
    ESP_LOGI(TAG, "WORD_RESOLVED word_id=%" PRIu32 " margin=%.3f reason=%s",
             word_id, (double)score_margin,
             reason == SPELL_COMMIT_REASON_EARLY ? "early" : "full_eow");
    esp_event_post(SPELL_DECODER_EVENT, SPELL_EVENT_WORD_RESOLVED,
                   &evt, sizeof(evt), 0);
}

static void publish_abstain(int n_positions, bool overflowed)
{
    atomic_fetch_add(&s_abstain_count, 1);
    ESP_LOGI(TAG, "WORD_ABSTAIN n_positions=%d overflowed=%d",
             n_positions, overflowed ? 1 : 0);
    esp_event_post(SPELL_DECODER_EVENT, SPELL_EVENT_WORD_ABSTAIN, NULL, 0, 0);
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

static void on_letter_recognized(void *arg, esp_event_base_t base,
                                 int32_t event_id, void *event_data)
{
    (void)arg; (void)base; (void)event_id;
    if (!s_ready) return;
    const spell_letter_recognized_event_t *evt =
        (const spell_letter_recognized_event_t *)event_data;

    // spell_letter_top_k_t and dec_top_k_t have identical layouts (see the
    // comment in decoder_core.h). Memcpy is safe; field-copy keeps the type
    // contract honest if either ever drifts.
    dec_top_k_t top_k;
    for (int i = 0; i < SPELL_LETTER_TOP_K; i++) {
        top_k.candidates[i].letter_index = evt->top_k.candidates[i].letter_index;
        top_k.candidates[i].probability  = evt->top_k.candidates[i].probability;
    }
    top_k.invoke_ms = evt->top_k.invoke_ms;

    dec_on_letter(&s_state, &top_k, evt->retract_count);

    dec_early_commit_eval_t e = dec_try_early_commit(&s_state,
                                                     /*silence_floor=*/false);
    if (e.kind == DEC_EARLY_COMMIT_RESOLVED) {
        publish_resolved(e.word_id, e.margin_runnerup,
                         SPELL_COMMIT_REASON_EARLY);
        dec_reset(&s_state);
    } else {
        ESP_LOGD(TAG, "letter pos=%d hold[ru=%d(%.2f) lo=%d(%.2f) sf=%d]",
                 s_state.n_positions,
                 e.margin_runnerup_ok, (double)e.margin_runnerup,
                 e.margin_longer_ok,   (double)e.margin_longer,
                 e.silence_floor_ok);
    }
}

static void on_early_commit_window(void *arg, esp_event_base_t base,
                                   int32_t event_id, void *event_data)
{
    (void)arg; (void)base; (void)event_id; (void)event_data;
    if (!s_ready) return;

    dec_early_commit_eval_t e = dec_try_early_commit(&s_state,
                                                     /*silence_floor=*/true);
    if (e.kind == DEC_EARLY_COMMIT_RESOLVED) {
        publish_resolved(e.word_id, e.margin_runnerup,
                         SPELL_COMMIT_REASON_EARLY);
        dec_reset(&s_state);
    } else {
        ESP_LOGD(TAG, "early-window pos=%d hold[ru=%d(%.2f) lo=%d(%.2f)]",
                 s_state.n_positions,
                 e.margin_runnerup_ok, (double)e.margin_runnerup,
                 e.margin_longer_ok,   (double)e.margin_longer);
    }
}

static void on_end_of_word(void *arg, esp_event_base_t base,
                           int32_t event_id, void *event_data)
{
    (void)arg; (void)base; (void)event_id; (void)event_data;
    if (!s_ready) return;

    dec_resolution_t r = dec_resolve_full_eow(&s_state);
    if (r.kind == DEC_OUTCOME_RESOLVED) {
        publish_resolved(r.word_id, r.score_margin,
                         SPELL_COMMIT_REASON_FULL_EOW);
    } else {
        publish_abstain(r.n_positions, r.overflowed);
    }
    dec_reset(&s_state);
}

// ---------------------------------------------------------------------------
// Partition helpers
// ---------------------------------------------------------------------------

/// Look up a partition by subtype + label and verify it isn't erased.
/// Sets *out_part on success; returns ESP_OK / ESP_ERR_NOT_FOUND /
/// ESP_ERR_INVALID_STATE.
static esp_err_t find_and_check(uint8_t subtype, const char *label,
                                const esp_partition_t **out_part)
{
    *out_part = NULL;
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        (esp_partition_subtype_t)subtype,
        label);
    if (!part) {
        ESP_LOGE(TAG, "partition '%s' (subtype 0x%02x) not found in flash",
                 label, subtype);
        return ESP_ERR_NOT_FOUND;
    }

    uint8_t magic[4] = { 0 };
    esp_err_t err = esp_partition_read(part, 0, magic, sizeof(magic));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "partition '%s' read header failed: %s",
                 label, esp_err_to_name(err));
        return err;
    }
    if (magic[0] == 0xFF && magic[1] == 0xFF &&
        magic[2] == 0xFF && magic[3] == 0xFF) {
        ESP_LOGE(TAG, "partition '%s' is erased (0xFF magic) — "
                      "flash a populated image before booting",
                 label);
        return ESP_ERR_INVALID_STATE;
    }

    *out_part = part;
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Confusion-matrix loader
// ---------------------------------------------------------------------------

static esp_err_t load_confusion(void)
{
    const esp_partition_t *part = NULL;
    esp_err_t err = find_and_check(0x81, SPELL_MATRIX_PARTITION_LABEL, &part);
    if (err != ESP_OK) return err;

    const size_t need = sizeof(dec_confusion_t);  // 26*26*4 = 2704 B
    if (part->size < need) {
        ESP_LOGE(TAG, "matrix partition too small: have %" PRIu32 " B, need %u B",
                 part->size, (unsigned)need);
        return ESP_ERR_INVALID_SIZE;
    }

    s_confusion = heap_caps_malloc(sizeof(dec_confusion_t),
                                   MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_confusion) {
        ESP_LOGE(TAG, "PSRAM alloc for confusion matrix failed (%u B)",
                 (unsigned)sizeof(dec_confusion_t));
        return ESP_ERR_NO_MEM;
    }

    err = esp_partition_read(part, 0, s_confusion, need);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "matrix partition read failed: %s", esp_err_to_name(err));
        return err;
    }

    // Strict row-sum validation: each of 26 rows must sum to 1.0 ± 1e-3.
    for (int r = 0; r < DEC_ALPHABET_SIZE; r++) {
        float sum = 0.0f;
        for (int c = 0; c < DEC_ALPHABET_SIZE; c++) {
            float v = s_confusion->m[r * DEC_ALPHABET_SIZE + c];
            if (!isfinite(v) || v < 0.0f) {
                ESP_LOGE(TAG, "matrix row %d col %d has invalid value %f",
                         r, c, (double)v);
                return ESP_ERR_INVALID_CRC;
            }
            sum += v;
        }
        if (fabsf(sum - 1.0f) > DECODER_ROW_SUM_TOL) {
            ESP_LOGE(TAG, "matrix row %d sum=%.6f outside 1.0 ± %.0e",
                     r, (double)sum, (double)DECODER_ROW_SUM_TOL);
            return ESP_ERR_INVALID_CRC;
        }
    }

    ESP_LOGI(TAG, "confusion matrix loaded (%u B, all rows sum to 1.0 ± %.0e)",
             (unsigned)need, (double)DECODER_ROW_SUM_TOL);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Dictionary loader
// ---------------------------------------------------------------------------

/// Read the entire dictionary partition into a temporary RAM buffer, parse
/// it into s_words. The temp buffer is freed before returning. Validation
/// failures all return ESP_ERR_INVALID_CRC with a precise log line.
static esp_err_t load_dictionary(void)
{
    const esp_partition_t *part = NULL;
    esp_err_t err = find_and_check(0x82, SPELL_DICT_PARTITION_LABEL, &part);
    if (err != ESP_OK) return err;

    if (part->size < DICT_HEADER_BYTES) {
        ESP_LOGE(TAG, "dict partition too small for header: %" PRIu32 " B",
                 part->size);
        return ESP_ERR_INVALID_SIZE;
    }

    uint8_t *raw = heap_caps_malloc(part->size,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!raw) {
        ESP_LOGE(TAG, "PSRAM alloc for dict raw buffer failed (%" PRIu32 " B)",
                 part->size);
        return ESP_ERR_NO_MEM;
    }
    err = esp_partition_read(part, 0, raw, part->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "dict partition read failed: %s", esp_err_to_name(err));
        heap_caps_free(raw);
        return err;
    }

    // Header.
    if (memcmp(raw, DICT_MAGIC, 4) != 0) {
        ESP_LOGE(TAG, "dict magic mismatch: have %02x %02x %02x %02x, "
                      "expected SPDC", raw[0], raw[1], raw[2], raw[3]);
        heap_caps_free(raw);
        return ESP_ERR_INVALID_CRC;
    }
    uint32_t version = 0, n_words = 0, reserved = 0;
    memcpy(&version,  raw + 4,  4);
    memcpy(&n_words,  raw + 8,  4);
    memcpy(&reserved, raw + 12, 4);
    if (version != DICT_FORMAT_VERSION) {
        ESP_LOGE(TAG, "dict version %" PRIu32 " not supported (expected %u)",
                 version, (unsigned)DICT_FORMAT_VERSION);
        heap_caps_free(raw);
        return ESP_ERR_INVALID_CRC;
    }
    if (reserved != 0) {
        ESP_LOGE(TAG, "dict reserved field non-zero: %" PRIu32, reserved);
        heap_caps_free(raw);
        return ESP_ERR_INVALID_CRC;
    }
    if (n_words == 0) {
        ESP_LOGE(TAG, "dict has zero words");
        heap_caps_free(raw);
        return ESP_ERR_INVALID_CRC;
    }

    // Allocate the parsed array.
    s_words = heap_caps_calloc(n_words, sizeof(dec_word_t),
                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_words) {
        ESP_LOGE(TAG, "PSRAM alloc for %" PRIu32 " parsed words failed",
                 n_words);
        heap_caps_free(raw);
        return ESP_ERR_NO_MEM;
    }

    // Walk the records.
    size_t off = DICT_HEADER_BYTES;
    for (uint32_t i = 0; i < n_words; i++) {
        // word_id (4 B) + length (1 B) header.
        if (off + 5 > part->size) {
            ESP_LOGE(TAG, "dict record %" PRIu32 ": header truncated at "
                          "offset %u (partition size %" PRIu32 ")",
                     i, (unsigned)off, part->size);
            heap_caps_free(s_words); s_words = NULL;
            heap_caps_free(raw);
            return ESP_ERR_INVALID_CRC;
        }
        uint32_t word_id = 0;
        memcpy(&word_id, raw + off, 4);
        uint8_t length = raw[off + 4];
        off += 5;

        if (length == 0 || length > SPELL_MAX_LETTERS_PER_WORD) {
            ESP_LOGE(TAG, "dict record %" PRIu32 " (id=%" PRIu32 "): "
                          "length=%u out of [1,%d]",
                     i, word_id, (unsigned)length, SPELL_MAX_LETTERS_PER_WORD);
            heap_caps_free(s_words); s_words = NULL;
            heap_caps_free(raw);
            return ESP_ERR_INVALID_CRC;
        }
        if (off + length > part->size) {
            ESP_LOGE(TAG, "dict record %" PRIu32 " letters truncated at "
                          "offset %u (need %u, partition size %" PRIu32 ")",
                     i, (unsigned)off, (unsigned)length, part->size);
            heap_caps_free(s_words); s_words = NULL;
            heap_caps_free(raw);
            return ESP_ERR_INVALID_CRC;
        }

        s_words[i].word_id = word_id;
        s_words[i].length  = length;
        for (uint8_t j = 0; j < length; j++) {
            uint8_t letter = raw[off + j];
            if (letter > 25) {
                ESP_LOGE(TAG, "dict record %" PRIu32 " (id=%" PRIu32 "): "
                              "letter %u at slot %u out of [0,25]",
                         i, word_id, (unsigned)letter, (unsigned)j);
                heap_caps_free(s_words); s_words = NULL;
                heap_caps_free(raw);
                return ESP_ERR_INVALID_CRC;
            }
            s_words[i].letters[j] = letter;
        }
        off += length;
    }

    s_dict.words   = s_words;
    s_dict.n_words = (int)n_words;

    heap_caps_free(raw);
    ESP_LOGI(TAG, "dictionary loaded (%" PRIu32 " words, %u B consumed)",
             n_words, (unsigned)off);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Public init
// ---------------------------------------------------------------------------

esp_err_t decoder_init(void)
{
    if (s_ready) {
        ESP_LOGW(TAG, "decoder_init() called twice — ignoring");
        return ESP_OK;
    }

    esp_err_t err = load_confusion();
    if (err != ESP_OK) return err;

    err = load_dictionary();
    if (err != ESP_OK) return err;

    dec_config_t cfg;
    dec_config_default(&cfg);
    if (dec_init(&s_state, &s_dict, s_confusion, &cfg) != 0) {
        ESP_LOGE(TAG, "dec_init() rejected the loaded tables");
        return ESP_ERR_INVALID_STATE;
    }

    err = esp_event_handler_register(
        SPELL_RECOGNIZER_EVENT, SPELL_EVENT_LETTER_RECOGNIZED,
        on_letter_recognized, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "register LETTER_RECOGNIZED handler failed: %s",
                 esp_err_to_name(err));
        return ESP_FAIL;
    }
    err = esp_event_handler_register(
        SPELL_SEGMENTER_EVENT, SPELL_EVENT_EARLY_COMMIT_WINDOW,
        on_early_commit_window, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "register EARLY_COMMIT_WINDOW handler failed: %s",
                 esp_err_to_name(err));
        return ESP_FAIL;
    }
    err = esp_event_handler_register(
        SPELL_SEGMENTER_EVENT, SPELL_EVENT_END_OF_WORD,
        on_end_of_word, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "register END_OF_WORD handler failed: %s",
                 esp_err_to_name(err));
        return ESP_FAIL;
    }

    s_ready = true;
    ESP_LOGI(TAG, "decoder ready: %d words, alpha=%.2f margin=%.2f "
                  "early[ru=%.2f, longer=%.2f, silence_ms=%d]",
             s_dict.n_words,
             (double)cfg.alpha, (double)cfg.decision_margin,
             (double)cfg.early_margin_runnerup,
             (double)cfg.early_margin_longer,
             SPELL_EARLY_MIN_SILENCE_MS);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Stat getters
// ---------------------------------------------------------------------------

uint32_t decoder_get_resolved_count(void) {
    return (uint32_t)atomic_load(&s_resolved_count);
}
uint32_t decoder_get_abstain_count(void) {
    return (uint32_t)atomic_load(&s_abstain_count);
}
uint32_t decoder_get_early_commit_count(void) {
    return (uint32_t)atomic_load(&s_early_commit_count);
}

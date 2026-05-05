/**
 * @file w_detector_core.c
 * @brief Pure-C trigger predicate for the W-recovery cycle.
 *
 * Companion to w_detector_core.h; see that header for the algorithm
 * summary and ADR-0005 / PRD references. No esp_event, no logging, no
 * allocation — the recognizer (Vikunja #30) owns event subscription, the
 * PCM ring (#27), and the merged-rerun call.
 */

#include "w_detector_core.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "spell_config.h"

void wdet_config_default(wdet_config_t *cfg)
{
    cfg->u_threshold        = SPELL_W_DETECT_U_THRESHOLD;
    cfg->low_conf_threshold = SPELL_W_DETECT_LOW_CONF_THRESHOLD;
}

int wdet_init(wdet_state_t *st, const wdet_config_t *cfg)
{
    if (!st) return -1;

    wdet_config_t local;
    if (cfg) {
        local = *cfg;
    } else {
        wdet_config_default(&local);
    }
    if (local.u_threshold        < 0.0f || local.u_threshold        > 1.0f) return -1;
    if (local.low_conf_threshold < 0.0f || local.low_conf_threshold > 1.0f) return -1;

    memset(st, 0, sizeof(*st));
    st->cfg = local;
    return 0;
}

void wdet_reset(wdet_state_t *st)
{
    if (!st) return;
    st->prev_top1_prob[0] = 0.0f;
    st->prev_top1_prob[1] = 0.0f;
    st->prior_count       = 0;
}

wdet_eval_t wdet_on_event(wdet_state_t *st, uint8_t top1_index, float top1_prob)
{
    wdet_eval_t e;
    memset(&e, 0, sizeof(e));
    e.current_top1      = top1_index;
    e.current_top1_prob = top1_prob;

    if (!st) return e;

    e.sufficient_history = (st->prior_count >= WDET_HISTORY_DEPTH);

    e.current_is_u      = (top1_index == WDET_LETTER_U);
    e.current_high_conf = (top1_prob  >= st->cfg.u_threshold);

    // Per-condition flags evaluate against whatever is currently in the
    // window, even when prior_count < 2 — the sufficient_history flag is
    // what gates the trigger. This keeps the diagnostic output well-defined
    // (slots not yet populated read 0.0f, which is < any sane threshold,
    // so prevN_low_conf reads true vacuously — useful for logging).
    e.prev1_low_conf = (st->prev_top1_prob[0] < st->cfg.low_conf_threshold);
    e.prev2_low_conf = (st->prev_top1_prob[1] < st->cfg.low_conf_threshold);

    e.trigger = e.sufficient_history
             && e.current_is_u
             && e.current_high_conf
             && e.prev1_low_conf
             && e.prev2_low_conf;

    // Shift the window AFTER evaluation: prev[1] becomes the new prev[2],
    // and the new event becomes the new prev[1].
    st->prev_top1_prob[1] = st->prev_top1_prob[0];
    st->prev_top1_prob[0] = top1_prob;
    if (st->prior_count < WDET_HISTORY_DEPTH) st->prior_count++;

    return e;
}

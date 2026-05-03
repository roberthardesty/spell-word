/**
 * @file segmenter_core.c
 * @brief Pure-C VAD state machine for letter segmentation.
 *
 * No ESP-IDF, no FreeRTOS. Compiles on a host with -std=c99 plus the math
 * library. The full algorithm is documented in segmenter_core.h's banner.
 *
 * One subtle ordering rule: per-frame, we run the state machine FIRST and
 * push the current frame into the pre-roll ring LAST. This way a transition
 * IDLE→IN_LETTER on this frame snapshots a pre-roll that contains only
 * pre-onset context (the previous frames), and accum_append then writes
 * the current frame as the first letter frame — without duplication. The
 * current frame becomes pre-roll context for the *next* onset, naturally.
 */

#include "segmenter_core.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float seg_frame_rms_dbfs(const int16_t *frame, int n)
{
    if (n <= 0) return -100.0f;
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double s = (double)frame[i] / 32768.0;
        sum_sq += s * s;
    }
    if (sum_sq < 1e-20) return -100.0f;
    return 10.0f * log10f((float)(sum_sq / n));
    // 10*log10(rms²) ≡ 20*log10(rms). Matches the training pipeline's gate.
}

/// Push @p n samples into the pre-roll ring buffer (continuous overwrite).
static void preroll_push(seg_state_t *s, const int16_t *frame, int n)
{
    int cap = s->cfg.preroll_samples;
    if (cap <= 0 || n <= 0) return;

    int wp = s->preroll_write_pos;
    for (int i = 0; i < n; i++) {
        s->cfg.preroll[wp] = frame[i];
        wp++;
        if (wp >= cap) wp = 0;
    }
    s->preroll_write_pos = wp;
    s->preroll_filled += n;
    if (s->preroll_filled > cap) s->preroll_filled = cap;
}

/// Copy the pre-roll content (oldest-to-newest order) into accum[0..filled).
/// Returns the number of samples copied.
static int preroll_snapshot_into_accum(seg_state_t *s)
{
    int cap    = s->cfg.preroll_samples;
    int filled = s->preroll_filled;
    if (filled <= 0 || cap <= 0 || s->cfg.accum_capacity <= 0) return 0;

    int copy = filled;
    if (copy > s->cfg.accum_capacity) copy = s->cfg.accum_capacity;

    if (filled < cap) {
        // Ring not yet wrapped — oldest is at index 0.
        memcpy(s->cfg.accum, s->cfg.preroll, copy * sizeof(int16_t));
    } else {
        // Ring wrapped — oldest is at preroll_write_pos (the next slot to
        // be overwritten); samples wrap around the end of the buffer.
        int wp = s->preroll_write_pos;
        int first_chunk = cap - wp;
        if (first_chunk > copy) first_chunk = copy;
        memcpy(s->cfg.accum, &s->cfg.preroll[wp],
               first_chunk * sizeof(int16_t));
        if (copy > first_chunk) {
            memcpy(&s->cfg.accum[first_chunk], s->cfg.preroll,
                   (copy - first_chunk) * sizeof(int16_t));
        }
    }

    s->accum_len = copy;
    return copy;
}

/// Append @p n samples to accum, truncating if it would overflow.
static int accum_append(seg_state_t *s, const int16_t *frame, int n)
{
    int space = s->cfg.accum_capacity - s->accum_len;
    if (space <= 0 || n <= 0) return 0;
    int copy = n < space ? n : space;
    memcpy(&s->cfg.accum[s->accum_len], frame, copy * sizeof(int16_t));
    s->accum_len += copy;
    return copy;
}

static seg_event_t make_letter_event(seg_state_t *s, seg_event_kind_t kind)
{
    seg_event_t evt;
    memset(&evt, 0, sizeof(evt));
    evt.kind        = kind;
    evt.pcm         = s->cfg.accum;
    evt.n_samples   = s->accum_len;
    evt.duration_ms = (s->letter_frames * s->cfg.frame_samples * 1000)
                       / s->cfg.sample_rate;
    evt.peak_dbfs   = s->letter_peak_dbfs;
    return evt;
}

static void enter_letter_state(seg_state_t *s, const int16_t *frame, int n,
                               float ema)
{
    // Snapshot pre-onset context, then append the current (onset) frame.
    preroll_snapshot_into_accum(s);
    s->letter_onset_in_accum = s->accum_len;
    accum_append(s, frame, n);

    s->state                = SEG_STATE_IN_LETTER;
    s->letter_frames        = 1;
    s->silent_frames_offset = 0;
    s->silent_frames_eow    = 0;
    s->letter_peak_dbfs     = ema;
}

static void clear_letter_state(seg_state_t *s)
{
    s->accum_len             = 0;
    s->letter_onset_in_accum = 0;
    s->letter_frames         = 0;
    s->letter_peak_dbfs      = -100.0f;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

int seg_init(seg_state_t *state, const seg_config_t *cfg)
{
    if (!state || !cfg) return -1;
    if (cfg->frame_samples <= 0)        return -1;
    if (cfg->sample_rate   <= 0)        return -1;
    if (cfg->ema_alpha_x100 < 1 || cfg->ema_alpha_x100 > 100) return -1;
    if (cfg->off_frames        <= 0)    return -1;
    if (cfg->min_letter_frames < 0)     return -1;
    if (cfg->max_letter_frames <= cfg->min_letter_frames) return -1;
    if (cfg->eow_frames        <= 0)    return -1;
    if (!cfg->preroll || cfg->preroll_samples <= 0)   return -1;
    if (!cfg->accum   || cfg->accum_capacity  < cfg->frame_samples) return -1;
    if (cfg->on_dbfs <= cfg->off_dbfs)  return -1;  // hysteresis must have margin

    memset(state, 0, sizeof(*state));
    state->cfg              = *cfg;
    state->state            = SEG_STATE_IDLE;
    state->ema_dbfs         = -100.0f;
    state->letter_peak_dbfs = -100.0f;
    return 0;
}

void seg_reset(seg_state_t *state)
{
    if (!state) return;
    state->state                 = SEG_STATE_IDLE;
    state->word_in_flight        = false;
    state->silent_frames_offset  = 0;
    state->silent_frames_eow     = 0;
    state->letter_frames         = 0;
    state->ema_dbfs              = -100.0f;
    state->accum_len             = 0;
    state->letter_onset_in_accum = 0;
    state->letter_peak_dbfs      = -100.0f;
    // Pre-roll is intentionally preserved across resets.
}

void seg_reset_stats(seg_state_t *state)
{
    if (!state) return;
    state->total_letters_emitted        = 0;
    state->total_letters_rejected_short = 0;
    state->total_words_completed        = 0;
    state->total_force_splits           = 0;
}

seg_event_t seg_process_frame(seg_state_t *s, const int16_t *frame, int n)
{
    seg_event_t evt;
    memset(&evt, 0, sizeof(evt));
    evt.kind = SEG_EVT_NONE;

    if (!s || !frame || n != s->cfg.frame_samples) return evt;

    // ── 1. Compute raw + smoothed RMS ────────────────────────────────────
    float raw = seg_frame_rms_dbfs(frame, n);
    float a   = s->cfg.ema_alpha_x100 / 100.0f;
    if (s->ema_dbfs <= -99.0f) {
        s->ema_dbfs = raw;        // seed EMA on first frame to avoid warm-up
    } else {
        s->ema_dbfs = a * raw + (1.0f - a) * s->ema_dbfs;
    }
    float ema = s->ema_dbfs;

    // ── 2. State machine ─────────────────────────────────────────────────
    switch (s->state) {

    case SEG_STATE_IDLE: {
        if (ema >= s->cfg.on_dbfs) {
            // Onset.
            enter_letter_state(s, frame, n, ema);
        } else if (s->word_in_flight) {
            if (++s->silent_frames_eow >= s->cfg.eow_frames) {
                evt.kind            = SEG_EVT_END_OF_WORD;
                evt.letters_in_word = s->total_letters_emitted;
                s->total_words_completed++;
                s->word_in_flight   = false;
                s->silent_frames_eow = 0;
            }
        }
        break;
    }

    case SEG_STATE_IN_LETTER: {
        accum_append(s, frame, n);
        s->letter_frames++;
        if (ema > s->letter_peak_dbfs) s->letter_peak_dbfs = ema;

        if (s->letter_frames >= s->cfg.max_letter_frames) {
            // Force-split. Emit what we have, stay in IN_LETTER for
            // continued capture without re-arming pre-roll.
            evt = make_letter_event(s, SEG_EVT_FORCE_SPLIT);
            s->total_force_splits++;
            s->total_letters_emitted++;
            s->word_in_flight        = true;
            s->silent_frames_eow     = 0;
            clear_letter_state(s);
            // Stay above onset → next frame keeps accumulating.
            // We mark this as letter_frames=0 so min-letter logic on the
            // continuation segment starts fresh.
            break;
        }

        if (ema < s->cfg.off_dbfs) {
            s->state                = SEG_STATE_POST_LETTER;
            s->silent_frames_offset = 1;
        }
        break;
    }

    case SEG_STATE_POST_LETTER: {
        // Keep accumulating so the trailing tail of the letter makes it
        // into the emitted window.
        accum_append(s, frame, n);
        s->letter_frames++;
        if (ema > s->letter_peak_dbfs) s->letter_peak_dbfs = ema;

        if (ema >= s->cfg.on_dbfs) {
            // Energy returned within the silence window — continuation.
            s->state                = SEG_STATE_IN_LETTER;
            s->silent_frames_offset = 0;
        } else if (++s->silent_frames_offset >= s->cfg.off_frames) {
            // Confirmed offset.
            if (s->letter_frames < s->cfg.min_letter_frames) {
                evt.kind        = SEG_EVT_LETTER_REJECTED_SHORT;
                evt.duration_ms = (s->letter_frames * s->cfg.frame_samples * 1000)
                                  / s->cfg.sample_rate;
                evt.peak_dbfs   = s->letter_peak_dbfs;
                s->total_letters_rejected_short++;
                // Don't set word_in_flight — a rejected blip shouldn't
                // sustain a stale word.
            } else {
                evt = make_letter_event(s, SEG_EVT_LETTER_EMITTED);
                s->total_letters_emitted++;
                s->word_in_flight = true;
            }

            s->state                = SEG_STATE_IDLE;
            s->silent_frames_offset = 0;
            s->silent_frames_eow    = 0;
            clear_letter_state(s);
        }
        break;
    }
    }

    // ── 3. Push current frame into pre-roll AFTER the state machine ─────
    // so that the pre-roll snapshot taken on this frame's onset transition
    // contains pre-onset context only (no duplication of the onset frame).
    preroll_push(s, frame, n);

    return evt;
}

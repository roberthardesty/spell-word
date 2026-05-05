/**
 * @file decoder_core.c
 * @brief Pure-C decoder — posterior assembly + bounded-edit dictionary scoring.
 *
 * Companion to decoder_core.h; see that header for the algorithm summary.
 *
 * No esp_event, no logging, no allocation. The caller (IDF wrapper or host
 * test harness) owns event subscription and the dict / confusion lifetime.
 */

#include "decoder_core.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// Floor for log(p) so a zero-probability letter doesn't poison the score
// with -INF (which makes the runner-up margin meaningless). 1e-10 ≈ -23.0
// in nats — well past any realistic margin threshold.
#define DEC_LOG_FLOOR 1e-10f

// =============================================================================
// Defaults + init
// =============================================================================

void dec_config_default(dec_config_t *cfg)
{
    cfg->alpha                  = SPELL_DECODER_ALPHA;
    cfg->decision_margin        = SPELL_DECISION_MARGIN;
    cfg->ins_penalty            = SPELL_EDIT_INS_PENALTY;
    cfg->del_penalty            = SPELL_EDIT_DEL_PENALTY;
    cfg->early_margin_runnerup  = SPELL_EARLY_MARGIN_RUNNERUP;
    cfg->early_margin_longer    = SPELL_EARLY_MARGIN_LONGER;
}

int dec_init(dec_state_t *st,
             const dec_dict_t *dict,
             const dec_confusion_t *confusion,
             const dec_config_t *cfg)
{
    if (!st || !dict || !confusion || !cfg) return -1;
    if (dict->n_words < 0) return -1;
    if (dict->n_words > 0 && !dict->words) return -1;

    // Coarse sanity check on confusion-matrix rows (loose so host tests can
    // use simplified matrices). Strict 1.0 ± 1e-3 validation lives in the
    // IDF partition loader.
    for (int j = 0; j < DEC_ALPHABET_SIZE; j++) {
        float row_sum = 0.0f;
        for (int i = 0; i < DEC_ALPHABET_SIZE; i++) {
            float v = confusion->m[j * DEC_ALPHABET_SIZE + i];
            if (v < 0.0f) return -1;
            row_sum += v;
        }
        if (row_sum < 0.5f || row_sum > 1.5f) return -1;
    }

    memset(st, 0, sizeof(*st));
    st->dict      = dict;
    st->confusion = confusion;
    st->cfg       = *cfg;
    return 0;
}

void dec_reset(dec_state_t *st)
{
    st->n_positions = 0;
    st->overflowed  = false;
    // Don't bother zeroing the position array — n_positions is the live bound.
}

// =============================================================================
// Unified letter-event handler
// =============================================================================

void dec_on_letter(dec_state_t *st,
                   const dec_top_k_t *top_k,
                   uint8_t retract_count)
{
    int retract = retract_count;
    if (retract > st->n_positions) retract = st->n_positions;
    st->n_positions -= retract;

    if (st->n_positions < SPELL_MAX_LETTERS_PER_WORD) {
        st->positions[st->n_positions++] = *top_k;
    } else {
        st->overflowed = true;
    }
}

// =============================================================================
// Posterior assembly  (PRD §"Posterior assembly")
// =============================================================================
//
//   net_prob[c] = top-K probability if c ∈ top_k
//                 else (1 - Σ top-K probabilities) / (26 - |top_k_unique|)
//   posterior[c] = (1 - α) · net_prob[c] + α · M[top1, c]
//
// PRD nominally calls out 21 outside-K letters (K=5, 26-5). We compute the
// outside count from the *unique* top-K letter indices to be defensive
// against duplicates (which shouldn't happen but would otherwise corrupt
// the uniform spread).

static void assemble_posterior(const dec_top_k_t *top_k,
                               const dec_confusion_t *confusion,
                               float alpha,
                               float posterior_out[DEC_ALPHABET_SIZE])
{
    float net_prob[DEC_ALPHABET_SIZE];
    bool  in_top_k[DEC_ALPHABET_SIZE] = { false };
    float top_k_mass = 0.0f;

    for (int k = 0; k < SPELL_LETTER_TOP_K; k++) {
        uint8_t idx = top_k->candidates[k].letter_index;
        if (idx >= DEC_ALPHABET_SIZE) continue;
        if (in_top_k[idx]) continue;          // dedup defensively
        in_top_k[idx] = true;
        net_prob[idx] = top_k->candidates[k].probability;
        top_k_mass   += top_k->candidates[k].probability;
    }

    int n_outside = 0;
    for (int c = 0; c < DEC_ALPHABET_SIZE; c++) if (!in_top_k[c]) n_outside++;

    float remaining = 1.0f - top_k_mass;
    if (remaining < 0.0f) remaining = 0.0f;
    float per_outside = (n_outside > 0) ? (remaining / (float)n_outside) : 0.0f;

    for (int c = 0; c < DEC_ALPHABET_SIZE; c++) {
        if (!in_top_k[c]) net_prob[c] = per_outside;
    }

    uint8_t top1 = top_k->candidates[0].letter_index;
    if (top1 >= DEC_ALPHABET_SIZE) top1 = 0;   // defensive
    const float *conf_row = &confusion->m[top1 * DEC_ALPHABET_SIZE];

    for (int c = 0; c < DEC_ALPHABET_SIZE; c++) {
        posterior_out[c] = (1.0f - alpha) * net_prob[c]
                         + alpha          * conf_row[c];
    }
}

// =============================================================================
// Bounded-edit alignment scoring (per-word)
// =============================================================================
//
// For utterance count N and dictionary-word length M:
//   M == N         → direct alignment; substitution implicit.
//   M == N - 1     → one utterance was spurious (cough / breath); pick the
//                    skip position that maximises score, add ins_penalty.
//   M == N + 1     → one letter was missed (swallowed pause); pick the
//                    word position to skip, add del_penalty.
//   |M - N| > 1    → out of bounds; word excluded from candidates.

static inline float safe_log(float p)
{
    return logf(p > DEC_LOG_FLOOR ? p : DEC_LOG_FLOOR);
}

// Score @p word against the @p n_positions utterance posteriors when the
// word has exactly (n_positions + dels) letters — i.e. the alignment skips
// `dels` word positions, charging del_penalty per skip. Supports dels=0..2.
// Returns -INFINITY if the word's length doesn't match the requested skip
// count or the word is malformed.
static float score_with_deletions(const float posteriors[][DEC_ALPHABET_SIZE],
                                  int n_positions,
                                  const dec_word_t *word,
                                  int dels,
                                  float del_penalty)
{
    int n = n_positions;
    int m = (int)word->length;

    if (m <= 0 || m > SPELL_MAX_LETTERS_PER_WORD) return -INFINITY;
    if (dels < 0 || dels > 2)                     return -INFINITY;
    if (m != n + dels)                            return -INFINITY;

    if (dels == 0) {
        float s = 0.0f;
        for (int i = 0; i < n; i++) {
            s += safe_log(posteriors[i][word->letters[i]]);
        }
        return s;
    }

    if (dels == 1) {
        float best = -INFINITY;
        for (int skip = 0; skip < m; skip++) {
            float s = del_penalty;
            int ui = 0;
            for (int j = 0; j < m; j++) {
                if (j == skip) continue;
                s += safe_log(posteriors[ui][word->letters[j]]);
                ui++;
            }
            if (s > best) best = s;
        }
        return best;
    }

    // dels == 2: pick the best pair of word positions to skip.
    float best = -INFINITY;
    for (int skip_a = 0; skip_a < m; skip_a++) {
        for (int skip_b = skip_a + 1; skip_b < m; skip_b++) {
            float s = 2.0f * del_penalty;
            int ui = 0;
            for (int j = 0; j < m; j++) {
                if (j == skip_a || j == skip_b) continue;
                s += safe_log(posteriors[ui][word->letters[j]]);
                ui++;
            }
            if (s > best) best = s;
        }
    }
    return best;
}

static float score_word(const float posteriors[][DEC_ALPHABET_SIZE],
                        int n_positions,
                        const dec_word_t *word,
                        float ins_penalty,
                        float del_penalty)
{
    int n = n_positions;
    int m = (int)word->length;

    if (m <= 0 || m > SPELL_MAX_LETTERS_PER_WORD) return -INFINITY;

    int diff = m - n;
    if (diff < -1 || diff > 1) return -INFINITY;

    if (diff == 0) return score_with_deletions(posteriors, n, word, 0, del_penalty);
    if (diff == 1) return score_with_deletions(posteriors, n, word, 1, del_penalty);

    // diff == -1, n = m + 1: skip one utterance, charge ins_penalty.
    float best = -INFINITY;
    for (int skip = 0; skip < n; skip++) {
        float s = ins_penalty;
        int wj = 0;
        for (int i = 0; i < n; i++) {
            if (i == skip) continue;
            s += safe_log(posteriors[i][word->letters[wj]]);
            wj++;
        }
        if (s > best) best = s;
    }
    return best;
}

// =============================================================================
// Full-EOW resolution
// =============================================================================

dec_resolution_t dec_resolve_full_eow(const dec_state_t *st)
{
    dec_resolution_t r;
    memset(&r, 0, sizeof(r));
    r.n_positions  = st->n_positions;
    r.overflowed   = st->overflowed;
    r.score_margin = 0.0f;

    if (st->overflowed || st->n_positions == 0 || st->dict->n_words == 0) {
        r.kind = DEC_OUTCOME_ABSTAIN;
        return r;
    }

    float posteriors[SPELL_MAX_LETTERS_PER_WORD][DEC_ALPHABET_SIZE];
    for (int i = 0; i < st->n_positions; i++) {
        assemble_posterior(&st->positions[i], st->confusion,
                           st->cfg.alpha, posteriors[i]);
    }

    float best_score      = -INFINITY;
    float runner_up_score = -INFINITY;
    int   best_idx        = -1;

    for (int w = 0; w < st->dict->n_words; w++) {
        float s = score_word(posteriors, st->n_positions,
                             &st->dict->words[w],
                             st->cfg.ins_penalty,
                             st->cfg.del_penalty);
        if (s > best_score) {
            runner_up_score = best_score;
            best_score      = s;
            best_idx        = w;
        } else if (s > runner_up_score) {
            runner_up_score = s;
        }
    }

    if (best_idx < 0 || best_score == -INFINITY) {
        r.kind = DEC_OUTCOME_ABSTAIN;
        return r;
    }

    float margin = (runner_up_score == -INFINITY)
                 ? INFINITY
                 : (best_score - runner_up_score);

    r.score        = best_score;
    r.score_margin = margin;

    if (margin < st->cfg.decision_margin) {
        r.kind = DEC_OUTCOME_ABSTAIN;
        return r;
    }

    r.kind    = DEC_OUTCOME_RESOLVED;
    r.word_id = st->dict->words[best_idx].word_id;
    return r;
}

// =============================================================================
// Aggressive early-commit predicate (PRD §"Aggressive early-commit predicate",
// ADR-0001). Three clauses: margin over same-length runner-up, margin over
// best length-(N+1)/(N+2) competitor, post-letter silence floor (caller-
// supplied). Pure read-only over @p st.
// =============================================================================

dec_early_commit_eval_t dec_try_early_commit(const dec_state_t *st,
                                             bool silence_floor_satisfied)
{
    dec_early_commit_eval_t e;
    memset(&e, 0, sizeof(e));
    e.kind             = DEC_EARLY_COMMIT_HOLD;
    e.n_positions      = st ? st->n_positions : 0;
    e.silence_floor_ok = silence_floor_satisfied;

    if (!st || st->overflowed || st->n_positions == 0 || st->dict->n_words == 0) {
        return e;
    }

    float posteriors[SPELL_MAX_LETTERS_PER_WORD][DEC_ALPHABET_SIZE];
    for (int i = 0; i < st->n_positions; i++) {
        assemble_posterior(&st->positions[i], st->confusion,
                           st->cfg.alpha, posteriors[i]);
    }

    // Same-length pool: best + runner-up.
    float best_same   = -INFINITY;
    float runner_same = -INFINITY;
    int   best_idx    = -1;

    // Longer-by-1-or-2 pool: just the best.
    float best_longer = -INFINITY;

    for (int w = 0; w < st->dict->n_words; w++) {
        const dec_word_t *word = &st->dict->words[w];
        int diff = (int)word->length - st->n_positions;

        if (diff == 0) {
            float s = score_with_deletions(posteriors, st->n_positions, word,
                                           0, st->cfg.del_penalty);
            if (s > best_same) {
                runner_same = best_same;
                best_same   = s;
                best_idx    = w;
            } else if (s > runner_same) {
                runner_same = s;
            }
        } else if (diff == 1 || diff == 2) {
            float s = score_with_deletions(posteriors, st->n_positions, word,
                                           diff, st->cfg.del_penalty);
            if (s > best_longer) best_longer = s;
        }
        // Other lengths (shorter than N, or longer by 3+) are not part of
        // the early-commit candidate pool.
    }

    if (best_idx < 0 || best_same == -INFINITY) {
        // No valid same-length candidate to commit to.
        e.margin_runnerup = 0.0f;
        e.margin_longer   = 0.0f;
        return e;
    }

    e.score        = best_same;
    e.word_id      = st->dict->words[best_idx].word_id;
    e.margin_runnerup = (runner_same == -INFINITY)
                      ? INFINITY
                      : (best_same - runner_same);
    e.margin_longer   = (best_longer == -INFINITY)
                      ? INFINITY
                      : (best_same - best_longer);

    e.margin_runnerup_ok = (e.margin_runnerup > st->cfg.early_margin_runnerup);
    e.margin_longer_ok   = (e.margin_longer   > st->cfg.early_margin_longer);

    if (e.margin_runnerup_ok && e.margin_longer_ok && e.silence_floor_ok) {
        e.kind = DEC_EARLY_COMMIT_RESOLVED;
    }
    return e;
}

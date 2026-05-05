/**
 * @file main.c
 * @brief CLI driver — replays a CSV fixture of synthetic top-K events
 *        through decoder_core and prints the resolution + per-evaluation
 *        early-commit predicate flags.
 *
 * Lets you sweep the decoder hyperparameters (`α`, decision margin, edit
 * penalties, early-commit margins) against captured event sequences without
 * re-flashing the device. The same decoder_core.c that runs on-device is
 * linked here, so the algorithm is bit-identical.
 *
 * ── Fixture format ──────────────────────────────────────────────────────
 *
 * Rows are processed in time order. The first comma-separated field is a
 * single-letter event tag (case-insensitive); whitespace around any field
 * is trimmed; lines starting with `#` and blank lines are skipped.
 *
 *   L, top1L, top1P, top2L, top2P, top3L, top3P, top4L, top4P, top5L, top5P, retract_count
 *      LETTER_RECOGNIZED. Letters are A..Z (case-insensitive) or 0..25.
 *      retract_count is 0 in the common case, 2 on a confirmed-W replacement
 *      (per ADR-0005, ADR-0006). Probs are floats in [0, 1].
 *
 *   W   EARLY_COMMIT_WINDOW. Fires the early-commit predicate with
 *       silence_floor_satisfied = true. No further columns.
 *
 *   E   END_OF_WORD. Calls dec_resolve_full_eow and resets state. No further
 *       columns. End-of-file is treated as an implicit E if any positions
 *       remain in the buffer.
 *
 *   R   RESET. Drops in-flight state without resolving (useful for stitching
 *       multiple words into one fixture without an EOW between them).
 *
 * ── Dictionary file (`--dict`) ─────────────────────────────────────────
 *
 * One word per line: `<word_id> <LETTERS>` (e.g. `1 CAT`). word_id is a
 * decimal uint32; LETTERS is A..Z, length ≤ SPELL_MAX_LETTERS_PER_WORD.
 * Comments start with `#`. Required.
 *
 * ── Confusion matrix file (`--matrix`) ─────────────────────────────────
 *
 * Optional. 26 lines each with 26 floats (space- or comma-separated).
 * Row j corresponds to top1=j (P(true | top1=j)). When absent, a neutral
 * matrix is used (diag 0.6, off 0.4/25 = 0.016) — same shape that
 * test_decoder_core.c builds for its baseline tests.
 *
 * ── Usage ──────────────────────────────────────────────────────────────
 *
 *   ./decoder_replay --dict words.txt fixture.csv
 *   cat fixture.csv | ./decoder_replay --dict words.txt -
 *
 * Every tunable in dec_config_t is exposed by a `--<name>` flag (see
 * --help for the list and defaults sourced from spell_config.h).
 */

#include "decoder_core.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LX(ch) ((uint8_t)((ch) - 'A'))

// Dictionary cap is generous — host build only, nothing is allocated up-front.
#define DICT_MAX_WORDS 1024

static void print_usage(const char *argv0)
{
    dec_config_t d; dec_config_default(&d);
    fprintf(stderr,
        "usage: %s [options] --dict FILE fixture.csv|-\n"
        "\n"
        "options:\n"
        "  --dict FILE                   dictionary (one `<word_id> <LETTERS>` per line)\n"
        "  --matrix FILE                 26x26 confusion matrix (rows = top1, cols = true);\n"
        "                                defaults to neutral diag=0.6, off=0.4/25\n"
        "  --alpha FLOAT                 confusion-matrix mix weight    (default %.3f)\n"
        "  --decision-margin FLOAT       full-EOW abstain threshold     (default %.3f)\n"
        "  --ins-penalty FLOAT           edit-alignment ins penalty     (default %.3f)\n"
        "  --del-penalty FLOAT           edit-alignment del penalty     (default %.3f)\n"
        "  --early-margin-runnerup FLOAT early-commit clause 1 threshold (default %.3f)\n"
        "  --early-margin-longer FLOAT   early-commit clause 2 threshold (default %.3f)\n"
        "  --quiet                       suppress per-event log; summary only\n"
        "  --help                        this message\n"
        "\n"
        "fixture rows:\n"
        "  L, top1L, top1P, top2L, top2P, top3L, top3P, top4L, top4P, top5L, top5P, retract\n"
        "  W                              (EARLY_COMMIT_WINDOW)\n"
        "  E                              (END_OF_WORD; implicit at EOF)\n"
        "  R                              (RESET in-flight state)\n",
        argv0,
        (double)d.alpha,
        (double)d.decision_margin,
        (double)d.ins_penalty,
        (double)d.del_penalty,
        (double)d.early_margin_runnerup,
        (double)d.early_margin_longer);
}

static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) return s;
    char *e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}

/// Pull the next comma-separated cell out of @p line in place. On entry,
/// @p *cursor points to the read position; on return, *cursor advances past
/// the separator. Returns the (trimmed) cell or NULL if the cursor is at
/// end-of-line.
static char *next_cell(char **cursor)
{
    if (!*cursor || !**cursor) return NULL;
    char *start = *cursor;
    char *comma = strchr(start, ',');
    if (comma) {
        *comma  = '\0';
        *cursor = comma + 1;
    } else {
        *cursor = start + strlen(start);
    }
    return trim(start);
}

/// Accept "A".."Z" (case-insensitive) or "0".."25". Writes the alphabet
/// index into @p out. Returns 0 on success, -1 on bad input.
static int parse_letter(const char *s, uint8_t *out)
{
    if (!s || !*s) return -1;
    if (isalpha((unsigned char)s[0]) && s[1] == '\0') {
        *out = (uint8_t)(toupper((unsigned char)s[0]) - 'A');
        return 0;
    }
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || (end && *end != '\0')) return -1;
    if (v < 0 || v > 25) return -1;
    *out = (uint8_t)v;
    return 0;
}

static int parse_prob(const char *s, float *out)
{
    if (!s || !*s) return -1;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || (end && *end != '\0')) return -1;
    if (v < 0.0 || v > 1.0) return -1;
    *out = (float)v;
    return 0;
}

static int parse_uint(const char *s, unsigned long *out)
{
    if (!s || !*s) return -1;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || (end && *end != '\0')) return -1;
    *out = v;
    return 0;
}

// ---------------------------------------------------------------------------
// Dictionary + confusion matrix loaders
// ---------------------------------------------------------------------------

/// Reads a dictionary file into @p words (length ≤ DICT_MAX_WORDS). Returns
/// the number of words on success or -1 on parse error.
static int load_dict(const char *path, dec_word_t *words, int max_words)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "open dict %s: %s\n", path, strerror(errno));
        return -1;
    }
    char line[256];
    int  line_no = 0;
    int  n = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = trim(line);
        if (!*p || *p == '#') continue;
        if (n >= max_words) {
            fprintf(stderr, "dict %s: more than %d words\n", path, max_words);
            fclose(fp); return -1;
        }
        char *id_str = p;
        char *space  = p;
        while (*space && !isspace((unsigned char)*space)) space++;
        if (!*space) {
            fprintf(stderr, "dict %s line %d: missing word\n", path, line_no);
            fclose(fp); return -1;
        }
        *space++ = '\0';
        char *word_str = trim(space);

        unsigned long id;
        if (parse_uint(id_str, &id) != 0) {
            fprintf(stderr, "dict %s line %d: bad word_id '%s'\n",
                    path, line_no, id_str);
            fclose(fp); return -1;
        }
        size_t wlen = strlen(word_str);
        if (wlen == 0 || wlen > (size_t)SPELL_MAX_LETTERS_PER_WORD) {
            fprintf(stderr, "dict %s line %d: word length %zu out of [1, %d]\n",
                    path, line_no, wlen, SPELL_MAX_LETTERS_PER_WORD);
            fclose(fp); return -1;
        }
        memset(&words[n], 0, sizeof(words[n]));
        words[n].word_id = (uint32_t)id;
        words[n].length  = (uint8_t)wlen;
        for (size_t i = 0; i < wlen; i++) {
            int c = toupper((unsigned char)word_str[i]);
            if (c < 'A' || c > 'Z') {
                fprintf(stderr, "dict %s line %d: non A-Z char in '%s'\n",
                        path, line_no, word_str);
                fclose(fp); return -1;
            }
            words[n].letters[i] = (uint8_t)(c - 'A');
        }
        n++;
    }
    fclose(fp);
    return n;
}

/// Fill @p c with the neutral diag=0.6 / off=0.4/25 matrix.
static void conf_neutral(dec_confusion_t *c)
{
    const float diag = 0.6f;
    const float off  = 0.4f / 25.0f;
    for (int j = 0; j < DEC_ALPHABET_SIZE; j++) {
        for (int i = 0; i < DEC_ALPHABET_SIZE; i++) {
            c->m[j * DEC_ALPHABET_SIZE + i] = (i == j) ? diag : off;
        }
    }
}

/// Load a 26×26 matrix from @p path. Cells are space- or comma-separated.
/// Returns 0 on success, -1 on parse error.
static int load_matrix(const char *path, dec_confusion_t *out)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "open matrix %s: %s\n", path, strerror(errno));
        return -1;
    }
    int row = 0;
    char line[2048];
    while (row < DEC_ALPHABET_SIZE && fgets(line, sizeof(line), fp)) {
        char *p = trim(line);
        if (!*p || *p == '#') continue;
        // Tokenise on whitespace OR comma.
        int col = 0;
        char *tok = p;
        while (col < DEC_ALPHABET_SIZE && *tok) {
            while (*tok && (isspace((unsigned char)*tok) || *tok == ',')) tok++;
            if (!*tok) break;
            char *end = tok;
            while (*end && !isspace((unsigned char)*end) && *end != ',') end++;
            char saved = *end;
            *end = '\0';
            char *parse_end = NULL;
            double v = strtod(tok, &parse_end);
            if (parse_end == tok || (parse_end && *parse_end != '\0')) {
                fprintf(stderr, "matrix %s row %d col %d: bad cell '%s'\n",
                        path, row, col, tok);
                fclose(fp); return -1;
            }
            out->m[row * DEC_ALPHABET_SIZE + col] = (float)v;
            *end = saved;
            tok = end;
            col++;
        }
        if (col != DEC_ALPHABET_SIZE) {
            fprintf(stderr, "matrix %s row %d: got %d cells, expected %d\n",
                    path, row, col, DEC_ALPHABET_SIZE);
            fclose(fp); return -1;
        }
        row++;
    }
    fclose(fp);
    if (row != DEC_ALPHABET_SIZE) {
        fprintf(stderr, "matrix %s: got %d rows, expected %d\n",
                path, row, DEC_ALPHABET_SIZE);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Pretty-printers
// ---------------------------------------------------------------------------

static const char *kind_str(dec_outcome_kind_t k)
{
    return (k == DEC_OUTCOME_RESOLVED) ? "RESOLVED" : "ABSTAIN";
}

static void fmt_margin(float m, char *buf, size_t buflen)
{
    if (isinf(m)) {
        snprintf(buf, buflen, "+inf");
    } else {
        snprintf(buf, buflen, "%.3f", (double)m);
    }
}

static void log_letter_row(int event_n, const dec_top_k_t *tk, uint8_t retract,
                           const dec_early_commit_eval_t *e)
{
    const dec_candidate_t *c = tk->candidates;
    char rb[16], lb[16];
    fmt_margin(e->margin_runnerup, rb, sizeof(rb));
    fmt_margin(e->margin_longer,   lb, sizeof(lb));
    printf("event %3d: LETTER  "
           "%c=%.2f %c=%.2f %c=%.2f %c=%.2f %c=%.2f  retract=%u  "
           "early=%s  [ru=%s(%s) lon=%s(%s) sf=%s]\n",
           event_n,
           'A' + c[0].letter_index, (double)c[0].probability,
           'A' + c[1].letter_index, (double)c[1].probability,
           'A' + c[2].letter_index, (double)c[2].probability,
           'A' + c[3].letter_index, (double)c[3].probability,
           'A' + c[4].letter_index, (double)c[4].probability,
           (unsigned)retract,
           e->kind == DEC_EARLY_COMMIT_RESOLVED ? "COMMIT" : "HOLD  ",
           e->margin_runnerup_ok ? "YES" : "no ", rb,
           e->margin_longer_ok   ? "YES" : "no ", lb,
           e->silence_floor_ok   ? "YES" : "no ");
}

static void log_window_row(int event_n, const dec_early_commit_eval_t *e)
{
    char rb[16], lb[16];
    fmt_margin(e->margin_runnerup, rb, sizeof(rb));
    fmt_margin(e->margin_longer,   lb, sizeof(lb));
    if (e->kind == DEC_EARLY_COMMIT_RESOLVED) {
        printf("event %3d: WINDOW  early=COMMIT  word_id=%u score=%.3f  "
               "[ru=%s(%s) lon=%s(%s) sf=%s]\n",
               event_n,
               (unsigned)e->word_id, (double)e->score,
               e->margin_runnerup_ok ? "YES" : "no ", rb,
               e->margin_longer_ok   ? "YES" : "no ", lb,
               e->silence_floor_ok   ? "YES" : "no ");
    } else {
        printf("event %3d: WINDOW  early=HOLD    "
               "[ru=%s(%s) lon=%s(%s) sf=%s]\n",
               event_n,
               e->margin_runnerup_ok ? "YES" : "no ", rb,
               e->margin_longer_ok   ? "YES" : "no ", lb,
               e->silence_floor_ok   ? "YES" : "no ");
    }
}

static void log_eow_row(int event_n, const dec_resolution_t *r)
{
    char mb[16];
    fmt_margin(r->score_margin, mb, sizeof(mb));
    if (r->kind == DEC_OUTCOME_RESOLVED) {
        printf("event %3d: EOW     %s  word_id=%u score=%.3f margin=%s  positions=%d\n",
               event_n, kind_str(r->kind),
               (unsigned)r->word_id, (double)r->score, mb, r->n_positions);
    } else {
        printf("event %3d: EOW     %s   margin=%s positions=%d overflow=%d\n",
               event_n, kind_str(r->kind), mb, r->n_positions,
               (int)r->overflowed);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    dec_config_t cfg;
    dec_config_default(&cfg);

    const char *dict_path   = NULL;
    const char *matrix_path = NULL;
    const char *fixture     = NULL;
    int         quiet       = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--dict")                  && i+1<argc) dict_path                = argv[++i];
        else if (!strcmp(a, "--matrix")                && i+1<argc) matrix_path              = argv[++i];
        else if (!strcmp(a, "--alpha")                 && i+1<argc) cfg.alpha                 = (float)atof(argv[++i]);
        else if (!strcmp(a, "--decision-margin")       && i+1<argc) cfg.decision_margin       = (float)atof(argv[++i]);
        else if (!strcmp(a, "--ins-penalty")           && i+1<argc) cfg.ins_penalty           = (float)atof(argv[++i]);
        else if (!strcmp(a, "--del-penalty")           && i+1<argc) cfg.del_penalty           = (float)atof(argv[++i]);
        else if (!strcmp(a, "--early-margin-runnerup") && i+1<argc) cfg.early_margin_runnerup = (float)atof(argv[++i]);
        else if (!strcmp(a, "--early-margin-longer")   && i+1<argc) cfg.early_margin_longer   = (float)atof(argv[++i]);
        else if (!strcmp(a, "--quiet"))                              quiet = 1;
        else if (!strcmp(a, "--help"))                             { print_usage(argv[0]); return 0; }
        else if ((a[0] != '-' || !strcmp(a, "-")) && !fixture)       fixture = a;
        else { fprintf(stderr, "unknown arg: %s\n", a); print_usage(argv[0]); return 2; }
    }
    if (!dict_path || !fixture) { print_usage(argv[0]); return 2; }

    static dec_word_t words[DICT_MAX_WORDS];
    int n_words = load_dict(dict_path, words, DICT_MAX_WORDS);
    if (n_words < 0) return 1;
    if (n_words == 0) {
        fprintf(stderr, "dict %s: empty\n", dict_path);
        return 1;
    }
    dec_dict_t dict = { .words = words, .n_words = n_words };

    dec_confusion_t conf;
    if (matrix_path) {
        if (load_matrix(matrix_path, &conf) != 0) return 1;
    } else {
        conf_neutral(&conf);
    }

    dec_state_t st;
    if (dec_init(&st, &dict, &conf, &cfg) != 0) {
        fprintf(stderr, "dec_init() rejected configuration "
                        "(check matrix row sums and tunable ranges)\n");
        return 1;
    }

    FILE *fp = (!strcmp(fixture, "-")) ? stdin : fopen(fixture, "r");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", fixture, strerror(errno));
        return 1;
    }

    if (!quiet) {
        printf("config: alpha=%.3f decision_margin=%.3f ins=%.3f del=%.3f "
               "early_runnerup=%.3f early_longer=%.3f\n",
               (double)cfg.alpha, (double)cfg.decision_margin,
               (double)cfg.ins_penalty, (double)cfg.del_penalty,
               (double)cfg.early_margin_runnerup,
               (double)cfg.early_margin_longer);
        printf("dict:   %d words from %s\n", n_words, dict_path);
        printf("matrix: %s\n",
               matrix_path ? matrix_path : "<neutral default>");
        printf("source: %s\n\n", (fp == stdin) ? "<stdin>" : fixture);
    }

    char line[1024];
    int  line_no    = 0;
    int  event_n    = 0;
    int  parse_err  = 0;
    int  letters    = 0;
    int  windows    = 0;
    int  resolved   = 0;
    int  abstained  = 0;
    int  early_hits = 0;

    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = trim(line);
        if (!*p || *p == '#') continue;

        char *cursor = p;
        char *tag    = next_cell(&cursor);
        if (!tag || !*tag) continue;
        char t = (char)toupper((unsigned char)tag[0]);
        if (tag[1] != '\0' || (t != 'L' && t != 'W' && t != 'E' && t != 'R')) {
            fprintf(stderr, "line %d: unknown row tag '%s' (expected L/W/E/R)\n",
                    line_no, tag);
            parse_err++;
            continue;
        }

        event_n++;

        if (t == 'L') {
            dec_top_k_t tk;
            memset(&tk, 0, sizeof(tk));
            int slot_err = 0;
            for (int k = 0; k < SPELL_LETTER_TOP_K; k++) {
                char *cl = next_cell(&cursor);
                char *cp = next_cell(&cursor);
                uint8_t li = 0; float pp = 0.0f;
                if (!cl || !cp ||
                    parse_letter(cl, &li) != 0 ||
                    parse_prob(cp,   &pp) != 0) {
                    fprintf(stderr,
                            "line %d: bad LETTER row at slot %d "
                            "(need 5x(letter,prob) + retract)\n",
                            line_no, k);
                    slot_err = 1;
                    break;
                }
                tk.candidates[k].letter_index = li;
                tk.candidates[k].probability  = pp;
            }
            if (slot_err) { parse_err++; continue; }

            char *cr = next_cell(&cursor);
            unsigned long retract = 0;
            if (!cr || parse_uint(cr, &retract) != 0 || retract > 255) {
                fprintf(stderr, "line %d: bad retract_count\n", line_no);
                parse_err++; continue;
            }

            dec_on_letter(&st, &tk, (uint8_t)retract);
            dec_early_commit_eval_t e = dec_try_early_commit(&st, false);
            letters++;
            if (e.kind == DEC_EARLY_COMMIT_RESOLVED) early_hits++;
            if (!quiet) log_letter_row(event_n, &tk, (uint8_t)retract, &e);
        } else if (t == 'W') {
            dec_early_commit_eval_t e = dec_try_early_commit(&st, true);
            windows++;
            if (e.kind == DEC_EARLY_COMMIT_RESOLVED) {
                early_hits++;
                if (!quiet) log_window_row(event_n, &e);
                dec_reset(&st);
            } else {
                if (!quiet) log_window_row(event_n, &e);
            }
        } else if (t == 'E') {
            dec_resolution_t r = dec_resolve_full_eow(&st);
            if (r.kind == DEC_OUTCOME_RESOLVED) resolved++; else abstained++;
            if (!quiet) log_eow_row(event_n, &r);
            dec_reset(&st);
        } else { // R
            dec_reset(&st);
            if (!quiet) printf("event %3d: RESET\n", event_n);
        }
    }

    if (fp != stdin) fclose(fp);

    // Implicit EOW at end-of-file when positions remain.
    if (st.n_positions > 0 || st.overflowed) {
        event_n++;
        dec_resolution_t r = dec_resolve_full_eow(&st);
        if (r.kind == DEC_OUTCOME_RESOLVED) resolved++; else abstained++;
        if (!quiet) log_eow_row(event_n, &r);
    }

    printf("%ssummary: %d event%s (%d LETTER, %d WINDOW), "
           "%d resolved, %d abstain, %d early-commit, %d parse error%s\n",
           quiet ? "" : "\n",
           event_n, event_n == 1 ? "" : "s",
           letters, windows,
           resolved, abstained, early_hits,
           parse_err, parse_err == 1 ? "" : "s");

    return parse_err ? 1 : 0;
}

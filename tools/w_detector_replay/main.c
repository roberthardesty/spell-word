/**
 * @file main.c
 * @brief CLI driver — replays a CSV of synthetic top-K events through
 *        w_detector_core and prints the trigger decision per row.
 *
 * Lets you sweep the W-recovery trigger thresholds against captured event
 * sequences without re-flashing the device. The same w_detector_core.c
 * that runs on-device is linked here, so the predicate is bit-identical.
 *
 * CSV format (one event per row):
 *
 *     event_n , top1_letter , top1_prob , [ extra columns ignored ... ]
 *
 *   - event_n     integer (display only — the predicate is positional)
 *   - top1_letter single ASCII letter A..Z (case-insensitive) OR a numeric
 *                 alphabet index 0..25
 *   - top1_prob   float in [0, 1]
 *   - extra columns (top2_letter, top2_prob, ...) are tolerated and skipped;
 *     the predicate only consumes top-1
 *
 * Lines starting with `#` and blank lines are skipped. A header line whose
 * first cell is the literal `event_n` is also skipped, so spreadsheets export
 * cleanly. Comma is the only field separator; whitespace around fields is
 * trimmed.
 *
 * Usage:
 *   ./w_detector_replay [options] events.csv
 *   ./w_detector_replay [options] -            # read from stdin
 *
 * Options (override the SPELL_W_DETECT_* defaults from spell_config.h):
 *   --u-threshold FLOAT          Current top-1 'U' must clear this (>=).
 *                                Default 0.50.
 *   --low-conf-threshold FLOAT   Each prior top-1 prob must be below this (<).
 *                                Default 0.40.
 *   --quiet                      Suppress per-event lines; print summary only.
 *   --help                       Show usage.
 */

#include "w_detector_core.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [options] events.csv|-\n"
        "\n"
        "options:\n"
        "  --u-threshold FLOAT         current 'U' prob must be >= this (default %.2f)\n"
        "  --low-conf-threshold FLOAT  each prior prob must be < this (default %.2f)\n"
        "  --quiet                     suppress per-event log; summary only\n"
        "  --help                      this message\n"
        "\n"
        "csv columns:  event_n , top1_letter , top1_prob [ , ... ]\n"
        "  - top1_letter is A..Z (case-insensitive) or 0..25\n"
        "  - extra columns are tolerated and ignored\n"
        "  - lines starting with # and the optional header row are skipped\n",
        argv0,
        (double)SPELL_W_DETECT_U_THRESHOLD,
        (double)SPELL_W_DETECT_LOW_CONF_THRESHOLD);
}

static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) return s;
    char *e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}

/// Parse a top-1 letter cell. Accepts a single ASCII letter or "0".."25".
/// Returns 0 on success and writes the alphabet index into @p out; -1 on bad input.
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

/// Pull the next comma-separated cell out of @p line in place. On entry,
/// @p *cursor points to the read position; on return, *cursor advances past
/// the cell separator (or to the trailing NUL). Returns the (trimmed) cell
/// or NULL if the cursor is already at end-of-line.
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

int main(int argc, char **argv)
{
    wdet_config_t cfg;
    wdet_config_default(&cfg);
    int   quiet = 0;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--u-threshold")        && i+1<argc) cfg.u_threshold        = (float)atof(argv[++i]);
        else if (!strcmp(a, "--low-conf-threshold") && i+1<argc) cfg.low_conf_threshold = (float)atof(argv[++i]);
        else if (!strcmp(a, "--quiet"))                          quiet = 1;
        else if (!strcmp(a, "--help"))                         { print_usage(argv[0]); return 0; }
        else if ((a[0] != '-' || !strcmp(a, "-")) && !path)      path = a;
        else { fprintf(stderr, "unknown arg: %s\n", a); print_usage(argv[0]); return 2; }
    }
    if (!path) { print_usage(argv[0]); return 2; }

    FILE *fp = (!strcmp(path, "-")) ? stdin : fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return 1;
    }

    wdet_state_t st;
    if (wdet_init(&st, &cfg) != 0) {
        fprintf(stderr, "wdet_init() rejected configuration "
                        "(thresholds must lie in [0.0, 1.0])\n");
        if (fp != stdin) fclose(fp);
        return 1;
    }

    if (!quiet) {
        printf("config: u_threshold=%.3f  low_conf_threshold=%.3f\n",
               (double)cfg.u_threshold, (double)cfg.low_conf_threshold);
        printf("source: %s\n\n", (fp == stdin) ? "<stdin>" : path);
    }

    char line[1024];
    int  line_no   = 0;
    int  events    = 0;
    int  triggers  = 0;
    int  parse_err = 0;
    bool header_seen = false;

    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        char *p = trim(line);
        if (!*p || *p == '#') continue;

        // Optional header row: first cell == "event_n". Only honored as the
        // first non-blank, non-comment line we see.
        if (!header_seen) {
            char copy[1024];
            strncpy(copy, p, sizeof(copy)-1);
            copy[sizeof(copy)-1] = '\0';
            char *cur = copy;
            char *first = next_cell(&cur);
            header_seen = true;
            if (first && !strcasecmp(first, "event_n")) continue;
        }

        char *cursor = p;
        char *c_event = next_cell(&cursor);
        char *c_letter = next_cell(&cursor);
        char *c_prob   = next_cell(&cursor);

        uint8_t top1 = 0;
        float   prob = 0.0f;
        if (!c_event || !c_letter || !c_prob ||
            parse_letter(c_letter, &top1) != 0 ||
            parse_prob  (c_prob,   &prob) != 0)
        {
            fprintf(stderr,
                    "line %d: bad row (need: event_n, top1_letter, top1_prob)\n",
                    line_no);
            parse_err++;
            continue;
        }

        wdet_eval_t e = wdet_on_event(&st, top1, prob);
        events++;
        if (e.trigger) triggers++;

        if (!quiet) {
            printf("event %4s: top1='%c'(%2u) prob=%.3f  trigger=%s  "
                   "[hist=%s is_u=%d hi=%d p1_lo=%d p2_lo=%d]\n",
                   c_event,
                   (char)('A' + e.current_top1),
                   (unsigned)e.current_top1,
                   (double)e.current_top1_prob,
                   e.trigger ? "YES" : "no ",
                   e.sufficient_history ? "2/2" : "<2 ",
                   (int)e.current_is_u,
                   (int)e.current_high_conf,
                   (int)e.prev1_low_conf,
                   (int)e.prev2_low_conf);
        }
    }

    if (fp != stdin) fclose(fp);

    printf("%ssummary: %d events, %d trigger%s, %d parse error%s\n",
           quiet ? "" : "\n",
           events, triggers, triggers == 1 ? "" : "s",
           parse_err, parse_err == 1 ? "" : "s");

    return parse_err ? 1 : 0;
}

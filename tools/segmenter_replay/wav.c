/**
 * @file wav.c
 * @brief Minimal WAV reader/writer — 16-bit PCM, mono, 16 kHz only.
 */

#include "wav.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// WAV is little-endian. Hosts we care about (x86_64, arm64-darwin, arm64-linux)
// are all LE, so we read raw. A portable build would byteswap.

#pragma pack(push, 1)
typedef struct {
    char     riff[4];        // "RIFF"
    uint32_t file_size;
    char     wave[4];        // "WAVE"
    char     fmt[4];         // "fmt "
    uint32_t fmt_size;       // 16 for PCM
    uint16_t fmt_format;     // 1 = PCM
    uint16_t fmt_channels;
    uint32_t fmt_rate;
    uint32_t fmt_byte_rate;
    uint16_t fmt_block_align;
    uint16_t fmt_bits;
    char     data[4];        // "data"
    uint32_t data_size;
} wav_header_t;
#pragma pack(pop)

int wav_read_16k_mono(const char *path,
                      int16_t **out_samples,
                      size_t   *out_n,
                      int      *out_rate)
{
    if (!path || !out_samples || !out_n) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "wav_read: cannot open '%s'\n", path);
        return -1;
    }

    wav_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "wav_read: short header in '%s'\n", path);
        fclose(f);
        return -1;
    }

    if (memcmp(hdr.riff, "RIFF", 4) != 0 ||
        memcmp(hdr.wave, "WAVE", 4) != 0 ||
        memcmp(hdr.fmt,  "fmt ", 4) != 0 ||
        memcmp(hdr.data, "data", 4) != 0) {
        fprintf(stderr, "wav_read: not a canonical WAV file: '%s' "
                        "(this reader only handles fmt-then-data layout)\n",
                path);
        fclose(f);
        return -1;
    }

    if (hdr.fmt_format != 1) {
        fprintf(stderr, "wav_read: '%s' is not PCM (format=%u)\n",
                path, hdr.fmt_format);
        fclose(f);
        return -1;
    }
    if (hdr.fmt_channels != 1) {
        fprintf(stderr, "wav_read: '%s' has %u channels, expected mono\n",
                path, hdr.fmt_channels);
        fclose(f);
        return -1;
    }
    if (hdr.fmt_bits != 16) {
        fprintf(stderr, "wav_read: '%s' is %u-bit, expected 16-bit\n",
                path, hdr.fmt_bits);
        fclose(f);
        return -1;
    }
    if (hdr.fmt_rate != 16000) {
        fprintf(stderr, "wav_read: '%s' is %u Hz, expected 16000 Hz\n",
                path, hdr.fmt_rate);
        fclose(f);
        return -1;
    }

    size_t n = hdr.data_size / sizeof(int16_t);
    int16_t *buf = malloc(n * sizeof(int16_t));
    if (!buf) {
        fprintf(stderr, "wav_read: malloc(%zu) failed\n",
                n * sizeof(int16_t));
        fclose(f);
        return -1;
    }

    if (fread(buf, sizeof(int16_t), n, f) != n) {
        fprintf(stderr, "wav_read: short data in '%s'\n", path);
        free(buf);
        fclose(f);
        return -1;
    }

    fclose(f);
    *out_samples = buf;
    *out_n       = n;
    if (out_rate) *out_rate = hdr.fmt_rate;
    return 0;
}

int wav_write_16k_mono(const char *path, const int16_t *samples, size_t n)
{
    if (!path || !samples) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "wav_write: cannot open '%s' for writing\n", path);
        return -1;
    }

    uint32_t data_size = (uint32_t)(n * sizeof(int16_t));

    wav_header_t hdr = {0};
    memcpy(hdr.riff, "RIFF", 4);
    hdr.file_size = 36 + data_size;
    memcpy(hdr.wave, "WAVE", 4);
    memcpy(hdr.fmt,  "fmt ", 4);
    hdr.fmt_size        = 16;
    hdr.fmt_format      = 1;        // PCM
    hdr.fmt_channels    = 1;
    hdr.fmt_rate        = 16000;
    hdr.fmt_byte_rate   = 16000 * 2;
    hdr.fmt_block_align = 2;
    hdr.fmt_bits        = 16;
    memcpy(hdr.data, "data", 4);
    hdr.data_size = data_size;

    if (fwrite(&hdr,    sizeof(hdr),     1, f) != 1 ||
        fwrite(samples, sizeof(int16_t), n, f) != n) {
        fprintf(stderr, "wav_write: short write to '%s'\n", path);
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

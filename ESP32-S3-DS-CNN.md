# I2S Audio Capture & DS-CNN Spoken Letter Classification on ESP32-S3

**Firmware:** placeholder name `FW_*` (ESP-IDF v5.3.5, ESP32-S3 N16R8 DevKitC-1)
**Microphone:** ICS-43434 MEMS (I2S, 24-bit left-justified)
**Model:** DS-CNN int8 quantized (~60 KB), 26-class softmax over A–Z (95% top-1 accuracy on the training distribution)
**Inference engine:** TFLite Micro 1.3.5 with esp-nn SIMD acceleration

This document covers the I2S audio input pipeline and the DS-CNN inference subsystem for a spoken-letter classifier running on-device. Everything here is adapted from the EARS bark-detection firmware. The audio capture path and the log-mel front-end carry over almost unchanged; the model output, the inference window length, and the event format are new. This document does not cover the downstream sequence-to-word post-processor (which consumes the per-window top-K letter probabilities and decodes the most-likely intended word), nor any networking or transport.

Names like `FW_*` and `fw_config.h` are placeholders — substitute your project's prefix and header names.

---

## Architecture overview

```
ICS-43434 MEMS mic
     │
     │  I2S (32-bit left-justified, Philips/I2S standard, LEFT slot)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  capture_task  (Core 0, priority 10)                │
│  ┌───────────────────────────────────┐              │
│  │ i2s_channel_read() — 256 frames  │              │
│  │ int32 → int16 via >> 16          │              │
│  └──────────────┬────────────────────┘              │
│                 │                                    │
│        Non-blocking fan-out to N subscribers         │
│                 │                                    │
│    ┌────────────┼────────────────┐                   │
│    ▼            ▼                ▼                   │
│  StreamBuf    StreamBuf      StreamBuf               │
│  "inference"  (optional)     (future)                │
│  ~20 KB       ...            ...                     │
│  (PSRAM)                                             │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  inference_task  (Core 1, priority 5)               │
│                                                     │
│  1. Read 9 600 samples (600 ms) from StreamBuffer   │
│  2. RMS energy gate (skip if < -45 dBFS)            │
│  3. Log-mel spectrogram (58×40, esp-dsp FFT)        │
│  4. Quantize float → int8                           │
│  5. TFLite Micro Invoke() — DS-CNN forward pass     │
│  6. Dequantize int8 → 26 softmax probabilities      │
│  7. Top-K selection → post FW_EVENT_LETTER_TOP_K    │
└─────────────────────────────────────────────────────┘
```

The capture task and inference task run on separate cores. Core 0 handles I2S reads and fan-out; Core 1 handles the entire ML pipeline. This mirrors the EARS topology and keeps the latency-sensitive I2S reads isolated from the multi-hundred-millisecond inference passes.

A 600 ms window is the starting point. Letters are short utterances (typically 200–400 ms when spelled clearly) and 600 ms gives enough trailing context for stop consonants without bloating the feature map. If field testing shows letters being clipped at boundaries, extend to 700–750 ms — the only constants that change are `FW_INFERENCE_WINDOW_SAMPLES`, `FW_MEL_N_FRAMES`, and the model's input shape.

---

## Source files at a glance

| File | Role |
|------|------|
| `components/fw_common/include/fw_config.h` | All pin definitions, audio format constants, mel/inference parameters |
| `components/audio_capture/include/audio_capture.h` | Public API: init → subscribe → start → read |
| `components/audio_capture/audio_capture.c` | I2S driver setup, 32→16 conversion, subscriber fan-out task |
| `components/letter_classifier/include/inference.h` | Public API: init, reload, event definitions, top-K event payload |
| `components/letter_classifier/inference.cpp` | TFLite Micro interpreter, energy gate, top-K event posting |
| `components/letter_classifier/include/log_mel.h` | Public API: init, compute, deinit |
| `components/letter_classifier/log_mel.c` | 512-pt FFT (esp-dsp), 40-band mel filterbank, log compression |
| `main/app_main.c` | Boot sequence — init ordering for the full pipeline |

---

## 1. ICS-43434 microphone configuration

The ICS-43434 is a bottom-port digital MEMS microphone that outputs 24-bit two's-complement audio left-justified in a 32-bit I2S slot. It supports standard I2S (Philips) timing. The SEL pin should be tied to GND on the PCB so the mic drives data on the LEFT channel only.

### 1.1 Pin assignments

```c
// I2S Microphone Pins (ICS-43434).
#define FW_I2S_SD_PIN         4    // Serial Data (DOUT from mic)
#define FW_I2S_SCK_PIN        5    // Bit Clock (BCLK)
#define FW_I2S_WS_PIN         6    // Word Select (LRCLK)
```

These three pins are all that's needed. The ICS-43434 generates its own clock from BCLK, so no MCLK is required (`I2S_GPIO_UNUSED` in the driver config).

### 1.2 Audio format constants

```c
#define FW_SAMPLE_RATE        16000  // Hz
#define FW_BITS_PER_SAMPLE    16
#define FW_CHANNELS           1
#define FW_BYTES_PER_SAMPLE   (FW_BITS_PER_SAMPLE / 8)
```

The ICS-43434 physically outputs 24-bit samples in 32-bit frames over I2S, but the firmware converts to 16-bit in software. The 16 kHz sample rate is dictated by the DS-CNN training pipeline — the model expects 600 ms windows of 9 600 samples at 16 kHz.

### 1.3 I2S capture geometry

```c
// Frames read per i2s_channel_read() call. 256 frames @ 16 kHz = 16 ms of
// audio per read — a good trade between syscall overhead and tap latency.
#define FW_CAPTURE_CHUNK_FRAMES   256

// Max concurrent audio_capture subscribers.
#define FW_CAPTURE_MAX_SUBS       4

// Inference subscriber's StreamBuffer capacity — 600 ms of PCM (19 200 B)
// plus one chunk of slack.
#define FW_INFERENCE_WINDOW_SAMPLES  9600  // 600 ms @ 16 kHz
#define FW_INFERENCE_WINDOW_BYTES \
    (FW_INFERENCE_WINDOW_SAMPLES * FW_BYTES_PER_SAMPLE)  // 19 200

#define FW_INFERENCE_SUB_BYTES \
    (FW_INFERENCE_WINDOW_BYTES + \
     FW_CAPTURE_CHUNK_FRAMES * (int)sizeof(int16_t))     // 19 712
```

The inference subscriber gets `19 200 + 512 = 19 712` bytes of PSRAM-backed StreamBuffer. The extra 512 bytes (one chunk) absorb scheduling jitter so the capture task's non-blocking writes don't drop samples under normal conditions. If you extend the window to 750 ms the buffer grows to ~24 KB — still trivially small for PSRAM.

---

## 2. I2S driver setup

Use IDF's native I2S standard driver (`driver/i2s_std.h`) — deliberately NOT ESP-ADF's `i2s_stream` / `audio_stream` pipeline. ADF's audio stream component drags in board support, speech recognition, and tone-partition dependencies that don't apply to a single-mic capture device, and introduces an esp-dsp version conflict that has bitten the EARS firmware in the past.

### 2.1 Channel and slot configuration

```c
// --- I2S channel ---
i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
// dma_desc_num × dma_frame_num defaults (6 × 240) give ~90 ms of DMA buffering
ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan),
                    TAG, "i2s_new_channel");
```

The `NULL` second argument means we're creating an RX-only channel (no TX handle needed — the ICS-43434 is input only).

```c
// --- Slot + clock config ---
i2s_std_config_t std_cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(FW_SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                    I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {
        .mclk = I2S_GPIO_UNUSED,
        .bclk = FW_I2S_SCK_PIN,
        .ws   = FW_I2S_WS_PIN,
        .dout = I2S_GPIO_UNUSED,
        .din  = FW_I2S_SD_PIN,
        .invert_flags = {
            .mclk_inv = false,
            .bclk_inv = false,
            .ws_inv   = false,
        },
    },
};
std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
```

Key points about this configuration:

- **`I2S_DATA_BIT_WIDTH_32BIT`** — read the full 32-bit I2S frame even though the ICS-43434 only puts 24 meaningful bits in it. The 32→16 bit conversion happens later in software.
- **`I2S_SLOT_MODE_MONO`** — single channel.
- **`I2S_STD_SLOT_LEFT`** — the ICS-43434's SEL pin is tied to GND, so data appears in the LEFT slot. This mask tells the driver to only deliver LEFT-slot frames.
- **`I2S_GPIO_UNUSED` for MCLK and DOUT** — no master clock needed (the ICS-43434 derives its clock from BCLK), and there is no speaker output.

After configuring, the channel is initialized but not yet enabled:

```c
ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx_chan, &std_cfg),
                    TAG, "i2s_channel_init_std_mode");
```

The channel is enabled later in `audio_capture_start()` after all subscribers have registered.

---

## 3. Subscriber fan-out pattern

The audio capture system uses a publish-subscribe pattern where a single capture task broadcasts PCM to multiple consumers, each with its own independent PSRAM-backed FreeRTOS StreamBuffer.

### 3.1 Subscriber data structure

```c
struct audio_capture_sub {
    StreamBufferHandle_t    stream;
    StaticStreamBuffer_t    stream_struct;  // small metadata, kept in internal RAM
    uint8_t                *storage;        // PSRAM — owned by this sub
    size_t                  capacity;
    uint32_t                overrun_count;
    char                    label[16];
    bool                    active;
};
```

Each subscriber owns a PSRAM byte array (`storage`) that backs the StreamBuffer. The `StaticStreamBuffer_t` metadata struct stays in internal SRAM (it's small and the FreeRTOS kernel accesses it frequently). The `overrun_count` tracks how many times the capture task tried to write into a full buffer — this is the subscriber's problem, not the capture task's.

### 3.2 Init sequence

The three-phase init sequence is enforced by the API:

```
1. audio_capture_init()          — configures I2S, zeros subscriber table
2. audio_capture_subscribe(...)  — one call per consumer (inference, recorder, etc.)
3. audio_capture_start()         — enables I2S, spawns capture task — data flows
```

Subscriptions after `start()` return NULL. This is intentional — it avoids runtime list mutation and locking in the hot capture loop. All subscribers must register between `init()` and `start()`.

In `app_main.c` this ordering is wired up:

```c
ESP_ERROR_CHECK(audio_capture_init());
ESP_ERROR_CHECK(inference_init());      // registers the 600-ms window subscriber
// ... any additional subscribers (audio dump, debugging tap, etc.)
ESP_ERROR_CHECK(audio_capture_start()); // spawns capture task; data flows
```

### 3.3 Subscriber registration

The `audio_capture_subscribe()` function allocates a PSRAM-backed StreamBuffer:

```c
// PSRAM allocation
sub->storage = heap_caps_malloc(byte_capacity,
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

// Static StreamBuffer creation
// trigger level = 1: wake xStreamBufferReceive as soon as any data arrives
sub->stream = xStreamBufferCreateStatic(byte_capacity,
                                        1,
                                        sub->storage,
                                        &sub->stream_struct);
```

The inference component subscribes in `inference_init()`:

```c
s_sub = audio_capture_subscribe(FW_INFERENCE_SUB_BYTES, "inference");
```

This gives it a ~20 KB buffer — enough for a full 600 ms window (19 200 B) plus one chunk of scheduling slack.

### 3.4 The capture task

The capture task runs on Core 0 at priority 10. Its inner loop:

1. **Read 256 frames of 32-bit I2S data** via `i2s_channel_read()`. This is a blocking call — it sleeps until the DMA has a full chunk ready.

2. **Convert 32-bit → 16-bit** by shifting right 16:

```c
for (size_t i = 0; i < n_frames; i++) {
    pcm_buf[i] = (int16_t)(i2s_buf[i] >> 16);
}
```

This keeps the top 16 bits of the ICS-43434's 24-bit output. The bottom 8 bits (noise floor) are discarded. This matches Cochlea/EARS's proven conversion and produces clean 16-bit PCM that the downstream pipeline expects.

3. **Fan out to every subscriber** with non-blocking sends:

```c
for (size_t i = 0; i < s_sub_count; i++) {
    struct audio_capture_sub *sub = &s_subs[i];
    if (!sub->active) continue;
    size_t sent = xStreamBufferSend(sub->stream,
                                    pcm_buf,
                                    pcm_bytes,
                                    0 /* no wait */);
    if (sent < pcm_bytes) {
        sub->overrun_count++;
    }
}
```

The `0` timeout means: if a subscriber's buffer is full, drop the data for that subscriber only and increment its overrun counter. Other subscribers are unaffected. This is critical — the inference task takes hundreds of milliseconds per pass, so it can't keep up sample-by-sample. Instead, it reads in 600 ms bursts from its StreamBuffer, which absorbs the capture task's continuous output.

4. **Log data rate** once per second for verification. Expected: 32 000 B/s (16 000 Hz × 2 bytes/sample).

---

## 4. Reading audio in the inference task

The inference task runs on Core 1 at priority 5. It reads 600 ms windows from its StreamBuffer subscriber in a fill loop:

```c
size_t filled = 0;
while (filled < FW_INFERENCE_WINDOW_BYTES) {
    size_t n = audio_capture_read(
        s_sub,
        reinterpret_cast<uint8_t *>(s_window_pcm) + filled,
        FW_INFERENCE_WINDOW_BYTES - filled,
        portMAX_DELAY);
    if (n == 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
    }
    filled += n;
}
```

`FW_INFERENCE_WINDOW_BYTES` is `9 600 × 2 = 19 200` bytes. The `portMAX_DELAY` means this blocks until data is available. The loop accumulates partial reads until 600 ms of audio is buffered in `s_window_pcm` — a ~19 KB PSRAM allocation.

### 4.1 Note on overlapping windows for streaming

For continuous letter spelling (a user dictating "C-A-T"), letter onsets that fall near a window boundary risk being split across two windows, which can lower confidence on both. The simplest mitigation is overlapping windows: instead of reading 600 ms back-to-back, slide the window forward by 200–300 ms each pass and re-use the trailing samples from the previous window. This costs a small ring-buffer in PSRAM and one extra `memcpy` per pass.

The downstream sequence-to-word post-processor will already see top-K probabilities for each window, so it can deduplicate adjacent windows that vote for the same letter. Whether to add overlap in firmware vs. let the post-processor handle boundary effects is a tuning decision; start without overlap and add it only if accuracy on boundary letters is poor.

---

## 5. RMS energy gate

Before running the model, the inference task checks whether the audio window contains enough energy to be worth classifying. This gate exists because the DS-CNN was trained on audio segments that contained spoken letters; feeding the model silence or low-level ambient noise — audio it never saw during training — produces uninformative softmax distributions and noisy top-K events. Skipping silent windows also saves substantial CPU cycles and power.

```c
{
    const int n_samples = FW_INFERENCE_WINDOW_SAMPLES;  // 9 600
    double sum_sq = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double s = static_cast<double>(s_window_pcm[i]) / 32768.0;
        sum_sq += s * s;
    }
    float rms_db = (sum_sq < 1e-20)
        ? -100.0f
        : 10.0f * log10f(static_cast<float>(sum_sq / n_samples));

    if (rms_db < FW_ENERGY_GATE_DB) {
        s_gate_skip_count++;
        if (s_gate_skip_count <= 3 || s_gate_skip_count % 30 == 0) {
            ESP_LOGD(TAG, "energy gate: %.1f dBFS < %.1f — skipping "
                          "(#%" PRIu32 " skips)",
                     rms_db, FW_ENERGY_GATE_DB, s_gate_skip_count);
        }
        continue;
    }
}
```

```c
#define FW_ENERGY_GATE_DB          (-45.0f)  // starting point, retune for letters
```

The math: normalize each int16 sample to [-1.0, 1.0], compute the mean of squares, convert to dBFS via `10 * log10(mean_sq)`. This is equivalent to `20 * log10(rms)` and matches the training pipeline's `compute_rms_db()` function. The log-throttled debug output avoids flooding the serial console during long periods of silence.

The −45 dBFS threshold is inherited from EARS and is a sensible starting point. Spoken letters at conversational distance typically run −30 to −15 dBFS. If users speak softly or far from the mic, lower the gate (e.g. to −50 dBFS); if there's significant ambient HVAC/fan noise, raise it. The training pipeline must use the same threshold when curating segments, otherwise the model sees a different audio distribution than the firmware feeds it.

---

## 6. Log-mel spectrogram

The log-mel spectrogram transforms 600 ms of raw PCM into the 58×40 feature map that the DS-CNN expects as input. This is implemented in `log_mel.c` using esp-dsp's SIMD-accelerated FFT.

### 6.1 Spectrogram parameters

All parameters must match the training pipeline exactly:

```c
#define FW_MEL_N_FFT               512    // FFT size
#define FW_MEL_HOP_LENGTH          160    // 10 ms hop @ 16 kHz
#define FW_MEL_WIN_LENGTH          400    // 25 ms window @ 16 kHz
#define FW_MEL_N_MELS              40     // number of mel filter bands
#define FW_MEL_N_FRAMES            58     // floor((9600 - 400) / 160) + 1
#define FW_MEL_FMIN                0.0f   // mel filter low edge
#define FW_MEL_FMAX                8000.0f // mel filter high edge (Nyquist)
```

58 frames come from: `floor((9 600 − 400) / 160) + 1 = 58`. Each frame is a 25 ms Hann-windowed segment, hopped forward by 10 ms, zero-padded to 512 samples, FFT'd, and projected onto 40 triangular mel-scaled filters.

If you extend the inference window: 700 ms → 11 200 samples → `floor((11 200 − 400) / 160) + 1 = 68` frames; 750 ms → 12 000 samples → 73 frames. The mel filterbank itself is independent of `n_frames` and does not change.

### 6.2 Memory allocation strategy

The log-mel workspace is carefully split between internal SRAM and PSRAM based on access patterns:

```c
// Hann window — 1.6 KB, internal SRAM for fast windowing loop.
s_hann_window = heap_caps_malloc(
    FW_MEL_WIN_LENGTH * sizeof(float),
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

// Mel filterbank — 40 × 257 floats ≈ 40 KB → PSRAM (too large for SRAM).
s_mel_filterbank = heap_caps_malloc(
    FW_MEL_N_MELS * N_FFT_BINS * sizeof(float),
    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

// Sparse filterbank bounds — 320 B, internal SRAM.
s_mel_bounds = heap_caps_malloc(
    FW_MEL_N_MELS * sizeof(mel_bound_t),
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

// FFT interleaved buffer — 4 KB, internal SRAM.
s_fft_buf = heap_caps_malloc(
    FW_MEL_N_FFT * 2 * sizeof(float),
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
```

| Buffer | Size | Location | Why |
|--------|------|----------|-----|
| `s_fft_buf` | 4 096 B | Internal SRAM | `dsps_fft2r_fc32` does random butterfly accesses — PSRAM's SPI latency kills performance (on EARS this single move sped log-mel up 3.7×) |
| `s_hann_window` | 1 600 B | Internal SRAM | Sequential but accessed once per frame (58×) in the inner loop |
| `s_mel_bounds` | 320 B | Internal SRAM | Tight inner-loop bounds lookup |
| `s_mel_filterbank` | ~40 KB | PSRAM | Too large for SRAM, sequential access pattern tolerates SPI latency |

Lesson from EARS: moving the FFT buffer from PSRAM to internal SRAM was the single biggest performance win — log-mel computation dropped from 184 ms to 50 ms on a 98-frame window from changing two allocation flags. For the smaller 58-frame window the gap is proportionally smaller in absolute terms but the same factor in relative terms; keep this allocation split.

### 6.3 FFT computation

The per-frame FFT loop in `log_mel_compute()` runs 58 times per inference (once per frame).

**Step 1 — Window and load into interleaved Re/Im buffer:**

```c
for (int i = 0; i < FW_MEL_N_FFT; i++) {
    if (i < FW_MEL_WIN_LENGTH && (start + i) < n_samples) {
        // Normalize int16 → [-1.0, 1.0] and apply Hann window
        s_fft_buf[2 * i]     = ((float)pcm[start + i] / 32768.0f)
                               * s_hann_window[i];
    } else {
        s_fft_buf[2 * i]     = 0.0f;  // zero-pad beyond window
    }
    s_fft_buf[2 * i + 1] = 0.0f;      // imaginary = 0
}
```

The esp-dsp FFT expects interleaved `[Re0, Im0, Re1, Im1, ...]` format. The first 400 samples get Hann-windowed; samples 400–511 are zero-padded.

**Step 2 — In-place FFT:**

```c
dsps_fft2r_fc32(s_fft_buf, FW_MEL_N_FFT);
dsps_bit_rev_fc32(s_fft_buf, FW_MEL_N_FFT);
```

`dsps_fft2r_fc32` is esp-dsp's radix-2 float32 FFT, which uses SIMD butterfly operations on the ESP32-S3. The bit-reversal permutation is a separate call. Twiddle factors are precomputed once during `log_mel_init()` via `dsps_fft2r_init_fc32(NULL, 512)`.

**Step 3 — Power spectrum:**

```c
float power[N_FFT_BINS];  // 257 bins (DC through Nyquist)
for (int k = 0; k < N_FFT_BINS; k++) {
    float re = s_fft_buf[2 * k];
    float im = s_fft_buf[2 * k + 1];
    power[k] = re * re + im * im;
}
```

### 6.4 Mel filterbank with sparse bounds

The mel filterbank applies 40 triangular filters to the 257 FFT magnitude bins. Each triangular filter only covers ~10–30 bins, but a naive implementation iterates all 257 bins per filter. The sparse bounds optimization precomputes the non-zero range for each filter.

The `mel_bound_t` struct and bounds array:

```c
typedef struct { int lo; int hi; } mel_bound_t;
static mel_bound_t *s_mel_bounds = NULL; // [FW_MEL_N_MELS]
```

These bounds are computed during `build_mel_filterbank()` alongside the filterbank weights. For each mel band, `lo` is the first non-zero bin and `hi` is one past the last non-zero bin.

The optimized inner loop:

```c
for (int m = 0; m < FW_MEL_N_MELS; m++) {
    float energy = 0.0f;
    const float *filt = &s_mel_filterbank[m * N_FFT_BINS];
    const int lo = s_mel_bounds[m].lo;
    const int hi = s_mel_bounds[m].hi;
    for (int k = lo; k < hi; k++) {
        energy += filt[k] * power[k];
    }
    out[frame * FW_MEL_N_MELS + m] = logf(fmaxf(energy, floor_val));
}
```

Instead of 40 × 257 = 10 280 multiply-accumulate operations per frame, this does roughly 40 × 20 = 800 — a ~12× reduction in the inner loop. The `floor_val` (1e-10) prevents `log(0)`.

---

## 7. DS-CNN model and TFLite Micro interpreter

### 7.1 Model overview

The DS-CNN (Depthwise Separable Convolutional Neural Network) is an int8-quantized TFLite model (~60 KB) trained for 26-way letter classification. It takes a 58×40 log-mel spectrogram as input and produces a 26-element softmax distribution over the letters A–Z (index 0 = 'A', ..., index 25 = 'Z'). The architecture uses depthwise separable convolutions, which factor a standard convolution into a depthwise convolution (per-channel) followed by a pointwise 1×1 convolution, dramatically reducing parameter count and compute.

The training pipeline reports 95% top-1 accuracy on its evaluation distribution. On-device accuracy will depend on mic placement, ambient noise, and how closely the user's speech matches the training data — measure and tune with realistic samples before judging the model.

The model is stored in a dedicated flash partition labeled `"model"`:

```c
#define FW_MODEL_PARTITION_LABEL   "model"
```

### 7.2 Op resolver

The DS-CNN uses 9 TFLite operations. Note the `Logistic` (sigmoid) op from the bark detector is gone, replaced by `Softmax` for the multi-class head:

```c
static tflite::MicroMutableOpResolver<9> s_resolver;

// DS-CNN ops:
s_resolver.AddConv2D();
s_resolver.AddDepthwiseConv2D();
s_resolver.AddFullyConnected();
s_resolver.AddAveragePool2D();
s_resolver.AddMean();
s_resolver.AddReshape();
s_resolver.AddSoftmax();           // 26-class output
s_resolver.AddQuantize();
s_resolver.AddDequantize();
```

Using `MicroMutableOpResolver<9>` instead of `AllOpsResolver` keeps the binary small — only the kernels needed for this specific model are linked.

If your model exporter emits raw logits (no in-graph Softmax) you can drop `AddSoftmax()` and apply softmax in firmware after dequantize. Both forms work; the in-graph variant is slightly faster because the kernel runs on int8 with esp-nn.

### 7.3 Model loading from flash

The model is loaded from a custom data partition in `load_model_from_partition()`:

```c
const esp_partition_t *part = esp_partition_find_first(
    ESP_PARTITION_TYPE_DATA,
    static_cast<esp_partition_subtype_t>(0x80),  // custom subtype
    FW_MODEL_PARTITION_LABEL);
```

The function checks for an erased (all 0xFF) partition — this handles the case where the device has been flashed but no model has been OTA'd yet. The model bytes are read into a PSRAM buffer, and the TFLite flatbuffer is validated:

```c
const tflite::Model *model = tflite::GetModel(s_model_buf);
if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "invalid TFLite model (version mismatch or corrupt)");
    return false;
}
```

### 7.4 Tensor arena

The interpreter's working memory is allocated in PSRAM:

```c
s_tensor_arena = static_cast<uint8_t *>(
    heap_caps_malloc(FW_TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM));
```

```c
#define FW_TENSOR_ARENA_SIZE       (512 * 1024)  // PSRAM
```

The 512 KB allocation is conservative — the actual usage depends on the model's largest intermediate tensor. With a 58×40 input (vs. EARS' 98×40), the early-stage activations are ~40% smaller, so arena usage will be lower than EARS' ~278 KB. Plan for 150–250 KB in practice. Once `AllocateTensors()` succeeds, the interpreter logs its actual usage:

```c
size_t arena_used = s_interpreter->arena_used_bytes();
ESP_LOGI(TAG, "interpreter ready — arena used %u / %d B (%.0f%% of alloc)",
         (unsigned)arena_used, FW_TENSOR_ARENA_SIZE,
         100.0f * arena_used / FW_TENSOR_ARENA_SIZE);
```

The arena lives in PSRAM because hundreds of kilobytes of activations far exceed the ESP32-S3's available contiguous internal SRAM. This is the primary reason DS-CNN inference takes hundreds of milliseconds rather than the 30–60 ms cited for very small DS-CNN variants whose activations fit in SRAM. After the model is finalized and arena usage is measured, consider tightening `FW_TENSOR_ARENA_SIZE` to actual_usage + ~10% slack to leave more PSRAM for other components.

### 7.5 Quantization and dequantization

The model uses per-tensor int8 quantization. Before invoke, the float mel spectrogram (58 × 40 = 2 320 floats) is quantized to match the input tensor's scale and zero point:

```c
if (s_input->type == kTfLiteInt8) {
    float scale = s_input->params.scale;
    int32_t zero_point = s_input->params.zero_point;
    int8_t *input_data = s_input->data.int8;

    for (int i = 0; i < FW_MEL_N_FRAMES * FW_MEL_N_MELS; i++) {
        int32_t q = static_cast<int32_t>(
            roundf(s_mel_spec[i] / scale)) + zero_point;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        input_data[i] = static_cast<int8_t>(q);
    }
}
```

After invoke, the 26-element int8 output is dequantized to a float vector:

```c
float probs[26];
if (s_output->type == kTfLiteInt8) {
    float scale = s_output->params.scale;
    int32_t zero_point = s_output->params.zero_point;
    for (int i = 0; i < 26; i++) {
        probs[i] = (s_output->data.int8[i] - zero_point) * scale;
    }
}
```

If the model includes an in-graph `Softmax`, `probs[]` is already a probability distribution (non-negative, sums to ~1.0). If the model emits raw logits, apply softmax in firmware:

```c
// Firmware-side softmax (only if model emits raw logits)
float max_logit = probs[0];
for (int i = 1; i < 26; i++) if (probs[i] > max_logit) max_logit = probs[i];
float sum = 0.0f;
for (int i = 0; i < 26; i++) {
    probs[i] = expf(probs[i] - max_logit);  // numerically-stable
    sum += probs[i];
}
for (int i = 0; i < 26; i++) probs[i] /= sum;
```

The quantization parameters (scale and zero_point) are embedded in the TFLite model file and read from the tensor metadata at runtime.

### 7.6 Invoke and top-K event posting

The TFLite Micro forward pass:

```c
int64_t t2 = esp_timer_get_time();
TfLiteStatus status = s_interpreter->Invoke();
int64_t dt_invoke_us = esp_timer_get_time() - t2;
```

Invoke runs the full DS-CNN forward pass using esp-nn's SIMD-accelerated kernels for Conv2D and DepthwiseConv2D. With a 58×40 input on the ESP32-S3 at 240 MHz with the tensor arena in PSRAM, expect roughly 200–300 ms — measure on your final model before relying on a number.

The inference task does NOT make a binary detection decision. Instead it forwards the top-K letter candidates and their probabilities to the downstream sequence-to-word post-processor, which will perform beam search or n-gram decoding over the stream of inferences. Top-K (rather than top-1) is essential because confusable pairs — B/D/E/G/P/T/V/Z (the "rhyming set"), F/S, M/N — are the dominant failure mode for spoken-letter classifiers, and the post-processor's vocabulary prior can disambiguate them.

```c
#define FW_LETTER_TOP_K 3   // tunable: 3 captures most rhyming-set ambiguity

typedef struct {
    uint8_t letter_index;   // 0..25 → 'A'..'Z'
    float   probability;    // softmax output in [0.0, 1.0]
} fw_letter_candidate_t;

typedef struct {
    fw_letter_candidate_t top_k[FW_LETTER_TOP_K];
    uint32_t              invoke_ms;
} fw_letter_topk_event_t;
```

Top-K selection is a partial-sort over 26 elements — not worth a heap, an insertion-sort pass is fastest at this size:

```c
static void top_k_select(const float *probs, int n,
                         fw_letter_candidate_t *out, int k) {
    for (int i = 0; i < k; i++) {
        out[i].letter_index = 255;
        out[i].probability  = -1.0f;
    }
    for (int i = 0; i < n; i++) {
        if (probs[i] <= out[k - 1].probability) continue;
        int j = k - 1;
        while (j > 0 && out[j - 1].probability < probs[i]) {
            out[j] = out[j - 1];
            j--;
        }
        out[j].letter_index = (uint8_t)i;
        out[j].probability  = probs[i];
    }
}
```

Posting the event:

```c
fw_letter_topk_event_t evt;
top_k_select(probs, 26, evt.top_k, FW_LETTER_TOP_K);
evt.invoke_ms = (uint32_t)(dt_invoke_us / 1000);

ESP_LOGI(TAG, "letters: %c=%.2f  %c=%.2f  %c=%.2f  (%lu ms)",
         'A' + evt.top_k[0].letter_index, evt.top_k[0].probability,
         'A' + evt.top_k[1].letter_index, evt.top_k[1].probability,
         'A' + evt.top_k[2].letter_index, evt.top_k[2].probability,
         (unsigned long)evt.invoke_ms);

esp_event_post(FW_INFERENCE_EVENT, FW_EVENT_LETTER_TOP_K,
               &evt, sizeof(evt), 0);
```

Any component can subscribe via `esp_event_handler_register(FW_INFERENCE_EVENT, FW_EVENT_LETTER_TOP_K, handler, ctx)`. The event base and ID are declared in `inference.h`. The energy gate already filters out silent windows, so every posted event corresponds to a window that contained meaningful audio; the post-processor uses the gap pattern (windows posted vs. windows skipped) as a soft signal for word boundaries.

There is no firmware-side confidence threshold — even low-confidence top-1 results carry information when paired with their alternates. The post-processor decides when a sequence is confident enough to commit to a word.

### 7.7 Hot-reload support

The inference task checks a `s_reload_requested` flag at the top of each loop iteration. After an OTA writes a new model to the flash partition, calling `inference_reload_model()` sets this flag. The next inference pass reloads the model and rebuilds the interpreter without restarting the device. This is especially valuable for letter classification: vocabularies and accent distributions vary per deployment, and you'll likely iterate the model many times after the firmware is otherwise stable.

---

## 8. Timing budget

Projected timing for a 58-frame, ~60 KB DS-CNN on ESP32-S3 N16R8 at 240 MHz. These are estimates derived from EARS measurements; remeasure once the model is finalized.

| Stage | Estimated time | Notes |
|-------|----------------|-------|
| Log-mel spectrogram | ~30 ms | 58 frames × 512-pt FFT (esp-dsp SIMD), FFT buffer in internal SRAM. Scales linearly with frames vs. EARS' 50 ms / 98 frames. |
| Quantize | ~1 ms | 2 320 float→int8 conversions |
| TFLite Invoke | ~250 ms | DS-CNN forward pass with arena in PSRAM. Smaller than EARS (360 ms) because of smaller input feature map. |
| Dequantize + softmax + top-K | <1 ms | 26 elements |
| **Total** | **~280 ms** | Comfortably fits the 600 ms window |

A 600 ms window with ~280 ms of compute leaves ~320 ms of headroom. That headroom is what enables overlapping windows (Section 4.1) at 300 ms stride if you decide you want them later — you pay one full inference per 300 ms instead of per 600 ms, doubling CPU duty but still fitting in real time. The energy gate skips compute entirely when the environment is quiet, saving CPU cycles and reducing power.

If your final model is slower than these estimates and inference exceeds the window length, you have three options in roughly increasing pain order: (1) shrink the model, (2) extend the window to 750 ms, (3) accept that frames will be dropped during inference and let the next window pick up — the post-processor sees a sparser sequence but can still decode.

---

## 9. FreeRTOS task topology

| Task | Core | Priority | Stack | Purpose |
|------|------|----------|-------|---------|
| `audio_cap` | 0 | 10 | 4 KB (internal) | I2S read + 32→16 + fan-out |
| `inference` | 1 | 5 | 8 KB (internal) | Energy gate + mel + TFLM invoke + top-K |

The capture task has the highest priority (10) to ensure I2S DMA buffers are drained promptly — if the capture task starves, all downstream subscribers lose audio. The inference task runs at the lowest priority (5) because its 600 ms window provides a large scheduling buffer.

Core assignment keeps the latency-sensitive I2S reads (Core 0) separated from the compute-heavy TFLite invokes (Core 1). Add additional tasks (a sequence-to-word decoder, output transport, debug audio dumper) on whichever core has spare cycles — the inference task on Core 1 is bursty (compute-then-block), so Core 1 has substantial idle time within each 600 ms window.

---

## 10. Lessons carried over from EARS (and why they matter here)

A summary of the EARS firmware decisions that transfer directly:

- **Native I2S driver, not ESP-ADF.** ADF brings esp-dsp version conflicts and pulls in board/codec/speech components that aren't relevant to a single-mic capture device.
- **32-bit I2S read, software shift to 16-bit.** Matches the ICS-43434's actual output and gives clean PCM. Don't try to coerce the driver into 16-bit slots.
- **`I2S_STD_SLOT_LEFT` mask.** With SEL tied to GND, only LEFT-slot frames are valid; without the mask, the driver delivers RIGHT-slot zeros interleaved.
- **Pub/sub fan-out from a single capture task.** All subscribers must register before `audio_capture_start()`; the capture loop never mutates the subscriber list. Locking-free, predictable.
- **Non-blocking `xStreamBufferSend` with per-subscriber overrun counts.** A slow subscriber drops its own data; it cannot stall the capture task or starve other subscribers.
- **PSRAM-backed StreamBuffers, SRAM-resident metadata.** Storage is large and benefits from PSRAM; FreeRTOS metadata is small and accessed from the kernel's hot path.
- **FFT buffer in internal SRAM.** The single biggest log-mel performance win (3.7×). Random butterfly access patterns make PSRAM's SPI latency catastrophic.
- **Sparse mel filterbank bounds.** ~12× speedup in the inner loop with no accuracy change.
- **Energy gate before mel/inference.** Don't waste compute on silence, and don't feed the model audio it never saw during training.
- **Match training-pipeline parameters exactly.** Sample rate, FFT size, hop, window, n_mels, fmin/fmax, RMS gate threshold — any drift between firmware and training degrades accuracy silently. Encode these in `fw_config.h` and reference the same constants from your model export tooling.
- **Hot-reload via partition swap.** Lets you iterate the model without firmware re-flashes — invaluable for a 26-class letter model that you'll retrain often.
- **`MicroMutableOpResolver` with the exact op set.** Keeps the binary small and the link error obvious if you accidentally add a model that uses an unregistered op.
- **Tensor arena in PSRAM.** Activations don't fit in SRAM for any non-trivial DS-CNN. Accept the SPI latency penalty as the price of fitting the model at all.

The new things specific to letter classification — 600 ms windows, 26-class softmax, top-K event payload, no firmware-side confidence threshold — are documented in Sections 4, 6.1, 7.2, and 7.6.

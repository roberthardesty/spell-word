/**
 * @file app_main.c
 * @brief Spell-Word firmware entry point — Day 3 bring-up scaffold.
 *
 * Init order (matters; mirrors the EARS pattern with networking dropped):
 *
 *   1. init_nvs()             — must come first; future config persistence
 *                               will use it, and many IDF subsystems need
 *                               NVS to be initialized even if we don't
 *                               explicitly read keys yet.
 *   2. log_boot_banner()      — proves we got past startup with PSRAM,
 *                               flash, and chip rev visible in the log.
 *   3. check_factory_reset()  — hold BOOT for 5 s at boot to erase NVS.
 *                               Useful as a "model OTA recovery" fallback.
 *   4. audio_capture_init()   — configures I2S RX, zeros sub table.
 *   5. inference_init()       — log-mel + TFLM + utterance queue.
 *   6. segmenter_init()       — subscribes to audio_capture, spawns task.
 *   7. audio_capture_start()  — enables I2S, spawns capture task.
 *                               After this call no further subscribers
 *                               can register.
 *   8. blink_forever()        — heartbeat LED so we know the scheduler
 *                               is still alive while the audio pipeline
 *                               churns. Replaced by a real status_led
 *                               component once the UI state machine
 *                               lands in Phase 4.
 *
 * Sequencing rule from EARS that survives verbatim: ALL audio_capture
 * subscribers must be registered between init() and start(). The capture
 * loop never mutates the subscriber list, so subscribing after start()
 * fails loudly (and would be a memory race anyway).
 */

#include <inttypes.h>

#include "audio_capture.h"
#include "audio_dump.h"
#include "driver/gpio.h"
#include "esp_app_desc.h"
#include "esp_chip_info.h"
#include "esp_event.h"
#include "esp_flash.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "inference.h"
#include "nvs_flash.h"
#include "sdkconfig.h"
#include "segmenter.h"
#include "spell_config.h"
#include "spell_events.h"

static const char *TAG = "spell";

// ---------------------------------------------------------------------------
// NVS — auto-erase on version mismatch / no free pages, same pattern as the
// IDF hello_world template and the EARS POC.
// ---------------------------------------------------------------------------
static void init_nvs(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS needs erasing (err=%s); erasing and retrying",
                 esp_err_to_name(err));
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

static void log_boot_banner(void)
{
    const esp_app_desc_t *app = esp_app_get_description();

    esp_chip_info_t chip;
    esp_chip_info(&chip);

    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);

    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, " Spell-Word firmware booting");
    ESP_LOGI(TAG, "   version:  %s", app->version);
    ESP_LOGI(TAG, "   built:    %s %s", app->date, app->time);
    ESP_LOGI(TAG, "   idf:      %s", app->idf_ver);
    ESP_LOGI(TAG, "   chip:     ESP32-S%d rev v%d.%d, %d core%s",
             chip.model == CHIP_ESP32S3 ? 3 : 0,
             chip.revision / 100, chip.revision % 100,
             chip.cores, chip.cores == 1 ? "" : "s");
    ESP_LOGI(TAG, "   flash:    %" PRIu32 " MB", flash_size / (1024 * 1024));
#if CONFIG_SPIRAM
    ESP_LOGI(TAG, "   psram:    octal, configured via Kconfig");
#else
    ESP_LOGW(TAG, "   psram:    DISABLED — bringup needs it for "
                  "subscriber buffers + tensor arena");
#endif
    ESP_LOGI(TAG, "================================================");
}

// ---------------------------------------------------------------------------
// Factory reset — hold BOOT (GPIO 0) for SPELL_FACTORY_RESET_HOLD_MS at boot
// to erase NVS and reboot. Inherited verbatim from the EARS POC.
// ---------------------------------------------------------------------------
static void check_factory_reset(void)
{
    gpio_config_t btn = {
        .pin_bit_mask = (1ULL << SPELL_FACTORY_RESET_PIN),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&btn);

    if (gpio_get_level(SPELL_FACTORY_RESET_PIN) != 0) {
        return;  // not pressed — normal boot
    }

    ESP_LOGW(TAG, "BOOT button held — hold for %d ms to factory-reset...",
             SPELL_FACTORY_RESET_HOLD_MS);

    const int poll_ms = 100;
    int held_ms = 0;
    while (held_ms < SPELL_FACTORY_RESET_HOLD_MS) {
        vTaskDelay(pdMS_TO_TICKS(poll_ms));
        held_ms += poll_ms;
        if (gpio_get_level(SPELL_FACTORY_RESET_PIN) != 0) {
            ESP_LOGI(TAG, "BOOT released after %d ms — normal boot", held_ms);
            return;
        }
    }

    ESP_LOGW(TAG, "=== FACTORY RESET ===");
    ESP_LOGW(TAG, "Erasing NVS...");
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_LOGW(TAG, "NVS erased. Rebooting...");
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_restart();
}

// ---------------------------------------------------------------------------
// Day-3 bring-up: log every top-K event from the inference task. This is
// the equivalent of EARS' bark-event handler, for letters. Replaced by the
// decoder subscription once Phase 5 lands.
// ---------------------------------------------------------------------------
static void on_letter_top_k(void *arg, esp_event_base_t base,
                            int32_t event_id, void *event_data)
{
    (void)arg; (void)base; (void)event_id;
    const spell_letter_topk_event_t *evt = (const spell_letter_topk_event_t *)event_data;

    ESP_LOGI(TAG, "TOPK  %c=%.2f  %c=%.2f  %c=%.2f  %c=%.2f  %c=%.2f  (%" PRIu32 "ms)",
             'A' + evt->top_k[0].letter_index, evt->top_k[0].probability,
             'A' + evt->top_k[1].letter_index, evt->top_k[1].probability,
             'A' + evt->top_k[2].letter_index, evt->top_k[2].probability,
             'A' + evt->top_k[3].letter_index, evt->top_k[3].probability,
             'A' + evt->top_k[4].letter_index, evt->top_k[4].probability,
             evt->invoke_ms);
}

static void on_end_of_word(void *arg, esp_event_base_t base,
                           int32_t event_id, void *event_data)
{
    (void)arg; (void)base; (void)event_id;
    const spell_eow_event_t *evt = (const spell_eow_event_t *)event_data;
    ESP_LOGI(TAG, "EOW   letters_so_far=%" PRIu32, evt->letter_count);
}

// ---------------------------------------------------------------------------
// Status LED heartbeat. The DevKitC-1 onboard LED on GPIO 48 is actually a
// WS2812; treating it as a plain digital output works as a heartbeat for
// bring-up but produces a different color/no light depending on what the
// LED chip happens to latch. Phase 4's status_led component does this
// properly via RMT.
// ---------------------------------------------------------------------------
static void blink_forever(void)
{
    gpio_reset_pin(SPELL_STATUS_LED_PIN);
    gpio_set_direction(SPELL_STATUS_LED_PIN, GPIO_MODE_OUTPUT);

    int level = 0;
    while (1) {
        gpio_set_level(SPELL_STATUS_LED_PIN, level);
        level = !level;
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void app_main(void)
{
    init_nvs();
    log_boot_banner();
    check_factory_reset();

    // Default event loop — required for esp_event_post / handler register.
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // Bring-up subscriptions. The decoder will replace the TOP_K handler
    // in Phase 5; the EOW handler stays as-is and just changes consumer.
    ESP_ERROR_CHECK(esp_event_handler_register(
        SPELL_INFERENCE_EVENT, SPELL_EVENT_LETTER_TOP_K,
        on_letter_top_k, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        SPELL_SEGMENTER_EVENT, SPELL_EVENT_END_OF_WORD,
        on_end_of_word, NULL));

    // --- Audio pipeline ----------------------------------------------------
    ESP_ERROR_CHECK(audio_capture_init());
    ESP_ERROR_CHECK(inference_init());      // log-mel + TFLM + utterance queue
    ESP_ERROR_CHECK(segmenter_init());      // subscribes to audio_capture
    ESP_ERROR_CHECK(audio_dump_init(10));   // 10 s ring for VAD-failure capture
    ESP_ERROR_CHECK(audio_capture_start()); // I2S enabled; data flows

    // For now, audio_dump_emit_b64() has no firmware-side trigger — call it
    // from a debug hook (button handler, console command, decoder-abstain
    // event handler, etc.) once the UI state machine lands in Phase 4.

    blink_forever();
}

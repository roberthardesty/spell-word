idf_component_register(
    SRCS
        "app_main.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        spell_common
        audio_capture
        letter_classifier
        segmenter
        spell_ui
        audio_dump
        nvs_flash
        driver
        esp_event
        esp_timer
        esp_app_format    # esp_app_desc.h, esp_app_get_description() (IDF 5.x)
)

# Spell-Word firmware — top-level project file.
#
# Targets ESP32-S3 (N16R8 DevKitC-1 for MVP bring-up). Built with ESP-IDF
# v5.3.5 + esp-tflite-micro. The component layout under components/ mirrors
# the structure documented in docs/MVP-Implementation-Plan.md §4.

cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(spell_word)

# After project() returns, the IDF Component Manager has populated
# managed_components/. Apply local patches against vendored dependencies —
# see docs/patches/README.md for the architecture and rationale, and
# cmake/apply_patches.cmake for the apply mechanics.
include(${CMAKE_SOURCE_DIR}/cmake/apply_patches.cmake)

apply_repo_patch(
    LABEL       "esp-nn #53 dispatcher-guard"
    TARGET_FILE "${CMAKE_SOURCE_DIR}/managed_components/espressif__esp-nn/src/convolution/esp_nn_conv_esp32s3.c"
    MARKER      "Vikunja #53"
    PATCH_FILE  "${CMAKE_SOURCE_DIR}/docs/patches/53-esp-nn-conv-dispatcher-guard.patch"
)

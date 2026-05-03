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

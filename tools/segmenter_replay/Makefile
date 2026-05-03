# Host build for the segmenter VAD — links the same segmenter_core.c that
# runs on-device, against a WAV reader and a small CLI driver.
#
# Build:
#   make
# Run (defaults):
#   ./segmenter_replay path/to/recording.wav
# Tweak thresholds:
#   ./segmenter_replay --on -36 --off -40 --eow 1500 path/to/recording.wav
# Run unit tests:
#   make test

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
LDFLAGS ?= -lm

CORE_DIR := ../../components/segmenter
INCLUDES := -I$(CORE_DIR)/include

SRCS := main.c wav.c $(CORE_DIR)/segmenter_core.c
OBJS := $(SRCS:.c=.o)

TEST_SRCS := test_segmenter_core.c $(CORE_DIR)/segmenter_core.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: segmenter_replay

segmenter_replay: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_segmenter_core: $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: test_segmenter_core
	./test_segmenter_core

clean:
	rm -f segmenter_replay test_segmenter_core $(OBJS) $(TEST_OBJS)

.PHONY: all clean test

# Host build for the word decoder — links the same decoder_core.c that
# runs on-device. Builds two binaries:
#
#   decoder_replay        CSV fixture replay CLI (Vikunja #24)
#   test_decoder_core     unit-test rig          (Vikunja #20, #21)
#
# Build everything:
#   make
# Build + run unit tests:
#   make test
# Run the CLI against the bundled sample:
#   ./decoder_replay --dict fixtures/sample_dict.txt fixtures/sample.csv

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
LDFLAGS ?= -lm

CORE_DIR   := ../../components/decoder
COMMON_DIR := ../../components/spell_common
INCLUDES   := -I$(CORE_DIR)/include -I$(COMMON_DIR)/include

CLI_SRCS  := main.c $(CORE_DIR)/decoder_core.c
CLI_OBJS  := $(CLI_SRCS:.c=.o)

TEST_SRCS := test_decoder_core.c $(CORE_DIR)/decoder_core.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: decoder_replay test_decoder_core

decoder_replay: $(CLI_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_decoder_core: $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: test_decoder_core
	./test_decoder_core

clean:
	rm -f decoder_replay test_decoder_core $(CLI_OBJS) $(TEST_OBJS)

.PHONY: all clean test

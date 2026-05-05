# Host build for the W-recovery trigger predicate — links the same
# w_detector_core.c that runs on-device. Builds two binaries:
#
#   w_detector_replay     CSV fixture replay CLI (Vikunja #29)
#   test_w_detector_core  unit-test rig          (Vikunja #28)
#
# Build everything:
#   make
# Build + run unit tests:
#   make test
# Run the CLI against the bundled sample:
#   ./w_detector_replay fixtures/sample.csv

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
LDFLAGS ?=

CORE_DIR   := ../../components/letter_recognizer
COMMON_DIR := ../../components/spell_common
INCLUDES   := -I$(CORE_DIR)/include -I$(COMMON_DIR)/include

CLI_SRCS  := main.c $(CORE_DIR)/w_detector_core.c
CLI_OBJS  := $(CLI_SRCS:.c=.o)

TEST_SRCS := test_w_detector_core.c $(CORE_DIR)/w_detector_core.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: w_detector_replay test_w_detector_core

w_detector_replay: $(CLI_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_w_detector_core: $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: test_w_detector_core
	./test_w_detector_core

clean:
	rm -f w_detector_replay test_w_detector_core $(CLI_OBJS) $(TEST_OBJS)

.PHONY: all clean test

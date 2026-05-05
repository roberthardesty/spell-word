# Host build for the W-recovery trigger predicate — links the same
# w_detector_core.c that runs on-device. Currently builds the unit-test
# rig only; the CSV fixture replay CLI (`w_detector_replay` binary,
# `main.c`) lands with Vikunja #29.
#
# Build the test rig + run:
#   make test
# Build only:
#   make

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
LDFLAGS ?=

CORE_DIR   := ../../components/letter_recognizer
COMMON_DIR := ../../components/spell_common
INCLUDES   := -I$(CORE_DIR)/include -I$(COMMON_DIR)/include

TEST_SRCS := test_w_detector_core.c $(CORE_DIR)/w_detector_core.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: test_w_detector_core

test_w_detector_core: $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: test_w_detector_core
	./test_w_detector_core

clean:
	rm -f test_w_detector_core $(TEST_OBJS)

.PHONY: all clean test

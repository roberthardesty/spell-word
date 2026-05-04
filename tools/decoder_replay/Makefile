# Host build for the word decoder — links the same decoder_core.c that
# runs on-device. Currently builds the unit-test rig only; the CSV fixture
# replay CLI (`decoder_replay` binary, `main.c`) lands with Vikunja #24.
#
# Build the test rig + run:
#   make test
# Build only:
#   make

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
LDFLAGS ?= -lm

CORE_DIR   := ../../components/decoder
COMMON_DIR := ../../components/spell_common
INCLUDES   := -I$(CORE_DIR)/include -I$(COMMON_DIR)/include

TEST_SRCS := test_decoder_core.c $(CORE_DIR)/decoder_core.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: test_decoder_core

test_decoder_core: $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: test_decoder_core
	./test_decoder_core

clean:
	rm -f test_decoder_core $(TEST_OBJS)

.PHONY: all clean test

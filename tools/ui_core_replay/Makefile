# Host build for the spell_ui transition table — links the same ui_core.c
# that runs on-device against a small unit-test driver.
#
# Build:
#   make
# Run unit tests:
#   make test

CC      ?= cc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
LDFLAGS ?=

CORE_DIR     := ../../components/spell_ui
COMMON_DIR   := ../../components/spell_common
INCLUDES     := -I$(CORE_DIR)/include -I$(COMMON_DIR)/include

# spell_events.h pulls esp_event.h on-device. Host build supplies a tiny
# shim so ui_core's include of <spell_events.h> resolves without IDF.
SHIM_INCLUDES := -Ishims

TEST_SRCS := test_ui_core.c $(CORE_DIR)/ui_core.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: test_ui_core

test_ui_core: $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(SHIM_INCLUDES) $(INCLUDES) -c -o $@ $<

test: test_ui_core
	./test_ui_core

clean:
	rm -f test_ui_core $(TEST_OBJS)

.PHONY: all clean test

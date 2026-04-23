# ============================================================================
# libkser Makefile - Serenity Serialization Format
# ============================================================================

CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC -I./include
LDFLAGS = -shared
AR = ar
ARFLAGS = rcs

# Source files
SRCS = src/kser_write.c src/kser_read.c src/kser_quantize.c src/kser_checksum.c
OBJS = $(SRCS:.c=.o)

# Targets
STATIC_LIB = libkser.a
SHARED_LIB = libkser.so

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHARED_LIB = libkser.dylib
    LDFLAGS = -dynamiclib
endif

.PHONY: all clean test install

all: $(STATIC_LIB) $(SHARED_LIB)

# Static library
$(STATIC_LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

# Shared library
$(SHARED_LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Tests
test: $(STATIC_LIB)
	$(CC) $(CFLAGS) -o tests/test_roundtrip tests/test_roundtrip.c $(STATIC_LIB) -lm
	./tests/test_roundtrip

# Install (system-wide)
install: all
	mkdir -p /usr/local/lib /usr/local/include
	cp $(STATIC_LIB) /usr/local/lib/
	cp $(SHARED_LIB) /usr/local/lib/
	cp include/kser.h /usr/local/include/

# Clean
clean:
	rm -f $(OBJS) $(STATIC_LIB) $(SHARED_LIB)
	rm -f tests/test_roundtrip tests/*.ser

# Debug build
debug: CFLAGS = -Wall -Wextra -g -O0 -I./include -DDEBUG
debug: all

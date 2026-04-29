# ============================================================================
# libkser Makefile - Serenity Serialization Format
# ============================================================================

CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC -I./include
AR = ar
ARFLAGS = rcs

# Source files
SRCS = src/kser_write.c src/kser_read.c src/kser_quantize.c src/kser_checksum.c
OBJS = $(SRCS:.c=.o)

# Targets
STATIC_LIB = libkser.a

.PHONY: all clean test install

all: $(STATIC_LIB)

# Static library (copiée dans le répertoire principal)
$(STATIC_LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^
	@cp $@ ../../
	@echo "✓ $(STATIC_LIB) prête (copiée dans ../../)"

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
	cp include/kser.h /usr/local/include/

# Clean
clean:
	rm -f $(OBJS) $(STATIC_LIB)
	rm -f tests/test_roundtrip tests/*.ser

# Debug build
debug: CFLAGS = -Wall -Wextra -g -O0 -I./include -DDEBUG
debug: all

# Makefile — libtrain.a (Training Library)
# Zero-dependency training library for k-mamba

CC = gcc
CFLAGS = -O3 -mavx2 -Wall -Wextra -fPIC -I./include -I../../include
LDFLAGS = -lm -lgomp

SRCS = src/trainer.c
OBJS = $(SRCS:.c=.o)

TARGET = libtrain.a

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^
	@cp $@ ../../
	@echo "✓ $(TARGET) prête (copiée dans ../../)"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

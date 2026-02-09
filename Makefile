CC = gcc
# enable OpenMP when available for parallel selective_scan
CFLAGS = -Wall -Wextra -O2 -std=c99 -lm -fopenmp
LDFLAGS = -lm -fopenmp

# Source files
SOURCES = mamba.c main.c
ADVANCED_SOURCES = mamba.c advanced_example.c
MILLION_SOURCES = mamba.c million_params.c
OBJECTS = $(SOURCES:.c=.o)
ADVANCED_OBJECTS = $(ADVANCED_SOURCES:.c=.o)
MILLION_OBJECTS = $(MILLION_SOURCES:.c=.o)
EXECUTABLE = mamba_demo
ADVANCED_EXECUTABLE = mamba_advanced
TRAIN_EXECUTABLE = mamba_train
MILLION_EXECUTABLE = mamba_million

# Default target
all: $(EXECUTABLE) $(ADVANCED_EXECUTABLE) $(MILLION_EXECUTABLE)

# Build training executable
$(TRAIN_EXECUTABLE): mamba.o train.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build basic executable
$(EXECUTABLE): mamba.o main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build advanced executable
$(ADVANCED_EXECUTABLE): mamba.o advanced_example.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build 1M parameter executable
$(MILLION_EXECUTABLE): mamba.o million_params.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(ADVANCED_OBJECTS) $(MILLION_OBJECTS) $(EXECUTABLE) $(ADVANCED_EXECUTABLE) $(TRAIN_EXECUTABLE) $(MILLION_EXECUTABLE) *.o

# Run the basic program
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run the advanced program
run-advanced: $(ADVANCED_EXECUTABLE)
	./$(ADVANCED_EXECUTABLE)

run-train: $(TRAIN_EXECUTABLE)
	./$(TRAIN_EXECUTABLE)

run-million: $(MILLION_EXECUTABLE)
	./$(MILLION_EXECUTABLE)

# Rebuild everything
rebuild: clean all

# Help target
help:
	@echo "Mamba State Space Model in C - Build Commands:"
	@echo "  make              - Build both executables"
	@echo "  make run          - Build and run the basic demo"
	@echo "  make run-advanced - Build and run advanced examples"
	@echo "  make run-train    - Build and run training example"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make help         - Show this help message"

.PHONY: all clean run run-advanced rebuild help

.PHONY: run-train


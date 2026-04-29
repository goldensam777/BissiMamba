# Makefile maître — k-mamba
# Structure inspirée de optimus/cpu/Makefile
#
# Usage:
#   make all              # Tout compiler (lib + modèles)
#   make lib              # Juste la bibliothèque
#   make k-mamba-train    # CLI d'entraînement
#   make all_models       # Tous les modèles
#   make tests            # Tests unitaires
#   make clean            # Nettoyage

.PHONY: all lib cpu cuda models all_models tests clean distclean help inference

# ═══════════════════════════════════════════════════════════════
# Compilateurs et flags
# ═══════════════════════════════════════════════════════════════
CC = gcc
CFLAGS = -O3 -mavx2 -Wall -Wextra -I./include -Ilibs/train_set/include -fopenmp
LDFLAGS = -lm -lgomp

ifdef FAST_EXP
CFLAGS += -DKMAMBA_FAST_EXP_APPROX
endif

# ═══════════════════════════════════════════════════════════════
# Rust Tokenizer
# ═══════════════════════════════════════════════════════════════
RUST_DIR = tokenizer_rs
RUST_LIB = $(RUST_DIR)/target/release/libkmamba_tokenizer.a
RUST_LDFLAGS = -lrt -ldl -lpthread

CARGO := $(shell which cargo 2>/dev/null)
RUST_AVAILABLE := $(if $(CARGO),1,0)

# ═══════════════════════════════════════════════════════════════
# CUDA Auto-Detection
# ═══════════════════════════════════════════════════════════════
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_AVAILABLE := $(if $(NVCC),1,0)

ifdef CPU_ONLY
CUDA_AVAILABLE := 0
endif

ifeq ($(CUDA_AVAILABLE),1)
CUDA_HOME ?= $(dir $(NVCC))..
CUDA_FLAGS = -O3 -arch=sm_70 -I./include -Ilibs/train_set/include -I$(CUDA_HOME)/include -DKMAMBA_BUILD_CUDA
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CFLAGS += -DKMAMBA_BUILD_CUDA
endif

# ═══════════════════════════════════════════════════════════════
# libkser Integration
KSER_DIR = libs/kser
KSER_LIB = $(KSER_DIR)/libkser.a
KSER_LDFLAGS = -L$(KSER_DIR) -lkser -Wl,-rpath,$(KSER_DIR)
CFLAGS += -I$(KSER_DIR)/include

# ═══════════════════════════════════════════════════════════════
# libtrain Integration
TRAIN_DIR = libs/train_set
TRAIN_LIB = $(TRAIN_DIR)/libtrain.a
TRAIN_LDFLAGS = -L$(TRAIN_DIR) -ltrain -Wl,-rpath,$(TRAIN_DIR)
CFLAGS += -I$(TRAIN_DIR)/include

# ═══════════════════════════════════════════════════════════════
# Fichiers source
# ═══════════════════════════════════════════════════════════════
SRCS = src/kmamba.c \
       src/mamba_block.c \
       src/kmamba_cuda_utils.c \
       src/kmamba_mixed_precision.c \
       src/kmamba_distributed.c \
       src/km_topology.c \
       src/wavefront_nd.c \
       src/wavefront_plan.c \
       src/scan_nd.c \
       src/convnd.c \
       src/km_memory_pool.c \
       src/kmamba_ser.c \
       kernels/gemm_f32.c \
       kernels/activations_f32.c \
       kernels/elementwise_f32.c \
       kernels/optimizer_f32.c \
       kernels/init_f32.c

CUDA_SRCS = cuda/scan_nd.cu \
            cuda/convnd.cu \
            cuda/convnd_separable.cu \
            cuda/mamba_block.cu \
            cuda/kmamba_cuda_utils.cu \
            cuda/kmamba_mixed_precision.cu \
            cuda/kmamba_distributed.cu \
            cuda/kmamba_kernels.cu

# ═══════════════════════════════════════════════════════════════
# Objets et cibles
# ═══════════════════════════════════════════════════════════════
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(patsubst %.cu,cuda/%.o,$(notdir $(CUDA_SRCS)))

TARGET = libkmamba.a

# k-mamba-train (CLI principal)
K_MAMBA_TRAIN_OBJS = train.o src/configs.o
MODEL_OBJS = model.o src/configs.o
TRAIN_OBJS = train.o src/configs.o

# Runtime bundle export
BUNDLE_DIR ?= dist/runtime
BUNDLE_CONFIG ?= configs/cifar10.json

# ═══════════════════════════════════════════════════════════════
# Cibles principales (cpu / cuda / all)
# ═══════════════════════════════════════════════════════════════

# Compilation CPU uniquement (lib + tests CPU)
cpu: lib k-mamba-train
	@echo "=== Compilation CPU terminée ==="
	@echo "Bibliothèque: $(TARGET)"
	@echo "CLI: k-mamba-train"
	@echo "Tests CPU: make tests"

# Compilation CUDA (lib + tests GPU si disponible)
cuda: lib k-mamba-train
ifeq ($(CUDA_AVAILABLE),1)
	@echo "=== Compilation CUDA terminée ==="
	@echo "Bibliothèque: $(TARGET)"
	@echo "CLI: k-mamba-train"
	@$(MAKE) bench-convnd-cuda test-convnd-separable-cuda 2>/dev/null || true
else
	@echo "✗ CUDA non disponible - impossible de compiler les tests GPU"
	@exit 1
endif

# Tout compiler (lib + CLI)
all: lib k-mamba-train
	@echo "=== Compilation complète terminée ==="

# Juste la bibliothèque
lib: check_cuda check_rust $(KSER_LIB) $(TRAIN_LIB) $(RUST_LIB) $(TARGET)
	@echo ""
	@echo "=== libkmamba.a prête ==="

# libtrain static library (auto-copiée via son Makefile)
$(TRAIN_LIB):
	@echo "Building libtrain..."
	@$(MAKE) -C $(TRAIN_DIR)

# libkser static library (auto-copiée via son Makefile)
$(KSER_LIB):
	@echo "Building libkser..."
	@$(MAKE) -C $(KSER_DIR)

# Tous les modèles
models: all_models

# Tous les modèles (uniquement k-mamba-train maintenant)
all_models: k-mamba-train
	@echo "✓ CLI k-mamba-train prêt"

# Tests
tests: lib
	@echo "=== Compilation des tests ==="
	@$(MAKE) test-mamba3
	@$(MAKE) test-gemm-atb-determinism
	@$(MAKE) test-scan-nd-regression
ifeq ($(CUDA_AVAILABLE),1)
	@$(MAKE) test-mamba3-gpu
endif

# ═══════════════════════════════════════════════════════════════
# Compilation de la lib
# ═══════════════════════════════════════════════════════════════

$(TARGET): $(OBJS) $(CUDA_OBJS)
	ar rcs $@ $^
	@echo "✓ $(TARGET) prête"

# C files (always use gcc, not nvcc)
%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# CUDA files
ifeq ($(CUDA_AVAILABLE),1)
cuda/%.o: cuda/%.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
endif

# Rust tokenizer
$(RUST_LIB):
ifeq ($(RUST_AVAILABLE),1)
	cd $(RUST_DIR) && cargo build --release
	@echo "✓ Tokenizer Rust compilé"
else
	@echo "✗ Rust indisponible — tokenizer ignoré"
endif

# ═══════════════════════════════════════════════════════════════
# Compilation des modèles
# ═══════════════════════════════════════════════════════════════

# Lien des libs pour k-mamba-train
APP_LDFLAGS = $(TARGET) $(KSER_LIB) $(TRAIN_LIB) $(LDFLAGS)
ifeq ($(RUST_AVAILABLE),1)
APP_LDFLAGS += $(RUST_LIB) $(RUST_LDFLAGS)
endif
ifeq ($(CUDA_AVAILABLE),1)
APP_LDFLAGS += $(CUDA_LDFLAGS)
endif

k-mamba-train: $(K_MAMBA_TRAIN_OBJS) $(TARGET) $(KSER_LIB) $(TRAIN_LIB) $(RUST_LIB)
	$(CC) $(CFLAGS) -o $@ $(K_MAMBA_TRAIN_OBJS) $(APP_LDFLAGS)
	@echo "Built: $@ (K-Mamba Training CLI)"

model: $(MODEL_OBJS) $(TARGET) $(KSER_LIB) $(TRAIN_LIB) $(RUST_LIB)
	$(CC) $(CFLAGS) -o $@ $(MODEL_OBJS) $(APP_LDFLAGS)
	@echo "Built: $@ (Model serialization CLI)"

train: $(TRAIN_OBJS) $(TARGET) $(KSER_LIB) $(TRAIN_LIB) $(RUST_LIB)
	$(CC) $(CFLAGS) -o $@ $(TRAIN_OBJS) $(APP_LDFLAGS)
	@echo "Built: $@ (Standalone training CLI)"

# ═══════════════════════════════════════════════════════════════
# Inference package — minimal deployment folder
# Usage: make inference
# Result: inference/ folder with model, train binaries and config template
# ═══════════════════════════════════════════════════════════════
inference: lib model train
	@mkdir -p inference
	@cp model train inference/
	@if [ ! -f inference/config.json ]; then \
		cp $(BUNDLE_CONFIG) inference/config.json; \
		echo "✓ Created inference/config.json (template)"; \
	fi
	@echo "✓ Inference package ready in inference/"
	@echo "  1. Edit inference/config.json with your model settings"
	@echo "  2. cd inference && ./model --config config.json --serialize ser"
	@echo "  3. cd inference && ./train --config config.json --data <dataset_dir> --epochs=10"

# Note: Les modèles standalone (cpu/cuda/azure/vision) ont été supprimés.
# Utiliser k-mamba-train avec les presets de config appropriés.

# ═══════════════════════════════════════════════════════════════
# Vérifications
# ═══════════════════════════════════════════════════════════════

check_cuda:
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ CUDA: $(NVCC)"
else
	@echo "✗ CUDA non détecté"
endif

check_rust:
ifeq ($(RUST_AVAILABLE),1)
	@echo "✓ Rust/Cargo disponible"
else
	@echo "✗ Rust indisponible"
endif

# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

test-mamba3: $(TARGET) tests/unit/test_mamba3_forward.c
ifeq ($(CUDA_AVAILABLE),1)
	$(CC) $(CFLAGS) -o test_mamba3 tests/unit/test_mamba3_forward.c $(TARGET) $(LDFLAGS) $(CUDA_LDFLAGS)
else
	$(CC) $(CFLAGS) -o test_mamba3 tests/unit/test_mamba3_forward.c $(TARGET) $(LDFLAGS)
endif
	./test_mamba3

test-mamba3-gpu: $(TARGET) tests/unit/test_mamba3_gpu.cu
ifeq ($(CUDA_AVAILABLE),1)
	$(NVCC) -O3 -arch=sm_70 -I./include -I$(CUDA_HOME)/include -o test_mamba3_gpu tests/unit/test_mamba3_gpu.cu $(TARGET) -L$(CUDA_HOME)/lib64 -lcudart -lcublas -Xcompiler -fopenmp -lgomp
	./test_mamba3_gpu
else
	@echo "SKIP: test GPU nécessite CUDA"
endif

test-gemm-atb-determinism: tests/unit/test_gemm_atb_determinism.c kernels/gemm_f32.c include/kmamba_kernels.h
	$(CC) $(CFLAGS) -o /tmp/test_gemm_atb_determinism tests/unit/test_gemm_atb_determinism.c kernels/gemm_f32.c $(LDFLAGS)
	/tmp/test_gemm_atb_determinism

test-scan-nd-regression: tests/unit/test_scan_nd.c lib
	$(CC) -no-pie $(CFLAGS) -o /tmp/test_scan_nd tests/unit/test_scan_nd.c libkmamba.a $(LDFLAGS) $(CUDA_LDFLAGS)
	/tmp/test_scan_nd

test-gradient: tests/test_gradient.c lib
	$(CC) $(CFLAGS) -no-pie -o /tmp/test_gradient tests/test_gradient.c libkmamba.a $(LDFLAGS) $(CUDA_LDFLAGS)
	/tmp/test_gradient

bench-gates:
	bash scripts/bench_cpu_gates.sh

# ═══════════════════════════════════════════════════════════════
# Benchmarks ConvND (CPU et CUDA)
# ═══════════════════════════════════════════════════════════════

# Benchmark ConvND CPU (dense vs séparable)
bench-convnd-cpu: tests/unit/bench_convnd.c src/convnd.c src/wavefront_plan.c src/wavefront_nd.c src/km_topology.c src/km_memory_pool.c src/kmamba_cuda_utils.c
	$(CC) -O3 -fopenmp -I include -I. tests/unit/bench_convnd.c src/convnd.c src/wavefront_plan.c src/wavefront_nd.c src/km_topology.c src/km_memory_pool.c src/kmamba_cuda_utils.c -o tests/unit/bench_convnd -lm
	@echo "✓ Benchmark ConvND CPU: tests/unit/bench_convnd"

# Benchmark ConvND CUDA (dense vs séparable) - nécessite CUDA
ifeq ($(CUDA_AVAILABLE),1)
bench-convnd-cuda: tests/unit/bench_convnd_cuda.cu cuda/convnd.cu cuda/convnd_separable.cu $(OBJS)
	$(NVCC) -O3 -arch=sm_70 -I include -I cuda tests/unit/bench_convnd_cuda.cu cuda/convnd.cu cuda/convnd_separable.cu src/wavefront_plan.c src/wavefront_nd.c src/km_topology.c src/km_memory_pool.c -o tests/unit/bench_convnd_cuda -lcudart
	@echo "✓ Benchmark ConvND CUDA: tests/unit/bench_convnd_cuda"

test-convnd-separable-cuda: tests/unit/test_convnd_separable.cu cuda/convnd_separable.cu
	$(NVCC) -O3 -arch=sm_70 -I include -I cuda tests/unit/test_convnd_separable.cu cuda/convnd_separable.cu src/wavefront_plan.c src/wavefront_nd.c src/km_topology.c src/km_memory_pool.c -o tests/unit/test_convnd_separable_cuda -lcudart
	@echo "✓ Test ConvND séparable CUDA: tests/unit/test_convnd_separable_cuda"
else
bench-convnd-cuda test-convnd-separable-cuda:
	@echo "✗ Cible $@ nécessite CUDA (non disponible)"
endif

# ═══════════════════════════════════════════════════════════════
# Nettoyage
# ═══════════════════════════════════════════════════════════════

test-trainer-gc: $(TARGET) tests/unit/test_trainer_gc.c
	$(CC) $(CFLAGS) -o test_trainer_gc tests/unit/test_trainer_gc.c $(TARGET) libs/kser/libkser.a $(LDFLAGS) $(RUST_LDFLAGS) $(CUDA_LDFLAGS)
	./test_trainer_gc

clean:
	rm -f $(OBJS)
	rm -f src/*.o kernels/*.o cpu/*.o
	rm -f cuda/*.o
	rm -f train.o model.o src/configs.o
	rm -f test_mamba3 test_mamba3_gpu test_trainer_gc
	rm -f model train
	rm -f tests/unit/bench_convnd tests/unit/bench_convnd_cuda tests/unit/test_convnd_separable_cuda
	rm -f libkser.a libtrain.a
	$(MAKE) -C $(KSER_DIR) clean 2>/dev/null || true
	$(MAKE) -C $(TRAIN_DIR) clean 2>/dev/null || true
ifeq ($(RUST_AVAILABLE),1)
	cd $(RUST_DIR) && cargo clean 2>/dev/null || true
endif

distclean: clean
	rm -f $(TARGET) $(RUST_LIB)
	rm -f k-mamba-train
	rm -f checkpoint.ser checkpoint.ser.opt checkpoint.ser.state
	rm -rf logs/

# ═══════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════

help:
	@echo "k-mamba Makefile maître"
	@echo ""
	@echo "Cibles principales:"
	@echo "  make all              - Tout compiler (lib + CLI)"
	@echo "  make lib              - Juste la bibliothèque libkmamba.a"
	@echo "  make k-mamba-train    - CLI d'entraînement principal"
	@echo "  make model            - CLI de sérialisation du modèle"
	@echo "  make train            - CLI d'entraînement standalone"
	@echo "  make export-runtime-bundle BUNDLE_DIR=dist/runtime BUNDLE_CONFIG=configs/cifar10.json"
	@echo "  make tests            - Tests unitaires"
	@echo "  make clean            - Nettoyage"
	@echo "  make distclean        - Nettoyage complet"
	@echo "  make help             - Cette aide"
	@echo ""
	@echo "Variables:"
	@echo "  CPU_ONLY=1            - Forcer compilation CPU sans CUDA"

train.o: train.c include/configs.h libs/train_set/include/trainer.h
	$(CC) $(CFLAGS) -c train.c -o train.o

src/configs.o: src/configs.c include/configs.h
	$(CC) $(CFLAGS) -c src/configs.c -o src/configs.o

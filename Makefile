# Makefile maître — k-mamba
# Structure inspirée de optimus/cpu/Makefile
#
# Usage:
#   make all              # Tout compiler (lib + modèles)
#   make lib              # Juste la bibliothèque
#   make cpu_lm_model     # Modèle CPU 500K
#   make cuda_lm_model    # Modèle GPU 500M
#   make hybrid_lm_model  # Modèle Hybrid 1.5M
#   make all_models       # Tous les modèles
#   make tests            # Tests unitaires
#   make clean            # Nettoyage

.PHONY: all lib cpu cuda models cpu_lm_model cuda_lm_model hybrid_lm_model vision_model all_models tests test-gemm-atb-determinism test-scan-nd-regression test-gradient bench-gates clean distclean help

# ═══════════════════════════════════════════════════════════════
# Compilateurs et flags
# ═══════════════════════════════════════════════════════════════
CC = gcc
CFLAGS = -O3 -mavx2 -Wall -Wextra -I./include -fopenmp
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
CUDA_FLAGS = -O3 -arch=sm_70 -I./include -I$(CUDA_HOME)/include -DKMAMBA_BUILD_CUDA
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CFLAGS += -DKMAMBA_BUILD_CUDA
endif

# ═══════════════════════════════════════════════════════════════
# libkser Integration
# ═══════════════════════════════════════════════════════════════
KSER_DIR = libs/kser
KSER_LIB = $(KSER_DIR)/libkser.a
KSER_LDFLAGS = -L$(KSER_DIR) -lkser
CFLAGS += -I$(KSER_DIR)/include

# ═══════════════════════════════════════════════════════════════
# Fichiers source
# ═══════════════════════════════════════════════════════════════
SRCS = src/kmamba.c \
       src/mamba_block.c \
       src/kmamba_cuda_utils.c \
       src/kmamba_mixed_precision.c \
       src/kmamba_checkpoint.c \
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
            cuda/mamba_scan.cu \
            cuda/mamba_block.cu \
            cuda/kmamba_cuda_utils.cu \
            cuda/kmamba_mixed_precision.cu \
            cuda/kmamba_checkpoint.cu \
            cuda/kmamba_distributed.cu

# ═══════════════════════════════════════════════════════════════
# Objets et cibles
# ═══════════════════════════════════════════════════════════════
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(patsubst %.cu,cuda/%.o,$(notdir $(CUDA_SRCS)))

TARGET = libkmamba.a

MODEL_CPU = models/kmamba_cpu
MODEL_CUDA = models/kmamba_cuda
MODEL_AZURE = models/kmamba_azure
MODEL_VISION = models/kmamba_vision

# ═══════════════════════════════════════════════════════════════
# Cibles principales (cpu / cuda / all)
# ═══════════════════════════════════════════════════════════════

# Compilation CPU uniquement (lib + tests CPU)
cpu: lib
	@echo "=== Compilation CPU terminée ==="
	@echo "Bibliothèque: $(TARGET)"
	@echo "Tests CPU: make tests"

# Compilation CUDA (lib + tests GPU si disponible)
cuda: lib
ifeq ($(CUDA_AVAILABLE),1)
	@echo "=== Compilation CUDA terminée ==="
	@$(MAKE) bench-convnd-cuda test-convnd-separable-cuda 2>/dev/null || true
else
	@echo "✗ CUDA non disponible - impossible de compiler les tests GPU"
	@exit 1
endif

# Tout compiler (CPU + CUDA si dispo)
all: lib all_models
	@echo "=== Compilation complète terminée ==="

# Juste la bibliothèque
lib: check_cuda check_rust $(KSER_LIB) $(RUST_LIB) $(TARGET)
	@echo ""
	@echo "=== libkmamba.a prête ==="

# libkser static library
$(KSER_LIB):
	@echo "Building libkser..."
	$(MAKE) -C $(KSER_DIR)
	@echo "✓ libkser.a prête"

# Tous les modèles
models: all_models

# Modèle CPU uniquement
cpu_lm_model: lib $(MODEL_CPU)
	@echo "✓ Modèle CPU prêt: $(MODEL_CPU)"

# Modèle CUDA uniquement (si dispo)
cuda_lm_model: lib $(MODEL_CUDA)
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ Modèle CUDA prêt: $(MODEL_CUDA)"
else
	@echo "✗ CUDA non disponible — modèle CUDA ignoré"
endif

# Modèle Azure uniquement (si dispo)
azure_lm_model: lib $(MODEL_AZURE)
ifeq ($(CUDA_AVAILABLE),1)
	@echo "✓ Modèle Azure prêt: $(MODEL_AZURE)"
else
	@echo "✗ CUDA non disponible — modèle Azure ignoré"
endif

# Tous les modèles (selon disponibilité CUDA)
all_models: cpu_lm_model vision_model
ifeq ($(CUDA_AVAILABLE),1)
	@echo "Compilation modèles CUDA..."
	@$(MAKE) $(MODEL_CUDA) $(MODEL_AZURE)
	@echo "✓ Tous les modèles sont prêts"
endif

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

# Crée le répertoire models
models_dir:
	@mkdir -p models

# Lien des libs
MODEL_LDFLAGS = $(TARGET) $(KSER_LIB) $(LDFLAGS)
ifeq ($(RUST_AVAILABLE),1)
MODEL_LDFLAGS += $(RUST_LIB) $(RUST_LDFLAGS)
endif
ifeq ($(CUDA_AVAILABLE),1)
MODEL_LDFLAGS += $(CUDA_LDFLAGS)
endif

# Modèle CPU
$(MODEL_CPU): models/kmamba_cpu.c $(TARGET) $(KSER_LIB) $(RUST_LIB) | models_dir
	$(CC) $(CFLAGS) -o $@ $< $(MODEL_LDFLAGS)
	@echo "Built: $@ (CPU 500K params, BPE 32K)"

# Modèle CUDA (fichier .cu compilé avec nvcc)
$(MODEL_CUDA): models/kmamba_cuda.cu $(TARGET) $(KSER_LIB) $(RUST_LIB) | models_dir
ifeq ($(CUDA_AVAILABLE),1)
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(TARGET) $(KSER_LIB) $(CUDA_LDFLAGS) $(RUST_LIB) $(RUST_LDFLAGS) -Xcompiler "$(CFLAGS)"
	@echo "Built: $@ (CUDA 350M params, BPE 32K)"
endif

# Modèle Azure
$(MODEL_AZURE): models/kmamba_azure.cu $(TARGET) $(KSER_LIB) $(RUST_LIB) | models_dir
ifeq ($(CUDA_AVAILABLE),1)
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(TARGET) $(KSER_LIB) $(CUDA_LDFLAGS) $(RUST_LIB) $(RUST_LDFLAGS) -Xcompiler "$(CFLAGS)"
	@echo "Built: $@ (Azure 7.5B params, cl100k 100K)"
endif

# Modèle Vision 2D (K-Mamba 2D pour CIFAR-10)
$(MODEL_VISION): models/kmamba_vision.c models/kmamba_vision.h $(TARGET) $(KSER_LIB) $(RUST_LIB) | models_dir
	$(CC) $(CFLAGS) -no-pie -o $@ $< $(MODEL_LDFLAGS)
	@echo "Built: $@ (K-Mamba 2D Vision, 96 dim, 192 state, 5 layers)"

vision_model: lib $(MODEL_VISION)
	@echo "✓ Modèle Vision 2D prêt: $(MODEL_VISION)"

# Alias pour kmamba_vision
kmamba_vision: vision_model

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

clean:
	rm -f $(OBJS)
	rm -f src/*.o kernels/*.o cpu/*.o
	rm -f test_mamba3 test_mamba3_gpu
	$(MAKE) -C $(KSER_DIR) clean 2>/dev/null || true
ifeq ($(CUDA_AVAILABLE),1)
	rm -f cuda/*.cu.o
endif
ifeq ($(RUST_AVAILABLE),1)
	cd $(RUST_DIR) && cargo clean 2>/dev/null || true
endif

distclean: clean
	rm -f $(TARGET) $(RUST_LIB)
	rm -rf models/
	rm -f examples/train_500k examples/train_500m examples/train_1_5m examples/chat

# ═══════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════

help:
	@echo "k-mamba Makefile maître"
	@echo ""
	@echo "Cibles principales:"
	@echo "  make all              - Tout compiler (lib + modèles)"
	@echo "  make lib              - Juste la bibliothèque libkmamba.a"
	@echo "  make cpu_lm_model     - Modèle CPU 500K params"
	@echo "  make cuda_lm_model    - Modèle GPU 500M params (si CUDA)"
	@echo "  make hybrid_lm_model  - Modèle Hybrid 1.5M params (si CUDA)"
	@echo "  make all_models       - Tous les modèles disponibles"
	@echo "  make tests            - Tests unitaires"
	@echo "  make clean            - Nettoyage"
	@echo "  make distclean        - Nettoyage complet"
	@echo "  make help             - Cette aide"
	@echo ""
	@echo "Variables:"
	@echo "  CPU_ONLY=1            - Forcer compilation CPU sans CUDA"

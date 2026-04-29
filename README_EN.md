<div align="center">

# k‑mamba

**Wavefront Topology for Universal N-Dimensional Causal Modeling & Mamba‑3 Alignment (2026)**

[![Version](https://img.shields.io/badge/version-0.3.0-blue)](https://github.com/goldensam777/k-mamba)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-≥11.0-brightgreen)](https://developer.nvidia.com/cuda-toolkit)
[![CPU](https://img.shields.io/badge/CPU-AVX2%2B-blueviolet)]()

[Paper](#citation) · [Theory](THEORY.md) · [API Docs](#quick-start) · [Benchmarks](#performance)

*Samuel YEVI* · IFRI-UAC, Benin · [github.com/goldensam777](https://github.com/goldensam777)

</div>

---

## Abstract

**k‑mamba** introduces a unified topological framework for N-dimensional state space modeling based on the **wavefront ordering**. This project is the first to provide a native C/CUDA implementation of **Mamba‑3 (March 2026)** innovations, including Exponential‑Trapezoidal discretization and complex dynamics (RoPE).

Unlike decomposition-based approaches (VMamba, Mamba‑ND), k‑mamba establishes **simultaneous ND recurrence** where latent states depend on all immediate topological predecessors at once, without sacrificing parallelism thanks to the wavefront topology theorem.

**Key Contributions:**
1. **Full Mamba‑3 Support:** Exponential‑Trapezoidal discretization (2nd order), complex‑valued RoPE (learned angles), and MIMO formulation.
2. **Wavefront Theory:** Characterization proving that ND causal operators are perfectly parallelizable by level (Corollary 4.3, THEORY.md).
3. **Unified Architecture:** A single topological skeleton shared between `scanND` (long memory) and `convND` (local interactions).
4. **Zero‑Dependency Philosophy:** Pure implementation (C, x86‑64 ASM AVX2, CUDA) with no external dependencies (no PyTorch, no OpenBLAS).

---

## Mamba‑3 Features (Native)

k‑mamba implements the three pillars of **Mamba‑3**:

*   **Exponential‑Trapezoidal Discretization:** Uses a second-order approximation via `lambda_proj`, capturing local patterns without requiring external short causal convolutions.
*   **Complex‑Valued State Updates (RoPE):** Applies learned rotations (`theta`) to state projections, doubling storage efficiency compared to Mamba‑2.
*   **MIMO (Multi‑Input Multi‑Output):** Supports `mimo_rank > 1` to increase arithmetic intensity and accuracy without increasing decoding latency.

---

## Quick Start

### Installation

```bash
git clone https://github.com/goldensam777/k-mamba.git
cd k-mamba

# Intelligent build (auto-detects CUDA/AVX2)
make

# Validation tests (Mamba-3 & GPU)
make test-mamba3-gpu
```

### Runtime bundle (libs + binaries + config)

```bash
make export-runtime-bundle BUNDLE_DIR=dist/runtime BUNDLE_CONFIG=configs/cifar10.json

cd dist/runtime
./model --config config.json --serialize ser
./train --config config.json --data data/
```

The bundle exports:
- `libkmamba.a`, `libkser.a`, `libtrain.a`
- `model`, `train`
- `config.json`
- `inference/` (target folder for `*.ser`)

### 1. Creating a Mamba‑3 Model

```c
#include <kmamba.h>

KMambaConfig cfg;
kmamba_config_set_defaults(&cfg);
cfg.dim = 512;
cfg.state_size = 64;
cfg.n_layers = 12;
cfg.mimo_rank = 2; // Enable MIMO (Mamba-3)

// Enable RoPE (Complex SSM)
cfg.use_rope = 1; 

// 2D Spatial Topology
cfg.spatial_ndims = 2;
cfg.spatial_dims[0] = 32;
cfg.spatial_dims[1] = 32;

KMamba *model = kmamba_create(&cfg);
kmamba_init(model, 42);
```

### 2. Training with Gradient Checkpointing

```c
#include <trainer.h>

TrainerGCConfig gc_cfg = {
    .policy = TRAINER_GC_MODERATE,
    .checkpoint_every_n = 2 // Saves 50% VRAM
};

Trainer *trainer = trainer_create(model, &gc_cfg);
trainer_run(trainer, tokens, targets, n_samples, ...);
```

### 3. ND Primitives

```c
// ScanND: simultaneous N-dimensional recurrence
long dims[] = {32, 32};
ScanNDParams scan = {
    .dims = dims,
    .ndims = 2,
    .D = 64, .M = 16,
    .x = input, .A = A_data, .B = B_data, .C = C_data,
    .delta = delta_data,
    .h = state_buffer,
    .y = output,
    .default_lambda = 0.5f
};
scannd(&scan);  // Automatic wavefront execution

// ConvND: dense N-dimensional convolution (same skeleton)
ConvNDParams conv = {
    .input = in, .kernel = K, .bias = NULL, .output = out,
    .dims = dims, .ndims = 2, .D = 64, .K = 3
};
convnd(&conv, CONVND_FORWARD);
```

---

## Scientific Positioning

### The Wavefront Topology Theorem

For a point `n = (n₁, ..., n_N)` on a regular ND grid, define the **wavefront level**:

```
l(n) = n₁ + n₂ + ... + n_N
```

**Theorem (Characterization of causal operators):** For any operator O on a regular ND grid, the following are equivalent:
- (i) O is executable by level-by-level wavefront traversal with exact intra-level parallelism
- (ii) The dependency graph of O is a subgraph of the causal DAG defined by l(m) < l(n)
- (iii) All dependencies of any point n point strictly to lower levels

**Proof:** See THEORY.md, Section 0.3. ∎

**Corollary (Intra-level parallelism):** All points at the same wavefront level are mutually independent and parallelizable. On a d×d grid, level width reaches Θ(d), providing substantial parallelism.

### Comparison with State of the Art

| Approach | Mechanism | Causality | Parallelism |
|----------|-----------|-----------|-------------|
| Mamba-1 (Gu & Dao, 2023) | Linear 1D chain | Sequential | Inter-batch only |
| VMamba (Liu et al., 2024) | 4 scans in 4 directions | Compositional | None intra-level |
| Mamba-ND (Li et al., 2024) | Alternating 1D scans per dim | Factorized | None intra-level |
| Mamba-2 (Dao & Gu, 2024) | Matrix SSD | Tensor cores | Hardware-only |
| **k-mamba (YEVI)** | **Simultaneous ND recurrence** | **Partial order (wavefront)** | **Exact intra-level (theorem)** |

---

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────┐
│  Orchestration (src/)                   │
│  Model logic, API, training loop        │
├─────────────────────────────────────────┤
│  Topology (src/km_topology.c)           │
│  ND indexing, wavefront scheduling      │
├─────────────────────────────────────────┤
│  Kernels (kernels/, cuda/, cpu/)        │
│  GEMM, activations, optimizers          │
└─────────────────────────────────────────┘
```

**Rule:** If it's trivial, it goes in `src/`. If it loops millions of times, it goes in `kernels/`.

### Project Structure

```
k-mamba/
├── include/              # Public API headers
│   ├── kmamba.h          # Main API
│   ├── scan_nd.h         # ND scan interface
│   ├── convnd.h          # ND convolution interface
│   └── wavefront_plan.h  # Wavefront generator
├── src/                  # Orchestration layer
│   ├── kmamba.c          # Model forward/backward
│   ├── mamba_block.c     # SSM block implementation
│   ├── kmamba_ser.c      # Serialization (.ser format)
│   └── scan_nd.c         # Reference CPU scanND
├── cuda/                 # GPU kernels
│   ├── scan_nd.cu        # CUDA wavefront scan
│   ├── mamba_block.cu    # Full GPU forward/backward
│   └── kmamba_kernels.cu # Embedding, head, loss
├── kernels/              # Zero-dependency compute
│   ├── gemm_f32.c        # Pure C GEMM/GEMV
│   ├── activations_f32.c # SiLU, ReLU, Sigmoid
│   └── optimizer_f32.c   # AdamW, MUON, Newton-Schulz
├── libs/
│   ├── kser/             # Binary serialization library
│   └── train_set/        # Trainer with gradient checkpointing
├── tokenizer_rs/         # Hybrid tokenizer (Rust FFI)
│   └── src/lib.rs        # Bytes32K + Tiktoken100K modes
├── configs/              # JSON configurations
│   ├── cifar10.json
│   └── synthetic_2d.json
└── tests/                # Unit and integration tests
```

---

## Advanced Features

### Mamba-3 Architecture

Implementation includes recent architectural improvements:

```c
// BCNorm: learned biases after RMSNorm
float *b_B;  // [state_size] — bias for B projection
float *b_C;  // [state_size] — bias for C projection

// Complex SSM / RoPE: learned rotation angles
float *theta;  // [state_size/2] — per-pair angles

// Exp-Trapezoidal discretization
MBMatrix lambda_proj;  // Projects x_t -> scalar lambda_t ∈ [0,1]
```

### Mixed Precision Training

| Format | Range | Precision | Loss Scaling | Speedup |
|--------|-------|-----------|--------------|---------|
| FP32 | ±3.4e38 | 23 bits | No | 1× |
| FP16 | ±65504 | 10 bits | **Required** | **16× (Tensor Cores)** |
| BF16 | ±3.4e38 | 7 bits | No | 16× |

```c
// FP16 with gradient scaling
cfg.use_fp16 = 1;
cfg.loss_scale = 65536.0f;

// BF16: better stability, no scaling needed
cfg.use_bf16 = 1;
```

### Gradient Checkpointing

```c
#include <trainer.h>

TrainerGCConfig gc_cfg = {
    .policy = TRAINER_GC_MODERATE,  // or AGGRESSIVE, NONE
    .checkpoint_every_n = 2
};

Trainer *trainer = trainer_create(model, &gc_cfg);
TrainerMetrics metrics = trainer_run(
    trainer,
    data, labels, num_samples,
    L, D, num_classes,
    batch_size, epochs,
    "checkpoint.ser",
    1  // verbose
);
```

**Theorem (Checkpoint optimal):** For L layers with activation memory M, uniform checkpointing every k = ⌈LM/B⌉ layers minimizes computational overhead for memory budget B. See THEORY.md, Section 7.3.

### Hybrid Tokenizer (Rust FFI)

```c
// Initialize tokenizer
kmamba_tokenizer_init("bytes");     // 32K tokens, local robustness
kmamba_tokenizer_init("cl100k");    // 100K Tiktoken, cloud deployment

// Encode/decode
size_t len;
uint32_t *tokens = kmamba_encode("Hello world", &len);
char *text = kmamba_decode(tokens, len);

// Cleanup
kmamba_free_tokens(tokens, len);
kmamba_free_string(text);
```

### Multiple Optimizers

```c
typedef enum {
    OPTIMIZER_ADAM_CLIP,  // AdamW + gradient clipping
    OPTIMIZER_MUON,       // Newton-Schulz orthogonalization
    OPTIMIZER_SGD,        // Vanilla with momentum
    OPTIMIZER_ADAMW       // Standard
} OptimizerType;

kmamba_enable_training_with_optimizer(
    model, OPTIMIZER_MUON, &opt, lr, wd
);
```

### Configuration via JSON

```json
{
  "model": {
    "vocab_size": 32768,
    "dim": 512,
    "state_size": 64,
    "n_layers": 12,
    "seq_len": 1024,
    "spatial_ndims": 2,
    "spatial_dims": [32, 32],
    "use_convnd": 1,
    "convnd_K": 3,
    "use_bf16": 1
  },
  "optim": {
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "clip_norm": 1.0
  },
  "backend": 2
}
```

```c
KMambaFullConfig cfg;
kmamba_configs_load_json(&cfg, "configs/cifar10.json");
KMamba *model = kmamba_configs_create_model(&cfg);
```

---

## Performance

### ConvND: Dense vs Separable

Benchmark on 2D grids (D=64, K=3, OpenMP parallel):

| Grid Size | Dense (ms) | Separable (ms) | Speedup |
|-----------|-----------|----------------|---------|
| 64×64 | 127.80 | 1.93 | **66×** |
| 256×256 | 135.51 | 34.62 | **3.9×** |
| 1024×1024 | 3497.63 | 807.60 | **4.3×** |

**Insight:** On CPU, separable wins (fewer ops). On GPU, dense wins (single kernel, better coalescing). k-mamba provides both.

### Build Options

```bash
# Fast exponential approximation
make FAST_EXP=1

# Force CPU-only (no CUDA)
make CPU_ONLY=1

# Debug build
make CFLAGS="-O0 -g -DDEBUG"
```

---

## Zero Dependencies

k-mamba requires only:
- **Build:** GCC ≥ 11 or Clang ≥ 12, NASM ≥ 2.15
- **GPU:** CUDA Toolkit ≥ 11.0, cuBLAS
- **Optional:** Cargo (for Rust tokenizer)

**Not required:** Python, PyTorch, TensorFlow, OpenBLAS, MKL, NCCL, CMake.

```bash
# Ubuntu/Debian
sudo apt-get install gcc nasm

# Arch
sudo pacman -S gcc nasm

# macOS
brew install gcc nasm
```

---

## Testing

```bash
# Unit tests
make tests

# Individual test suites
make test-mamba3          # Forward/backward correctness
make test-mamba3-gpu      # GPU numerical match
make test-scan-nd-regression  # Wavefront regression
make test-gradient        # Gradient checking
make test-trainer-gc      # Gradient checkpointing

# Benchmarks
make bench-gates          # CPU gate benchmarks
make bench-convnd-cpu     # ConvND performance
make bench-convnd-cuda    # GPU ConvND
```

---

## Citation

```bibtex
@software{k_mamba_2024,
  author = {YEVI, Samuel},
  title = {k-mamba: Wavefront Topology for Universal N-Dimensional Causal Modeling},
  year = {2024},
  url = {https://github.com/goldensam777/k-mamba},
  note = {Zero-dependency C/CUDA implementation of simultaneous ND state space models}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{li2024mamband,
  title={Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data},
  author={Li, S and Singh, H and Grover, A},
  booktitle={ECCV},
  year={2024}
}
```

---

## References

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
2. Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024.
3. Liu, Y. et al. (2024). *VMamba: Visual State Space Model*. arXiv:2401.10166.
4. Li, S., Singh, H., & Grover, A. (2024). *Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data*. ECCV 2024.
5. Blelloch, G. (1990). *Prefix Sums and Their Applications*. CMU-CS-90-190.
6. Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost*. arXiv:1604.06174.

---

<div align="center">

**Optima, Immo, Absoluta Perfectio**

</div>

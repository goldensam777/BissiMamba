# k-mamba

**Zero-dependency C library for native N-dimensional Mamba.**

Unified ScanND + ConvND. CLI model/train. Native Gradient Checkpointing.

[![Build](https://img.shields.io/badge/build-makefile-blue)](Makefile)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Language](https://img.shields.io/badge/language-C-blue)
![CUDA](https://img.shields.io/badge/CUDA-supported-green)
![Zero-Dependency](https://img.shields.io/badge/zero--dependency-✓-success)
![Platform](https://img.shields.io/badge/platform-Linux-blue)

---

## Table of Contents

- [Innovations](#innovations)
- [Architecture](#architecture)
- [Build](#build)
- [API](#api)
- [Documentation](#documentation)
- [Benchmarks](#benchmarks)

---

## Innovations

### 1. Native Mamba-ND (N-dimensional)

Native extension from Mamba 1D to N dimensions via **simultaneous recurrence**:

```math
h(n) = Σ_{k=1}^{N} A_k · h(n − e_k) + B(n) · x(n)
```

| Operator | Implementation | Wavefront | Parallelism |
|----------|---------------|-----------|--------------|
| **Scan 1D** | ASM AVX2 | N/A | Sequential |
| **Scan 2D** | ASM AVX2 | Anti-diagonal | Intra-diagonal |
| **Scan ND** | Pure C | Implicit geometric | Optional OpenMP |
| **ConvND Dense** | Pure C | Unified K^N wavefront | Optional OpenMP |
| **ConvND Separable** | Pure C | 1D cascade wavefront | Optional OpenMP | **4–5× faster** |

### 2. Wavefront Unification

**ScanND** and **ConvND** share the same topological skeleton:

- Same wavefront generator (`KMWavefrontPlan`)
- Same level-by-level ordering
- Same intra-level parallelism (OpenMP)

```c
// ConvND Dense — complete K^N kernel
convnd_forward_wavefront(p, plan);

// ConvND Separable — cascade of N 1D convolutions (Mamba-classic)
convnd_separable_forward_wavefront(p, plans_per_axis);  // 4–5× faster
```

**Theorem**: For a $d \times d$ grid, wavefront scheduling requires $2d - 1$ sequential steps, each exposing up to $d$ parallel tasks:

$$S(d) = \frac{d^2}{2d - 1} \approx \frac{d}{2} \quad (d \gg 1)$$

Measured: **32.25×** speedup at $d = 64$ (Table `bench_paper.c`).

**Benchmark** (2D grid 256×256, D=64, K=3): Dense 135ms → Separable 35ms = **3.9× speedup**. See `figures/convnd_dense_vs_separable.png` and section 8 of THEORY.md.

### 3. Native CPU MUON

Pure C implementation of the MUON optimizer:

- Newton-Schulz (5 iterations)
- Nesterov momentum + gradient clipping
- AdamW with weight decay
- **Zero dependency**

### 4. GPU Optimizations (CUDA)

#### Parallel Scan (Blelloch)

Work-efficient parallel SSM scan using Blelloch's algorithm over the monoid $(\otimes, (1,0))$:

$$h_t = A_t · h_{t-1} + B_t · x_t  → (A_t, B_t·x_t) ⊗ (A_{t-1}, B_{t-1}·x_{t-1})$$

**Complexity**: Depth $O(\log L)$, Work $O(L)$ — $51\times$ reduction at $L=1024$.

| Method | Depth | Parallelism |
|--------|-------|-------------|
| CPU sequential | $O(L)$ | $O(1)$ |
| CUDA sequential | $O(L)$ | $O(D \times M)$ |
| **Blelloch CUDA** | $O(\log L)$ | $O(L \times D \times M)$ |

Theoretical speedup: **790×** for $L=1024$, $D=128$, $M=16$.

#### Mixed Precision FP16/BF16
- **FP16**: Dynamic loss scaling (65536.0f) to prevent underflow
- **BF16**: Native FP32 range, no scaling required
- **Tensor Cores**: Accelerated GEMM via cuBLAS

#### Gradient Checkpointing
- Memory reduction O(L×N×D) → O(N×D)
- Policies: `none`, `per-layer`, `per-block`
- Recompute forward during backward

#### Multi-GPU (Optional NCCL)
- Data parallelism: split batch
- Pipeline parallelism: split layers
- **Zero dependency**: NCCL optional

### 5. Separable vs Dense: CPU vs GPU

Surprising finding from our benchmarks:

| Platform | Winner | Speedup |
|----------|--------|---------|
| **CPU** | Separable | 3–4× faster |
| **GPU** | Dense | 1.3× faster |

**Why?** On GPU, the dense kernel exploits parallelism better (single kernel launch, better memory coalescing), while the separable cascade has overhead from multiple kernel launches and ping-pong buffers. On CPU, the reduced computational complexity (K^N vs N×K) dominates.

See benchmark graphs:
- CPU: `figures/convnd_dense_vs_separable.png`
- GPU: `figures/convnd_cuda_dense_vs_separable.png`

### 6. Zero Dependency

- **CPU**: `gcc`, `nasm`, `libc`, `libm`
- **Build**: Simple Makefile (no CMake)
- **Kernels**: Pure inline C (no external BLAS)
- **Optional**: OpenMP, CUDA, NCCL

---

## Architecture

### Layered Design

The project follows a clean separation of concerns across three layers:

| Layer | Role | Location |
|-------|------|----------|
| **Orchestration** | Model logic, API, training loop | `src/kmamba.c`, `src/mamba_block.c` |
| **Topology** | ND indexing, wavefront scheduling | `src/km_topology.c`, `src/wavefront_*.c` |
| **Kernels** | Compute-intensive operations | `kernels/*.c`, `cuda/*.cu`, `cpu/*.asm` |
| **Training** | Gradient Checkpointing, trainer logic | `libs/train_set/trainer.h` |

**Design Principle**: High-level model code (5-10 lines per operation) stays in `src/`. Performance-critical loops (millions of iterations) go in `kernels/`. Gradient Checkpointing is handled by the Trainer in `libs/train_set/`.

---

## Build

### Requirements

- `gcc >= 11`
- `nasm >= 2.15` (for ASM kernels)
- `make`
- Optional: `nvcc` (for CUDA)
- Optional: `cargo` (for Rust tokenizer)

### Quick Start

```bash
# Clone
git clone https://github.com/goldensam777/k-mamba.git
cd k-mamba

# Build everything
make all

# CPU only
make cpu

# With CUDA (if available)
make cuda

# Library only
make lib

# Run tests
make tests
```

### Build Targets

| Target | Description |
|--------|-------------|
| `make all` | Full build (lib + models + tests) |
| `make cpu` | CPU-only build |
| `make cuda` | CUDA build (fails if no CUDA) |
| `make lib` | Library `libkmamba.a` only |
| `make model` | Build `model` CLI (model creation) |
| `make train` | Build `train` CLI (training) |
| `make tests` | Run test suite |
| `make bench-convnd-cpu` | CPU ConvND benchmark |
| `make bench-convnd-cuda` | CUDA ConvND benchmark |
| `make clean` | Clean build artifacts |

---

## API

### Quick Example

```c
#include <kmamba.h>

// Configuration
KMambaConfig cfg = {
    .vocab_size = 256,
    .dim = 384,
    .state_size = 1024,
    .seq_len = 128,
    .n_layers = 1
};

// Create model
KMamba *model = kmamba_create(&cfg);

// Forward pass
kmamba_forward(model, tokens, logits);

// Enable training
MBOptimConfig opt = {.lr = 1e-3f, .clip_norm = 1.0f};
kmamba_enable_training(model, &opt, 1e-3f, 1e-5f);

// Training step
float loss = kmamba_train_step(model, tokens_plus1);

// Cleanup
kmamba_free(model);
```

### CLI Usage

Create a model from JSON config and train it:

```bash
# Build CLI tools
make model train

# Create model from config
./model configs/cifar10.json

# Train the model
./train configs/cifar10.json --batch_size=16 --epochs=10 --backend=cpu

# Or use the pipeline script
./scripts/train.sh configs/cifar10.json --batch_size=16 --epochs=10
```

### JSON Configuration

```json
{
    "model_name": "k-mamba-cifar10",
    "dim": 128,
    "state_size": 16,
    "n_layers": 4,
    "seq_len": 64,
    "spatial_ndims": 2,
    "spatial_dims": [8, 8],
    "backend": 1,
    "lr": 0.0003,
    "weight_decay": 0.05
}
```

### Trainer API

```c
#include "trainer.h"

// Load config and create model
KMambaFullConfig cfg;
kmamba_configs_load_json(&cfg, "configs/cifar10.json");
KMamba *m = kmamba_configs_create_model(&cfg);

// Create trainer and run training
TrainerGCConfig gc = {.policy = TRAINER_GC_NONE};
Trainer *t = trainer_create(m, &gc);
trainer_run(t, data, labels, n_samples, L, D, num_classes,
            batch_size, epochs, "checkpoint.ser", verbose);
```

### ConvND API

```c
#include <convnd.h>

// Dense convolution
ConvNDParams p = {
    .input = input,      // [spatial, D]
    .output = output,    // [spatial, D]
    .kernel = kernel,    // [K^ndims, D]
    .bias = bias,        // [D] or NULL
    .dims = dims,        // shape [ndims]
    .ndims = 2,          // 2D, 3D, etc.
    .D = 64,             // channels
    .K = 3               // kernel size
};
convnd_forward_wavefront(&p, plan);

// Separable convolution (faster on CPU)
float *kernel_axes[2] = {kernel_x, kernel_y};
ConvNDSeparableParams p_sep = {
    .input = input,
    .output = output,
    .kernel_axes = kernel_axes,  // [ndims] 1D kernels
    .dims = dims,
    .ndims = 2,
    .D = 64,
    .K = 3
};
convnd_separable_forward_wavefront(&p_sep, NULL);
```

---

## Documentation

| File | Content |
|------|---------|
| `THEORY.md` | Mathematical foundations, wavefront topology, proofs |
| `ARCHITECTURE.md` | Code structure, Volontés/Puissance philosophy |
| `AGENTS.md` | Project context for AI agents |
| `README.md` | This file (French) |
| `README_EN.md` | This file (English) |

---

## Benchmarks

### CPU Benchmarks (2D grids, D=64, K=3)

Run: `make bench-convnd-cpu && ./tests/unit/bench_convnd 256 256 64 3`

| Grid | Dense (ms) | Separable (ms) | Speedup |
|------|-----------|---------------|---------|
| 64×64 | 127.8 | 1.9 | **66×** |
| 128×128 | 35.0 | 10.7 | **3.3×** |
| 256×256 | 135.5 | 34.6 | **3.9×** |
| 512×512 | 927.4 | 204.4 | **4.5×** |
| 1024×1024 | 3497.6 | 807.6 | **4.3×** |

_*Note: The 66× anomaly at 64×64 is due to L1/L2 cache fitting the entire separable computation.*_

### GPU Benchmarks (NVIDIA GeForce MX450)

Run: `make bench-convnd-cuda && ./tests/unit/bench_convnd_cuda`

| Grid | Dense (ms) | Separable (ms) | Speedup |
|------|-----------|---------------|---------|
| 64×64 | 1.14 | 1.51 | **0.76×** |
| 128×128 | 2.23 | 2.69 | **0.83×** |
| 256×256 | 5.30 | 7.13 | **0.74×** |
| 512×512 | 15.1 | 19.9 | **0.76×** |

**Key Finding**: On GPU, Dense $K^N$ is ~1.3× faster than Separable! This is the opposite of CPU behavior.

### Generated Graphs

- [`figures/convnd_dense_vs_separable.png`](figures/convnd_dense_vs_separable.png) — CPU comparison
- [`figures/convnd_cuda_dense_vs_separable.png`](figures/convnd_cuda_dense_vs_separable.png) — GPU comparison

---

## Project Structure

```
k-mamba/
├── include/          # Public API headers
│   ├── kmamba.h
│   ├── convnd.h
│   ├── scan_nd.h
│   └── ...
├── src/              # Source code (Volontés/Will)
│   ├── kmamba.c
│   ├── mamba_block.c
│   ├── convnd.c
│   └── ...
├── kernels/          # Compute kernels (Puissance/Power)
│   ├── gemm_f32.c
│   ├── activations_f32.c
│   └── ...
├── cpu/              # Assembly AVX2 kernels
│   ├── scan1d.asm
│   └── scan2d.asm
├── cuda/             # GPU implementations
│   ├── convnd.cu
│   ├── convnd_separable.cu
│   └── ...
├── tests/            # Unit tests
│   └── unit/
├── figures/          # Benchmark graphs
└── Makefile          # Master build file
```

---

## Citation

If you use k-mamba in your research, please cite:

```bibtex
@software{k-mamba,
  author = {YEVI, Mawuli Peniel Samuel},
  title = {k-mamba: Zero-dependency C library for N-dimensional Mamba},
  url = {https://github.com/goldensam777/k-mamba},
  year = {2026}
}
```

---

## Author

**YEVI Mawuli Peniel Samuel** — IFRI-UAC (Benin)

Motto: **"Optima, Immo Absoluta Perfectio"**

---

## License

MIT License — See [LICENSE](LICENSE) file.

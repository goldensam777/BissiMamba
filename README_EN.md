# k-mamba

**Zero-dependency C framework for Mamba-ND training.**

CLI model/train · Native Gradient Checkpointing · .ser serialization

[![Build](https://img.shields.io/badge/build-makefile-blue)](Makefile)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Language](https://img.shields.io/badge/language-C-blue)
![CUDA](https://img.shields.io/badge/CUDA-supported-green)
![Zero-Dependency](https://img.shields.io/badge/zero--dependency-✓-success)
![Platform](https://img.shields.io/badge/platform-Linux-blue)

---

## Table of Contents

- [Philosophy](#philosophy)
- [Architecture](#architecture)
- [CLI Workflow](#cli-workflow)
- [JSON Configuration](#json-configuration)
- [Build](#build)
- [Programmatic API](#programmatic-api)
- [Documentation](#documentation)

---

## Philosophy

**Three-layer architecture** with clear separation of concerns:

| Layer | Role | Location | Complexity |
|-------|------|----------|------------|
| **Orchestration** | Model logic, API, CLI, training loop | `src/*.c`, `model.c`, `train.c` | Trivial (5-10 lines/op) |
| **Topology** | ND indexing, wavefront scheduling | `src/km_topology.c`, `src/wavefront_*.c` | ND geometry |
| **Kernels** | Compute engine, pure math | `kernels/*.c`, `cuda/*.cu` | Intensive (millions of iterations) |

**Golden rule**: If it's trivial, it goes in `src/`. If it loops millions of times, it goes in `kernels/`.

### Technical Innovations

1. **Native Mamba-ND**: N-dimensional extension via simultaneous recurrence
2. **Wavefront Unification**: ScanND and ConvND share the same topological skeleton
3. **Zero Dependency**: Just `gcc`, `nasm`, `libc` — no BLAS, no CMake

---

## Architecture

### Three-Layer Separation

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION (src/)                                       │
│  model.c, train.c, configs.c, kmamba.c                      │
│  → API, CLI, training loop, checkpoint I/O                  │
├─────────────────────────────────────────────────────────────┤
│  TOPOLOGY (src/km_topology.c, src/wavefront_*.c)          │
│  → ND indexing, wavefront scheduling, execution plans       │
├─────────────────────────────────────────────────────────────┤
│  KERNELS (kernels/, cuda/, cpu/)                            │
│  → GEMM, activations, scan ND, ConvND, optimizers (AdamW)  │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Files | Function |
|--------|-------|----------|
| **CLI** | `model.c`, `train.c`, `scripts/train.sh` | Model creation → Training |
| **Config** | `src/configs.c`, `include/configs.h` | Unified JSON (model + optimizer + backend) |
| **Trainer** | `libs/train_set/src/trainer.c` | Gradient Checkpointing, `trainer_run()`, progress tables |
| **Serialization** | `libs/kser/`, `src/kmamba_ser.c` | `.ser` format (model + vocab + tensors) |
| **Backends** | `include/kmamba_cuda_utils.h` | Auto-detect CPU/GPU |

---

## CLI Workflow

### 1. Create the model

```bash
./model configs/cifar10.json
```

Creates the model from JSON config and saves to `checkpoint.ser`.

### 2. Train the model

```bash
./train configs/cifar10.json --batch_size=16 --epochs=10 --backend=cpu
```

Options:
- `--batch_size=N` : Batch size (default: 8)
- `--epochs=N` : Number of epochs (default: 3)
- `--backend=cpu|gpu` : Force backend (default: from JSON or auto)

### 3. Complete pipeline script

```bash
./scripts/train.sh configs/cifar10.json --batch_size=16 --epochs=10
```

Displays a progress table during training:

```
┌───────┬─────────┬─────────┬───────────┬───────────┬───────────┐
│ Epoch │  Loss   │ Acc (%) │ Samples/s │ Time (ms) │    LR     │
├───────┼─────────┼─────────┼───────────┼───────────┼───────────┤
│ 1     │  0.6931 │   50.20 │    1245.3 │    803.1  │  3.00e-04 │
│ 2     │  0.5421 │   65.40 │    1289.2 │    775.6  │  3.00e-04 │
└───────┴─────────┴─────────┴───────────┴───────────┴───────────┘
```

---

## JSON Configuration

Unified JSON format for architecture, optimizer and backend:

```json
{
    "model_name": "k-mamba-cifar10",
    "dim": 128,
    "state_size": 16,
    "n_layers": 4,
    "seq_len": 64,
    "spatial_ndims": 2,
    "spatial_dims": [8, 8],
    "use_convnd": 1,
    "convnd_K": 3,
    "convnd_ndims": 2,
    "mimo_rank": 1,
    "default_lambda": 0.5,
    "use_a_log_clamp": 1,
    "a_log_min": -5.0,
    "dt_min": 0.0001,
    "dt_max": 0.01,
    "lr": 0.0003,
    "mu": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "clip_norm": 1.0,
    "weight_decay": 0.05,
    "backend": 1,
    "gpu_device": -1
}
```

| Parameter | Description | Values |
|-----------|-------------|--------|
| `backend` | Computation backend | `0`=AUTO, `1`=CPU, `2`=GPU |
| `spatial_ndims` | Spatial dimensions | `1`, `2`, `3` |
| `use_convnd` | Enable ConvND | `0`=no, `1`=yes |

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

## Programmatic API

### Create from JSON

```c
#include "configs.h"

KMambaFullConfig cfg;
kmamba_configs_load_json(&cfg, "configs/cifar10.json");
KMamba *m = kmamba_configs_create_model(&cfg);  // Creates + enables AdamW
kmamba_save(m, "checkpoint.ser");  // .ser serialization
```

### Complete Training

```c
#include "trainer.h"

// Load model
KMamba *m = kmamba_load("checkpoint.ser", 1, &optim_cfg, lr, wd);

// Configure Gradient Checkpointing
TrainerGCConfig gc = {
    .policy = TRAINER_GC_EVERY_N,  // NONE, EVERY_N, ALL
    .checkpoint_every_n = 2
};

// Create trainer
Trainer *t = trainer_create(m, &gc);

// Run training with progress table
trainer_run(t, data, labels, n_samples, L, D, num_classes,
            batch_size, epochs, "checkpoint.ser", verbose=1);

// Save checkpoint (model + training state)
trainer_save_checkpoint(t, "checkpoint.ser");
```

### Gradient Checkpointing Policies

| Policy | Memory | Speed | Use Case |
|--------|--------|-------|----------|
| `TRAINER_GC_NONE` | High | Fast | GPUs with sufficient memory |
| `TRAINER_GC_EVERY_N` | Medium | Medium | Balanced (recommended) |
| `TRAINER_GC_ALL` | Minimal | Slow | Memory-limited GPUs |

---

## Documentation

| File | Content |
|------|---------|
| `THEORY.md` | Mathematical foundations, wavefront topology, proofs |
| `ARCHITECTURE.md` | Code structure, three-layer philosophy |
| `AGENTS.md` | Project context for AI agents |
| `README.md` | This file (French) |
| `README_EN.md` | This file (English) |

---

## Author

**YEVI Mawuli Peniel Samuel** — IFRI-UAC, Bénin

_*Optima, Immo Absoluta Perfectio*_

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

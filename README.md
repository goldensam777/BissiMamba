# k-mamba

**Bibliothèque C pour State Space Models Mamba en dimensions N — CPU pur, ASM AVX2.**

Architecture dualiste : **k-mamba** orchestre les Volontés (logique modèle), **optimatrix** fournit la Puissance (kernels ASM).

[![CMake](https://img.shields.io/badge/build-cmake-blue)](CMakeLists.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table des matières

- [Innovations](#innovations)
- [Structure](#structure)
- [Build](#build)
- [API](#api)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Citations](#citations)

---

## Innovations

### 1. Mamba-ND natif (N-dimensionnel)

Extension native de Mamba 1D vers N dimensions via **recurrence simultanée** :

```
h(n) = Σ_{k=1}^{N} A_k · h(n − e_k) + B(n) · x(n)
y(n) = C(n) · h(n)
```

- **Scan 1D** : séquentiel le long d'un axe
- **Scan 2D** : ordonnancement wavefront (diagonales anti), parallélisme intra-diagonale
- **Scan ND** : DAG N-dimensionnel complet

Différence avec l'état de l'art :
- **VMamba** (2024) : 4 scans 1D dans 4 directions (pas de vraie 2D)
- **Mamba-ND** (Li et al.) : scans 1D alternés par couche
- **k-mamba** : récurrence **native ND**, pas d'approximation séquentielle

### 2. Architecture Volontés/Puissance

```
k-mamba/              ← Volontés (intentions, orchestration)
├── Embedding lookup
├── Stack MambaBlocks  
├── Checkpoint I/O
└── Training loop

optimatrix/           ← Puissance (calcul brut, kernels)
├── GEMM/GEMV AVX2
├── Scan 1D/2D ASM
├── ConvND separable
└── MUONCLIP
```

Séparation philosophique : la logique modèle (triviale, 5-10 lignes) reste en C lisible ; le calcul lourd (millions d'itérations) est en assembleur AVX2 optimisé.

### 3. MUONCLIP natif CPU

Implémentation C/ASM de l'optimiseur MUON (arXiv:2502.16982, Moonshot AI) :
- Newton-Schulz orthogonalisation (5 itérations)
- Momentum Nesterov + gradient clipping
- Weight decay découplé
- **Pas de dépendance PyTorch** — production pure C

### 4. Zero-Dependency SSM

- Pas de Python, pas de GPU, pas de CUDA
- libc + libm seulement
- Temps de démarrage ~0, footprint minimal
- Déployable sur CPU edge (embarqué, IoT)

### 5. Théorie des Volontés

Cadre conceptuel original : les systèmes doivent opérer par **intentions** (Volontés) qui convergent vers un équilibre, pas par instructions séquentielles.

- Chaque MambaBlock = une Volonté qui transforme la séquence
- MUONCLIP = arbitre des tensions entre gradients
- Un bug = un **conflit de Volontés non résolu**

---

## Structure

```
k-mamba/
├── include/
│   └── kmamba.h              # API publique
├── src/
│   └── kmamba.c              # Orchestration (forward, backward, checkpoint)
├── optimatrix/               # Submodule — kernels ASM AVX2
│   ├── include/optimatrix.h  # API calcul
│   └── src/
│       ├── gemm_avx2.asm
│       ├── scan1d.asm
│       ├── scan2d.asm
│       ├── conv1d_avx2.asm
│       └── mamba_block.c
├── cmake/
│   └── k-mambaConfig.cmake.in
├── CMakeLists.txt
├── THEORY.md                 # Fondement mathématique Mamba-ND
├── ESTIMATIONS.md            # Complexité et benchmarks
└── ARCHITECTURE.md           # Séparation Volontés/Puissance
```

---

## Build

### Prérequis

- `gcc >= 11`
- `nasm >= 2.15`
- `cmake >= 3.18`
- CPU avec AVX2 (Intel Haswell+ / AMD Ryzen+)

### Compilation

```bash
git clone --recursive https://github.com/user/k-mamba
cd k-mamba && mkdir build && cd build
cmake ..
make -j
sudo make install  # Optionnel
```

### Usage dans un projet CMake

```cmake
find_package(k-mamba REQUIRED)
target_link_libraries(mon_app PRIVATE k-mamba::k-mamba)
```

---

## API

### Création

```c
#include <kmamba.h>

KMambaConfig cfg = {
    .vocab_size = 256,      // byte-level
    .dim        = 384,
    .state_size = 1024,
    .seq_len    = 128,
    .n_layers   = 1,
    .dt_scale   = 1.0f,
    .dt_min     = 0.001f,
    .dt_max     = 0.1f
};

KMamba *m = kmamba_create(&cfg);
kmamba_init(m, 1234);       // Xavier init
```

### Entraînement

```c
MBOptimConfig opt = {
    .lr = 1e-3f, .mu = 0.9f, .beta2 = 0.999f,
    .eps = 1e-8f, .clip_norm = 1.0f, .weight_decay = 1e-5f
};
kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);

// Une séquence (seq_len+1 bytes)
float loss = kmamba_train_step(m, tokens_plus1);

// Batch
float loss = kmamba_train_batch(m, batch_tokens, batch_size);
```

### Inférence

```c
uint8_t tokens[seq_len];
float logits[seq_len * vocab_size];
kmamba_forward(m, tokens, logits);
```

### Checkpoint

```c
kmamba_save(m, "checkpoint.bin");
KMamba *m = kmamba_load("checkpoint.bin", 1, &opt, lr, wd);
kmamba_free(m);
```

---

## Architecture

### Séparation des responsabilités

| k-mamba (Volontés) | optimatrix (Puissance) |
|-------------------|------------------------|
| Embedding (memcpy) | GEMM/GEMV AVX2 |
| Softmax, cross-entropy | Scan selectif 1D/2D |
| Batch training loop | Scan backward ASM+C |
| Checkpoint I/O | MambaBlock forward/backward |
| LM head projection | MUONCLIP optimizer |
| | ConvND separable |
| | Activations (SiLU, etc.) |

### Pipeline MambaBlock

```
input [seq_len × dim]
    │
    ▼
W_in : dim → state_size (GEMV)
    │
    ▼
SiLU (gate)
    │
    ▼
delta_proj + softplus + clamp → dt_t
    │
    ▼
Selective Scan (1D ou 2D wavefront)
    h_t = exp(dt · A) · h_{t−1} + dt · B · u_t
    y_t = C · h_t
    │
    ▼
W_out : state_size → dim (GEMV)
    │
    ▼
output [seq_len × dim]
```

---

## Documentation

- **[THEORY.md](THEORY.md)** — Fondement mathématique du scan Mamba-ND
- **[ESTIMATIONS.md](ESTIMATIONS.md)** — Complexité théorique et benchmarks mesurés
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Philosophie Volontés/Puissance

---

## Citations

```bibtex
@software{k-mamba,
  author = {YEVI, Mawuli Peniel Samuel},
  title = {k-mamba: Native N-dimensional Mamba State Space Models in C/ASM},
  url = {https://github.com/user/k-mamba},
  year = {2025}
}

@software{optimatrix,
  author = {YEVI, Mawuli Peniel Samuel},
  title = {optimatrix: High-performance compute kernels for Mamba-ND},
  url = {https://github.com/user/optimatrix},
  year = {2025}
}
```

---

## Auteur

**YEVI Mawuli Peniel Samuel** — IFRI-UAC, Bénin

**Devise**: *Ego Sum Optimus Optimus*  
**Conviction**: *On est assez grand pour voir des unités, il faut voir des structures.*

---

## License

MIT

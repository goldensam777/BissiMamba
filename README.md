# k-mamba

**Framework C zero-dependency pour entraînement Mamba-ND.**

CLI model/train · Gradient Checkpointing natif · Sérialisation .ser

[![Build](https://img.shields.io/badge/build-makefile-blue)](Makefile)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Language](https://img.shields.io/badge/language-C-blue)
![CUDA](https://img.shields.io/badge/CUDA-supported-green)
![Zero-Dependency](https://img.shields.io/badge/zero--dependency-✓-success)
![Platform](https://img.shields.io/badge/platform-Linux-blue)

---

## Table des matières

- [Philosophie](#philosophie)
- [Architecture](#architecture)
- [Workflow CLI](#workflow-cli)
- [Configuration JSON](#configuration-json)
- [Build](#build)
- [API Programmatique](#api-programmatique)
- [Documentation](#documentation)

---

## Philosophie

**Architecture en trois couches** avec séparation claire des responsabilités :

| Couche | Rôle | Localisation | Complexité |
|--------|------|--------------|------------|
| **Orchestration** | Logique modèle, API, CLI | `src/*.c`, `model.c`, `train.c` | Triviale (5-10 lignes/op) |
| **Topologie** | Indexation ND, wavefront scheduling | `src/km_topology.c`, `src/wavefront_*.c` | Géométrie ND |
| **Kernels** | Compute engine, math pure | `kernels/*.c`, `cuda/*.cu` | Intensive (millions d'itérations) |


### Innovations techniques

1. **Mamba-ND natif** : Extension N-dimensionnelle via récurrence simultanée
2. **Unification Wavefront** : ScanND et ConvND partagent le même squelette topologique
3. **Zero Dependency** : Juste `gcc`, `nasm`, `libc` — pas de BLAS, pas de CMake

---

## Architecture

### Trois couches de séparation

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION (src/)                                       │
│  model.c, train.c, configs.c, kmamba.c                      │
│  → API, CLI, boucle d'entraînement, I/O checkpoints          │
├─────────────────────────────────────────────────────────────┤
│  TOPOLOGIE (src/km_topology.c, src/wavefront_*.c)          │
│  → Indexation ND, wavefront scheduling, plans d'exécution    │
├─────────────────────────────────────────────────────────────┤
│  KERNELS (kernels/, cuda/, cpu/)                            │
│  → GEMM, activations, scan ND, ConvND, optimizers (AdamW)   │
└─────────────────────────────────────────────────────────────┘
```

### Modules clés

| Module | Fichiers | Fonction |
|--------|----------|----------|
| **CLI** | `model.c`, `train.c`, `scripts/train.sh` | Création modèle → Entraînement |
| **Config** | `src/configs.c`, `include/configs.h` | JSON unifié (modèle + optimizer + backend) |
| **Trainer** | `libs/train_set/src/trainer.c` | Gradient Checkpointing, `trainer_run()`, tables de progression |
| **Sérialisation** | `libs/kser/`, `src/kmamba_ser.c` | Format `.ser` (modèle + vocab + tensors) |
| **Backends** | `include/kmamba_cuda_utils.h` | Auto-détection CPU/GPU |

### Workflow complet

```
JSON config ──→ ./model ──→ checkpoint.ser ──→ ./train ──→ modèle entraîné
     │              │              │              │
     │              │              │              └── trainer_run()
     │              │              │                  ├── Gradient Checkpointing
     │              │              │                  ├── Tables de progression
     │              │              │                  └── Resume checkpoint
     │              │              └── kmamba_save()
     │              └── kmamba_configs_create_model()
     └── kmamba_configs_load_json()
```

---

## Build

### Prérequis

- `gcc >= 11`
- `nasm >= 2.15`
- `libc`, `libm`
- CPU AVX2 (Intel Haswell+ / AMD Ryzen+)
- **Optionnel** : CUDA Toolkit (pour GPU)

### Cibles Makefile

```bash
make              # Bibliothèque libkmamba.a
make model        # CLI création modèle
make train        # CLI entraînement
make model train  # Les deux CLI
make clean        # Nettoyage
```

---

## Workflow CLI

### 1. Créer le modèle

```bash
./model configs/cifar10.json
```

Crée le modèle depuis la config JSON et sauvegarde dans `checkpoint.ser`.

### 2. Entraîner le modèle

```bash
./train configs/cifar10.json --batch_size=16 --epochs=10 --backend=cpu
```

Options :
- `--batch_size=N` : Taille du batch (défaut: 8)
- `--epochs=N` : Nombre d'époques (défaut: 3)
- `--backend=cpu|gpu` : Forcer le backend (défaut: depuis JSON ou auto)

### 3. Script pipeline complet

```bash
./scripts/train.sh configs/cifar10.json --batch_size=16 --epochs=10
```

Affiche une table de progression pendant l'entraînement :

```
┌───────┬─────────┬─────────┬───────────┬───────────┬───────────┐
│ Epoch │  Loss   │ Acc (%) │ Samples/s │ Time (ms) │    LR     │
├───────┼─────────┼─────────┼───────────┼───────────┼───────────┤
│ 1     │  0.6931 │   50.20 │    1245.3 │    803.1  │  3.00e-04 │
│ 2     │  0.5421 │   65.40 │    1289.2 │    775.6  │  3.00e-04 │
└───────┴─────────┴─────────┴───────────┴───────────┴───────────┘
```

---

## Configuration JSON

Le format JSON unifie architecture, optimizer et backend :

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

| Paramètre | Description | Valeurs |
|-----------|-------------|---------|
| `backend` | Backend computation | `0`=AUTO, `1`=CPU, `2`=GPU |
| `spatial_ndims` | Dimensions spatiales | `1`, `2`, `3` |
| `use_convnd` | Activer ConvND | `0`=non, `1`=oui |

---

## API Programmatique

### Création depuis JSON

```c
#include "configs.h"

KMambaFullConfig cfg;
kmamba_configs_load_json(&cfg, "configs/cifar10.json");
KMamba *m = kmamba_configs_create_model(&cfg);  // Crée + active AdamW
kmamba_save(m, "checkpoint.ser");  // Sérialisation .ser
```

### Entraînement complet

```c
#include "trainer.h"

// Charger le modèle
KMamba *m = kmamba_load("checkpoint.ser", 1, &optim_cfg, lr, wd);

// Configurer le Gradient Checkpointing
TrainerGCConfig gc = {
    .policy = TRAINER_GC_EVERY_N,  // NONE, EVERY_N, ALL
    .checkpoint_every_n = 2
};

// Créer le trainer
Trainer *t = trainer_create(m, &gc);

// Lancer l'entraînement avec table de progression
trainer_run(t, data, labels, n_samples, L, D, num_classes,
            batch_size, epochs, "checkpoint.ser", verbose=1);

// Sauvegarder le checkpoint (modèle + état entraînement)
trainer_save_checkpoint(t, "checkpoint.ser");
```

### Gradient Checkpointing

| Politique | Mémoire | Vitesse | Usage |
|-----------|---------|---------|-------|
| `TRAINER_GC_NONE` | Élevée | Rapide | GPUs avec mémoire suffisante |
| `TRAINER_GC_EVERY_N` | Moyenne | Moyenne | Équilibré (recommandé) |
| `TRAINER_GC_ALL` | Minimale | Lente | GPUs mémoire limitée |

---

## Documentation

- **[THEORY.md](THEORY.md)** — Fondement mathématique
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Architecture technique
- **[AGENTS.md](AGENTS.md)** — Contexte technique

---

## Auteur

**YEVI Mawuli Peniel Samuel** — IFRI-UAC, Bénin

_*Optima, Immo Absoluta Perfectio*_

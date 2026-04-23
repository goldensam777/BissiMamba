# AGENTS.md — k-mamba

> Lis ce fichier avant de toucher au code. C'est le contexte technique et philosophique du projet.

---

## Qu'est-ce que k-mamba ?

Une **bibliothèque C zero-dependency** pour State Space Models Mamba en dimensions N.

**Philosophie** : Aucune dépendance externe — juste `gcc`, `nasm` et `libc/libm`.

**Innovations** : 
1. **Scan Mamba-ND natif** (N-dimensionnel) avec ordonnancement wavefront
2. **Générateur de wavefront ND** — primitive topologique générique réutilisable par scanND et convND
3. **Couche topologique commune** — normalisation des dimensions, strides, indexation ND
4. **Kernels inline** — GEMM, activations, MUON/AdamW en C pur (pas de BLAS externe)

---

## Architecture (zero-dependency)

```
k-mamba/
├── include/
│   ├── kmamba.h              # API publique
│   ├── kmamba_kernels.h      # Kernels zero-dependency (GEMM, activations, optimizers)
│   ├── km_topology.h         # Couche topologique ND
│   ├── wavefront_nd.h        # Générateur wavefront ND
│   ├── wavefront_plan.h      # Plans exécutables
│   ├── scan_nd.h             # Interface scan ND
│   └── convnd.h              # Interface convND
├── src/
│   ├── kmamba.c              # Forward, backward, training loop
│   ├── mamba_block.c         # Bloc SSM (projections, scan dispatch)
│   ├── km_topology.c         # Normalisation topologique
│   ├── wavefront_nd.c        # Générateur wavefront
│   ├── wavefront_plan.c      # Plans wavefront
│   ├── scan_nd.c             # Scan ND
│   └── convnd.c              # Convolution ND
├── kernels/                  # Kernels compute zero-dependency
│   ├── gemm_f32.c            # GEMM/GEMV en C pur
│   ├── activations_f32.c     # SiLU, ReLU, Sigmoid, Softplus
│   ├── elementwise_f32.c     # Hadamard, vector ops
│   ├── optimizer_f32.c       # Gradient clip, Newton-Schulz, MUON, AdamW
│   └── init_f32.c            # Xavier/Kaiming init
├── cpu/                      # Scan SSM en assembleur
│   ├── scan1d.asm            # Scan 1D AVX2
│   ├── scan2d.asm            # Scan 2D wavefront
│   └── mamba_scan.c          # Dispatch CPU
├── cuda/                     # GPU optimizations (optionnel)
│   ├── scan1d.cu             # Parallel scan (Blelloch)
│   ├── scan1d_backward.cu    # Backward parallel scan
│   ├── mamba_block.cu        # Full GPU forward/backward
│   ├── kmamba_mixed_precision.cu   # FP16/BF16 Tensor Cores
│   ├── kmamba_checkpoint.cu        # Gradient checkpointing
│   └── kmamba_distributed.cu       # Multi-GPU NCCL (optional)
├── Makefile                  # Build simple (pas de CMake)
└── build.sh                  # Script build style Karpathy
```

---

## Build (Zero Dependency)

```bash
# Build simple
make

# Ou avec le script
./build.sh

# Clean
make clean
```

**Requiert** : `gcc >= 11`, `nasm >= 2.15`, `libc`, `libm`

**Pas de** : CMake, OpenBLAS, optimatrix, Python, PyTorch

---

## Kernels Inline

Tous les kernels sont maintenant dans `kernels/` — C pur sans dépendances :

| Opération | Fichier | Fonction |
|-----------|---------|----------|
| GEMM | `gemm_f32.c` | `gemm_f32()`, `gemv_f32()` |
| Activations | `activations_f32.c` | `silu_f32()`, `relu_f32()` |
| Elementwise | `elementwise_f32.c` | `hadamard_f32()`, `vec_add_f32()` |
| Optimizers | `optimizer_f32.c` | `adamw_step_f32()`, `muon_update_mat_f32()` |
| Init | `init_f32.c` | `init_xavier_uniform_f32()` |

---

## API

```c
#include <kmamba.h>

// Création
KMambaConfig cfg = {
    .vocab_size = 256, .dim = 384, .state_size = 1024,
    .seq_len = 128, .n_layers = 1
};
KMamba *m = kmamba_create(&cfg);

// Forward
kmamba_forward(m, tokens, logits_out);

// Training
MBOptimConfig opt = {.lr = 1e-3f, .clip_norm = 1.0f};
kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);
float loss = kmamba_train_step(m, tokens_plus1);
```

---

## Séparation Volontés/Puissance

| Couche | Rôle | Localisation |
|--------|------|--------------|
| **Volontés** | Orchestration modèle | `src/kmamba.c`, `src/mamba_block.c` |
| **Topologie** | ND indexing, wavefront | `src/km_topology.c`, `src/wavefront_*.c` |
| **Puissance** | Kernels compute | `kernels/*.c`, `cpu/*.asm` |

---

## Conventions

1. **Zero dependency** — Pas de bibliothèque externe, pas de package manager
2. **Inline kernels** — Fonctions simples en C, pas de BLAS complexe
3. **Makefile simple** — 20 lignes, pas de CMake
4. **NASM + C** — Assembleur pour hot paths, C pour le reste

---

## Ce qu'il ne faut PAS faire

- Ajouter des dépendances externes (OpenBLAS, MKL, etc.)
- Utiliser CMake ou autre build system complexe
- Créer des abstractions prématurées
- Dépendre de Python ou PyTorch

---

## Auteur

**YEVI Mawuli Peniel Samuel** — IFRI-UAC (Bénin)

Devise : **"Ego Sum Optimus Optimus"**

---

## Journal des Sessions

### Session 10 Avril 2026 — Implémentation du Modèle Hybrid CPU/GPU

**Objectif** : Créer un modèle hybride où embedding/head tournent sur CPU et les blocs Mamba sur GPU.

**Travail effectué** :

1. **Renommage des kernels CUDA** (`cuda/mamba_block.cu`)
   - Éviter conflits de symboles avec version CPU (`mamba_block.c`)
   - `silu_fwd_kernel` → `cuda_silu_fwd_kernel`
   - `sigmoid_fwd_kernel` → `cuda_sigmoid_fwd_kernel`
   - `add_inplace_kernel` → `cuda_add_inplace_kernel`
   - `gpu_block_forward` → `cuda_block_forward`
   - `gpu_block_backward` → `cuda_block_backward`

2. **Implémentation `kmamba_train_batch_hybrid`** (`src/kmamba.c`)
   - Embedding lookup sur CPU
   - Upload vers GPU via `cudaMemcpy`
   - Forward des blocs Mamba sur GPU via `hybrid_block_forward`
   - Download activations cachées vers CPU
   - Head GEMM et calcul de loss sur CPU
   - Backward des blocs sur GPU via `hybrid_block_backward`
   - Accumulation des gradients sur CPU
   - Optimizer step CPU pour embedding/head

3. **Création des wrappers** (`src/kmamba.c`)
   - `hybrid_block_forward()` : alloue paramètres GPU, appelle `cuda_block_forward`, libère
   - `hybrid_block_backward()` : alloue paramètres + gradients GPU, appelle `cuda_block_backward`, accumulate CPU
   - `gpu_param_alloc()` : helper allocation + upload
   - `get_cublas_handle()` : initialise handle cuBLAS avec workspace 16MB

4. **Mise à jour `models/kmamba_hybrid.c`**
   - Utilise `kmamba_train_batch_hybrid()` au lieu de `kmamba_train_batch()`
   - Correction du backend GPU pour les blocs

5. **Implémentation `gpu_optimizer_step`** (`cuda/mamba_block.cu`)
   - Download gradients GPU vers CPU
   - Application AdamW/MUON sur CPU
   - Upload paramètres mis à jour vers GPU

6. **Fix compilation**
   - Casts `(void**)` pour `cudaMalloc`
   - Déclarations `extern "C"` pour linkage C/CUDA
   - Inclusion `<cublas_v2.h>` dans `src/kmamba.c`

7. **Debug cuBLAS error 13** (non résolu)
   - Erreur `CUBLAS_STATUS_ALLOC_FAILED` à la ligne 62 de `mamba_block.cu`
   - Tentatives : `cudaSetDevice(0)`, `cudaFree(0)`, workspace 16MB, vérifications pointeurs
   - Build réussi mais erreur runtime persiste
   - Hypothèse : problème de version CUDA/cuBLAS ou mémoire GPU insuffisante

**Fichiers modifiés** :
- `cuda/mamba_block.cu` — Renommage kernels, ajout `gpu_optimizer_step`, vérifications
- `src/kmamba.c` — Implémentation complète training hybrid
- `models/kmamba_hybrid.c` — Appel fonction hybrid

**État** : Compilation OK. Exécution échoue sur cuBLAS error 13. À investiguer plus profondément (versions CUDA, mémoire, ou abandonner cuBLAS pour kernels CPU).



### Session 23 Avril 2026 — Refonte libkser (Sérialisation .ser v1)

**Objectif** : Stabiliser le format de sérialisation `.ser` et corriger les bugs de roundtrip.

**Problèmes identifiés et corrigés** :
1. **Layout fichier incohérent** — Le writer mélangeait vocab et tensor data
2. **SHA256 dupliqué** — Stub incomplet dans `kser_write.c`, implémentation complète dans `kser_checksum.c`
3. **Fread warnings** — `-Werror=unused-result` sur tous les appels `fread()`
4. **Forward declaration manquante** — `kser_reader_close()` utilisée avant définition

**Changements majeurs** :

| Fichier | Changement |
|---------|-----------|
| `libs/kser/include/kser.h` | Nouveau layout documenté (16+96+4+V+D+4+T+32), structures packed avec `_pad[]` |
| `libs/kser/src/kser_write.c` | Ordre d'écriture strict : header→config→vocab_count→vocab→tensors→tensor_index→SHA256. Atomic rename sur `.tmp` |
| `libs/kser/src/kser_checksum.c` | SHA256 optimisé, ajout de `kser_sha256_file()` pour streaming |
| `libs/kser/src/kser_quantize.c` | Code simplifié, fonctions FP16/BF16/INT8 compactées |

**Format .ser v1 (final)** :
```
[0]       16 bytes  Magic (SERENITY + η + version)
[16]      96 bytes  KSerConfig (packed, 96 bytes)
[112]      4 bytes  vocab_count (uint32)
[116]   variable    Vocab entries: (id:u32)(len:u16)(token)
[116+V] variable    Tensor data (offsets absolus)
[116+V+D] 4 bytes   tensor_count (uint32)
[+4]    variable    KSerTensorEntry[] (72 bytes chacun)
[end-32]  32 bytes  SHA256 de tout le précédent
```

**Règles du writer** :
- `add_vocab()` doit être appelé **avant** le premier `add_tensor()`
- Écriture atomique : `.tmp` → rename → fichier final
- Checksum SHA256 calculé sur tout le fichier sauf les 32 derniers bytes

**Fichiers créés** :
- `libs/kser/include/kser_checksum.h` — Déclarations SHA256 publiques

**État** : ✅ Compilation propre (0 warnings), tests roundtrip à valider

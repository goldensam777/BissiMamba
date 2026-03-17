# CONTEXT.md — k-mamba & optimatrix

> Document autonome. Si vous arrivez sans contexte sur ce projet — humain ou IA —
> lisez ce fichier en entier avant de toucher quoi que ce soit.
> Il vous explique les deux composants du projet depuis zéro.

---

## Vue d'ensemble : deux projets, une seule finalité

Ce dépôt contient deux projets distincts qui fonctionnent ensemble :

1. **optimatrix** — un moteur de calcul bas niveau (kernels x86-64 ASM AVX2 + CUDA)
2. **k-mamba** — une bibliothèque de modèle de langage Mamba N-dimensionnel, construite au-dessus d'optimatrix

Ces deux projets ont une relation précise : optimatrix est un **submodule git** inclus dans k-mamba.
k-mamba utilise optimatrix comme moteur de calcul. optimatrix n'a aucune connaissance de k-mamba.

**Auteur** : YEVI Mawuli Peniel Samuel — étudiant en L1 Systèmes Embarqués & IoT, IFRI-UAC (Bénin).
**Devise** : *"Ego Sum Optimus Optimus"*
**Licence** : MIT

---

---

# PARTIE I — optimatrix

---

## Qu'est-ce qu'optimatrix ?

optimatrix est une **bibliothèque C de primitives de calcul haute performance**.
Elle ne sait rien de Mamba, des modèles de langage, ni d'apprentissage automatique en général.
Son seul rôle : effectuer des opérations mathématiques élémentaires le plus vite possible.

Concrètement, optimatrix fournit :

- Multiplication matrice-vecteur et matrice-matrice (GEMV, GEMM) en AVX2
- Activations non-linéaires vectorisées (SiLU, Sigmoid, Softplus, ReLU)
- Convolution 1D depthwise causale et convolution ND séparable
- Produit de Hadamard (élément par élément)
- Optimiseurs : gradient clipping, AdamW, MUON (CPU et CUDA)

C'est un **outil générique**. On pourrait l'utiliser dans un autre projet que k-mamba.

---

## Pourquoi optimatrix existe séparément ?

Le principe : le code qui tourne des millions de fois dans une boucle interne
doit être séparé du code qui décrit *ce que fait* le programme.

Dans k-mamba, une passe forward sur une séquence de 128 tokens appelle `gemv_avx2`
des centaines de fois. Ce kernel doit être le plus rapide possible — écrit en assembleur
x86-64 avec les instructions vectorielles AVX2 (8 flottants 32-bit traités en parallèle).
Ce code n'a pas sa place mélangé à la logique du modèle.

C'est la séparation **Puissance** (optimatrix) / **Volontés** (k-mamba).

---

## Architecture d'optimatrix

```
optimatrix/
├── include/
│   └── optimatrix.h          ← API publique complète, guard extern "C" pour NVCC
│
├── cpu/                      ← Kernels CPU (ASM x86-64 AVX2 + C)
│   ├── gemv.asm              ← GEMV scalaire (référence)
│   ├── gemv_avx2.asm         ← GEMV vectorisé AVX2 (4.2× speedup vs scalaire)
│   ├── gemm.asm              ← GEMM scalaire (référence)
│   ├── gemm_avx2.asm         ← GEMM vectorisé AVX2 (~6 GFLOPS sur 64×128×256)
│   ├── activations.asm       ← SiLU, Sigmoid, Softplus, ReLU (AVX2)
│   ├── hadamard.asm          ← Produit élément par élément (AVX2)
│   ├── conv1d_avx2.asm       ← Conv1D depthwise causale (AVX2)
│   ├── generic_ops.c         ← ConvND séparable : forward + backward (C pur)
│   └── optimizer_utils.c     ← Gradient clipping, AdamW, MUON (C pur, CPU)
│
├── cuda/
│   └── optimizer_utils.cu    ← Gradient clipping, AdamW, MUON (CUDA) ✅ testé
│
└── CMakeLists.txt            ← Exporte optimatrix::optimatrix
```

---

## Les kernels d'optimatrix en détail

### GEMV et GEMM

`gemv(A, x, y, m, n)` — Multiplie la matrice A [m×n] par le vecteur x [n] → y [m].
`gemm(A, B, C, m, k, n)` — Multiplie A [m×k] par B [k×n] → **accumule** dans C [m×n].

**Attention critique** : `gemm_avx2` et `gemm` font `C += A*B`, pas `C = A*B`.
L'appelant **doit zéroïser C avant l'appel** s'il veut une multiplication pure.
Ce comportement est intentionnel (accumulation de gradients).

Les variantes `_avx2` traitent 8 flottants en parallèle avec les registres YMM.
Les variantes sans suffixe sont des implémentations scalaires de référence.

### Activations

Toutes les activations opèrent sur des tableaux `float *x` → `float *y` de longueur `n`.
Implémentées en ASM AVX2 : 8 éléments par instruction.

- `silu_f32(x, y, n)` — SiLU : `y = x * sigmoid(x)`
- `sigmoid_f32(x, y, n)` — Sigmoid : `y = 1 / (1 + exp(-x))`
- `softplus_f32(x, y, n)` — Softplus : `y = log(1 + exp(x))`
- `relu_f32(x, y, n)` — ReLU : `y = max(0, x)`
- `hadamard_avx2(x, y, z, n)` — `z = x * y` élément par élément

### Conv1D depthwise causale

```c
typedef struct {
    float *input;   /* [L, D] */
    float *kernel;  /* [K, D] */
    float *bias;    /* [D] ou NULL */
    float *output;  /* [L, D] */
    long   L;       /* longueur de séquence */
    long   D;       /* nombre de canaux */
    long   K;       /* taille du noyau */
} Conv1DParams;

void conv1d_depthwise_avx2(Conv1DParams *p);
```

"Depthwise" = chaque canal est filtré indépendamment (pas de mélange inter-canaux).
"Causale" = à la position t, la convolution ne voit que les positions ≤ t (zéro-padding à gauche).

### ConvND séparable

Généralise Conv1D à N dimensions spatiales (2D, 3D, etc.).
L'implémentation est séparable : une Conv1D par axe, du dernier axe au premier.

```c
void convnd(ConvNDParams *p, ConvNDMode mode, ConvNDWorkspace *ws);
// mode : CONVND_FORWARD | CONVND_BACKWARD | CONVND_COMPLETE
```

Le workspace stocke les intermédiaires de la passe forward pour les réutiliser en backward.

### Optimiseurs (CPU)

```c
// Norme L2 du gradient
float gradient_norm(const float *grad, size_t n);

// Clipping par norme maximale (in-place)
void gradient_clip_inplace(float *grad, size_t n, float max_norm);

// Config partagée entre CPU et CUDA
typedef struct {
    float lr;           // taux d'apprentissage
    float mu;           // momentum
    float beta2;        // second moment (Adam)
    float eps;          // epsilon numérique
    float clip_norm;    // norme maximale du gradient
    float weight_decay; // régularisation L2
} MBOptimConfig;
```

MUON (Momentum + Orthogonalisation Newton-Schulz) est l'optimiseur principal.
Il orthogonalise les directions de gradient pour obtenir des mises à jour isotropiques.

### Optimiseurs (CUDA)

Mêmes opérations, sur GPU via `cuda/optimizer_utils.cu` :
```c
// Uniquement disponibles si compilé avec NVCC
void gradient_clip_inplace_cuda(float *grad, size_t n, float max_norm);
void adamw_update_cuda(float *param, float *grad, float *m, float *v,
                       size_t n, const MBOptimConfig *conf, size_t step);
void muon_update_cuda(float *param, float *grad, float *m,
                      size_t n, const MBOptimConfig *conf);
```

---

## Ce qu'optimatrix ne contient PAS

Les scans sélectifs Mamba (`scan1d`, `scan2d`, `scan1d_backward`…) ne sont **pas** dans
optimatrix. Ils encodent la structure du monoid SSM et la logique ZOH (Zero-Order Hold) —
c'est de la logique propre à Mamba, pas du calcul matriciel générique.
Ils vivent dans `k-mamba/cpu/` et `k-mamba/cuda/`.

---

## Prérequis et build d'optimatrix seul

```bash
# Prérequis : gcc >= 11, nasm >= 2.15, cmake >= 3.18, CPU AVX2
# CUDA optionnel : nvcc >= 12.0, GPU sm_75+

cmake -B build-opt
cmake --build build-opt -j

# Depuis un autre projet CMake :
find_package(optimatrix REQUIRED)
target_link_libraries(mon_app PRIVATE optimatrix::optimatrix)
```

---

---

# PARTIE II — k-mamba

---

## Qu'est-ce que k-mamba ?

k-mamba est une **bibliothèque C pour entraîner et inférer des modèles de type Mamba**
en N dimensions. Elle s'appuie sur optimatrix pour tout le calcul intensif.

Mamba est une architecture de modèle de séquence basée sur les State Space Models (SSM).
Contrairement aux Transformers (attention quadratique), Mamba utilise une récurrence
sélective — coût linéaire en la longueur de séquence.

**Innovation principale de k-mamba** : scan Mamba nativement N-dimensionnel.
Là où VMamba (2024) et Mamba-ND (2024) font des scans 1D dans différentes directions,
k-mamba implémente une vraie récurrence simultanée dans toutes les dimensions,
avec ordonnancement wavefront (diagonales anti) pour exposer le parallélisme.

---

## Rappel théorique : le scan sélectif 1D

Le SSM discret (après discrétisation ZOH) calcule, pour chaque pas t :

```
h_t[d,m] = exp(dt_t[d] * A[d,m]) * h_{t-1}[d,m]  +  dt_t[d] * B_t[d,m] * x_t[d]
y_t[d]   = sum_m  C_t[d,m] * h_t[d,m]
```

- `h_t` — état caché (mémoire du passé)
- `x_t` — entrée à l'instant t
- `A` — matrice de transition (diagonale, partagée sur L)
- `B_t`, `C_t`, `dt_t` — **sélectifs** : dépendent de l'entrée → le modèle choisit quoi retenir

Ce scan est séquentiel par nature (h_t dépend de h_{t-1}).
Sur CPU, il s'implémente avec des instructions AVX2 dans `cpu/scan1d.asm`.
Sur GPU, il utilise l'algorithme de Blelloch (parallel prefix scan) dans `cuda/scan1d.cu`.

### Extension à 2D (wavefront)

```
h(i,j,d,m) = dA1 * h(i-1,j,d,m)  +  dA2 * h(i,j-1,d,m)  +  dB * x(i,j,d)
y(i,j,d)   = sum_m  C(i,j,d,m) * h(i,j,d,m)
```

Les positions sur la même diagonale `k = i+j` sont indépendantes → parallélisables.
L'ordonnancement wavefront traite une diagonale à la fois, de k=0 à k=(d1+d2-2).

---

## Architecture de k-mamba

```
k-mamba/
│
├── include/
│   ├── kmamba.h              ← API publique complète (types + fonctions)
│   ├── scan.h                ← Tous les types des scans SSM
│   ├── mamba_scan.h          ← Alias typedef → MambaScan1DParams, etc.
│   ├── mamba_scan_2d.h       ← Alias typedef → MambaScan2DParams
│   └── mamba_scan_cuda.h     ← Alias typedef pour les scans CUDA
│
├── src/
│   ├── kmamba.c              ← Orchestration : embedding, softmax, cross-entropy,
│   │                            training loop, checkpoints, batch accumulation
│   ├── mamba_block.c         ← Un bloc SSM complet : projections W_in/W_out,
│   │                            dispatch scan, état optimiseur, forward + backward
│   └── convnd.c              ← Wrappeur ConvND appelant optimatrix
│
├── cpu/                      ← Scans SSM — logique Mamba, PAS dans optimatrix
│   ├── scan1d.asm            ← Scan 1D forward (ASM AVX2)
│   ├── scan2d.asm            ← Scan 2D wavefront (ASM)
│   ├── scan1d_backward.c     ← Backward 1D générique [L,D,M] (C pur)
│   ├── scan1d_backward_m.c   ← Backward 1D pour M > 1 (C pur)
│   ├── scan1d_backward_m1_shared_bc.asm      ← ⚠️ bug "two index registers" — DÉSACTIVÉ
│   ├── scan1d_backward_m1_shared_bc_simple.asm  ← ⚠️ même bug — DÉSACTIVÉ
│   └── mamba_scan.c          ← Dispatch CPU : choisit la routine selon les params
│
├── cuda/                     ← Scans SSM CUDA — logique Mamba, PAS dans optimatrix
│   ├── scan1d.cu             ← Blelloch parallel prefix scan 1D
│   ├── scan1d_backward.cu    ← Backward scan 1D CUDA
│   └── mamba_scan.cu         ← Dispatch CUDA
│
├── optimatrix/               ← Submodule git (voir Partie I)
│
├── bench/
│   └── bench_paper.c         ← Benchmarks G1-G7 (GEMM, wavefront, Blelloch, roofline)
│
├── paper/
│   ├── kmamba.tex             ← Paper arXiv LaTeX (deux colonnes)
│   └── kmamba.bib             ← Bibliographie BibTeX
│
└── CMakeLists.txt             ← Exporte k-mamba::k-mamba, gère CPU/CUDA optionnel
```

---

## Les types centraux (`include/kmamba.h`)

### KMambaConfig — configuration du modèle

```c
typedef struct {
    size_t vocab_size;   // 256 par défaut (byte-level)
    size_t dim;          // dimension du modèle : 384
    size_t state_size;   // taille de l'état caché : 1024
    size_t seq_len;      // longueur de contexte : 128
    size_t n_layers;     // nombre de MambaBlocks empilés : 1

    float dt_scale;      // 1.0
    float dt_min;        // 0.001
    float dt_max;        // 0.1

    int   use_convnd;    // activer la ConvND avant le scan (0 ou 1)
    long  convnd_K;      // taille du noyau ConvND
    long  convnd_ndims;  // nombre de dimensions spatiales (1, 2, 3)
} KMambaConfig;
```

### KMamba — le modèle complet

```c
typedef struct {
    KMambaConfig cfg;

    float *embedding;    // table d'embedding [vocab_size × dim]
    float *head;         // LM head [dim × vocab_size]

    MambaBlock **layers; // stack de n_layers blocs SSM

    int         for_training;
    MBOptimConfig opt_blocks;
    float         lr_embed_head;
    float         weight_decay;
} KMamba;
```

### MambaBlock — un seul bloc SSM

```c
typedef struct {
    MBConfig config;

    MBMatrix W_in;        // projection d'entrée [state_size × dim]
    MBMatrix W_out;       // projection de sortie [dim × state_size]
    MBMatrix A_log;       // log de la matrice A diagonale [state_size]
    MBMatrix B_mat;       // matrice B partagée [state_size]
    MBMatrix C_mat;       // matrice C partagée [state_size]
    MBMatrix delta_proj;  // projection delta [1 × dim]

    float *convnd_kernel; // noyau ConvND [ndims × K × dim]
    float *convnd_bias;   // biais ConvND [dim]

    // Buffers runtime
    float *hidden;        // [dim]
    float *delta;         // [seq_len]
    float *scan_B;        // [seq_len × state_size]
    float *scan_C;        // [seq_len × state_size]
    float *scan_delta;    // [seq_len × state_size]
    float *scan_h;        // [state_size] — état caché final
} MambaBlock;
```

### Types scan (`include/scan.h`)

```c
// Scan 1D forward
typedef struct {
    float *x;      // entrée  [L, D]
    float *A;      // matrice A [D, M]  — partagée sur L
    float *B;      // [L, D, M] — sélective
    float *C;      // [L, D, M] — sélective
    float *delta;  // [L, D]
    float *h;      // états cachés sortie [L, D, M]
    float *y;      // sortie [L, D]
    long   L, D, M;
} ScanParams;

// Scan 1D backward
typedef struct {
    float *x, *A, *B, *C, *delta, *h0, *h, *dy;
    float *dx, *dA, *dB, *dC, *ddelta;
    long   L, D, M;
} ScanBackwardParams;

// Scan 2D wavefront
typedef struct {
    float *x;
    float *A1, *A2;        // deux matrices A (une par axe)
    float *B, *C;
    float *delta1, *delta2;
    float *h;              // états cachés [d1, d2, D, M]
    float *y;              // sortie [d1, d2, D]
    long   d1, d2, D, M;
} Scan2DParams;
```

**Règle** : Ces types vivent dans `include/scan.h` uniquement. Jamais dans `optimatrix.h`.

---

## Pipeline d'exécution — passe forward

```
tokens [seq_len, uint8]
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  kmamba_forward()  [src/kmamba.c]                      │
│                                                        │
│  1. Embedding lookup                                   │
│     memcpy de la ligne tokens[t] dans la table         │
│     embedding[256 × dim] → hidden [seq_len × dim]      │
│                                                        │
│  2. Pour chaque layer i = 0..n_layers-1 :              │
│     mamba_block_forward(layers[i], ...)                │
│     (voir détail ci-dessous)                           │
│                                                        │
│  3. LM Head : GEMM [dim × vocab_size]                  │
│     → gemm_avx2 (optimatrix)                           │
│     → logits [seq_len × vocab_size]                    │
│                                                        │
│  4. Retourne logits                                    │
└───────────────────────────────────────────────────────┘
        │
        ▼ (step 2 détaillé)
┌───────────────────────────────────────────────────────┐
│  mamba_block_forward()  [src/mamba_block.c]            │
│                                                        │
│  1. (optionnel) ConvND sur l'entrée                    │
│     → convnd() (optimatrix)                            │
│                                                        │
│  2. Projection W_in : [dim → state_size]               │
│     → gemv_avx2(W_in, x, u)  (optimatrix)             │
│                                                        │
│  3. Gate SiLU                                          │
│     → silu_f32_avx2(u, u)  (optimatrix)               │
│                                                        │
│  4. Calcul de delta (pas de temps adaptatif)           │
│     → gemv_avx2(delta_proj, x, dt)                     │
│     → softplus_f32(dt, dt)  (optimatrix)               │
│     → clamp [dt_min, dt_max]                           │
│                                                        │
│  5. Scan sélectif                                      │
│     → scan1d() si 1D  (cpu/scan1d.asm)                │
│     → scan2d() si 2D  (cpu/scan2d.asm)                │
│     Ou version CUDA si compilé avec KMAMBA_BUILD_CUDA  │
│                                                        │
│  6. Projection W_out : [state_size → dim]              │
│     → gemv_avx2(W_out, h, out)  (optimatrix)          │
└───────────────────────────────────────────────────────┘
```

---

## Pipeline d'exécution — passe backward (batch)

```
kmamba_train_batch(m, batch_tokens, B)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Pour chaque séquence b = 0..B-1 :                      │
│                                                         │
│  1. Forward complet avec sauvegarde des activations     │
│     par couche (pour le backward)                       │
│                                                         │
│  2. Cross-entropy loss                                  │
│     dlogits[t,v] = softmax[t,v] - one_hot[t, target_v] │
│     scalé par 1/B                                       │
│                                                         │
│  3. Gradient du LM Head                                 │
│     d_hidden = dlogits @ head^T  (GEMM AVX2)           │
│     g_head += hidden^T @ dlogits (GEMM AVX2, accumulé) │
│                                                         │
│  4. Backward couche par couche (ordre inverse)          │
│     mamba_backward(layers[i], dY, input, d_input)      │
│     → backward W_out     (GEMM AVX2, optimatrix)       │
│     → scan1d_backward()  (cpu/scan1d_backward.c)       │
│     → backward SiLU      (optimatrix)                  │
│     → backward W_in      (GEMM AVX2, optimatrix)       │
│     → gradients accumulés dans MBOptimState             │
│                                                         │
│  5. Accumulation du gradient embedding (scatter-add)   │
│                                                         │
│ Après les B séquences :                                 │
│                                                         │
│  6. mamba_optimizer_step (MUONCLIP) — un seul step     │
│     → Newton-Schulz sur les gradients des blocs        │
│                                                         │
│  7. SGD sur embedding et head                           │
│     → param -= lr * grad                               │
└────────────────────────────────────────────────────────┘
```

---

## API publique de k-mamba

### Créer et initialiser un modèle

```c
#include "kmamba.h"

KMambaConfig cfg = {
    .vocab_size  = 256,
    .dim         = 384,
    .state_size  = 1024,
    .seq_len     = 128,
    .n_layers    = 1,
    .dt_scale    = 1.0f,
    .dt_min      = 0.001f,
    .dt_max      = 0.1f,
    .use_convnd  = 0,     // pas de ConvND
};

KMamba *m = kmamba_create(&cfg);
kmamba_init(m, 1234);          // initialise les poids (seed)
```

### Activer l'entraînement

```c
MBOptimConfig opt = {
    .lr          = 1e-3f,
    .mu          = 0.9f,
    .beta2       = 0.999f,
    .eps         = 1e-8f,
    .clip_norm   = 1.0f,
    .weight_decay = 1e-5f
};

kmamba_enable_training(m, &opt, /*lr_embed_head=*/1e-3f, /*weight_decay=*/1e-5f);
```

### Entraîner

```c
// Une séquence (seq_len + 1 tokens : seq_len entrées, seq_len cibles décalées)
float loss = kmamba_train_step(m, tokens_plus1);

// Un batch complet
float loss = kmamba_train_batch(m, batch_tokens, /*batch_size=*/8);
```

### Inférer

```c
float logits[128 * 256];
kmamba_forward(m, tokens, logits);
// logits[t * 256 + v] = score du token v à la position t
```

### Sauvegarder / charger

```c
kmamba_save(m, "checkpoint.bin");   // Magic "KMAMBA" + version 1

// Charger pour l'inférence
KMamba *m2 = kmamba_load("checkpoint.bin", /*for_training=*/0, NULL, 0.0f, 0.0f);

// Charger pour reprendre l'entraînement
KMamba *m3 = kmamba_load("checkpoint.bin", 1, &opt, 1e-3f, 1e-5f);

kmamba_free(m);
```

---

## Build de k-mamba

```bash
# Cloner avec le submodule optimatrix
git clone --recursive <url>
cd k-mamba

# CPU seul (standard)
cmake -B build -DKMAMBA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build

# CPU + CUDA (MX450 = sm_75)
cmake -B build-cuda -DKMAMBA_BUILD_CUDA=ON -DKMAMBA_BUILD_TESTS=ON
# IMPORTANT : le flag CUDA d'optimatrix ne passe pas via CLI (cache déjà créé)
# Il faut forcer dans le cache :
sed -i 's/OPTIMATRIX_BUILD_CUDA:BOOL=OFF/OPTIMATRIX_BUILD_CUDA:BOOL=ON/' build-cuda/CMakeCache.txt
cmake build-cuda && cmake --build build-cuda -j
ctest --test-dir build-cuda

# Test spécifique
ctest --test-dir build -R OptimizersTest -V
ctest --test-dir build -R optimatrix_kernels_unit -V

# Benchmarks paper
cmake -B build-bench -DKMAMBA_BUILD_BENCH=ON
cmake --build build-bench --target bench_paper
./build-bench/bench/bench_paper
```

**Prérequis** : `gcc >= 11`, `nasm >= 2.15`, `cmake >= 3.18`, CPU AVX2.
**CUDA optionnel** : `nvcc >= 12.0`, GPU sm_75+ (Turing). Testé sur NVIDIA MX450.

---

## État des tests (17/03/2026)

| Suite | Tests | Statut |
|-------|-------|--------|
| `test_optimatrix_kernels` | 5/5 | ✅ PASS |
| `test_optimizers` (CPU) | 2/2 | ✅ PASS |
| `CudaOptimizersTest` | 4/4 | ✅ PASS |
| Scan CUDA | — | Compile, pas encore dans CTest |

**Plateforme** : x86-64 Linux, NVIDIA MX450 (sm_75), CUDA 12.0.
**Benchmark GEMM** : AVX2 ASM ≈ 6.0 GFLOPS sur 64×128 × 128×256 (8× vs scalaire).

---

## Problèmes connus

- `cpu/scan1d_backward_m1_shared_bc.asm` — bug "two index registers" (désactivé, TODO)
- `cpu/scan1d_backward_m1_shared_bc_simple.asm` — même bug (désactivé)
- Les scans CUDA (`cuda/scan1d.cu`, etc.) compilent mais ne sont pas enregistrés dans CTest

---

## Règles absolues

1. **Les scans ne vont PAS dans optimatrix** — ils encodent la logique SSM (monoid, ZOH).
2. **Les types scan vivent dans `include/scan.h`** — jamais dans `optimatrix.h`.
3. **`gemm_avx2` accumule** : `C += A*B`. Zéroïser C avant si nécessaire.
4. **`optimatrix.h` a un guard `extern "C"`** pour compatibilité NVCC — ne pas le retirer.
5. **Exécutables de test** : compilés avec `-no-pie` (ASM 32-bit incompatible avec PIE).
6. **`OPTIMATRIX_BUILD_CUDA` s'active via `sed` sur CMakeCache**, pas via flag CLI.
7. **Toujours `-O3 -mavx2`** — sans ça, les kernels ASM perdent leur intérêt.
8. **Pas de Python, pas de dépendances externes** (libc + libm uniquement).
9. **k-mamba est une bibliothèque** — pas de CLI dans le projet principal.
10. **Ne rien mettre de "lourd" dans k-mamba** — si ça boucle des millions de fois, ça va dans optimatrix.

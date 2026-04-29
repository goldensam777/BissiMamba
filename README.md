<div style="text-align:center">

# k‑mamba

**Topologie Wavefront pour la Modélisation Causale N‑Dimensionnelle Universelle & Alignement Mamba‑3 (2026)**

[![Version](https://img.shields.io/badge/version-0.3.0-blue)](https://github.com/goldensam777/k-mamba)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-≥11.0-brightgreen)](https://developer.nvidia.com/cuda-toolkit)
[![CPU](https://img.shields.io/badge/CPU-AVX2%2B-blueviolet)]()

[Article](#citation) · [Théorie](THEORY.md) · [API](#démarrage-rapide) · [Benchmarks](#performance)

*Samuel YEVI* · IFRI-UAC, Bénin · [goldensam777@github](https://github.com/goldensam777)

</div>
---

## Résumé

**k‑mamba** introduit un cadre topologique unifié pour la modélisation par espaces d'états en N dimensions, fondé sur l'**ordonnancement wavefront**. Ce projet est le premier à proposer une implémentation C/CUDA native des innovations de **Mamba‑3 (Mars 2026)**, notamment la discrétisation Exponential‑Trapezoidal et les dynamiques complexes (RoPE).

Contrairement aux approches par décomposition (VMamba, Mamba‑ND), k‑mamba établit une **récurrence ND simultanée** où les états latents dépendent de tous leurs prédécesseurs topologiques immédiats à la fois, sans sacrifier le parallélisme grâce au théorème de topologie wavefront.

**Contributions principales :**
1. **Support Mamba‑3 Complet :** Discrétisation Exponential‑Trapezoidal (2nd ordre), complexe‑valued RoPE (angles appris) et formulation MIMO.
2. **Théorie Wavefront :** Caractérisation prouvant que les opérateurs causaux ND sont parfaitement parallélisables par niveau (Corollaire 4.3, THEORY.md).
3. **Architecture Unifiée :** Un seul squelette topologique partagé entre `scanND` (mémoire longue) et `convND` (interactions locales).
4. **Philosophie Zero‑Dependency :** Implémentation pure (C, x86‑64 ASM AVX2, CUDA) sans dépendances externes (pas de PyTorch, pas d'OpenBLAS).

---

## Fonctionnalités Mamba‑3 (Natif)

k‑mamba implémente les trois piliers de **Mamba‑3** :

*   **Discrétisation Exponential‑Trapezoidal :** Utilise une approximation de second ordre via `lambda_proj`, capturant les patterns locaux sans nécessiter de convolutions courtes externes.
*   **Complex‑Valued State Updates (RoPE) :** Applique des rotations appris (`theta`) sur les projections d'état, doublant l'efficacité de stockage par rapport à Mamba‑2.
*   **MIMO (Multi‑Input Multi‑Output) :** Support d'un `mimo_rank > 1` pour augmenter l'intensité arithmétique et la précision sans augmenter la latence de décodage.

---

## Démarrage Rapide

### Installation

```bash
git clone https://github.com/goldensam777/k-mamba.git
cd k-mamba

# Compilation intelligente (auto-détecte CUDA/AVX2)
make

# Tests de validation (Mamba-3 & GPU)
make test-mamba3-gpu
```

### Bundle runtime (libs + binaires + config)

```bash
make export-runtime-bundle BUNDLE_DIR=dist/runtime BUNDLE_CONFIG=configs/cifar10.json

cd dist/runtime
./model --config config.json --serialize ser
./train --config config.json --data data/
```

Le bundle exporte:
- `libkmamba.a`, `libkser.a`, `libtrain.a`
- `model`, `train`
- `config.json`
- `inference/` (dossier cible pour `*.ser`)

### 1. Création d'un Modèle Mamba‑3

```c
#include <kmamba.h>

KMambaConfig cfg;
kmamba_config_set_defaults(&cfg);
cfg.dim = 512;
cfg.state_size = 64;
cfg.n_layers = 12;
cfg.mimo_rank = 2; // Activation MIMO (Mamba-3)

// Activation RoPE (Complex SSM)
cfg.use_rope = 1; 

// Topologie spatiale 2D
cfg.spatial_ndims = 2;
cfg.spatial_dims[0] = 32;
cfg.spatial_dims[1] = 32;

KMamba *model = kmamba_create(&cfg);
kmamba_init(model, 42);
```

### 2. Entraînement avec Gradient Checkpointing

```c
#include <trainer.h>

TrainerGCConfig gc_cfg = {
    .policy = TRAINER_GC_MODERATE,
    .checkpoint_every_n = 2 // Économise 50% de VRAM
};

Trainer *trainer = trainer_create(model, &gc_cfg);
trainer_run(trainer, tokens, targets, n_samples, ...);
```

---

## Architecture en Trois Couches

k‑mamba repose sur une séparation stricte des responsabilités :

1.  **Orchestration (src/) :** Logique modèle et API (C pur). Code trivial et lisible.
2.  **Topologie (src/km_topology.c) :** Indexation ND et scheduling wavefront. Primitive mère partagée.
3.  **Kernels (kernels/, cuda/, cpu/) :** Moteur de calcul ultra‑optimisé (ASM, CUDA, C inline).

**Règle d'or :** Si ça boucle des millions de fois, ça va dans `kernels/`. Sinon, ça va dans `src/`.

---

## Philosophie Zero‑Dependency

Le projet refuse l'obésité logicielle :
*   ❌ **Pas de PyTorch/TensorFlow** : Backprop et optimiseurs (MUON, AdamW) écrits à la main.
*   ❌ **Pas de CMake** : Un Makefile simple de 20 lignes.
*   ❌ **Pas de BLAS externe** : GEMM optimisé via AVX2/FMA dans `kernels/`.

Le résultat est un binaire compact, prévisible et portable, capable de tourner aussi bien sur un serveur Azure A100 que sur une machine locale modeste.

### 3. Primitives ND

```c
// ScanND : récurrence N-dimensionnelle simultanée
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
scannd(&scan);  // Exécution wavefront automatique

// ConvND : convolution N-dimensionnelle dense (même squelette)
ConvNDParams conv = {
    .input = in, .kernel = K, .bias = NULL, .output = out,
    .dims = dims, .ndims = 2, .D = 64, .K = 3
};
convnd(&conv, CONVND_FORWARD);
```

---

## Positionnement Scientifique

### Le Théorème de Topologie Wavefront

Pour un point `n = (n₁, ..., n_N)` sur une grille ND régulière, définissons le **niveau wavefront** :

```
l(n) = n₁ + n₂ + ... + n_N
```

**Théorème (Caractérisation des opérateurs causaux) :** Pour tout opérateur O sur une grille ND régulière, les propositions suivantes sont équivalentes :
- (i) O est exécutable par parcours wavefront niveau par niveau avec parallélisme intra-niveau exact
- (ii) Le graphe de dépendances de O est un sous-graphe du DAG causal défini par l(m) < l(n)
- (iii) Toutes les dépendances de tout point n pointent strictement vers des niveaux inférieurs

**Preuve :** Voir THEORY.md, Section 0.3. ∎

**Corollaire (Parallélisme intra-niveau) :** Tous les points d'un même niveau wavefront sont mutuellement indépendants et parallélisables. Sur une grille d×d, la largeur du niveau atteint Θ(d), fournissant un parallélisme substantiel.

### Comparaison avec l'État de l'Art

| Approche | Mécanisme | Causalité | Parallélisme |
|----------|-----------|-----------|--------------|
| Mamba‑1 (Gu & Dao, 2023) | Chaîne 1D linéaire | Séquentielle | Inter-batch seulement |
| VMamba (Liu et al., 2024) | 4 scans dans 4 directions | Compositionnelle | Aucun intra-niveau |
| Mamba‑ND (Li et al., 2024) | Scans 1D alternés par dimension | Factorisée | Aucun intra-niveau |
| Mamba‑2 (Dao & Gu, 2024) | SSD matriciel | Tensor cores | Hardware uniquement |
| **k‑mamba (YEVI)** | **Récurrence ND simultanée** | **Ordre partiel (wavefront)** | **Exact intra-niveau (théorème)** |

---

## Architecture

### Conception en Trois Couches

```
┌─────────────────────────────────────────┐
│  Orchestration (src/)                   │
│  Logique modèle, API, boucle entraînement │
├─────────────────────────────────────────┤
│  Topologie (src/km_topology.c)          │
│  Indexation ND, ordonnancement wavefront │
├─────────────────────────────────────────┤
│  Kernels (kernels/, cuda/, cpu/)        │
│  GEMM, activations, optimiseurs          │
└─────────────────────────────────────────┘
```

**Règle d'or :** Si c'est trivial, ça va dans `src/`. Si ça boucle des millions de fois, ça va dans `kernels/`.

### Structure du Projet

```
k-mamba/
├── include/              # Headers API publique
│   ├── kmamba.h          # API principale
│   ├── scan_nd.h         # Interface scan ND
│   ├── convnd.h          # Interface convolution ND
│   └── wavefront_plan.h  # Générateur wavefront
├── src/                  # Couche orchestration
│   ├── kmamba.c          # Forward/backward modèle
│   ├── mamba_block.c     # Implémentation bloc SSM
│   ├── kmamba_ser.c      # Sérialisation (format .ser)
│   └── scan_nd.c         # ScanND CPU de référence
├── cuda/                 # Kernels GPU
│   ├── scan_nd.cu        # Scan wavefront CUDA
│   ├── mamba_block.cu    # Forward/backward GPU complet
│   └── kmamba_kernels.cu # Embedding, head, loss
├── kernels/              # Calcul sans dépendance
│   ├── gemm_f32.c        # GEMM/GEMV en C pur
│   ├── activations_f32.c # SiLU, ReLU, Sigmoid
│   └── optimizer_f32.c   # AdamW, MUON, Newton-Schulz
├── libs/
│   ├── kser/             # Bibliothèque sérialisation binaire
│   └── train_set/        # Trainer avec gradient checkpointing
├── tokenizer_rs/         # Tokenizer hybride (Rust FFI)
│   └── src/lib.rs        # Modes Bytes32K + Tiktoken100K
├── configs/              # Configurations JSON
│   ├── cifar10.json
│   └── synthetic_2d.json
└── tests/                # Tests unitaires et d'intégration
```

---

## Fonctionnalités Avancées

### Architecture Mamba-3

L'implémentation inclut des améliorations architecturales récentes :

```c
// BCNorm : biais appris après RMSNorm
float *b_B;  // [state_size] — biais pour projection B
float *b_C;  // [state_size] — biais pour projection C

// SSM Complexe / RoPE : angles de rotation appris
float *theta;  // [state_size/2] — angles par paire

// Discrétisation Exp-Trapezoïdale
MBMatrix lambda_proj;  // Projette x_t -> scalaire lambda_t ∈ [0,1]
```

### Entraînement en Précision Mixte

| Format | Plage | Précision | Loss Scaling | Accélération |
|--------|-------|-----------|--------------|--------------|
| FP32 | ±3.4e38 | 23 bits | Non | 1× |
| FP16 | ±65504 | 10 bits | **Obligatoire** | **16× (Tensor Cores)** |
| BF16 | ±3.4e38 | 7 bits | Non | 16× |

```c
// FP16 avec gradient scaling
cfg.use_fp16 = 1;
cfg.loss_scale = 65536.0f;

// BF16 : meilleure stabilité, pas de scaling nécessaire
cfg.use_bf16 = 1;
```

### Gradient Checkpointing

```c
#include <trainer.h>

TrainerGCConfig gc_cfg = {
    .policy = TRAINER_GC_MODERATE,  // ou AGGRESSIVE, NONE
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

**Théorème (Checkpoint optimal) :** Pour L couches avec mémoire d'activations M, le placement uniforme de checkpoints tous les k = ⌈LM/B⌉ couches minimise le surcoût computationnel pour un budget mémoire B. Voir THEORY.md, Section 7.3.

### Tokenizer Hybride (Rust FFI)

```c
// Initialisation du tokenizer
kmamba_tokenizer_init("bytes");     // 32K tokens, robustesse locale
kmamba_tokenizer_init("cl100k");    // 100K Tiktoken, déploiement cloud

// Encodage/décodage
size_t len;
uint32_t *tokens = kmamba_encode("Hello world", &len);
char *text = kmamba_decode(tokens, len);

// Libération
kmamba_free_tokens(tokens, len);
kmamba_free_string(text);
```

### Optimiseurs Multiples

```c
typedef enum {
    OPTIMIZER_ADAM_CLIP,  // AdamW + gradient clipping
    OPTIMIZER_MUON,       // Orthogonalisation Newton-Schulz
    OPTIMIZER_SGD,        // Vanilla avec momentum
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

### ConvND : Dense vs Séparable

Benchmark sur grilles 2D (D=64, K=3, parallélisme OpenMP) :

| Taille Grille | Dense (ms) | Séparable (ms) | Accélération |
|---------------|-----------|----------------|--------------|
| 64×64 | 127,80 | 1,93 | **66×** |
| 256×256 | 135,51 | 34,62 | **3,9×** |
| 1024×1024 | 3497,63 | 807,60 | **4,3×** |

**Constat :** Sur CPU, la séparable l'emporte (moins d'opérations). Sur GPU, la dense gagne (un seul kernel, meilleure coalescence). k‑mamba fournit les deux.

### Options de Compilation

```bash
# Approximation exponentielle rapide
make FAST_EXP=1

# Forcer CPU uniquement (pas de CUDA)
make CPU_ONLY=1

# Compilation debug
make CFLAGS="-O0 -g -DDEBUG"
```

---

## Zéro Dépendance

k‑mamba nécessite uniquement :
- **Compilation :** GCC ≥ 11 ou Clang ≥ 12, NASM ≥ 2.15
- **GPU :** CUDA Toolkit ≥ 11.0, cuBLAS
- **Optionnel :** Cargo (pour le tokenizer Rust)

**Non requis :** Python, PyTorch, TensorFlow, OpenBLAS, MKL, NCCL, CMake.

```bash
# Ubuntu/Debian
sudo apt-get install gcc nasm

# Arch
sudo pacman -S gcc nasm

# macOS
brew install gcc nasm
```

---

## Tests

```bash
# Tests unitaires
make tests

# Suites de tests individuelles
make test-mamba3              # Correction forward/backward
make test-mamba3-gpu          # Correspondance numérique GPU
make test-scan-nd-regression  # Régression wavefront
make test-gradient            # Vérification gradients
make test-trainer-gc          # Gradient checkpointing

# Benchmarks
make bench-gates              # Benchmarks CPU gates
make bench-convnd-cpu         # Performance ConvND
make bench-convnd-cuda        # ConvND GPU
```

---

## Citation

```bibtex
@software{k_mamba_2024,
  author = {YEVI, Samuel},
  title = {k-mamba : Topologie Wavefront pour la Modélisation Causale N‑Dimensionnelle Universelle},
  year = {2024},
  url = {https://github.com/goldensam777/k-mamba},
  note = {Implémentation C/CUDA sans dépendances des modèles d'espaces d'états ND simultanés}
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

## Références

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
2. Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024.
3. Liu, Y. et al. (2024). *VMamba: Visual State Space Model*. arXiv:2401.10166.
4. Li, S., Singh, H., & Grover, A. (2024). *Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data*. ECCV 2024.
5. Blelloch, G. (1990). *Prefix Sums and Their Applications*. CMU-CS-90-190.
6. Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost*. arXiv:1604.06174.

---


**Optima, Immo, Absoluta Perfectio**


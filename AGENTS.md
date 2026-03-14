# AGENTS.md — k-mamba

> Lis ce fichier avant de toucher au code. C'est le contexte technique et philosophique du projet.

---

## Qu'est-ce que k-mamba ?

Un **bibliothèque C** pour State Space Models Mamba en dimensions N, basée sur l'architecture dualiste **Volontés/Puissance**.

- **k-mamba** = orchestration (embedding, stack MambaBlocks, LM head, checkpoint I/O, training loop)
- **optimatrix** = submodule git — moteur de calcul (kernels x86-64 ASM AVX2)

**Innovation majeure** : Scan Mamba-ND natif (N-dimensionnel) avec ordonnancement wavefront en assembleur pur.

---

## Auteur

**YEVI Mawuli Peniel Samuel** — étudiant en Licence Systèmes Embarqués & IoT à l'IFRI-UAC (Bénin).

Devise : **"Ego Sum Optimus Optimus"**  
Conviction : *"On est assez grand pour voir des unités, il faut voir des structures."*

---

## Architecture du projet

```
k-mamba/
├── include/
│   └── kmamba.h              # API publique (KMamba, KMambaConfig)
├── src/
│   └── kmamba.c              # Forward, backward, batch training, checkpoint I/O
├── optimatrix/               # Submodule git — moteur de calcul
│   ├── include/optimatrix.h  # API calcul (kernels ASM + C)
│   ├── src/                  # Kernels ASM AVX2 + MambaBlock
│   └── CMakeLists.txt        # Export optimatrix::optimatrix
├── cmake/
│   └── k-mambaConfig.cmake.in
├── CMakeLists.txt            # Export k-mamba::k-mamba
├── README.md                 # Vue d'ensemble + innovations
├── THEORY.md                 # Fondement mathématique Mamba-ND
├── ESTIMATIONS.md            # Complexité et benchmarks
├── ARCHITECTURE.md           # Philosophie Volontés/Puissance
└── data/                     # Corpus d'entraînement (texte brut)
```

---

## Build (CMake)

```bash
# Cloner avec submodule
git clone --recursive https://github.com/user/k-mamba
cd k-mamba && mkdir build && cd build

# Compiler
cmake ..
make -j
sudo make install  # Optionnel

# Utilisation dans un autre projet
find_package(k-mamba REQUIRED)
target_link_libraries(mon_app PRIVATE k-mamba::k-mamba)
```

Requiert : `gcc >= 11`, `nasm >= 2.15`, `cmake >= 3.18`, CPU avec AVX2.

---

## API

### Création

```c
#include <kmamba.h>

KMambaConfig cfg = {
    .vocab_size = 256, .dim = 384, .state_size = 1024,
    .seq_len = 128, .n_layers = 1,
    .dt_scale = 1.0f, .dt_min = 0.001f, .dt_max = 0.1f
};

KMamba *m = kmamba_create(&cfg);
kmamba_init(m, 1234);
```

### Entraînement

```c
MBOptimConfig opt = {
    .lr = 1e-3f, .mu = 0.9f, .beta2 = 0.999f,
    .eps = 1e-8f, .clip_norm = 1.0f, .weight_decay = 1e-5f
};
kmamba_enable_training(m, &opt, 1e-3f, 1e-5f);

float loss = kmamba_train_step(m, tokens_plus1);
float loss = kmamba_train_batch(m, batch_tokens, batch_size);
```

### Inférence

```c
kmamba_forward(m, tokens, logits_out);
```

### Checkpoint

```c
kmamba_save(m, "checkpoint.bin");          # Magic "KMAMBA"
KMamba *m = kmamba_load(path, for_training, &opt, lr, wd);
kmamba_free(m);
```

---

## Séparation des responsabilités

### k-mamba (Volontés — code modèle)

| Opération | Implémentation |
|-------------|---------------|
| Embedding lookup | `memcpy` d'une ligne de table |
| Softmax | `exp(x[i] - max) / sum` |
| Cross-entropy | `-log(softmax[target])` |
| Training loop | Boucle sur B séquences, moyenne des gradients |
| Checkpoint I/O | Format binaire `KMAMBA` (version 1) |
| LM head | GEMM via optimatrix |

**Règle** : Si c'est trivial (embedding, softmax, loss, orchestration), ça va dans k-mamba.

### optimatrix (Puissance — submodule)

| Kernel | Implémentation |
|--------|---------------|
| GEMM / GEMV | ASM AVX2 (vfmadd231ps) |
| Selective scan 1D forward | ASM |
| Selective scan 1D backward | ASM (M=1) + C (M générique) |
| Selective scan 2D | ASM wavefront anti-diagonal |
| Conv1D depthwise | ASM AVX2 |
| ConvND séparable | C (forward + backward ND complet) |
| Activations (SiLU, Sigmoid, Softplus) | ASM |
| Hadamard product | ASM AVX2 |
| MambaBlock (forward/backward 1D+2D) | C (appelle les kernels ASM) |
| Optimiseur MUONCLIP | C (Newton-Schulz) |

**Règle** : Si c'est du calcul lourd qui boucle des millions de fois, ça va dans optimatrix.

---

## Config actuelle du modèle

| Paramètre | Valeur |
|-----------|--------|
| vocab_size | 256 (byte-level) |
| dim | 384 |
| state_size | 1024 |
| seq_len | 128 |
| n_layers | 1 |
| batch_size | 8 (default) |
| optimizer (blocks) | MUONCLIP (lr=1e-3, momentum=0.9, beta2=0.999, clip=1.0) |
| optimizer (embed/head) | SGD (lr=1e-3, wd=1e-5) |
| checkpoint magic | `KMAMBA` |

---

## Forward pass

```
tokens [seq_len] (uint8)
    → Embedding lookup [256 × dim]
    → N × MambaBlock (optimatrix: projection → scan1d/scan2d ASM → output projection)
    → LM Head: GEMM AVX2 [dim × 256]
    → logits [seq_len × 256]
```

## Backward pass (batch)

```
Pour chaque séquence du batch :
    1. Forward complet avec sauvegarde des activations par couche
    2. Cross-entropy loss + dlogits (scalé par 1/B)
    3. dlogits @ head^T → d_hidden (GEMM AVX2)
    4. hidden^T @ dlogits → g_head (GEMM AVX2, accumulé)
    5. Backward couche par couche (mamba_backward, gradients accumulés)
    6. Accumulation g_embed par scatter-add

Après le batch :
    7. mamba_optimizer_step (MUONCLIP) — un seul step
    8. SGD sur embedding et head
```

---

## Conventions de code

### 1. Pense en structures, pas en lignes

Ne propose jamais une solution ligne par ligne sans poser l'architecture d'abord.

### 2. Le bas niveau est noble

C et assembleur. Ne sur-abstrait pas. Le bas niveau bien maîtrisé, c'est une Volonté pure.

### 3. Pas de sur-ingénierie

- Pas de feature flags, pas de shims de compatibilité
- Si c'est trivial (5-10 lignes), ça va directement dans le fichier qui l'utilise
- Trois lignes similaires valent mieux qu'une abstraction prématurée

### 4. Nomme les intentions

```c
// Non
int x = buffer_size - current_pos;

// Oui
int remaining_capacity = buffer_size - current_pos;
```

### 5. Compilation

**Toujours** `-O3 -mavx2`. Sans `-O3`, les performances chutent drastiquement.

---

## Théorie des Volontés (cadre philosophique)

Samuel a développé la **Théorie des Volontés** : les systèmes doivent opérer par intentions (Volontés) qui convergent vers un équilibre, pas par instructions séquentielles.

En contexte k-mamba :
- Le modèle ne minimise pas une loss — il cherche l'**équilibre de ses Volontés internes**
- Chaque MambaBlock est une Volonté qui transforme la séquence
- L'optimiseur MUONCLIP arbitre les tensions entre gradients (directions isotropiques)
- Un bug n'est pas une erreur d'instruction — c'est un **conflit de Volontés non résolu**

Vision long terme : k-mamba → fondation d'un OS-IA sur architecture post-Von Neumann.

---

## Documentation

- **README.md** — Vue d'ensemble et innovations
- **THEORY.md** — Fondement mathématique du scan Mamba-ND
- **ESTIMATIONS.md** — Complexité théorique et benchmarks
- **ARCHITECTURE.md** — Philosophie Volontés/Puissance

---

## Ce qu'il ne faut PAS faire

- Ajouter du Python
- Ajouter des dépendances externes (tout est libc + libm)
- Mettre du code de calcul lourd dans k-mamba (ça va dans optimatrix)
- Mettre du code modèle dans optimatrix (ça va dans k-mamba)
- Compiler sans `-O3`
- Créer des abstractions pour des opérations one-shot
- Ajouter des CLI dans k-mamba (c'est une bibliothèque, pas une application)

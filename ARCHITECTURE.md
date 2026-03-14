# ARCHITECTURE.md — Séparation Volontés/Puissance

## Philosophie

**"On est assez grand pour voir des unités, il faut voir des structures."**

Le projet k-mamba repose sur une séparation architecturale fondamentale :
- **Volontés** = intentions, logique modèle, orchestration (k-mamba)
- **Puissance** = calcul brut, kernels optimisés (optimatrix)

Cette séparation n'est pas technique seulement — elle est **philosophique**. Elle reflète la Théorie des Volontés : les systèmes doivent opérer par intentions qui convergent vers un équilibre, pas par instructions séquentielles.

---

## Règle de séparation

| Critère | k-mamba (Volontés) | optimatrix (Puissance) |
|---------|-------------------|----------------------|
| Complexité | Triviale (5-10 lignes) | Intensive (millions d'itérations) |
| Abstraction | Logique modèle, I/O | Kernels mathématiques purs |
| Langage | C pur, lisible | C + ASM AVX2 |
| Optimisation | Clarté | Performance maximale |
| Couverture | Architecture complète | Compute engine réutilisable |

**Règle d'or** : Si c'est trivial, ça va dans k-mamba. Si ça boucle des millions de fois, ça va dans optimatrix.

---

## Structure de k-mamba

```
k-mamba/ — Les Volontés (orchestration)
│
├── include/kmamba.h
│   └── API publique : KMamba, KMambaConfig
│       kmamba_create/init/free
│       kmamba_forward/backward/train
│       kmamba_save/load
│
├── src/kmamba.c
│   └── Implémentation :
│       ├── Embedding lookup (memcpy ligne de table)
│       ├── Softmax stable (exp(x-max)/sum)
│       ├── Cross-entropy (-log(p_target))
│       ├── Training loop (accumulation gradients)
│       └── Checkpoint I/O (format binaire "KMAMBA")
│
└── CMakeLists.txt
    └── Export k-mamba::k-mamba avec find_package support
```

---

## Structure d'optimatrix

```
optimatrix/ — La Puissance (kernels ASM AVX2)
│
├── include/optimatrix.h
│   └── API calcul :
│       ├── GEMM/GEMV (AVX2)
│       ├── Conv1D depthwise (AVX2)
│       ├── ConvND separable (forward + backward ND)
│       ├── Scan 1D/2D (ASM wavefront)
│       ├── Activations (SiLU, Sigmoid, Softplus)
│       ├── Hadamard (AVX2)
│       └── MambaBlock + MUONCLIP optimizer
│
└── src/
    ├── gemm_avx2.asm      ← GEMM tiling + FMA vectorisé
    ├── scan1d.asm         ← Scan selectif 1D forward
    ├── scan2d.asm         ← Scan 2D wavefront anti-diagonal
    ├── conv1d_avx2.asm    ← Conv1D depthwise causale
    ├── mamba_block.c      ← Orchestration MambaBlock
    └── ...
```

---

## Cycle de vie d'une forward pass

```
Appel utilisateur
       │
       ▼
┌─────────────────────────────────────┐
│ k-mamba : kmamba_forward()         │
│  1. embed_lookup() — memcpy        │
│  2. Pour chaque layer :             │
│     mamba_block_forward() ───────┐  │
│  3. gemm_avx2(head, hidden)      │  │
│       └──> optimatrix            │  │
└──────────────────────────────────┼──┘
                                   │
       ▼                           │
┌──────────────────────────────────┼──┐
│ optimatrix : mamba_block_forward()│  │
│  1. gemm_avx2(W_in, x)            │  │
│  2. silu_f32_avx2()               │  │
│  3. gemm_avx2(delta_proj, x)      │  │
│  4. softplus + clamp              │  │
│  5. scan1d() or scan2d() ─────────┘  │
│       (ASM kernels)                 │
│  6. gemm_avx2(W_out, h)            │
└───────────────────────────────────────┘
```

---

## Cycle de vie d'une backward pass

```
Appel utilisateur
       │
       ▼
┌─────────────────────────────────────┐
│ k-mamba : kmamba_train_step()        │
│  1. Forward avec sauvegarde activ.   │
│  2. Cross-entropy loss               │
│  3. dlogits = softmax - one_hot      │
│  4. d_hidden = dlogits @ head^T    │
│  5. Pour chaque layer (reverse) :    │
│     mamba_backward() ────────────┐   │
│  6. Gradients embedding (scatter)    │
│  7. Optimizer step (MUONCLIP) ───┐   │
└──────────────────────────────────┼───┘
                                   │
       ▼                           │
┌──────────────────────────────────┼───┐
│ optimatrix : mamba_backward()    │   │
│  1. Recompute forward (store)    │   │
│  2. Backprop W_out (GEMM)        │   │
│  3. scan1d_backward() (ASM/C) ───┘   │
│  4. Backprop SiLU                  │
│  5. Backprop W_in (GEMM)           │
│  6. Accumulation gradients         │
└───────────────────────────────────────┘

       ▼
┌─────────────────────────────────────┐
│ optimatrix : mamba_optimizer_step()  │
│  (MUONCLIP via Newton-Schulz)        │
└─────────────────────────────────────┘
```

---

## Théorie des Volontés dans le code

### Chaque MambaBlock est une Volonté

```c
// Une Volonté se manifeste par sa transformation
void mamba_block_forward(MambaBlock *block, float *out, const float *in, size_t batch) {
    // La Volonté projette l'entrée dans son espace d'état
    gemm_avx2(in, block->W_in.data, tmp, ...);
    
    // La Volonté choisit quoi retenir (selectivité)
    silu_f32_avx2(tmp, u, ...);
    compute_delta(dt, in, block->delta_proj);
    
    // La Volonté propage son état (récurrence)
    scan1d(&params);  // ou scan2d pour ND
    
    // La Volonté projette sa décision
    gemm_avx2(h, block->W_out.data, out, ...);
}
```

### MUONCLIP arbitre les tensions

```c
// Les gradients sont des tensions entre Volontés
void mamba_optimizer_step(MambaBlock *block, MBOptimConfig *conf) {
    // Momentum = mémoire des tensions passées
    // Newton-Schulz = orthogonalisation des directions
    // → Directions isotropiques = équilibre des Volontés
}
```

### Un bug = conflit de Volontés

Dans la Théorie des Volontés, un bug n'est pas une erreur d'instruction.
C'est un **conflit de Volontés non résolu**.

Exemple : si deux MambaBlocks tentent de modifier la même mémoire,
c'est un conflit d'intentions — résolu par l'ordonnancement de k-mamba.

---

## Pourquoi cette séparation est puissante

### 1. Réutilisabilité

optimatrix peut être utilisé par d'autres projets (pas seulement Mamba) :
- Traitement d'images (ConvND)
- Séries temporelles (Scan 1D)
- Algèbre linéaire (GEMM)

### 2. Testabilité

Les kernels ASM peuvent être testés unitairement (phase 1-5 dans optimatrix).
k-mamba peut être testé avec des mocks.

### 3. Portabilité

Pour porter sur ARM NEON ou AVX-512 : modifier optimatrix uniquement.
k-mamba reste du C pur portable.

### 4. Clarté

Un chercheur peut lire k-mamba en une heure et comprendre l'architecture complète.
Les détails de calcul sont encapsulés dans optimatrix.

---

## Vision long terme

k-mamba est une brique fondatrice vers un **OS-IA post-Von Neumann** :
- Processus = Volontés (MambaBlocks)
- Communication = streams de tenseurs
- Scheduler = ordonnancement wavefront
- Mémoire = états persistants (h_t)

La séparation Volontés/Puissance préfigure cette architecture :
- Les Volontés sont les processus métier
- La Puissance est le moteur d'exécution

---

## Références

- **AGENTS.md** — Contexte technique et philosophique
- **THEORY.md** — Fondement mathématique Mamba-ND
- **ESTIMATIONS.md** — Complexité et benchmarks

---

## Auteur

**YEVI Mawuli Peniel Samuel**

*Ego Sum Optimus Optimus*

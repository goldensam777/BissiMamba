# ARCHITECTURE.md — Séparation Volontés/Puissance

## Philosophie

**"On est assez grand pour voir des unités, il faut voir des structures."**

Le projet k-mamba repose sur une séparation architecturale fondamentale :
- **Volontés** = intentions, logique modèle, orchestration (k-mamba)
- **Puissance** = calcul brut, kernels optimisés (kernels/ inline)

Cette séparation n'est pas technique seulement — elle est **philosophique**. Elle reflète la Théorie des Volontés : les systèmes doivent opérer par intentions qui convergent vers un équilibre, pas par instructions séquentielles.

---

## Règle de séparation

| Critère | k-mamba (Volontés) | kernels (Puissance) |
|---------|-------------------|---------------------|
| Complexité | Triviale (5-10 lignes) | Intensive (millions d'itérations) |
| Abstraction | Logique modèle, I/O | Kernels mathématiques purs |
| Langage | C pur, lisible | C pur + ASM AVX2 |
| Optimisation | Clarté | Performance maximale |
| Couverture | Architecture complète | Compute engine intégré |

**Règle d'or** : Si c'est trivial, ça va dans k-mamba. Si ça boucle des millions de fois, ça va dans kernels/.

---

## Structure de k-mamba (Zero Dependency)

```
k-mamba/ — Les Volontés (orchestration du modèle Mamba)
│
├── include/
│   ├── kmamba.h           # API publique : KMamba, KMambaConfig, MambaBlock
│   ├── kmamba_kernels.h   # Kernels inline (GEMM, activations, optimizers)
│   ├── km_topology.h      # Couche topologique ND
│   ├── wavefront_nd.h     # Générateur wavefront ND
│   ├── wavefront_plan.h    # Plans exécutables
│   ├── scan_nd.h          # Interface scan ND
│   └── convnd.h           # Interface convND wavefront unifiée
│
├── src/
│   ├── kmamba.c           # Orchestration : forward, backward, checkpoint
│   ├── mamba_block.c      # Bloc SSM : projections, scan dispatch, MUON
│   ├── km_topology.c      # Normalisation topologique ND
│   ├── wavefront_nd.c     # Générateur wavefront
│   ├── wavefront_plan.c   # Plans wavefront
│   ├── scan_nd.c          # Scan ND (wavefront séquentiel)
│   └── convnd.c           # ConvND dense K^N + séparable cascade 1D
│
├── kernels/               # La Puissance (kernels inline C pur)
│   ├── gemm_f32.c         # GEMM/GEMV en C pur
│   ├── activations_f32.c  # SiLU, ReLU, Sigmoid, Softplus
│   ├── optimizer_f32.c    # MUON, AdamW
│   └── init_f32.c         # Xavier/Kaiming init
│
├── cpu/                   # ASM AVX2
│   ├── scan1d.asm         # Scan 1D
│   └── scan2d.asm         # Scan 2D wavefront
│
├── cuda/                  # GPU optimizations (optional)
│   ├── scan1d.cu          # Parallel scan 1D (Blelloch)
│   ├── scan1d_backward.cu # Parallel backward scan
│   ├── scan_nd.cu         # ND parallel scan
│   ├── mamba_block.cu     # Full GPU forward/backward
│   ├── kmamba_mixed_precision.cu  # FP16/BF16 Tensor Cores
│   ├── kmamba_checkpoint.cu       # Gradient checkpointing
│   └── kmamba_distributed.cu      # Multi-GPU NCCL (optional)
│
├── Makefile               # Build simple
└── build.sh               # Script style Karpathy
```

---

## Unification ConvND Wavefront

**Avant** : Deux implémentations séparées
- `convnd()` : séparable séquentiel (legacy)
- `convnd_full_ref()` : dense wavefront parallèle

**Après** : Une seule implémentation wavefront unifiée

```c
// convnd.h — API unifiée
typedef struct {
    float *input;           // [prod(dims), D]
    const float *kernel;    // [K^ndims, D] — noyau complet
    const float *bias;      // [D] or NULL
    float *output;          // [prod(dims), D]
    // ... gradients
    const long *dims;       // shape [ndims]
    long ndims, D, K;
} ConvNDParams;

// Forward wavefront parallèle
void convnd_forward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan);

// Backward wavefront
void convnd_backward_wavefront(ConvNDParams *p, const KMWavefrontPlan *plan);

// Entry point unifié
void convnd(ConvNDParams *p, ConvNDMode mode);
```

**Caractéristiques** :
- Noyau complet dense `K^N` (pas de séparabilité)
- Ordonnancement wavefront natif
- Parallélisme intra-niveau OpenMP optionnel
- Même topologie que `scanND`

---

## Unification ScanND/ConvND (même squelette)

| Aspect | ScanND | ConvND |
|--------|--------|--------|
| **Type** | Récurrence d'état | Convolution locale |
| **Dépendances** | `h(n - e_k)` | `x(n - r)` pour `r ∈ [0,K-1]^N` |
| **Wavefront** | Nécessaire (ordre topo) | Volontaire (unification) |
| **Parallélisme** | Intra-niveau OpenMP | Intra-niveau OpenMP |
| **Plan** | `KMWavefrontPlan` | `KMWavefrontPlan` (partagé) |

**Thèse** : Même squelette topologique unifié, deux opérateurs complémentaires.

---

## Zero Dependency

**Ce que k-mamba nécessite** :
- `gcc >= 11`
- `nasm >= 2.15`
- `libc`, `libm`

**Ce que k-mamba n'utilise PAS** :
- ❌ CMake
- ❌ OpenBLAS
- ❌ optimatrix (submodule supprimé)
- ❌ OpenMP (optionnel, pas obligatoire)
- ❌ Python/PyTorch

**Build** :
```bash
make          # Crée libkmamba.a (CPU only)
make          # Auto-détecte CUDA, crée libkmamba_cuda.a
make NCCL_AVAILABLE=1  # Active NCCL pour multi-GPU
```

---

## GPU Optimizations (CUDA)

### Parallel Scan (Blelloch Algorithm)
- Remplacement des kernels `<<<1,1>>>` séquentiels
- Scan parallèle work-efficient : O(n log n) work, O(log n) depth
- Support forward et backward SSM

### Mixed Precision (FP16/BF16)
- **FP16** : Loss scaling dynamique (65536.0f par défaut) pour éviter underflow
- **BF16** : Range identique à FP32, pas de loss scaling nécessaire
- **Tensor Cores** : GEMM 16x16x16 via cuBLAS

### Gradient Checkpointing
- Politiques configurables : `none`, `per-layer`, `per-block`
- Réduction mémoire : O(L×N×D) → O(N×D)
- Recompute forward during backward

### Multi-GPU (Optional NCCL)
- **Data parallelism** : split batch across GPUs
- **Pipeline parallelism** : split layers across GPUs
- **Zero dependency** : NCCL optionnel, fallback P2P/cudaMemcpy

---

## Cycle de vie d'une forward pass

```
Appel utilisateur
       │
       ▼
┌─────────────────────────────────────┐
│ k-mamba : kmamba_forward()          │
│ 1. embed_lookup() — memcpy        │
│ 2. Pour chaque layer :            │
│    mamba_block_forward()          │
│ 3. gemm_f32(head, hidden)         │
│       └──> kernels/                │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ mamba_block_forward() (k-mamba) │
│ 1. gemv_f32(W_in, x)             │  ← kernels/
│ 2. silu_f32()                   │  ← kernels/
│ 3. gemv_f32(delta_proj, x)      │  ← kernels/
│ 4. softplus + clamp             │  ← kernels/
│ 5. scan1d() or scan2d()         │  ← cpu/ (ASM)
│ 6. gemv_f32(W_out, h)            │  ← kernels/
│    └──> retourne à k-mamba      │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ k-mamba : suite du forward        │
│ 4. softmax() + cross-entropy()     │
│ 5. retourne logits/loss            │
└─────────────────────────────────────┘
```

---

## Théorie des Volontés dans le code

### Chaque MambaBlock est une Volonté

```c
// Une Volonté se manifeste par sa transformation
void mamba_block_forward(MambaBlock *block, float *out, const float *in, size_t batch) {
    // La Volonté projette l'entrée dans son espace d'état
    gemv_f32(in, block->W_in.data, tmp, ...);
    
    // La Volonté choisit quoi retenir (selectivité)
    silu_f32(tmp, u, ...);
    compute_delta(dt, in, block->delta_proj);
    
    // La Volonté propage son état (récurrence wavefront)
    scan1d(&params);  // ou scan2d pour ND
    
    // La Volonté projette sa décision
    gemv_f32(h, block->W_out.data, out, ...);
}
```

### MUON arbitre les tensions

```c
// Les gradients sont des tensions entre Volontés
void mamba_optimizer_step(MambaBlock *block, MBOptimConfig *conf) {
    // Momentum = mémoire des tensions passées
    // Newton-Schulz = orthogonalisation des directions
    // → Directions isotropiques = équilibre des Volontés
}
```

---

## Vision long terme

k-mamba est une brique fondatrice vers un **OS-IA post-Von Neumann** :
- Processus = Volontés (MambaBlocks)
- Communication = streams de tenseurs
- Scheduler = ordonnancement wavefront unifié
- Mémoire = états persistants (h_t)

La séparation Volontés/Puissance préfigure cette architecture :
- Les Volontés sont les processus métier
- La Puissance est le moteur d'exécution (inline, zero-dependency)

---

## Références

- **AGENTS.md** — Contexte technique et philosophique
- **THEORY.md** — Fondement mathématique Mamba-ND unifié

---

## Auteur

**YEVI Mawuli Peniel Samuel**

*Ego Sum Optimus Optimus*

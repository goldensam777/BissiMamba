# SOURCES.md — Références bibliographiques de K-Mamba

Références complètes pour les algorithmes, architectures et théories utilisés
dans K-Mamba et optimatrix.

---

## State Space Models (SSM)

**[SSM-1] Gu et al. (2021)**
*Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers.*
NeurIPS 2021.
Première formalisation pratique des SSM pour le deep learning.

**[SSM-2] Gu et al. (2022)**
*Efficiently Modeling Long Sequences with Structured State Spaces (S4).*
ICLR 2022.
Diagonalisation de A, paramétrage HiPPO, complexité O(L log L) via FFT.

**[SSM-3] Gu & Dao (2023)**
*Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
arXiv:2312.00752.
SSM sélectif (B, C, δ dépendent de l'entrée), scan parallèle hardware-aware,
complexité O(L) temps et mémoire. **Fondement de K-Mamba.**

**[SSM-4] Dao & Gu (2024)**
*Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality.*
ICML 2024.
SSD (State Space Duality) — unification SSM/attention.

---

## Extensions multi-dimensionnelles de Mamba

**[ND-1] Liu et al. (2024)**
*VMamba: Visual State Space Models.*
arXiv:2401.13260.
4 scans 1D dans 4 directions pour la vision. Approche : 4× Mamba1D, pas de vraie 2D native.
**K-Mamba se différencie par la récurrence native ND simultanée.**

**[ND-2] Li et al. (2024)**
*Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data.*
arXiv:2402.05892.
Scans 1D alternés par axe et par couche. Toujours séquentiel par axe, pas simultané.
**K-Mamba généralise par la récurrence dans toutes les dimensions au même pas.**

**[ND-3] Zhu et al. (2024)**
*Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model.*
arXiv:2401.13320.
Bidirectional Mamba pour la vision — double scan gauche/droite.

**[ND-4] Pei et al. (2024)**
*MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection.*
arXiv:2403.19888.

---

## Algorithmes de scan parallèle (Blelloch)

**[BLE-1] Blelloch, G.E. (1990)**
*Prefix Sums and Their Applications.*
Technical Report CMU-CS-90-190, Carnegie Mellon University.
**L'algorithme de K-Mamba sur CUDA.** Prefix scan parallèle en O(log L) profondeur et O(L) travail.
Opérateur associatif : `(a₁,b₁) ⊗ (a₂,b₂) = (a₁·a₂, a₂·b₁ + b₂)`.

**[BLE-2] Blelloch, G.E. (1993)**
*Segmented Operations for Parallel Processing.*
IEEE Transactions on Parallel and Distributed Systems.

**[BLE-3] Harris, M. et al. (2007)**
*Parallel Prefix Sum (Scan) with CUDA.*
GPU Gems 3, Chapter 39, NVIDIA.
Implémentation pratique en CUDA avec up-sweep / down-sweep et shared memory.
**Référence directe de notre implémentation `scan1d_blelloch_kernel`.**

**[BLE-4] Merrill, D. & Grimshaw, A. (2009)**
*Revisiting Sorting for GPGPU Stream Architectures.*
CS Technical Report CS2010-03, University of Virginia.
Optimisations CUDA pour le prefix scan large.

---

## Optimiseurs

**[OPT-1] Kingma & Ba (2015)**
*Adam: A Method for Stochastic Optimization.*
ICLR 2015.
AdamW est la variante avec weight decay découplé — implémenté dans `optimizer_utils.c/cu`.

**[OPT-2] Loshchilov & Hutter (2019)**
*Decoupled Weight Decay Regularization (AdamW).*
ICLR 2019.
`adamw_update` et `adamw_update_cuda` dans optimatrix.

**[OPT-3] Jordan, K. et al. (2025)**
*MUON: Momentum + Orthogonalization.*
Moonshot AI, arXiv:2502.16982.
Newton-Schulz orthogonalisation des gradients (5 itérations), momentum Nesterov.
**Implémenté nativement en C dans K-Mamba via MUONCLIP.**

**[OPT-4] Björck, Å. & Bowie, C. (1971)**
*An Iterative Algorithm for Computing the Best Estimate of an Orthogonal Matrix.*
SIAM Journal on Numerical Analysis 8(2):358–364.
Newton-Schulz converge en quelques itérations vers la matrice orthogonale la plus proche.

**[OPT-5] Pascanu et al. (2013)**
*On the difficulty of training recurrent neural networks.*
ICML 2013.
Justification théorique du gradient clipping — explosions dans les RNN/SSM profonds.

---

## Transformers (référence comparative)

**[TR-1] Vaswani et al. (2017)**
*Attention Is All You Need.*
NeurIPS 2017.
Complexité O(L²) — K-Mamba atteint O(L) avec qualité comparable.

---

## Architectures matérielles

**[HW-1] Intel Corporation (2013)**
*Intel Advanced Vector Extensions 2 (AVX2) Programming Reference.*
Document 319433-022US.
FMA (`vfmadd231ps`), broadcast, 256-bit SIMD — base des kernels optimatrix.

**[HW-2] NVIDIA Corporation (2023)**
*CUDA C++ Programming Guide, Release 12.0.*
Warp shuffles, shared memory, `__syncthreads()`, `atomicAdd` — base de scan1d.cu.

**[HW-3] Williams et al. (2009)**
*Roofline: An Insightful Visual Performance Model for Multicore Architectures.*
Communications of the ACM 52(4):65–76.
Modèle roofline utilisé pour l'analyse compute-bound vs memory-bound des kernels.

---

## Convolutions dépthwise séparables

**[CONV-1] Howard et al. (2017)**
*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.*
arXiv:1704.04861.
Conv dépthwise séparable — base de la ConvND séparable dans optimatrix.

**[CONV-2] Chollet, F. (2017)**
*Xception: Deep Learning with Depthwise Separable Convolutions.*
CVPR 2017.

---

## ZOH (Zero-Order Hold)

**[ZOH-1] Gu et al. (2022) [SSM-2]** — voir ci-dessus.
Discrétisation ZOH : `Ā = exp(δA)`, `B̄ ≈ δ·B`.

**[ZOH-2] Franklin, G.F. et al. (2019)**
*Feedback Control of Dynamic Systems, 8th ed.* Pearson.
Théorie des systèmes dynamiques discrets — fondement du ZOH.

---

## Références secondaires (SSM classiques)

**[CLS-1] Kalman, R.E. (1960)**
*A New Approach to Linear Filtering and Prediction Problems.*
Journal of Basic Engineering 82(1):35–45.
Filtre de Kalman — ancêtre des SSM modernes.

**[CLS-2] Hippo: Gu et al. (2020)**
*HiPPO: Recurrent Memory with Optimal Polynomial Projections.*
NeurIPS 2020.
Matrice HiPPO pour initialiser A — utilisée dans S4 et descendants.

---

## Format de citation dans le code

Les commentaires dans le code utilisent les balises `[REF]` :
- `[BLE-1]` / `[BLE-3]` — scan1d.cu (Blelloch up/down-sweep)
- `[SSM-3]` — scan1d.cu, scan1d_backward.cu (formule SSM sélectif)
- `[OPT-3]` — mamba_block.c (MUONCLIP)
- `[OPT-1/2]` — optimizer_utils.c/cu (Adam/AdamW)
- `[HW-1]` — gemm_avx2.asm, gemv_avx2.asm (vfmadd231ps)
- `[ZOH-1]` — scan1d.cu (discrétisation Ā = exp(δA))

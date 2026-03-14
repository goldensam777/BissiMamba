# ESTIMATIONS.md — Complexite et Efficacite de k-mamba

## Notations

```text
m, n, k   dimensions des matrices
L         longueur de la sequence (scan 1D)
d1, d2    dimensions spatiales (scan 2D)
D         dimension du modele (canaux)
M         dimension de l'etat cache
P         nombre total de positions : P = L  (1D) ou P = d1 x d2 (2D)
W         largeur SIMD : 8 pour AVX2 float32
K         taille du noyau de convolution
```

---

## 1. GEMV — Produit Matrice x Vecteur

### Complexite theorique

```
Total flops : 2mn
Memoire     : O(mn + m + n)
```

### Scalaire vs AVX2

| Implementation | Speedup mesure (64x64) |
|---|---|
| Scalaire | 1x (reference) |
| AVX2 | ~8x theorique (memory-bound) |

---

## 2. GEMM — Produit Matrice x Matrice

### Complexite theorique

```
Total flops : 2mnk
Memoire     : O(mk + kn + mn)
```

### Resultats mesures

| Implementation | Temps (64x64, 200 iter) | Speedup |
|---|---|---|
| Scalaire | 45.41 ms | 1x |
| AVX2     |  6.58 ms | **x6.9** |

Le speedup de x6.9 vient du pipeline ILP + cache-friendly outer-product + FMA vectorise.

---

## 3. Hadamard — Produit Element par Element

```
Total flops : n
Intensite   : 1 flop / 24 octets — tres memory-bound
```

Speedup AVX2 attendu : x2 a x4 (limite par la latence memoire).

---

## 4. Activations

| Activation | Goulot | Flops approx |
|---|---|---|
| ReLU | trivial | 1 |
| Sigmoid | exp | ~20 |
| SiLU | exp | ~22 |
| Softplus | exp + log | ~40 |

---

## 5. Selective Scan 1D

```
Appels exp  : L x D x M
Flops       : L x D x M x 6
Complexite  : O(L x D x M)
```

L=1024, D=128, M=16 -> 2 097 152 appels exp. L'exp est le facteur limitant.

---

## 6. Selective Scan 2D — Wavefront

```
Appels exp   : 2 x P x D x M  (2 par dimension)
Complexite   : O(P x D x M)
Memoire      : O(P x D x M)   — tous les etats stockes
```

| | Scan 1D | Scan 2D |
|---|---|---|
| Appels exp | P x D x M | 2 x P x D x M |
| Memoire h | D x M (1 etat) | P x D x M (tous) |
| Parallelisme | sequentiel | par diagonale |

---

## 7. Conv1D Depthwise Causale

```
Total flops : L x D x K x 2
Memoire     : O(L x D + K x D)
```

Pas d'appel a exp() — purement arithmetique. AVX2 vectorise sur D.

| Configuration | Verifications | Erreur max |
|---|---|---|
| L=4, D=3, K=2 | 24/24 | 0 |
| L=128, D=64, K=4 | 8192/8192 | 5.96e-08 |

### ConvND Separable

```
Forward  : O(P x D x K x ndims)
Backward : O(P x D x K x ndims) + recompute forward si pas de workspace
```

| Configuration | Verifications | Erreur max |
|---|---|---|
| 2D: H=4, W=5, D=3, K=2 | 60/60 | < 1e-5 |
| 4D: 3x4x3x5, D=8, K=2 | 1440/1440 | 7.45e-09 |
| Backward 1D: L=5, D=3, K=3 | 27/27 grad checks | < 6e-3 |
| Backward 2D: H=3, W=4, D=2, K=2 | 32/32 grad checks | < 6e-3 |

---

## 8. MambaBlock — Complexite du pipeline complet

### Forward 1D (par position t)

```
W_in projection  : O(state_size x dim)        — GEMV
SiLU activation  : O(state_size)
Delta projection : O(dim)                      — GEMV + softplus
Scan selectif    : O(state_size)               — 1 exp + recurrence
W_out projection : O(dim x state_size)         — GEMV

Total par position : O(dim x state_size)
Total sequence     : O(seq_len x dim x state_size)
```

### Forward 2D

```
Meme pipeline, scan2d au lieu de scan1d.
Total : O(d1 x d2 x dim x state_size)
Parallelisme wavefront : jusqu'a min(d1, d2) positions par diagonale.
```

### Backward

```
Recompute forward (store intermediaires) : O(seq_len x state_size)
Backprop scan : O(seq_len x state_size)
Backprop W_in, W_out : O(seq_len x dim x state_size) — GEMM
Total : O(seq_len x dim x state_size)
```

---

## 9. Comparaison Globale

| Operation | Complexite | Goulot | Vectorise |
|---|---|---|---|
| GEMV | O(mn) | memoire | AVX2 |
| GEMM | O(mnk) | calcul | AVX2 x6.9 |
| Hadamard | O(n) | memoire | AVX2 |
| ReLU | O(n) | trivial | maxsd |
| Sigmoid/SiLU | O(n) | exp() | — |
| Conv1D | O(L x D x K) | calcul | AVX2 vfmadd231ps |
| ConvND | O(P x D x K x ndims) | calcul | via Conv1D AVX2 |
| ConvND backward | O(P x D x K x ndims) | calcul | C (backward) |
| Scan 1D | O(L x D x M) | exp() | forward ASM |
| Scan 1D backward | O(L x D x M) | exp() + memoire | C/AVX2 |
| Scan 2D | O(P x D x M) | exp() + memoire | — (wavefront) |
| MambaBlock fwd | O(L x dim x state_size) | scan + GEMV | ASM partiel |
| MambaBlock bwd | O(L x dim x state_size) | GEMM + scan | C + ASM |

---

## 10. Roadmap des optimisations

```
Actuel :
  GEMM AVX2, Conv1D AVX2, ConvND separable forward+backward,
  Scans 1D/2D forward ASM, Backward 1D ASM+C,
  MambaBlock complet (forward/backward 1D+2D + MUONCLIP)

Prochaines etapes :
  1. exp() rapide AVX2 (approximation polynomiale) -> vectoriser scan2d
  2. Conv1D backward ASM -> sortir le hot path de C
  3. Tiling GEMM (cache L1) -> x2 supplementaire
  4. Parallelisme wavefront OpenMP -> scan 2D x n coeurs
  5. AVX-512 (si disponible) -> x2 vs AVX2
```

---

## 11. Comparaison avec l'etat de l'art

| Systeme | Approche | Scan ND | Conv ND | Backward ND |
|---|---|---|---|---|
| Mamba (2023) | CUDA, 1D seulement | non | 1D CUDA | 1D |
| VMamba (2024) | CUDA, 4 directions 1D | partiel | 1D CUDA | 1D |
| **k-mamba** | ASM x86-64, vrai 2D wavefront | **oui** | **oui, separable ND** | **oui, complet** |

k-mamba est le premier module a implementer un scan selectif 2D natif
avec ordonnancement wavefront en assembleur x86-64, une convolution ND
separable avec backward complet, et un MambaBlock entrainable integre.

---

## References

- Optimatrix benchmark suite : `optimatrix/tests/`
- Test ConvND : `test_phase5.c`
- Test MambaBlock : `test_mamba_block.c`

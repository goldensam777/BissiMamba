# OPTIMIZATIONS.md — Optimisations Matricielles de BissiMamba

> Ce document trace l'évolution du moteur de calcul de BissiMamba,
> de la boucle naïve au kernel AVX2 vectorisé, et au-delà.

---

## 1. Problème Initial : le coût du forward pass

Dans `mamba_forward`, pour chaque séquence de longueur `L` :

- **2 × L appels à `matrix_vec_mult`** (projections W_in et W_out)
- **L itérations de `selective_scan`** avec Hadamard elementwise
- **L appels à `matrix_vec_mult`** pour delta_proj

Soit, pour L=128, dim=64, state_size=32 :
- ~384 GEMV de taille variable
- ~128 paires de boucles scalaires pour le state update

Complexité totale : `O(L × (state_size × dim) + L × state_size)`

---

## 2. Phase 1 — Remplacement de `matrix_vec_mult` par `gemv_avx2`

### Avant
```c
void matrix_vec_mult(real_t *out, const Matrix *m, const real_t *v) {
    for (size_t i = 0; i < m->rows; i++) {
        real_t sum = 0.0;
        for (size_t j = 0; j < m->cols; j++) {
            sum += m->data[i * m->cols + j] * v[j];
        }
        out[i] = sum;
    }
}
```
Scalaire pur. 1 multiply-add par cycle.

### Après
```c
void matrix_vec_mult(real_t *out, const Matrix *m, const real_t *v) {
    gemv_avx2(m->data, (real_t *)v, out, (long)m->rows, (long)m->cols);
}
```
Le kernel `gemv_avx2` utilise `vfmadd231ps` sur des registres YMM (256-bit).
**8 multiply-adds par cycle.** Speedup mesuré : **~12×** sur GEMM 64×64.

### Mécanisme interne de `gemv_avx2`
```
Pour chaque ligne i de A :
  accum = [0,0,0,0,0,0,0,0]       ← YMM, 8 floats
  Pour j = 0, 8, 16, ... (n8) :
    ymm1 = A[i][j..j+7]           ← vmovups
    ymm2 = x[j..j+7]              ← vmovups
    accum += ymm1 * ymm2           ← vfmadd231ps
  Réduction : vextractf128 + vaddps + 2×vhaddps
  y[i] = accum
```

---

## 3. Phase 2 — Vectorisation du Selective Scan

### Avant (double boucle parallèle OpenMP)
```c
#pragma omp parallel for
for (size_t i = 0; i < state_size; i++) {
    state[i] = A_diag_t[i] * temp_state[i] + B_bar_t[i] * u_t[i];
}
```
OpenMP a un coût de synchronisation non nul. Pour `state_size` < 256,
le overhead de fork/join dépasse souvent le gain de parallélisme.

### Après (Hadamard AVX2 + vec_add)
```c
hadamard_avx2(A_diag_t, state, temp_state, state_size);   // temp = A ⊙ state
hadamard_avx2(B_bar_t, u_t, state, state_size);            // state = B ⊙ u
vec_add(state, temp_state, state_size);                    // state += temp
```
Pas de synchronisation. Chaque `hadamard_avx2` traite 8 floats/cycle.

---

## 4. Phase 3 (à faire) — Batching GEMV → GEMM

### Observation clé
Le forward pass appelle `matrix_vec_mult(z, W_in, x_t)` pour chaque
timestep `t` séparément. C'est `L` appels GEMV indépendants avec **le
même W_in**.

Or `L` GEMV avec la même matrice = 1 GEMM :

```
L appels :  gemv(W_in, x_t, u_t)      pour t = 0..L-1
≡ 1 appel : gemm(X, W_in^T, U)
```

où :
- `X`     : (L × dim)       — séquence d'entrée
- `W_in^T`: (dim × state_size) — transposée de W_in
- `U`     : (L × state_size)  — projections en une passe

### Code cible
```c
/* Au lieu de la boucle sur t : */
gemm_avx2(batch_input, W_in_T, u_seq,
          (long)seq_len, (long)dim, (long)state_size);
```

**Gain attendu** : `L` fois moins de kernel launches, meilleure utilisation
du cache L2/L3 (W_in reste chaud en cache sur toute la séquence).

### Complexité
| Méthode        | Opérations      | Kernel launches |
|----------------|-----------------|-----------------|
| L × GEMV       | L × 2×m×n      | L               |
| 1 × GEMM       | 2×L×m×n        | 1               |

Même nombre d'opérations flottantes, mais le GEMM est
**memory-bound-optimal** grâce au tiling de cache interne.

---

## 5. Phase 4 (vision) — Kernel Fusionné Forward Complet

L'idéal : un seul kernel ASM qui enchaîne :
1. Projection W_in (GEMM)
2. SiLU elementwise
3. Selective scan (recurrence vectorisée)
4. Projection W_out (GEMM)

Sans jamais écrire les résultats intermédiaires en mémoire principale.
Tout reste dans les registres YMM / cache L1.

```
Coût mémoire actuel  : 4 passages sur les données (write+read × 2)
Coût kernel fusionné : 1 passage (read input, write output)
```

C'est le régime que les librairies comme cuBLAS atteignent sur GPU.
Sur CPU AVX2, c'est atteignable pour les petits modèles (dim < 256).

---

## Récapitulatif des Gains

| Opération             | Avant         | Après          | Gain  |
|-----------------------|---------------|----------------|-------|
| `matrix_vec_mult`     | scalaire      | `gemv_avx2`    | ~12×  |
| State update (scan)   | OpenMP loops  | `hadamard_avx2`| ~4×   |
| W_in projection (seq) | L × GEMV      | 1 × GEMM       | (todo)|
| Forward complet       | —             | kernel fusionné| (todo)|

---

## Dépendances

Les kernels ASM vivent dans `optimatrix/` (submodule git).
Interface via `optimatrix_bridge.h`.

```
BissiMamba/
├── optimatrix/          ← submodule (master)
│   ├── src/gemv_avx2.asm
│   ├── src/gemm_avx2.asm
│   ├── src/hadamard.asm
│   └── src/activations.asm
├── optimatrix_bridge.h  ← déclarations C
└── mamba.c              ← consomme les kernels
```

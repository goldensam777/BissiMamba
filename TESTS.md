# TESTS.md — État des tests k-mamba

---

## Suites actives (CTest — 6/6 ✅)

### test_optimizers — 15/15 ✅

Fichier : `tests/test_optimizers.c`

Couvre :
- Gradient clipping L2 global (3 tests)
- Newton-Schulz orthogonalisation sur matrices 4×8 (3 tests)
- MUON step sur MambaBlock réel (1 test)
- AdamW pour embedding/head : différence avec SGD, allocation des moments, incrément du step (6 tests)
- AdamW sans double weight decay (2 tests)

### test_optimatrix_kernels — 5/5 ✅

Fichier : `tests/unit/test_optimatrix_kernels.c`

Couvre : GEMM small/medium/edge, GEMV small/large.
Benchmark : AVX2 ≈ 6 GFLOPS sur 64×128×128×256 (8× vs scalaire).

### test_scan1d_backward_asm — 1/1 ✅

Fichier : `tests/unit/test_scan1d_backward_asm.c`

Compare le backward scan1d ASM (M=1, shared B/C) avec l'implémentation C de référence.

### test_scan1d — 7/7 ✅

Fichier : `tests/unit/test_scan1d.c`

Compare `scan1d()` (ASM AVX2) avec une référence C scalaire.

Cas testés :
- M=1 : small (L=4 D=8), medium (L=16 D=16), large (L=64 D=32), D non multiple de 8 (D=24)
- M=2 : small, medium
- Cas dégénéré : x=0 B=0 → y=0 h=0

Assertions par cas : y fini, y == référence, état final h == référence.

Note : `scan1d()` ASM maintient `h` comme état courant [D, M] (pas [L, D, M]).
Le test compare `h_asm[0..DM-1]` avec la tranche finale `h_ref[(L-1)*DM..]`.

### test_scan1d_backward — 4/4 ✅

Fichier : `tests/unit/test_scan1d_backward.c`

Gradient check numérique sur `scan1d_backward()` : différences finies centrées (eps=1e-3)
contre gradients analytiques pour x, A, B, C, delta. Tolérance 1e-3.

Cas testés : M=1 (small, medium), M=2 (small, medium).
Erreurs observées : toutes < 2e-4 (bien en dessous de la tolérance).

### test_kmamba_e2e — 4/4 ✅

Fichier : `tests/integration/test_kmamba_e2e.c`

Test end-to-end sur micro-modèle (dim=16, state=8, n_layers=1, seq_len=8) :
1. `kmamba_forward` : 2048 logits tous finis
2. `kmamba_train_step` : loss finie et positive (~5.5)
3. Décroissance : loss à 0 < loss à 20 steps (lr=1e-2, séquence fixe)
4. `kmamba_train_batch` : loss finie sur batch_size=4

---

## Lancer les tests

```bash
cd k-mamba
cmake -B build -DKMAMBA_BUILD_TESTS=ON
cmake --build build -j

# Tous les tests
ctest --test-dir build

# Un test spécifique avec sortie détaillée
ctest --test-dir build -R Scan1DForwardASM -V
./build/tests/test_scan1d
```

---

## Fichiers de test non enregistrés dans CTest

| Fichier | Contenu | État |
|---------|---------|------|
| `tests/unit/test_conv1d_final.c` | Conv1D depthwise | Compile — non enregistré |
| `tests/test_cuda_scan.cu` | Scan1D CUDA vs CPU ref | Compile sous `KMAMBA_BUILD_CUDA` |

---

## Bugs corrigés

| Bug | Fichier | Correctif |
|-----|---------|-----------|
| `logits` non remis à 0 entre séquences du batch | `src/kmamba.c` / `kmamba_train_batch` | `memset(logits, 0, ...)` avant `gemm_avx2` |
| `d_hidden` non remis à 0 entre séquences du batch | `src/kmamba.c` / `kmamba_train_batch` | `memset(d_hidden, 0, ...)` avant `gemm_avx2` |

Cause du `-nan` : `gemm_avx2` accumule (C += A@B). Les deux buffers n'étaient réinitialisés
qu'une fois (xcalloc) avant la boucle batch — l'accumulation croisait les séquences du batch.
Résultat après fix : descente monotone 5.38 → 3.74 → 3.07 sur `train.txt` (batch_size=32).

---

## Problèmes résiduels

| Fichier | État |
|---------|------|
| `cpu/scan1d_backward_m1_shared_bc.asm` | Assemblé et testé (7/7 ✅) — dispatch C ne l'appelle pas |
| `tests/unit/test_conv1d_final.c` | Expected values incorrectes, ne teste pas l'ASM — non intégré |

---

## Prochaines étapes

1. Test CUDA scan1D sous `KMAMBA_BUILD_CUDA`
2. Activer `scan1d_backward_m1_shared_bc_asm` dans le dispatch C si gain de perf mesuré

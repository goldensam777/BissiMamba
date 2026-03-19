# TEST_RESULTS.md — Résultats des Tests k-mamba

**Date** : 18 Mars 2026
**Plateforme** : x86-64 Linux, AVX2, GCC 11

---

## Builds testés

| Build | Flags | Statut |
|-------|-------|--------|
| CPU seul | `KMAMBA_BUILD_CPU=ON KMAMBA_BUILD_TESTS=ON` | ✅ |
| CPU + CUDA | `KMAMBA_BUILD_CUDA=ON` (optimizer steps GPU) | ✅ compile |

---

## Résultats (build CPU)

### test_optimatrix_kernels — GEMM / GEMV AVX2

| Test | Description | Résultat |
|------|-------------|----------|
| GEMM small | 3×4 × 4×2 | ✅ PASS |
| GEMM medium | 64×128 × 128×256 | ✅ PASS |
| GEMM edge cases | 1×1 et 1×4 × 4×1 | ✅ PASS |
| GEMV small | 4×3 × 3 | ✅ PASS |
| GEMV large | 1024×1024 × 1024 | ✅ PASS |

**5/5 tests passés.**

Tolérance : `EPSILON = 2e-5f`. `gemm_avx2` accumule dans C (`C += A@B`) : zéroïser avant appel.

#### Benchmark GEMM (64×128 × 128×256)

| Implémentation | GFLOPS | Speedup |
|----------------|--------|---------|
| Référence C | ~0.72 | 1× |
| AVX2 ASM | ~6.0 | ~8× |

---

### test_optimizers — Optimiseurs CPU

| Test | Description | Résultat |
|------|-------------|----------|
| clip_no_op | norme < max → pas de modification | ✅ |
| clip_actif | norme après clip == max_norm | ✅ |
| clip_direction | direction préservée après clip | ✅ |
| ns_diagonale | G·Gᵀ diagonale ≈ 1 | ✅ |
| ns_orthogonalite | G·Gᵀ hors-diag ≈ 0 (max err 0.003) | ✅ |
| ns_gradient_nul | gradient nul : pas de crash | ✅ |
| muon_block | W_in modifié après MUON step (NS + momentum) | ✅ |
| adam_diff_sgd | Adam != SGD (variance adaptative active) | ✅ |
| adam_m_embed | m_embedding alloué | ✅ |
| adam_v_embed | v_embedding alloué | ✅ |
| adam_m_head | m_head alloué | ✅ |
| adam_v_head | v_head alloué | ✅ |
| adam_step_init | step_embed_head initialisé à 0 | ✅ |
| adam_step_inc | step_embed_head == 1 après train_step | ✅ |
| adam_m_nonzero | m_embedding non nul après step | ✅ |
| adamw_no_double_wd | formule correcte != double WD | ✅ |
| adamw_w_in_evolue | W_in évolue à chaque step (WD simple) | ✅ |

**15/15 tests passés.**

Bugs corrigés lors de cette session (18/03/2026) :
- MUON : était un stub SGDM sans Newton-Schulz → remplacé par vrai MUON (NS 5 itérations)
- AdamW : double weight decay (wd appliqué dans la formule ET dans le gradient) → supprimé
- Embedding/head : SGD pur → AdamW avec moments m, v et correction de biais
- `gemm_avx2` accumule : zéroïser les buffers A, AG avant chaque appel NS

---

### test_scan1d_backward_asm — Backward ASM vs C (7/7)

Compare `scan1d_backward_m1_shared_bc_asm` (AVX2) et `scan1d_backward_m1_shared_bc_simple_asm`
avec la référence C. Tolérance 1e-4.

| Test | Résultat |
|------|----------|
| Simple ASM small (L=8 D=4) | ✅ |
| Simple ASM medium (L=32 D=16) | ✅ |
| Simple ASM large (L=64 D=32) | ✅ |
| AVX2 ASM small (L=8 D=4) | ✅ |
| AVX2 ASM medium (L=32 D=16) | ✅ |
| AVX2 ASM D=17 (queue scalaire) | ✅ |
| AVX2 ASM large (L=64 D=32) | ✅ |

---

### test_scan1d — ASM forward vs référence C (7/7)

| Test | Résultat |
|------|----------|
| small M=1 (L=4 D=8) | ✅ |
| medium M=1 (L=16 D=16) | ✅ |
| large M=1 (L=64 D=32) | ✅ |
| D non multiple de 8 (D=24) | ✅ |
| small M=2 (L=4 D=8) | ✅ |
| medium M=2 (L=16 D=16) | ✅ |
| Cas dégénéré x=0 B=0 | ✅ |

Note: l'ASM maintient h comme état courant [D,M] (pas [L,D,M]).

---

### test_scan1d_backward — Gradient check numérique (4/4)

Différences finies centrées (eps=1e-3, tol=1e-3) sur x, A, B, C, delta.

| Test | max_err |
|------|---------|
| small M=1 | < 2e-5 |
| medium M=1 | < 2e-5 |
| small M=2 | < 5e-6 |
| medium M=2 | < 1e-4 |

---

### test_kmamba_e2e — End-to-end (4/4)

Micro-modèle dim=16, state=8, n_layers=1, seq_len=8.

| Test | Résultat |
|------|----------|
| forward : logits finis | ✅ |
| train_step : loss finie ~5.5 | ✅ |
| Décroissance 5.57 → 4.54 sur 20 steps | ✅ |
| train_batch loss finie (batch_size=4) | ✅ |

---

## CTest global : 6/6 suites ✅

---

## Bugs corrigés (18/03/2026)

| Bug | Correctif |
|-----|-----------|
| MUON : stub SGDM sans Newton-Schulz | Vrai MUON (NS 5 itérations) |
| AdamW double weight decay | Supprimé |
| Embedding/head : SGD → AdamW | AdamW avec moments m, v |
| `gemm_avx2` accumule : buffers A, AG | memset avant chaque appel NS |
| `kmamba_train_batch` : `logits` accumule sur le batch | `memset(logits, 0, ...)` avant gemm |
| `kmamba_train_batch` : `d_hidden` accumule sur le batch | `memset(d_hidden, 0, ...)` avant gemm |

Le bug `-nan` sur `train.txt` (batch_size=32) était causé par les deux derniers : `gemm_avx2`
accumule et les buffers n'étaient pas réinitialisés entre séquences. Après fix : descente
monotone 5.38 → 3.74 → 3.07 sur `data/train.txt` (327 KB, DIM=128, STATE=256, N_LAYERS=2).

---

## Prochaines étapes

- Test CUDA scan1D sous `KMAMBA_BUILD_CUDA`
- Activer `scan1d_backward_m1_shared_bc_asm` dans le dispatch si gain perf mesuré

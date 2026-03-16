# TEST_RESULTS.md — Résultats des Tests k-mamba

**Date** : 17 Mars 2026
**Version** : k-mamba v0.1.0
**Plateforme** : x86-64 Linux, NVIDIA MX450 (sm_75), CUDA 12.0

---

## Builds testés

| Build | Flags | Statut |
|-------|-------|--------|
| CPU seul | `KMAMBA_BUILD_CPU=ON` | ✅ |
| CPU + CUDA | `KMAMBA_BUILD_CUDA=ON OPTIMATRIX_BUILD_CUDA=ON` | ✅ |

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

Tolérance numérique : `EPSILON = 2e-5f` — justifiée par l'accumulation float32 sur 128 éléments
(erreur max théorique ≈ 6e-5). `gemm_avx2` accumule dans C (`C += A*B`) : l'appelant doit
zéroïser C avant appel (contrat documenté).

#### Benchmark GEMM (64×128 × 128×256)

| Implémentation | GFLOPS | Speedup |
|----------------|--------|---------|
| Référence C | ~0.72 | 1× |
| AVX2 ASM | ~6.0 | ~8× |

---

### test_optimizers — Optimiseurs CPU

| Test | Résultat |
|------|----------|
| Gradient clipping (no-op) | ✅ |
| Gradient clipping (actif) | ✅ norme clampée à max_norm |
| ADAM_CLIP + clipping | ✅ |
| MUON + clipping | ✅ |

**2/2 tests passés.**

Fix appliqué : `mamba_attach_optimizer` n'allouait les buffers `m_*` (first moment) que pour
`OPTIMIZER_ADAM_CLIP` / `OPTIMIZER_ADAMW`. MUON et SGD utilisent aussi `m_*` → SEGFAULT.
Correction : condition étendue à `|| OPTIMIZER_MUON || OPTIMIZER_SGD`.

---

## Résultats (build CPU + CUDA)

### test_cuda_optimizers — Optimiseurs GPU

| Test | Résultat | Détail |
|------|----------|--------|
| Gradient clipping CUDA | ✅ | norme 18518 → 5.0 |
| AdamW CUDA | ✅ | 3 steps, convergence monotone |
| MUON CUDA | ✅ | momentum stable après step 1 |
| Consistance CPU/CUDA | ✅ | max diff = 2.98e-08 |

**4/4 tests passés.**

Fix appliqué : `optimatrix.h` manquait de `extern "C"`. NVCC compilant les `.cu` en C++,
les symboles C (`gradient_norm`, `gradient_clip_inplace`) n'étaient pas résolus. Ajout du guard
`#ifdef __cplusplus extern "C" { } #endif` autour de toutes les déclarations.

---

## Tests désactivés / Known Issues

| Fichier | Raison | Priorité |
|---------|--------|----------|
| `cpu/scan1d_backward_m1_shared_bc.asm` | Bug ASM "two index registers" (pré-existant) | TODO |
| `cpu/scan1d_backward_m1_shared_bc_simple.asm` | Même bug | TODO |
| CUDA scan (`scan1d.cu`, `scan1d_backward.cu`) | Compilent, non testés dans la suite actuelle | Prochain |

---

## Prochaines étapes

- Écrire tests pour scan1D/2D (forward + backward)
- Tester MambaBlock complet (forward → backward → optimizer step)
- Benchmark Scan1D CPU ASM vs CUDA
- Fixer les bugs ASM dans `scan1d_backward_m1_shared_bc.asm`

# K-Mamba Refactoring Plan

## Objectives
1.  **Dynamic Configuration:** Remove compile-time limits (`MAX_NDIMS`, `MAX_STATE`) and use runtime values.
2.  **Runtime Optimization:** Move `FAST_EXP` from a compile-time flag to a runtime configuration option.
3.  **ConvND Generality:** Support asymmetric padding and even kernel sizes in separable convolutions.
4.  **Strict Configuration:** Remove hidden fallbacks and enforce explicit initialization or use of `set_defaults`.
5.  **Wavefront Abstraction:** Use `KMWavefrontPlan` iterators for SSM forward/backward passes to simplify code.

## Progress Status

| Task | Status | Notes |
| :--- | :--- | :--- |
| **1. Remove MAX constants** | ✅ Completed | Replaced with dynamic fields + pre-allocated thread-local scratch. |
| **2. Fast Exp Option** | ✅ Completed | Centralized dispatch via `km_scan_exp`. |
| **3. ConvND Padding** | ✅ Completed | Added `pad_left`/`pad_right` support; implemented zero-padding. |
| **4. Remove Fallbacks** | ✅ Completed | `set_defaults` implemented; core logic now asserts valid config. |
| **5. Backward Wavefront** | ✅ Completed | Refactored using `km_wavefront_plan_iter_reverse` callback. |
| **6. Validation** | ✅ Completed | All test suites updated and passing with strict configuration rules. |
| **7. ASM Cleanup** | ✅ Completed | Removed scan1d.asm and scan2d.asm; unified on C wavefront implementation. |
| **8. DEBUG Cleanup** | ✅ Completed | Removed all DBG_SCAN debug statements from scan_nd.c. |
| **9. Gradient Checkpointing**| ✅ Completed | Added `libs/train_set/Trainer` with layer-wise GC and recompute. |
| **10. Full GPU Backward**| ✅ Completed | Mamba-3 backward pass implemented and aligned with ScanND. |

## Completed Steps
- [x] Fix `tests/unit/test_mamba3_backward.c` (optimizer config initialization).
- [x] Update `tests/test_optimizers.c`, `tests/test_gradient.c`, and `tests/unit/test_mamba3_gpu.cu`.
- [x] Final verification of `KMAMBA_BACKEND=cpu` on all unit tests.
- [x] Document/Update `kser` serialization for new config fields.
- [x] Remove ASM scan backends (scan1d.asm, scan2d.asm) and unify on C wavefront.
- [x] Remove DEBUG statements from scan_nd.c.
- [x] Implement `libs/train_set/Trainer` for Gradient Checkpointing (GC).
- [x] Unify CUDA backend with Mamba-3 ND formulation and implement full GPU backward.
- [x] Create GPU kernels for embedding, head, and loss (cuda/kmamba_kernels.cu).
- [x] Validate Trainer GC numerical equivalence with unit tests.

## Upcoming Tasks (Serialization & Large Scale Training)
- [ ] **Full Optimizer Serialization**: Save Adam momentum `m` and `v` buffers in `.ser` format (critical for Resume Training).
- [ ] **Extended Header Config**: Persist `use_pgf`, `pgf_block_size`, and `weight_tying` in serialization headers.
- [ ] **Trainer State Persistence**: Support saving/loading the `Trainer` object and its GC policy.
- [ ] **BFloat16 Tuning**: Fine-tune BF16 kernels for stability on large models (1B+).

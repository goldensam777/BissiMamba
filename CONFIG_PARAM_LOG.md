# Config Parametrization Log

Date: 2026-04-25

## Done

- Runtime lambda fallback is configurable via:
  - `KMambaConfig.default_lambda`
  - `MBConfig.default_lambda`
  - `ScanNDParams.default_lambda`
- A-log clamp behavior is configurable via:
  - `KMambaConfig.use_a_log_clamp`, `KMambaConfig.a_log_min`
  - `MBConfig.use_a_log_clamp`, `MBConfig.a_log_min`
  - `ScanNDParams.use_a_log_clamp`, `ScanNDParams.a_log_min`
- CUDA device selection is configurable via:
  - `KMambaConfig.gpu_device` (`-1` keep current/default, `>=0` explicit device)
- AdamW hyperparameters for embed/head are no longer hardcoded in `kmamba_train_batch`:
  - now read from `MBOptimConfig.mu`, `MBOptimConfig.beta2`, `MBOptimConfig.eps`
- AdamW hyperparameters in `mamba_optimizer_step` (block params) are no longer hardcoded:
  - now read from `MBOptimConfig.mu`, `MBOptimConfig.beta2`, `MBOptimConfig.eps`
- Serialization naming was unified (no legacy alias path):
  - canonical tensor names only (`W_in`, `W_out`, `A_log`, `W_B`, `W_C`, `delta_proj`, `lambda_proj`, `b_B`, `b_C`, `theta`)
- Header/API mismatch was fixed by implementing declared KMamba API symbols in `src/kmamba.c`.

## Remaining Hardcoded / Open Decisions

- `KMAMBA_MAX_NDIMS` and `KMAMBA_MAX_STATE` are compile-time constants.
- `scan_nd` fast exp approximation (`KMAMBA_FAST_EXP_APPROX`) is compile-time feature-gated.
- Separable ConvND currently assumes odd K (`kernel_radius = K/2`) and fixed border behavior (copy input on edges).
- Some defaults still exist for backward compatibility when invalid/unset config is provided:
  - fallback lambda `0.5`
  - fallback `a_log_min = -1e-5`
  - fallback Adam `beta1=0.9`, `beta2=0.999`, `eps=1e-8`
- ND backward wavefront path still contains simplified math sections and should be revisited for full correctness in high dimensions.

## Architecture Changes

- **ASM Backends Removed**: scan1d.asm and scan2d.asm removed; unified on C wavefront implementation.
- **Unified Scan Formula**: Only Mamba-3 exp-trapezoidal formula remains (removed backward-compatible path).
- **DEBUG Removed**: All DBG_SCAN statements removed from scan_nd.c; macros kept as no-ops for future use.

## Verification Notes

- `make -j4` passes.
- `make tests` passes (all 7/7 scan_nd tests, all optimizer tests, all gradient tests, all Mamba-3 tests).
- `KMAMBA_BACKEND=cpu ./test_mamba3` passes.
- GPU tests pass with CPU fallback when no GPU available.

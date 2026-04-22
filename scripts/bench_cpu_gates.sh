#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bench-gates] build cpu lib"
make lib CPU_ONLY=1 >/dev/null

echo "[bench-gates] run gemm determinism regression test"
gcc -O3 -mavx2 -Wall -Wextra -I./include -fopenmp \
  tests/unit/test_gemm_atb_determinism.c kernels/gemm_f32.c -lm -lgomp \
  -o /tmp/test_gemm_atb_determinism
/tmp/test_gemm_atb_determinism

echo "[bench-gates] run scan_nd regression test"
gcc -no-pie -O3 -mavx2 -Wall -Wextra -I./include -fopenmp \
  tests/unit/test_scan_nd.c src/scan_nd.c src/wavefront_plan.c src/wavefront_nd.c \
  src/km_topology.c src/km_memory_pool.c src/kmamba_cuda_utils.c cpu/scan1d.o cpu/scan2d.o \
  -lm -lgomp -o /tmp/test_scan_nd
/tmp/test_scan_nd

echo "[bench-gates] run cpu gates benchmark"
gcc -O3 -mavx2 -Wall -Wextra -I./include -fopenmp \
  tests/unit/bench_cpu_gates.c kernels/gemm_f32.c -lm -lgomp \
  -o /tmp/bench_cpu_gates
/tmp/bench_cpu_gates

echo "[bench-gates] PASS"

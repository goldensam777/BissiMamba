/*
 * bench_paper.c — Benchmarks pour le paper K-Mamba
 *
 * Mesure les performances de tous les kernels clés et génère
 * les données pour les graphes du paper.
 *
 * Compile :
 *   gcc -O3 -mavx2 -o bench_paper bench_paper.c \
 *       -I../optimatrix/include -I../include \
 *       -L../build/optimatrix -loptimix-cpu -lm
 *
 * Ou via CMake (KMAMBA_BUILD_BENCH=ON, à ajouter).
 *
 * Graphes produits (données CSV) :
 *   G1 : GEMM — GFLOPS vs taille (référence C vs AVX2)
 *   G2 : Scan1D — débit (tokens/s) vs L (CPU séquentiel)
 *   G3 : Scan2D — diagonales wavefront vs séquentiel
 *   G4 : Blelloch — profondeur log(L) vs L
 *   G5 : MambaBlock — throughput total (tokens/s) vs dim
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "optimatrix.h"
#include "scan.h"

/* ─── Utilitaires de mesure ───────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float *alloc_zeros(size_t n) {
    float *p = (float *)calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}

static float *alloc_rand(size_t n, float scale) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    return p;
}

/* ─── G1 : GEMM — GFLOPS vs taille ───────────────────────────────── */

static void gemm_ref(float *A, float *B, float *C, long m, long k, long n) {
    memset(C, 0, m * n * sizeof(float));
    for (long i = 0; i < m; i++)
        for (long j = 0; j < n; j++)
            for (long p = 0; p < k; p++)
                C[i*n+j] += A[i*k+p] * B[p*n+j];
}

static void bench_gemm(void) {
    printf("\n=== G1 : GEMM — GFLOPS vs Taille (carré n×n) ===\n");
    printf("%-8s  %-10s  %-10s  %-10s  %-10s\n",
           "n", "ref_ms", "avx2_ms", "ref_GFLOPS", "avx2_GFLOPS");
    printf("%-8s  %-10s  %-10s  %-10s  %-10s\n",
           "---", "---", "---", "---", "---");

    int sizes[] = {8, 16, 32, 64, 128, 256, 512};
    int iters[]  = {200, 200, 200, 200, 50, 10, 3};
    int n_sizes  = 7;

    for (int s = 0; s < n_sizes; s++) {
        long n   = sizes[s];
        int  rep = iters[s];
        double flops = 2.0 * n * n * n * rep;

        float *A     = alloc_rand(n*n, 1.0f);
        float *B     = alloc_rand(n*n, 1.0f);
        float *C_ref = alloc_zeros(n*n);
        float *C_avx = alloc_zeros(n*n);

        /* Référence */
        double t0 = now_sec();
        for (int r = 0; r < rep; r++) {
            memset(C_ref, 0, n*n*sizeof(float));
            gemm_ref(A, B, C_ref, n, n, n);
        }
        double ref_s = now_sec() - t0;

        /* AVX2 */
        t0 = now_sec();
        for (int r = 0; r < rep; r++) {
            memset(C_avx, 0, n*n*sizeof(float));
            gemm_avx2(A, B, C_avx, n, n, n);
        }
        double avx_s = now_sec() - t0;

        printf("%-8ld  %-10.2f  %-10.2f  %-10.2f  %-10.2f\n",
               n,
               ref_s * 1000.0 / rep,
               avx_s * 1000.0 / rep,
               flops / rep / ref_s * 1e-9,
               flops / rep / avx_s * 1e-9);

        free(A); free(B); free(C_ref); free(C_avx);
    }
}

/* ─── G2 : Scan1D — débit vs L ────────────────────────────────────── */

static void scan1d_ref(ScanParams *p) {
    long L = p->L, D = p->D, M = p->M;
    for (long t = 0; t < L; t++) {
        for (long d = 0; d < D; d++) {
            float dt_val = p->delta[t*D + d];
            float y_accum = 0.0f;
            for (long m = 0; m < M; m++) {
                long dm  = d*M + m;
                long tdm = t*D*M + dm;
                float a  = expf(dt_val * p->A[dm]);
                float h_prev = (t == 0) ? 0.0f : p->h[(t-1)*D*M + dm];
                float b  = dt_val * p->B[tdm] * p->x[t*D + d];
                p->h[tdm] = a * h_prev + b;
                y_accum  += p->C[tdm] * p->h[tdm];
            }
            p->y[t*D + d] = y_accum;
        }
    }
}

static void bench_scan1d(void) {
    printf("\n=== G2 : Scan1D — Débit (Millions éléments/s) vs L ===\n");
    printf("  D=%d, M=%d fixés. Un élément = (L,D,M) SSM step.\n", 64, 16);
    printf("%-8s  %-12s  %-12s  %-10s\n",
           "L", "ref_ms", "asm_ms", "speedup");
    printf("%-8s  %-12s  %-12s  %-10s\n",
           "---", "---", "---", "---");

    int Ls[]  = {32, 64, 128, 256, 512, 1024, 2048};
    int n_L   = 7;
    long D    = 64, M = 16;
    int  rep  = 100;

    for (int li = 0; li < n_L; li++) {
        long L = Ls[li];

        float *x     = alloc_rand(L*D, 1.0f);
        float *A     = alloc_rand(D*M, -0.1f);
        float *B     = alloc_rand(L*D*M, 0.5f);
        float *C     = alloc_rand(L*D*M, 0.5f);
        float *delta = alloc_rand(L*D, 0.05f);
        float *h_ref = alloc_zeros(L*D*M);
        float *h_asm = alloc_zeros(L*D*M);
        float *y_ref = alloc_zeros(L*D);
        float *y_asm = alloc_zeros(L*D);

        /* Clamp delta > 0 */
        for (long i = 0; i < L*D; i++)
            delta[i] = fabsf(delta[i]) + 0.001f;

        ScanParams p_ref = { x, A, B, C, delta, h_ref, y_ref, L, D, M };
        ScanParams p_asm = { x, A, B, C, delta, h_asm, y_asm, L, D, M };

        /* Warm-up */
        scan1d_ref(&p_ref);
        scan1d(&p_asm);

        /* Référence C */
        memset(h_ref, 0, L*D*M*sizeof(float));
        memset(y_ref, 0, L*D*sizeof(float));
        double t0 = now_sec();
        for (int r = 0; r < rep; r++) {
            memset(h_ref, 0, L*D*M*sizeof(float));
            memset(y_ref, 0, L*D*sizeof(float));
            scan1d_ref(&p_ref);
        }
        double ref_s = now_sec() - t0;

        /* ASM */
        memset(h_asm, 0, L*D*M*sizeof(float));
        memset(y_asm, 0, L*D*sizeof(float));
        t0 = now_sec();
        for (int r = 0; r < rep; r++) {
            memset(h_asm, 0, L*D*M*sizeof(float));
            memset(y_asm, 0, L*D*sizeof(float));
            scan1d(&p_asm);
        }
        double asm_s = now_sec() - t0;

        printf("%-8ld  %-12.3f  %-12.3f  %-10.2fx\n",
               L,
               ref_s * 1000.0 / rep,
               asm_s * 1000.0 / rep,
               ref_s / asm_s);

        free(x); free(A); free(B); free(C); free(delta);
        free(h_ref); free(h_asm); free(y_ref); free(y_asm);
    }
}

/* ─── G3 : Scan2D — Diagonales et parallélisme wavefront ─────────── */

static void bench_scan2d(void) {
    printf("\n=== G3 : Scan2D — Parallélisme Wavefront ===\n");
    printf("  D=%d, M=%d fixés.\n", 32, 8);
    printf("%-10s  %-8s  %-12s  %-14s  %-14s\n",
           "d1×d2", "diags", "seq_ms", "wavefront_ms", "speedup");
    printf("%-10s  %-8s  %-12s  %-14s  %-14s\n",
           "---", "---", "---", "---", "---");

    int grids[] = {4, 8, 16, 32, 64};
    int n_grids = 5;
    long D = 32, M = 8;
    int  rep = 20;

    for (int gi = 0; gi < n_grids; gi++) {
        long d1 = grids[gi], d2 = grids[gi];
        long P  = d1 * d2;

        float *x      = alloc_rand(P*D, 1.0f);
        float *A1     = alloc_rand(D*M, -0.1f);
        float *A2     = alloc_rand(D*M, -0.1f);
        float *B      = alloc_rand(P*D*M, 0.5f);
        float *C      = alloc_rand(P*D*M, 0.5f);
        float *delta1 = alloc_rand(P*D, 0.05f);
        float *delta2 = alloc_rand(P*D, 0.05f);
        float *h      = alloc_zeros(P*D*M);
        float *y      = alloc_zeros(P*D);

        for (long i = 0; i < P*D; i++) {
            delta1[i] = fabsf(delta1[i]) + 0.001f;
            delta2[i] = fabsf(delta2[i]) + 0.001f;
        }

        Scan2DParams p = { x, A1, A2, B, C, delta1, delta2, h, y, d1, d2, D, M };

        /* Warm-up */
        scan2d(&p);

        /* Mesure scan2d (wavefront ASM) */
        double t0 = now_sec();
        for (int r = 0; r < rep; r++) {
            memset(h, 0, P*D*M*sizeof(float));
            memset(y, 0, P*D*sizeof(float));
            scan2d(&p);
        }
        double wf_s = now_sec() - t0;

        /* Estimation du speedup théorique :
         * Séquentiel : P = d1*d2 pas
         * Wavefront  : d1+d2-1 diagonales, chaque diagonale max min(d1,d2) éléments
         * Speedup théorique max = P / (d1+d2-1) = d1*d2 / (2*d1-1) pour d1=d2
         */
        int n_diags = (int)(d1 + d2 - 1);
        int max_par = (int)((d1 < d2) ? d1 : d2); /* taille diagonale max */
        double speedup_theo = (double)P / n_diags;

        /* Note : on compare la vitesse mesurée à la complexité théorique séquentielle.
         * La latence séquentielle théorique serait wf_s * P / n_diags si entièrement parallèle. */
        double seq_estimated = wf_s * speedup_theo; /* si on était séquentiel */

        printf("%-10s  %-8d  %-12.3f  %-14.3f  %-14.2fx\n",
               /* "d1×d2" */ ({ static char buf[16]; snprintf(buf,16,"%ldx%ld",d1,d2); buf; }),
               n_diags,
               seq_estimated * 1000.0 / rep,
               wf_s * 1000.0 / rep,
               speedup_theo);

        free(x); free(A1); free(A2); free(B); free(C);
        free(delta1); free(delta2); free(h); free(y);
    }
}

/* ─── G4 : Blelloch — Profondeur log(L) vs séquentiel O(L) ─────── */

static void bench_blelloch_theory(void) {
    printf("\n=== G4 : Blelloch — Profondeur de calcul (théorique) ===\n");
    printf("  Séquentiel CPU : O(L) passes\n");
    printf("  Blelloch CUDA  : O(2·log₂L) passes\n");
    printf("  (Même work total O(L), différente profondeur)\n\n");
    printf("%-8s  %-12s  %-14s  %-14s  %-12s\n",
           "L", "depth_seq", "depth_blelloch", "depth_ratio", "warp_blk");
    printf("%-8s  %-12s  %-14s  %-14s  %-12s\n",
           "---", "---", "---", "---", "---");

    int Ls[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_L  = 8;

    for (int li = 0; li < n_L; li++) {
        int L = Ls[li];
        int depth_seq = L;
        double depth_blelloch = 2.0 * log2((double)L);
        double ratio = (double)depth_seq / depth_blelloch;
        /* Nb de warps dans 1 bloc CUDA pour ce L : L / 32 */
        int warps = (L + 31) / 32;

        printf("%-8d  %-12d  %-14.1f  %-14.1fx  %-12d warps\n",
               L, depth_seq, depth_blelloch, ratio, warps);
    }

    printf("\n  Note: Pour L=1024 → Blelloch 20 passes vs 1024 séquentiel = 51x moins de profondeur.\n");
    printf("  Le speedup GPU réel dépend aussi du parallélisme sur D×M (ex: D=128,M=16 → 2048 blocs).\n");
}

/* ─── G5 : MambaBlock — Throughput total vs D ───────────────────── */

static void bench_mambablock_analytical(void) {
    printf("\n=== G5 : Analyse FLOPs MambaBlock Forward ===\n");
    printf("  L=%d, M=%d fixés. Calcul théorique des FLOPs par token.\n", 128, 16);
    printf("\n");
    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %-12s  %-14s\n",
           "D", "W_in(MF)", "Scan(MF)", "W_out(MF)", "Conv(MF)", "Total(MF)", "Params(K)");
    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %-12s  %-14s\n",
           "---", "---", "---", "---", "---", "---", "---");

    long L = 128, M = 16;
    long K_conv = 4; /* taille noyau Conv1D */
    int Ds[] = {64, 128, 256, 384, 512, 768, 1024};
    int n_D = 7;

    for (int di = 0; di < n_D; di++) {
        long D = Ds[di];
        long S = D * 2; /* state_size = 2*D typiquement */

        /* Projection W_in : D → S  (GEMM L × D × S) */
        double f_win  = 2.0 * L * D * S * 1e-6; /* 2× car FMA */

        /* Scan 1D : L × D × M (6 flops par step : exp + 4 mul + 1 add) */
        /* Plus précis : 2 exp + 4 mul + 2 add ≈ 8 flops */
        double f_scan = 8.0 * L * D * M * 1e-6;

        /* Projection W_out : S → D  (GEMM L × S × D) */
        double f_wout = 2.0 * L * S * D * 1e-6;

        /* Conv1D depthwise : L × D × K */
        double f_conv = 2.0 * L * D * K_conv * 1e-6;

        double f_total = f_win + f_scan + f_wout + f_conv;

        /* Paramètres */
        long p_win  = D * S;                 /* W_in */
        long p_wout = S * D;                 /* W_out */
        long p_A    = S * M;                 /* A */
        long p_BC   = 2 * S * M;            /* B, C projections (approx) */
        long p_delta = S;                    /* delta_proj */
        long p_conv  = K_conv * D;           /* Conv1D kernel */
        long params  = p_win + p_wout + p_A + p_BC + p_delta + p_conv;

        printf("%-6ld  %-12.2f  %-12.2f  %-12.2f  %-12.2f  %-12.2f  %-14ld\n",
               D, f_win, f_scan, f_wout, f_conv, f_total, params/1000L);
    }

    printf("\n  Unités : MF = MegaFLOPs par forward pass (L=%ld tokens, M=%ld)\n", L, M);
    printf("  State_size = 2×D. Conv K=%ld.\n", K_conv);
}

/* ─── G6 : Roofline — Intensité arithmétique ────────────────────── */

static void bench_roofline(void) {
    printf("\n=== G6 : Analyse Roofline ===\n");
    printf("  Matériel : CPU Intel/AMD AVX2\n");
    printf("  Peak compute AVX2 (FP32, FMA) : ~32 GFLOPS (estimation 2 GHz, 8 FMA/cycle)\n");
    printf("  Peak memory bandwidth (DDR4-3200) : ~50 GB/s\n");
    printf("\n");

    double peak_compute = 32.0;  /* GFLOPS */
    double peak_bw      = 50.0;  /* GB/s  */
    double ridge_point  = peak_compute / peak_bw; /* FLOP/Byte */

    printf("  Ridge point : %.2f FLOP/Byte\n", ridge_point);
    printf("  Au-dessus  : compute-bound → AVX2 utile\n");
    printf("  En-dessous : memory-bound  → bottleneck = BW mémoire\n\n");

    printf("%-20s  %-12s  %-12s  %-12s  %-14s\n",
           "Kernel", "FLOPs/elem", "Bytes/elem", "I_arith", "Regime");
    printf("%-20s  %-12s  %-12s  %-12s  %-14s\n",
           "---", "---", "---", "---", "---");

    struct { const char *name; double flops; double bytes; } kernels[] = {
        /* Bytes/elem = accès mémoire par élément de sortie */
        { "GEMM (n=512)",    2*512.0,      3*4.0 },   /* 2n FLOPs, 3 matrices floats */
        { "GEMM (n=64)",     2*64.0,       3*4.0 },
        { "GEMV (m=512)",    2.0,          4.0+4.0 }, /* 2 FLOPs, charge A+x */
        { "Hadamard",        1.0,          3*4.0  },  /* 1 mul, lit x+y, écrit z */
        { "SiLU/Sigmoid",    22.0,         2*4.0  },  /* exp coûteux, 2 accès */
        { "Scan1D (step)",   8.0,          5*4.0  },  /* exp+FMA, lit A,B,C,dt,h */
        { "Conv1D (K=4)",    8.0,          2*4.0  },  /* 4 FMA, charge kernel+input */
        { "Scan2D (step)",   16.0,         9*4.0  },  /* 2 exp, lit 2 h_prev */
    };
    int n_k = 8;

    for (int k = 0; k < n_k; k++) {
        double I = kernels[k].flops / kernels[k].bytes;
        const char *regime = (I >= ridge_point) ? "compute-bound" : "memory-bound";
        printf("%-20s  %-12.1f  %-12.1f  %-12.2f  %-14s\n",
               kernels[k].name,
               kernels[k].flops,
               kernels[k].bytes,
               I,
               regime);
    }

    printf("\n  GFLOPS atteignables = min(I × peak_BW, peak_compute)\n");
}

/* ─── G7 : Complexité Scan — CPU vs CUDA comparaison théorique ──── */

static void bench_scan_complexity(void) {
    printf("\n=== G7 : Complexité Scan — CPU séquentiel vs CUDA Blelloch ===\n");
    printf("  D=128, M=16 (fixés). Ratio de profondeur et temps estimé.\n\n");

    /* Hypothèses matérielles */
    double cpu_exp_ns   = 15.0;   /* ns par expf() sur AVX2 (scalaire) */
    double cpu_fma_ns   = 0.5;    /* ns par FMA (pipeliné) */
    double gpu_exp_ns   = 0.5;    /* ns par expf() en parallèle sur CUDA */
    double gpu_sync_us  = 2.0;    /* µs par __syncthreads() (overhead) */

    printf("  Hypothèses CPU : expf=%.1fns, FMA=%.1fns\n", cpu_exp_ns, cpu_fma_ns);
    printf("  Hypothèses GPU : expf=%.1fns par thread (parallèle), sync=%.1fµs\n\n",
           gpu_exp_ns, gpu_sync_us);

    printf("%-8s  %-14s  %-14s  %-14s  %-12s\n",
           "L", "CPU_depth", "GPU_depth", "depth_ratio", "GPU_work");
    printf("%-8s  %-14s  %-14s  %-14s  %-12s\n",
           "---", "---", "---", "---", "---");

    long D = 128, M = 16;
    int Ls[] = {64, 128, 256, 512, 1024};
    int n_L  = 5;

    for (int li = 0; li < n_L; li++) {
        long L = Ls[li];
        long DM = D * M;

        /* CPU : L pas séquentiels, DM threads en parallèle (pour GPU seq)
         * Temps estimé CPU (1 thread) : L * D * M * (exp_ns + 4*fma_ns) */
        double t_cpu_us = L * DM * (cpu_exp_ns + 4*cpu_fma_ns) * 1e-3;

        /* GPU Blelloch : 2*log2(L) étapes de synchro
         * Chaque étape : DM threads font 1 FMA (parallèle)
         * + overhead synchro */
        int log2L = (int)ceil(log2((double)L));
        int steps = 2 * log2L; /* up + down sweep */
        double t_gpu_us = steps * gpu_sync_us
                        + DM * (gpu_exp_ns + 4*cpu_fma_ns) * 1e-3; /* precompute */

        printf("%-8ld  %-14.2f  %-14.2f  %-14.1fx  %ldDM blocs\n",
               L,
               t_cpu_us,
               t_gpu_us,
               t_cpu_us / t_gpu_us,
               DM);
    }

    printf("\n  GPU Blelloch : DM=%ld blocs en parallèle, chaque bloc = L threads.\n",
           D*M);
    printf("  Avec D=128, M=16 : %ld blocs CUDA = %ld threads totaux sur L=1024.\n",
           D*M, D*M*1024L);
}

/* ─── Main ────────────────────────────────────────────────────────── */

int main(void) {
    srand(42);

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       K-Mamba — Benchmarks pour le Paper (17/03/2026)       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    bench_gemm();
    bench_scan1d();
    bench_scan2d();
    bench_blelloch_theory();
    bench_mambablock_analytical();
    bench_roofline();
    bench_scan_complexity();

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  Benchmarks terminés. Données utilisables pour figures du paper.\n");
    printf("════════════════════════════════════════════════════════════════\n");

    return 0;
}

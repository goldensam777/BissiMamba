/*
 * test_scan1d.c — Forward scan1d : ASM vs référence C
 *
 * Teste scan1d() (cpu/scan1d.asm) contre une implémentation scalaire
 * pour M=1 (cas utilisé par le modèle) et M=2 (cas générique).
 *
 * Layout mémoire rappel :
 *   x     [L, D]        — entrée
 *   A     [D, M]        — matrice de transition (partagée sur L)
 *   B     [L, D, M]     — sélectif
 *   C     [L, D, M]     — sélectif
 *   delta [L, D]        — pas de temps adaptatif
 *   h     [L, D, M]     — états cachés (sortie)
 *   y     [L, D]        — sortie
 *
 * Récurrence :
 *   h_t[d,m] = exp(dt_t[d] * A[d,m]) * h_{t-1}[d,m]
 *            + dt_t[d] * B_t[d,m] * x_t[d]
 *   y_t[d]   = sum_m  C_t[d,m] * h_t[d,m]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "scan_nd.h"

#define PASS_TAG  "  [PASS]"
#define FAIL_TAG  "  [FAIL]"

/* ── Référence C scalaire — layout général [L, D, M] ──────────── */
static void scan1d_ref(
    const float *x,      /* [L, D]     */
    const float *A,      /* [D, M]     */
    const float *B,      /* [L, D, M]  */
    const float *C,      /* [L, D, M]  */
    const float *delta,  /* [L, D]     */
    float       *h,      /* [L, D, M]  out */
    float       *y,      /* [L, D]     out */
    long L, long D, long M)
{
    float *state = (float *)calloc((size_t)(D * M), sizeof(float));
    memset(y, 0, (size_t)(L * D) * sizeof(float));
    memset(h, 0, (size_t)(L * D * M) * sizeof(float));

    for (long t = 0; t < L; t++) {
        for (long d = 0; d < D; d++) {
            float dt = delta[t * D + d];
            float xt = x[t * D + d];
            float yt = 0.0f;

            for (long m = 0; m < M; m++) {
                long dm  = d * M + m;
                long tdm = (t * D + d) * M + m;

                float dA   = expf(dt * A[dm]);
                float bbar = dt * B[tdm];
                state[dm]  = dA * state[dm] + bbar * xt;
                h[tdm]     = state[dm];
                yt        += C[tdm] * state[dm];
            }

            y[t * D + d] = yt;
        }
    }
    free(state);
}

/* ── Utilitaires ──────────────────────────────────────────────── */
static void fill_rand(float *p, long n, float lo, float hi) {
    for (long i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static float max_abs_diff(const float *a, const float *b, long n) {
    float worst = 0.0f;
    for (long i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/* ── Vérification que aucune valeur n'est NaN/Inf ─────────────── */
static int all_finite(const float *v, long n) {
    for (long i = 0; i < n; i++)
        if (!isfinite(v[i])) return 0;
    return 1;
}

/* ────────────────────────────────────────────────────────────────
 * run_test : ASM scan1d vs C reference
 * ──────────────────────────────────────────────────────────────── */
static int run_test(const char *label, long L, long D, long M, float eps) {
    printf("\n--- %s (L=%ld D=%ld M=%ld) ---\n", label, L, D, M);

    long LD   = L * D;
    long DM   = D * M;
    long LDM  = L * D * M;

    float *x      = (float *)malloc((size_t)LD  * sizeof(float));
    float *A      = (float *)malloc((size_t)DM  * sizeof(float));
    float *B      = (float *)malloc((size_t)LDM * sizeof(float));
    float *C      = (float *)malloc((size_t)LDM * sizeof(float));
    float *delta  = (float *)malloc((size_t)LD  * sizeof(float));
    float *h_ref  = (float *)malloc((size_t)LDM * sizeof(float));
    float *y_ref  = (float *)malloc((size_t)LD  * sizeof(float));
    /* h_asm : le kernel ASM utilise h comme état courant [D, M] seulement.
     * On alloue [L, D, M] pour sécurité mais on ne compare que [D, M]. */
    float *h_asm  = (float *)malloc((size_t)LDM * sizeof(float));
    float *y_asm  = (float *)malloc((size_t)LD  * sizeof(float));

    if (!x || !A || !B || !C || !delta || !h_ref || !y_ref || !h_asm || !y_asm) {
        fprintf(stderr, "  alloc failed\n");
        return 0;
    }

    /* A doit être négatif pour la stabilité du scan */
    fill_rand(x,     LD,  -1.0f,  1.0f);
    fill_rand(A,     DM,  -1.0f, -0.01f);
    fill_rand(B,     LDM, -0.5f,  0.5f);
    fill_rand(C,     LDM, -0.5f,  0.5f);
    fill_rand(delta, LD,   0.01f, 0.5f);

    /* Référence C */
    scan1d_ref(x, A, B, C, delta, h_ref, y_ref, L, D, M);

    /* Kernel ASM via scan1d() */
    ScanParams sp = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h = h_asm, .y = y_asm,
        .L = L, .D = D, .M = M
    };
    memset(h_asm, 0, (size_t)LDM * sizeof(float));
    memset(y_asm, 0, (size_t)LD  * sizeof(float));
    scan1d(&sp);

    int ok = 1;

    /* Test 1 : y est fini */
    if (!all_finite(y_asm, LD)) {
        printf("%s y contient NaN/Inf\n", FAIL_TAG);
        ok = 0;
    } else {
        printf("%s y est fini\n", PASS_TAG);
    }

    /* Test 2 : y correspond à la référence */
    float diff_y = max_abs_diff(y_ref, y_asm, LD);
    if (diff_y <= eps) {
        printf("%s y ASM == y_ref  (max_err=%.2e)\n", PASS_TAG, diff_y);
    } else {
        printf("%s y ASM != y_ref  (max_err=%.2e > tol=%.2e)\n",
               FAIL_TAG, diff_y, eps);
        ok = 0;
    }

    /* Test 3 : état final h_asm[0..DM] == h_ref[(L-1)*DM..(L)*DM]
     * Le kernel ASM maintient h comme état courant [D, M] (pas [L, D, M]).
     * Après le scan, h_asm[dm] = état au pas L-1 pour le canal dm. */
    float diff_h = max_abs_diff(h_ref + (long)(L - 1) * DM, h_asm, DM);
    if (diff_h <= eps) {
        printf("%s h_final ASM == h_ref[L-1]  (max_err=%.2e)\n", PASS_TAG, diff_h);
    } else {
        printf("%s h_final ASM != h_ref[L-1]  (max_err=%.2e > tol=%.2e)\n",
               FAIL_TAG, diff_h, eps);
        ok = 0;
    }

    free(x); free(A); free(B); free(C); free(delta);
    free(h_ref); free(y_ref); free(h_asm); free(y_asm);
    return ok;
}

/* ── Cas degenéré : zéros ─────────────────────────────────────── */
static int test_zero_input(void) {
    printf("\n--- Cas dégénéré : x=0, B=0 → h=0, y=0 ---\n");
    long L=4, D=8, M=1;
    long LD=L*D;

    float *x = calloc(LD, sizeof(float));
    float A[8]; for (int i=0;i<8;i++) A[i]=-0.5f;
    float *B = calloc(LD, sizeof(float));
    float *C = malloc(LD * sizeof(float)); for(int i=0;i<LD;i++) C[i]=1.0f;
    float *delta = malloc(LD*sizeof(float)); for(int i=0;i<LD;i++) delta[i]=0.1f;
    float *h = calloc(LD, sizeof(float));
    float *y = calloc(LD, sizeof(float));

    ScanParams sp = {.x=x,.A=A,.B=B,.C=C,.delta=delta,.h=h,.y=y,.L=L,.D=D,.M=M};
    scan1d(&sp);

    float s = 0.0f;
    for (int i = 0; i < LD; i++) s += fabsf(y[i]) + fabsf(h[i]);
    int ok = (s < 1e-12f);
    printf("%s x=0 B=0 → y=0 h=0 (sum=%.2e)\n", ok ? PASS_TAG : FAIL_TAG, s);

    free(x); free(B); free(C); free(delta); free(h); free(y);
    return ok;
}

/* ── Main ─────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Tests scan1d : ASM vs référence C ===\n");
    srand(42);

    int passed = 0, total = 0;

#define RUN(label, L, D, M, eps) do { \
    total++; passed += run_test(label, L, D, M, eps); \
} while(0)

    /* M=1 — cas principal du modèle */
    RUN("small M=1",   4,  8, 1, 1e-5f);
    RUN("medium M=1", 16, 16, 1, 1e-5f);
    RUN("large M=1",  64, 32, 1, 1e-5f);
    RUN("D multiple 8", 32, 24, 1, 1e-5f); /* queue scalaire */

    /* M=2 — cas générique */
    RUN("small M=2",   4,  8, 2, 1e-5f);
    RUN("medium M=2", 16, 16, 2, 1e-5f);

    /* Cas dégénéré */
    total++; passed += test_zero_input();

#undef RUN

    printf("\n=== %d/%d tests passés ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}

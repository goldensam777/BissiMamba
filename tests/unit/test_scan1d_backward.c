/*
 * test_scan1d_backward.c — Gradient check numérique sur scan1d_backward
 *
 * Vérifie que les gradients analytiques (scan1d_backward) correspondent
 * aux gradients numériques (différences finies centrées) pour chaque
 * paramètre d'entrée : x, A, B, C, delta.
 *
 * Layout mémoire (scan1d_backward — générique) :
 *   x     [L, D]        delta [L, D]
 *   A     [D, M]        B     [L, D, M]
 *   C     [L, D, M]     h     [L, D, M]
 *
 * Loss scalaire : sum(y) = sum_{t,d} y_t[d]
 * donc dy[t,d] = 1 pour tout (t,d).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "scan.h"

#define PASS_TAG "  [PASS]"
#define FAIL_TAG "  [FAIL]"

/* ── Forward de référence C ────────────────────────────────────── */
static void scan_forward_ref(
    const float *x,     /* [L, D]    */
    const float *A,     /* [D, M]    */
    const float *B,     /* [L, D, M] */
    const float *C,     /* [L, D, M] */
    const float *delta, /* [L, D]    */
    float       *h,     /* [L, D, M] out */
    float       *y,     /* [L, D]    out */
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

/* loss = sum(y) — alloue h/y en interne */
static float compute_loss(
    const float *x,  const float *A,  const float *B,
    const float *C,  const float *delta,
    long L, long D, long M)
{
    long LDM = L * D * M;
    long LD  = L * D;
    float *h = (float *)malloc((size_t)LDM * sizeof(float));
    float *y = (float *)malloc((size_t)LD  * sizeof(float));

    scan_forward_ref(x, A, B, C, delta, h, y, L, D, M);

    float s = 0.0f;
    for (long i = 0; i < LD; i++) s += y[i];

    free(h);
    free(y);
    return s;
}

/* ── Utilitaires ───────────────────────────────────────────────── */
static void fill_rand(float *p, long n, float lo, float hi) {
    for (long i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

/*
 * Gradient check différences finies centrées.
 * Perturbe chaque élément i de `param` (taille n), recompute loss,
 * et compare avec le gradient analytique `ana`.
 * Renvoie l'erreur absolue maximale.
 *
 * Note : param doit pointer dans l'un des buffers x/A/B/C/delta,
 * de sorte que la perturbation soit visible par compute_loss.
 */
static float grad_check_param(
    float *param, long n, const float *ana,
    const float *x,  const float *A,  const float *B,
    const float *C,  const float *delta,
    long L, long D, long M, float eps)
{
    float worst = 0.0f;
    for (long i = 0; i < n; i++) {
        float orig = param[i];

        param[i] = orig + eps;
        float lp = compute_loss(x, A, B, C, delta, L, D, M);

        param[i] = orig - eps;
        float lm = compute_loss(x, A, B, C, delta, L, D, M);

        param[i] = orig;

        float num = (lp - lm) / (2.0f * eps);
        float err = fabsf(ana[i] - num);
        if (err > worst) worst = err;
    }
    return worst;
}

/* ── Test principal ────────────────────────────────────────────── */
static int run_grad_test(const char *label, long L, long D, long M,
                         float tol, float eps)
{
    printf("\n--- %s (L=%ld D=%ld M=%ld) ---\n", label, L, D, M);

    long LD  = L * D;
    long DM  = D * M;
    long LDM = L * D * M;

    float *x     = (float *)malloc((size_t)LD  * sizeof(float));
    float *A     = (float *)malloc((size_t)DM  * sizeof(float));
    float *B     = (float *)malloc((size_t)LDM * sizeof(float));
    float *C     = (float *)malloc((size_t)LDM * sizeof(float));
    float *delta = (float *)malloc((size_t)LD  * sizeof(float));
    float *h     = (float *)malloc((size_t)LDM * sizeof(float));
    float *y     = (float *)malloc((size_t)LD  * sizeof(float));
    float *dy    = (float *)malloc((size_t)LD  * sizeof(float));
    float *dx    = (float *)malloc((size_t)LD  * sizeof(float));
    float *dA    = (float *)malloc((size_t)DM  * sizeof(float));
    float *dB    = (float *)malloc((size_t)LDM * sizeof(float));
    float *dC    = (float *)malloc((size_t)LDM * sizeof(float));
    float *ddelta= (float *)malloc((size_t)LD  * sizeof(float));

    if (!x||!A||!B||!C||!delta||!h||!y||!dy||!dx||!dA||!dB||!dC||!ddelta) {
        fprintf(stderr, "  alloc failed\n");
        return 0;
    }

    fill_rand(x,     LD,  -1.0f,  1.0f);
    fill_rand(A,     DM,  -1.0f, -0.01f);  /* A négatif pour stabilité */
    fill_rand(B,     LDM, -0.5f,  0.5f);
    fill_rand(C,     LDM, -0.5f,  0.5f);
    fill_rand(delta, LD,   0.01f, 0.5f);

    /* Forward + stockage h (requis par backward) */
    scan_forward_ref(x, A, B, C, delta, h, y, L, D, M);

    /* dy = 1 partout (loss = sum y) */
    for (long i = 0; i < LD; i++) dy[i] = 1.0f;

    /* Gradients analytiques */
    ScanBackwardParams bp = {
        .x = x, .A = A, .B = B, .C = C, .delta = delta,
        .h0 = NULL, .h = h, .dy = dy,
        .dx = dx, .dA = dA, .dB = dB, .dC = dC, .ddelta = ddelta,
        .L = L, .D = D, .M = M
    };
    scan1d_backward(&bp);

    int ok = 1;
    float err;

    err = grad_check_param(x,     LD,  dx,     x, A, B, C, delta, L, D, M, eps);
    if (err <= tol) printf("%s grad_x     ok  (max_err=%.2e)\n", PASS_TAG, err);
    else { printf("%s grad_x     KO  (max_err=%.2e > tol=%.2e)\n", FAIL_TAG, err, tol); ok = 0; }

    err = grad_check_param(A,     DM,  dA,     x, A, B, C, delta, L, D, M, eps);
    if (err <= tol) printf("%s grad_A     ok  (max_err=%.2e)\n", PASS_TAG, err);
    else { printf("%s grad_A     KO  (max_err=%.2e > tol=%.2e)\n", FAIL_TAG, err, tol); ok = 0; }

    err = grad_check_param(B,     LDM, dB,     x, A, B, C, delta, L, D, M, eps);
    if (err <= tol) printf("%s grad_B     ok  (max_err=%.2e)\n", PASS_TAG, err);
    else { printf("%s grad_B     KO  (max_err=%.2e > tol=%.2e)\n", FAIL_TAG, err, tol); ok = 0; }

    err = grad_check_param(C,     LDM, dC,     x, A, B, C, delta, L, D, M, eps);
    if (err <= tol) printf("%s grad_C     ok  (max_err=%.2e)\n", PASS_TAG, err);
    else { printf("%s grad_C     KO  (max_err=%.2e > tol=%.2e)\n", FAIL_TAG, err, tol); ok = 0; }

    err = grad_check_param(delta, LD,  ddelta, x, A, B, C, delta, L, D, M, eps);
    if (err <= tol) printf("%s grad_delta ok  (max_err=%.2e)\n", PASS_TAG, err);
    else { printf("%s grad_delta KO  (max_err=%.2e > tol=%.2e)\n", FAIL_TAG, err, tol); ok = 0; }

    free(x); free(A); free(B); free(C); free(delta);
    free(h); free(y); free(dy);
    free(dx); free(dA); free(dB); free(dC); free(ddelta);
    return ok;
}

/* ── Main ──────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Tests scan1d_backward : gradient check numérique ===\n");
    srand(42);

    int passed = 0, total = 0;

#define RUN(label, L, D, M, tol, eps) do { \
    total++; passed += run_grad_test(label, L, D, M, tol, eps); \
} while(0)

    /* M=1 — cas principal */
    RUN("small M=1",   4,  8, 1, 1e-3f, 1e-3f);
    RUN("medium M=1",  8, 16, 1, 1e-3f, 1e-3f);

    /* M=2 — cas générique */
    RUN("small M=2",   4,  8, 2, 1e-3f, 1e-3f);
    RUN("medium M=2",  8, 16, 2, 1e-3f, 1e-3f);

#undef RUN

    printf("\n=== %d/%d tests passés ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}

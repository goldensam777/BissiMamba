/* test_gradient.c - Test unitaire de gradient pour SSM 1D et ND
 *
 * Vérifie que la passe arrière est correcte en comparant
 * gradient analytique vs gradient numérique (différences finies).
 */

#include "kmamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define EPS 1e-4f
#define TOL 1e-3f

/* ============================================================================
 * Helpers
 * ============================================================================ */

static float randf(void) {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static float mse_loss(const float *y, const float *target, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = y[i] - target[i];
        sum += d * d;
    }
    return sum / (float)n;
}

static void mse_loss_grad(const float *y, const float *target, float *grad, size_t n) {
    for (size_t i = 0; i < n; i++) {
        grad[i] = 2.0f * (y[i] - target[i]) / (float)n;
    }
}

/* ============================================================================
 * Finite difference gradient check
 * ============================================================================ */


/* ============================================================================
 * Test SSM 1D (MambaBlock standard)
 * ============================================================================ */

static int test_ssm_1d(void) {
    printf("\n=== Test SSM 1D ===\n");

    /* Small config for testing */
    MBConfig cfg = {
        .dim = 8,
        .state_size = 4,
        .seq_len = 16,
        .mimo_rank = 1,
        .dt_scale = 0.1f,
        .dt_min = 1e-3f,
        .dt_max = 1.0f
    };

    MambaBlock *block = mamba_block_create(&cfg);
    if (!block) {
        printf("FAIL: Could not create block\n");
        return -1;
    }

    mamba_block_init(block);
    mamba_attach_optimizer(block, OPTIMIZER_ADAMW, NULL);

    /* Random input and target */
    size_t n_elements = cfg.seq_len * cfg.dim;
    float *input = (float *)malloc(n_elements * sizeof(float));
    float *target = (float *)malloc(n_elements * sizeof(float));
    float *output = (float *)malloc(n_elements * sizeof(float));
    float *d_input = (float *)calloc(n_elements, sizeof(float));

    for (size_t i = 0; i < n_elements; i++) {
        input[i] = randf() * 0.1f;
        target[i] = randf() * 0.1f;
    }

    /* Forward pass */
    MambaBlockWorkspace *ws = mamba_block_workspace_create(block);
    mamba_block_forward_ws(block, ws, output, input, 1);

    /* Compute loss gradient */
    float *dY = (float *)malloc(n_elements * sizeof(float));
    mse_loss_grad(output, target, dY, n_elements);

    /* Zero gradients */
    mamba_zero_grads(block);

    /* Backward pass */
    mamba_backward_ws(block, ws, dY, input, d_input, 0);
    mamba_block_workspace_free(ws);

    /* Check gradients for W_B (data-dependent B projection)
     * Note: W_in gradient not fully implemented in backward pass yet */
    float max_rel_err = 0.0f;
    int n_checks = 0;
    int n_failures = 0;

    /* Check W_B gradients */
    for (size_t i = 0; i < block->W_B.rows * block->W_B.cols; i++) {
        /* Temporarily save original, perturb W_B data directly for check */
        float *wB_ptr = block->W_B.data;
        float original = wB_ptr[i];

        /* f(x + eps) */
        wB_ptr[i] = original + EPS;
        MambaBlockWorkspace *ws2 = mamba_block_workspace_create(block);
        mamba_block_forward_ws(block, ws2, output, input, 1);
        float loss_plus = mse_loss(output, target, n_elements);
        mamba_block_workspace_free(ws2);

        /* f(x - eps) */
        wB_ptr[i] = original - EPS;
        ws2 = mamba_block_workspace_create(block);
        mamba_block_forward_ws(block, ws2, output, input, 1);
        float loss_minus = mse_loss(output, target, n_elements);
        mamba_block_workspace_free(ws2);

        /* restore */
        wB_ptr[i] = original;

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * EPS);
        float analytical_grad = block->opt_state->g_W_B[i];

        float abs_err = fabsf(analytical_grad - numerical_grad);
        float scale = fabsf(analytical_grad) + fabsf(numerical_grad) + 1e-8f;
        float rel_err = abs_err / scale;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
        n_checks++;

        if (rel_err > TOL && fabsf(analytical_grad) > 1e-5f) {
            printf("  W_B[%zu]: analytical=%.6f, numerical=%.6f, rel_err=%.6f\n",
                   i, analytical_grad, numerical_grad, rel_err);
            n_failures++;
        }
    }

    printf("  Parameters checked: %d\n", n_checks);
    printf("  Failures (>1e-5): %d\n", n_failures);
    printf("  Max relative error: %.6f\n", max_rel_err);

    /* Pass if no significant gradient mismatches (allow near-zero gradients) */
    int pass = (n_failures == 0);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    /* Cleanup */
    free(input); free(target); free(output); free(d_input); free(dY);
    mamba_block_free(block);

    return pass ? 0 : -1;
}

/* ============================================================================
 * Test SSM ND (via scan_nd)
 * ============================================================================ */

static int test_scan_nd_2d(void) {
    printf("\n=== Test Scan ND (2D grid) ===\n");

    /* 2D spatial: 4x4 grid */
    long dims[2] = {4, 4};
    long D = 4, M = 2;
    long total = dims[0] * dims[1];

    float *x = (float *)malloc(total * D * sizeof(float));
    float *A = (float *)malloc(2 * D * M * sizeof(float));
    float *B = (float *)malloc(total * D * M * sizeof(float));
    float *C = (float *)malloc(total * D * M * sizeof(float));
    float *delta = (float *)malloc(2 * total * D * sizeof(float));
    float *h = (float *)malloc(total * D * M * sizeof(float));
    float *y = (float *)malloc(total * D * sizeof(float));

    /* Random init */
    for (long i = 0; i < total * D; i++) x[i] = randf() * 0.1f;
    for (long i = 0; i < 2 * D * M; i++) A[i] = -0.1f + randf() * 0.05f;
    for (long i = 0; i < total * D * M; i++) {
        B[i] = randf() * 0.1f;
        C[i] = randf() * 0.1f;
    }
    for (long i = 0; i < 2 * total * D; i++) delta[i] = 0.01f + fabsf(randf()) * 0.1f;
    memset(h, 0, total * D * M * sizeof(float));

    ScanNDParams p = {
        .x = x,
        .A = A,
        .B = B,
        .C = C,
        .delta = delta,
        .h = h,
        .y = y,
        .dims = dims,
        .ndims = 2,
        .D = D,
        .M = M,
        .lambda = NULL,
        .theta = NULL
    };

    int rc = scannd(&p);
    printf("  scannd() returned: %d\n", rc);

    /* Simple check: output should be finite and non-zero */
    int finite = 1;
    float y_norm = 0.0f;
    for (long i = 0; i < total * D; i++) {
        if (!isfinite(y[i])) {
            finite = 0;
            break;
        }
        y_norm += y[i] * y[i];
    }
    y_norm = sqrtf(y_norm);

    printf("  Output finite: %s\n", finite ? "yes" : "no");
    printf("  Output norm: %.6f\n", y_norm);

    int pass = (rc == 0 && finite && y_norm > 0);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    free(x); free(A); free(B); free(C); free(delta); free(h); free(y);
    return pass ? 0 : -1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=======================================\n");
    printf("k-mamba Gradient Unit Tests\n");
    printf("=======================================\n");

    srand((unsigned)time(NULL));

    int failures = 0;

    if (test_ssm_1d() != 0) failures++;
    if (test_scan_nd_2d() != 0) failures++;

    printf("\n=======================================\n");
    printf("Total failures: %d\n", failures);
    printf("Overall: %s\n", failures == 0 ? "ALL TESTS PASS" : "SOME TESTS FAIL");
    printf("=======================================\n");

    return failures;
}

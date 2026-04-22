
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmamba.h"
#include "kmamba_kernels.h"

#define TEST_SEQ_LEN 16
#define TEST_DIM 32
#define TEST_STATE 8

int main(void) {
    printf("=== Test Mamba-3 Backward (CPU) ===\n");

    MBConfig cfg = {
        .dim = TEST_DIM,
        .state_size = TEST_STATE,
        .seq_len = TEST_SEQ_LEN,
        .mimo_rank = 1,
        .dt_scale = 0.01f,
        .dt_min = 1e-3f,
        .dt_max = 0.1f,
        .spatial_ndims = 0,
        .use_convnd = 0
    };

    MambaBlock *block = mamba_block_create(&cfg);
    mamba_block_init(block);
    
    MBOptimConfig opt_cfg = { .lr = 0.1f, .eps = 1e-8f, .weight_decay = 0.0f };
    mamba_attach_optimizer(block, OPTIMIZER_ADAMW, &opt_cfg);

    float *input = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    float *output = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    float *dY = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));
    float *dX = (float *)malloc(TEST_SEQ_LEN * TEST_DIM * sizeof(float));

    for (int i = 0; i < TEST_SEQ_LEN * TEST_DIM; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        dY[i] = 1.0f; /* Constant gradient for testing */
    }

    /* 1. Forward */
    MambaBlockWorkspace *ws = mamba_block_workspace_create(block);
    mamba_block_forward_ws(block, ws, output, input, 1);
    printf("[OK] Forward pass (CPU)\n");

    /* 2. Backward */
    mamba_zero_grads(block);
    mamba_backward_ws(block, ws, dY, input, dX, 0);
    printf("[OK] Backward pass (CPU)\n");
    
    mamba_block_workspace_free(ws);

    /* 3. Check gradients */
    MBOptimState *s = (MBOptimState *)block->opt_state;
    float gn = mamba_block_grad_sqnorm(block);
    printf("[INFO] Gradient sqnorm: %f\n", gn);

    if (gn > 0.0f) {
        printf("[OK] Gradients are non-zero\n");
    } else {
        printf("FAIL: Gradients are zero\n");
        return 1;
    }

    /* Check for NaN in some critical gradients */
    for (size_t i = 0; i < TEST_STATE; i++) {
        if (isnan(s->g_A_log[i])) {
            printf("FAIL: NaN in g_A_log[%zu]\n", i);
            return 1;
        }
    }
    printf("[OK] No NaN in A_log gradients\n");

    /* 4. Optimizer Step */
    float old_A0 = block->A_log.data[0];
    mamba_optimizer_step(block, &opt_cfg);
    float new_A0 = block->A_log.data[0];
    
    if (fabsf(new_A0 - old_A0) > 0.0f) {
        printf("[OK] Parameters updated (A_log[0]: %f -> %f)\n", old_A0, new_A0);
    } else {
        printf("FAIL: Parameters did not update\n");
        return 1;
    }

    mamba_block_free(block);
    free(input); free(output); free(dY); free(dX);

    printf("\n=== All Backward tests PASSED ===\n");
    return 0;
}

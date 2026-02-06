#include "mamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    (void)0;  /* Suppress unused parameter warnings */
    printf("=== Mamba State Space Model in C ===\n\n");
    
    /* Configuration for a simple Mamba block */
    MambaConfig config = {
        .dim = 64,              /* Model dimension */
        .state_size = 16,       /* State space dimension (N) */
        .seq_len = 10,          /* Sequence length */
        .dt_rank = 0.1f,        /* Delta time rank */
        .dt_scale = 1.0f,       /* Delta time scale */
        .dt_init = 0.001f,      /* Delta time initialization */
        .dt_min = 0.001f,       /* Minimum delta time */
        .dt_max = 0.1f          /* Maximum delta time */
    };
    
    printf("Mamba Configuration:\n");
    printf("  - Model dimension: %zu\n", config.dim);
    printf("  - State size: %zu\n", config.state_size);
    printf("  - Sequence length: %zu\n", config.seq_len);
    printf("  - dt range: [%f, %f]\n\n", config.dt_min, config.dt_max);
    
    /* Create and initialize Mamba block */
    printf("Creating Mamba block...\n");
    MambaBlock *mamba = mamba_block_create(&config);
    if (!mamba) {
        fprintf(stderr, "Error: Failed to create Mamba block\n");
        return 1;
    }
    
    printf("Initializing parameters...\n");
    srand((unsigned int)time(NULL));
    mamba_block_init(mamba);
    
    /* Create test input (batch_size=1, seq_len=10, dim=64) */
    size_t batch_size = 1;
    size_t total_input_size = batch_size * config.seq_len * config.dim;
    real_t *input = (real_t *)malloc(total_input_size * sizeof(real_t));
    real_t *output = (real_t *)malloc(total_input_size * sizeof(real_t));
    
    if (!input || !output) {
        fprintf(stderr, "Error: Failed to allocate input/output buffers\n");
        mamba_block_free(mamba);
        return 1;
    }
    
    printf("Generating random input data...\n");
    for (size_t i = 0; i < total_input_size; i++) {
        input[i] = ((real_t)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    /* Run forward pass */
    printf("Running forward pass through Mamba...\n");
    mamba_forward(mamba, output, input, batch_size);
    
    /* Display some results */
    printf("\n=== Input Sample ===\n");
    printf("First timestep, first 10 dimensions:\n");
    for (size_t i = 0; i < 10 && i < config.dim; i++) {
        printf("  input[0][%zu] = %10.6f\n", i, input[i]);
    }
    
    printf("\n=== Output Sample ===\n");
    printf("First timestep, first 10 dimensions:\n");
    for (size_t i = 0; i < 10 && i < config.dim; i++) {
        printf("  output[0][%zu] = %10.6f\n", i, output[i]);
    }
    
    /* Display state information */
    printf("\n=== Internal State Information ===\n");
    printf("A matrix (diagonal, first 10 values):\n");
    for (size_t i = 0; i < 10 && i < mamba->A_log.rows; i++) {
        printf("  A_log[%zu] = %10.6f\n", i, mamba->A_log.data[i]);
    }
    
    printf("\nB vector:\n");
    for (size_t i = 0; i < 10 && i < mamba->B_mat.rows; i++) {
        printf("  B[%zu] = %10.6f\n", i, mamba->B_mat.data[i]);
    }
    
    printf("\nC vector:\n");
    for (size_t i = 0; i < 10 && i < mamba->C_mat.rows; i++) {
        printf("  C[%zu] = %10.6f\n", i, mamba->C_mat.data[i]);
    }
    
    /* Cleanup */
    printf("\n=== Cleanup ===\n");
    free(input);
    free(output);
    mamba_block_free(mamba);
    
    printf("Mamba model test completed successfully!\n");
    
    return 0;
}

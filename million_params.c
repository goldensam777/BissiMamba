#include "mamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    /* 1 Million Parameter Mamba Model */
    MambaConfig config = {
        .dim = 706,              /* Model dimension */
        .state_size = 512,       /* State space dimension */
        .seq_len = 32,           /* Sequence length for inference */
        .dt_rank = 16,
        .dt_scale = 1.0f,
        .dt_init = 0.001f,
        .dt_min = 0.001f,
        .dt_max = 0.1f,
    };
    
    printf("=== Mamba 1M Parameter Model ===\n\n");
    printf("Configuration:\n");
    printf("  Model dimension:   %zu\n", config.dim);
    printf("  State size:        %zu\n", config.state_size);
    printf("  Sequence length:   %zu\n", config.seq_len);
    
    /* Compute parameter count */
    size_t params = 2 * config.dim * config.dim + 3 * config.state_size + config.dim;
    printf("  Total parameters:  %zu (~%.2fM)\n\n", params, (double)params / 1e6);
    
    /* Create block */
    printf("Creating Mamba block...\n");
    MambaBlock *block = mamba_block_create(&config);
    if (!block) {
        fprintf(stderr, "Failed to create Mamba block\n");
        return 1;
    }
    
    printf("Initializing parameters...\n");
    mamba_block_init(block);
    
    /* Memory analysis */
    size_t model_memory = params * sizeof(real_t);
    printf("Model memory:      %.2f MB\n", (double)model_memory / (1024*1024));
    
    /* Allocate input/output for inference */
    size_t batch_size = 1;
    size_t input_size = config.seq_len * config.dim * batch_size;
    size_t output_size = config.seq_len * config.dim * batch_size;
    
    real_t *input = (real_t *)malloc(input_size * sizeof(real_t));
    real_t *output = (real_t *)malloc(output_size * sizeof(real_t));
    
    if (!input || !output) {
        fprintf(stderr, "Failed to allocate I/O buffers\n");
        return 1;
    }
    
    printf("Input buffer:      %.2f MB\n", (double)input_size * sizeof(real_t) / (1024*1024));
    printf("Output buffer:     %.2f MB\n", (double)output_size * sizeof(real_t) / (1024*1024));
    
    /* Generate random input */
    printf("\nGenerating random input...\n");
    srand(42);
    for (size_t i = 0; i < input_size; i++) {
        input[i] = ((real_t)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    /* Forward pass with timing */
    printf("Running forward pass...\n");
    clock_t start = clock();
    mamba_forward(block, output, input, batch_size);
    clock_t end = clock();
    
    double elapsed_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  Time: %.2f ms\n", elapsed_ms);
    printf("  Throughput: %.2f timesteps/ms\n", config.seq_len / elapsed_ms);
    printf("  Throughput: %.2f sequences/sec\n", 1000.0 / elapsed_ms);
    
    /* Compute flops (rough estimate: W_in, selective_scan, W_out) */
    double flops = 0.0;
    flops += config.seq_len * config.dim * config.dim;  /* W_in projection */
    flops += config.seq_len * config.state_size * 2;    /* selective_scan */
    flops += config.seq_len * config.dim * config.dim;  /* W_out projection */
    
    printf("  Estimated FLOPs:  %.2e\n", flops);
    if (elapsed_ms > 0) {
        printf("  GFLOP/s:          %.2f\n", flops / (elapsed_ms * 1e6));
    }
    
    /* Sample output */
    printf("\nOutput sample (first timestep, first 10 dims):\n");
    for (size_t i = 0; i < 10; i++) {
        printf("  output[0][%zu] = %.6f\n", i, output[i]);
    }
    
    /* Cleanup */
    printf("\nCleaning up...\n");
    mamba_block_free(block);
    free(input);
    free(output);
    
    printf("=== 1M Parameter Model Test Complete ===\n");
    return 0;
}

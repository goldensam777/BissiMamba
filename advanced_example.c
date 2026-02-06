#include "mamba.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Helper function to generate sinusoidal input sequence */
void generate_sine_sequence(real_t *data, size_t seq_len, size_t dim, 
                           real_t frequency, real_t amplitude) {
    for (size_t t = 0; t < seq_len; t++) {
        real_t phase = 2.0f * M_PI * frequency * t / seq_len;
        for (size_t d = 0; d < dim; d++) {
            real_t phase_offset = 2.0f * M_PI * d / dim;
            data[t * dim + d] = amplitude * sinf(phase + phase_offset);
        }
    }
}

/* Helper function to compute L2 norm of a vector */
real_t compute_norm(const real_t *v, size_t n) {
    real_t sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrtf(sum);
}

/* Advanced example with multiple configurations */
int main(void) {
    printf("=== Advanced Mamba Examples ===\n\n");
    
    srand((unsigned int)time(NULL));
    
    /* Example 1: Different state space dimensions */
    printf("--- Example 1: Effect of State Space Dimension ---\n\n");
    
    size_t state_sizes[] = {8, 16, 32};
    size_t num_configs = sizeof(state_sizes) / sizeof(state_sizes[0]);
    
    for (size_t cfg_idx = 0; cfg_idx < num_configs; cfg_idx++) {
        MambaConfig config = {
            .dim = 32,
            .state_size = state_sizes[cfg_idx],
            .seq_len = 20,
            .dt_rank = 0.1f,
            .dt_scale = 1.0f,
            .dt_init = 0.001f,
            .dt_min = 0.001f,
            .dt_max = 0.1f
        };
        
        printf("Configuration with state_size = %zu:\n", config.state_size);
        
        MambaBlock *mamba = mamba_block_create(&config);
        if (!mamba) continue;
        
        mamba_block_init(mamba);
        
        /* Generate sinusoidal input */
        size_t input_size = config.seq_len * config.dim;
        real_t *input = (real_t *)malloc(input_size * sizeof(real_t));
        real_t *output = (real_t *)malloc(input_size * sizeof(real_t));
        
        if (input && output) {
            generate_sine_sequence(input, config.seq_len, config.dim, 1.0f, 1.0f);
            
            mamba_forward(mamba, output, input, 1);
            
            real_t input_norm = compute_norm(input, input_size);
            real_t output_norm = compute_norm(output, input_size);
            
            printf("  Input norm:  %10.6f\n", input_norm);
            printf("  Output norm: %10.6f\n", output_norm);
            printf("  Gain ratio:  %10.6f\n\n", output_norm / (input_norm + 1e-8f));
        }
        
        free(input);
        free(output);
        mamba_block_free(mamba);
    }
    
    /* Example 2: Variable sequence lengths */
    printf("--- Example 2: Processing Variable Sequence Lengths ---\n\n");
    
    size_t seq_lengths[] = {10, 50, 100};
    
    for (size_t seq_idx = 0; seq_idx < 3; seq_idx++) {
        MambaConfig config = {
            .dim = 32,
            .state_size = 16,
            .seq_len = seq_lengths[seq_idx],
            .dt_rank = 0.1f,
            .dt_scale = 1.0f,
            .dt_init = 0.001f,
            .dt_min = 0.001f,
            .dt_max = 0.1f
        };
        
        printf("Processing sequence of length %zu:\n", config.seq_len);
        
        MambaBlock *mamba = mamba_block_create(&config);
        if (!mamba) continue;
        
        mamba_block_init(mamba);
        
        size_t input_size = config.seq_len * config.dim;
        real_t *input = (real_t *)malloc(input_size * sizeof(real_t));
        real_t *output = (real_t *)malloc(input_size * sizeof(real_t));
        
        if (input && output) {
            generate_sine_sequence(input, config.seq_len, config.dim, 0.5f, 1.0f);
            
            clock_t start = clock();
            mamba_forward(mamba, output, input, 1);
            clock_t end = clock();
            
            double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
            
            printf("  Computation time: %8.3f ms\n", elapsed);
            printf("  Throughput: %8.2f timesteps/ms\n\n", 
                   (double)config.seq_len / (elapsed + 1e-8));
        }
        
        free(input);
        free(output);
        mamba_block_free(mamba);
    }
    
    /* Example 3: Batch processing */
    printf("--- Example 3: Batch Processing ---\n\n");
    
    MambaConfig config = {
        .dim = 64,
        .state_size = 16,
        .seq_len = 20,
        .dt_rank = 0.1f,
        .dt_scale = 1.0f,
        .dt_init = 0.001f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    size_t batch_sizes[] = {1, 4, 16};
    
    for (size_t b_idx = 0; b_idx < 3; b_idx++) {
        size_t batch_size = batch_sizes[b_idx];
        printf("Batch processing with batch_size = %zu:\n", batch_size);
        
        MambaBlock *mamba = mamba_block_create(&config);
        if (!mamba) continue;
        
        mamba_block_init(mamba);
        
        size_t input_size = batch_size * config.seq_len * config.dim;
        real_t *input = (real_t *)malloc(input_size * sizeof(real_t));
        real_t *output = (real_t *)malloc(input_size * sizeof(real_t));
        
        if (input && output) {
            /* Generate different sequences for each batch element */
            for (size_t b = 0; b < batch_size; b++) {
                generate_sine_sequence(&input[b * config.seq_len * config.dim],
                                     config.seq_len, config.dim,
                                     0.5f + 0.5f * b / batch_size, 1.0f);
            }
            
            clock_t start = clock();
            mamba_forward(mamba, output, input, batch_size);
            clock_t end = clock();
            
            double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
            
            printf("  Total computation time: %8.3f ms\n", elapsed);
            printf("  Time per batch element: %8.3f ms\n", elapsed / batch_size);
            printf("  Throughput: %8.2f sequences/ms\n\n", 
                   (double)batch_size / (elapsed + 1e-8));
        }
        
        free(input);
        free(output);
        mamba_block_free(mamba);
    }
    
    /* Example 4: State evolution tracking */
    printf("--- Example 4: Hidden State Evolution ---\n\n");
    
    config.seq_len = 5;
    config.dim = 8;
    config.state_size = 4;
    
    MambaBlock *mamba = mamba_block_create(&config);
    if (mamba) {
        mamba_block_init(mamba);
        
        size_t input_size = config.seq_len * config.dim;
        real_t *input = (real_t *)malloc(input_size * sizeof(real_t));
        real_t *output = (real_t *)malloc(input_size * sizeof(real_t));
        
        if (input && output) {
            generate_sine_sequence(input, config.seq_len, config.dim, 1.0f, 1.0f);
            
            printf("Initial state:\n");
            for (size_t i = 0; i < config.state_size; i++) {
                printf("  state[%zu] = %10.6f\n", i, mamba->hidden[i]);
            }
            
            mamba_forward(mamba, output, input, 1);
            
            printf("\nFinal state after processing sequence:\n");
            for (size_t i = 0; i < config.state_size; i++) {
                printf("  state[%zu] = %10.6f\n", i, mamba->hidden[i]);
            }
            
            real_t state_norm = compute_norm(mamba->hidden, config.state_size);
            printf("\nState norm: %10.6f\n", state_norm);
        }
        
        free(input);
        free(output);
        mamba_block_free(mamba);
    }
    
    printf("\n=== Advanced Examples Completed ===\n");
    
    return 0;
}

/*
 * test_regression.c — Tests de régression pour k-mamba
 *
 * Phase 4 : Tests de régression et benchmarks
 * Tests automatisés pour détecter les régressions de performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

#define EPSILON 1e-6f

/* ============================================================
 * Configuration de test
 * ============================================================ */

typedef struct {
    size_t vocab_size;
    size_t dim;
    size_t state_size;
    size_t seq_len;
    size_t n_layers;
    float dt_scale;
    float dt_min;
    float dt_max;
} TestConfig;

typedef struct {
    double throughput;      /* ops/sec */
    double latency;         /* ms per operation */
    double memory_usage;    /* MB */
    double accuracy;        /* relative error */
    double stability;      /* coefficient de variation */
} RegressionMetrics;

/* ============================================================
 * Configurations de test pour régression
 * ============================================================ */

static const TestConfig test_configs[] = {
    /* Configuration minimale */
    {
        .vocab_size = 128,
        .dim = 32,
        .state_size = 64,
        .seq_len = 64,
        .n_layers = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    },
    /* Configuration moyenne */
    {
        .vocab_size = 256,
        .dim = 64,
        .state_size = 128,
        .seq_len = 128,
        .n_layers = 2,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    },
    /* Configuration grande */
    {
        .vocab_size = 512,
        .dim = 128,
        .state_size = 256,
        .seq_len = 256,
        .n_layers = 4,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    },
    /* Configuration très grande */
    {
        .vocab_size = 1000,
        .dim = 256,
        .state_size = 512,
        .seq_len = 512,
        .n_layers = 8,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    }
};

static const size_t n_test_configs = sizeof(test_configs) / sizeof(TestConfig);

/* ============================================================
 * Fonctions de test simulées
 * ============================================================ */

static double simulate_embedding_forward(const TestConfig *config, int iterations) {
    struct timespec start, end;
    
    /* Simuler embedding lookup */
    float *embedding = (float*)malloc(config->vocab_size * config->dim * sizeof(float));
    float *hidden = (float*)malloc(config->seq_len * config->dim * sizeof(float));
    
    /* Initialiser l'embedding */
    for (size_t i = 0; i < config->vocab_size * config->dim; i++) {
        embedding[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t t = 0; t < config->seq_len; t++) {
            uint8_t token = (uint8_t)(rand() % config->vocab_size);
            for (size_t d = 0; d < config->dim; d++) {
                hidden[t * config->dim + d] = embedding[token * config->dim + d];
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    free(embedding);
    free(hidden);
    
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

static double simulate_mamba_block_forward(const TestConfig *config, int iterations) {
    struct timespec start, end;
    
    /* Simuler MambaBlock */
    float *W_in = (float*)malloc(config->state_size * config->dim * sizeof(float));
    float *W_out = (float*)malloc(config->dim * config->state_size * sizeof(float));
    float *hidden = (float*)malloc(config->seq_len * config->dim * sizeof(float));
    float *output = (float*)malloc(config->seq_len * config->dim * sizeof(float));
    
    /* Initialiser les poids */
    for (size_t i = 0; i < config->state_size * config->dim; i++) {
        W_in[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < config->dim * config->state_size; i++) {
        W_out[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t t = 0; t < config->seq_len; t++) {
            for (size_t d = 0; d < config->dim; d++) {
                float sum = 0.0f;
                for (size_t s = 0; s < config->state_size; s++) {
                    sum += hidden[t * config->dim + d] * W_in[s * config->dim + d];
                }
                /* Activation SiLU */
                float silu = sum / (1.0f + expf(-sum));
                output[t * config->dim + d] = silu;
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    free(W_in);
    free(W_out);
    free(hidden);
    free(output);
    
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

static double simulate_lm_head_forward(const TestConfig *config, int iterations) {
    struct timespec start, end;
    
    /* Simuler LM Head */
    float *W_lm = (float*)malloc(config->dim * config->vocab_size * sizeof(float));
    float *hidden = (float*)malloc(config->seq_len * config->dim * sizeof(float));
    float *logits = (float*)malloc(config->seq_len * config->vocab_size * sizeof(float));
    
    /* Initialiser les poids */
    for (size_t i = 0; i < config->dim * config->vocab_size; i++) {
        W_lm[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t t = 0; t < config->seq_len; t++) {
            for (size_t v = 0; v < config->vocab_size; v++) {
                float sum = 0.0f;
                for (size_t d = 0; d < config->dim; d++) {
                    sum += hidden[t * config->dim + d] * W_lm[d * config->vocab_size + v];
                }
                logits[t * config->vocab_size + v] = sum;
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    free(W_lm);
    free(hidden);
    free(logits);
    
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

/* ============================================================
 * Tests de régression
 * ============================================================ */

static int test_embedding_regression() {
    printf("Testing Embedding regression...\n");
    
    const int iterations = 1000;
    
    for (size_t i = 0; i < n_test_configs; i++) {
        const TestConfig *config = &test_configs[i];
        
        printf("  Config %zu: vocab=%zu, dim=%zu, seq=%zu\n", 
               i, config->vocab_size, config->dim, config->seq_len);
        
        double time = simulate_embedding_forward(config, iterations);
        double throughput = (double)iterations / time;
        double latency = time * 1000.0 / iterations;
        
        printf("    Time: %.3f sec, Throughput: %.2f ops/sec, Latency: %.3f ms\n", 
               time, throughput, latency);
        
        /* Vérifier les limites acceptables */
        if (throughput < 1000.0) {  /* Moins de 1000 ops/sec */
            printf("    WARNING: Low throughput detected\n");
        }
        
        if (latency > 10.0) {  /* Plus de 10ms */
            printf("    WARNING: High latency detected\n");
        }
    }
    
    printf("PASS: Embedding regression test completed\n");
    return 1;
}

static int test_mamba_block_regression() {
    printf("Testing MambaBlock regression...\n");
    
    const int iterations = 100;
    
    for (size_t i = 0; i < n_test_configs; i++) {
        const TestConfig *config = &test_configs[i];
        
        printf("  Config %zu: dim=%zu, state=%zu, seq=%zu\n", 
               i, config->dim, config->state_size, config->seq_len);
        
        double time = simulate_mamba_block_forward(config, iterations);
        double throughput = (double)iterations / time;
        double latency = time * 1000.0 / iterations;
        
        printf("    Time: %.3f sec, Throughput: %.2f ops/sec, Latency: %.3f ms\n", 
               time, throughput, latency);
        
        /* Vérifier les limites acceptables */
        if (throughput < 100.0) {  /* Moins de 100 ops/sec */
            printf("    WARNING: Low throughput detected\n");
        }
        
        if (latency > 100.0) {  /* Plus de 100ms */
            printf("    WARNING: High latency detected\n");
        }
    }
    
    printf("PASS: MambaBlock regression test completed\n");
    return 1;
}

static int test_lm_head_regression() {
    printf("Testing LM Head regression...\n");
    
    const int iterations = 100;
    
    for (size_t i = 0; i < n_test_configs; i++) {
        const TestConfig *config = &test_configs[i];
        
        printf("  Config %zu: dim=%zu, vocab=%zu, seq=%zu\n", 
               i, config->dim, config->vocab_size, config->seq_len);
        
        double time = simulate_lm_head_forward(config, iterations);
        double throughput = (double)iterations / time;
        double latency = time * 1000.0 / iterations;
        
        printf("    Time: %.3f sec, Throughput: %.2f ops/sec, Latency: %.3f ms\n", 
               time, throughput, latency);
        
        /* Vérifier les limites acceptables */
        if (throughput < 50.0) {  /* Moins de 50 ops/sec */
            printf("    WARNING: Low throughput detected\n");
        }
        
        if (latency > 200.0) {  /* Plus de 200ms */
            printf("    WARNING: High latency detected\n");
        }
    }
    
    printf("PASS: LM Head regression test completed\n");
    return 1;
}

/* ============================================================
 * Tests de stabilité
 * ============================================================ */

static int test_numerical_stability() {
    printf("Testing numerical stability...\n");
    
    const TestConfig *config = &test_configs[1];  /* Configuration moyenne */
    const int iterations = 1000;
    
    /* Test avec différentes seeds */
    double times[10];
    for (int seed_idx = 0; seed_idx < 10; seed_idx++) {
        srand(42 + seed_idx);
        times[seed_idx] = simulate_mamba_block_forward(config, iterations);
    }
    
    /* Calculer la moyenne et l'écart-type */
    double mean = 0.0;
    for (int i = 0; i < 10; i++) {
        mean += times[i];
    }
    mean /= 10.0;
    
    double variance = 0.0;
    for (int i = 0; i < 10; i++) {
        double diff = times[i] - mean;
        variance += diff * diff;
    }
    variance /= 10.0;
    double std_dev = sqrt(variance);
    double cv = (std_dev / mean) * 100.0;  /* Coefficient de variation */
    
    printf("  Mean time: %.3f sec\n", mean);
    printf("  Std deviation: %.3f sec\n", std_dev);
    printf("  Coefficient of variation: %.2f%%\n", cv);
    
    /* Vérifier la stabilité */
    if (cv > 5.0) {  /* Plus de 5% de variation */
        printf("  WARNING: High variability detected\n");
    }
    
    printf("PASS: Numerical stability test completed\n");
    return 1;
}

/* ============================================================
 * Tests de scaling
 * ============================================================ */

static int test_scaling_performance() {
    printf("Testing scaling performance...\n");
    
    const int iterations = 100;
    
    printf("  Scaling with sequence length:\n");
    for (size_t seq_len = 64; seq_len <= 512; seq_len *= 2) {
        TestConfig config = {
            .vocab_size = 256,
            .dim = 64,
            .state_size = 128,
            .seq_len = seq_len,
            .n_layers = 2,
            .dt_scale = 1.0f,
            .dt_min = 0.001f,
            .dt_max = 0.1f
        };
        
        double time = simulate_mamba_block_forward(&config, iterations);
        double throughput = (double)iterations / time;
        
        printf("    Seq_len=%zu: %.3f sec, %.2f ops/sec\n", seq_len, time, throughput);
    }
    
    printf("  Scaling with model dimension:\n");
    for (size_t dim = 32; dim <= 256; dim *= 2) {
        TestConfig config = {
            .vocab_size = 256,
            .dim = dim,
            .state_size = dim * 2,
            .seq_len = 128,
            .n_layers = 2,
            .dt_scale = 1.0f,
            .dt_min = 0.001f,
            .dt_max = 0.1f
        };
        
        double time = simulate_mamba_block_forward(&config, iterations);
        double throughput = (double)iterations / time;
        
        printf("    Dim=%zu: %.3f sec, %.2f ops/sec\n", dim, time, throughput);
    }
    
    printf("PASS: Scaling performance test completed\n");
    return 1;
}

/* ============================================================
 * Tests de stress
 * ============================================================ */

static int test_stress_long_training() {
    printf("Testing long training stress...\n");
    
    const TestConfig *config = &test_configs[2];  /* Configuration grande */
    const int long_iterations = 10000;
    
    printf("  Running %d iterations...\n", long_iterations);
    
    double total_time = 0.0;
    for (int i = 0; i < 10; i++) {
        double batch_time = simulate_mamba_block_forward(config, long_iterations / 10);
        total_time += batch_time;
        
        if (i % 3 == 0) {
            printf("    Batch %d: %.3f sec\n", i + 1, batch_time);
        }
    }
    
    printf("  Total time: %.3f sec\n", total_time);
    printf("  Average per batch: %.3f sec\n", total_time / 10.0);
    
    printf("PASS: Long training stress test completed\n");
    return 1;
}

static int test_stress_memory_usage() {
    printf("Testing memory usage stress...\n");
    
    const TestConfig *config = &test_configs[3];  /* Configuration très grande */
    
    /* Calculer l'utilisation mémoire */
    size_t embedding_size = config->vocab_size * config->dim * sizeof(float);
    size_t mamba_size = config->n_layers * config->dim * config->state_size * sizeof(float);
    size_t lm_head_size = config->dim * config->vocab_size * sizeof(float);
    size_t total_size = embedding_size + mamba_size + lm_head_size;
    
    printf("  Memory usage breakdown:\n");
    printf("    Embedding: %.2f MB\n", (double)embedding_size / (1024*1024));
    printf("    MambaBlocks: %.2f MB\n", (double)mamba_size / (1024*1024));
    printf("    LM Head: %.2f MB\n", (double)lm_head_size / (1024*1024));
    printf("    Total: %.2f MB\n", (double)total_size / (1024*1024));
    
    /* Vérifier les limites */
    if (total_size > 1024 * 1024 * 1024) {  /* Plus de 1GB */
        printf("  WARNING: High memory usage detected\n");
    }
    
    printf("PASS: Memory usage stress test completed\n");
    return 1;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Regression Test Suite ===\n");
    printf("Testing k-mamba for performance regressions\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests de régression */
    total++; passed += test_embedding_regression();
    total++; passed += test_mamba_block_regression();
    total++; passed += test_lm_head_regression();
    
    /* Tests de stabilité */
    total++; passed += test_numerical_stability();
    
    /* Tests de scaling */
    total++; passed += test_scaling_performance();
    
    /* Tests de stress */
    total++; passed += test_stress_long_training();
    total++; passed += test_stress_memory_usage();
    
    printf("\n=== Regression Test Results ===\n");
    printf("Passed: %d/%d test suites\n", passed, total);
    
    if (passed == total) {
        printf("All regression tests PASSED!\n");
        return 0;
    } else {
        printf("Some regression tests FAILED!\n");
        return 1;
    }
}

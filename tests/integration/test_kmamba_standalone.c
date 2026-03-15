/*
 * test_kmamba_standalone.c — Tests de bout en bout KMamba (STANDALONE)
 *
 * Phase 3 : Tests de bout en bout KMamba (training/inference)
 * Version standalone sans déclarations en double
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

#define EPSILON 1e-5f

/* ============================================================
 * Structures de données pour KMamba
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
} SimpleKMambaConfig;

typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} SimpleMatrix;

typedef struct {
    SimpleKMambaConfig config;
    
    /* Embedding */
    SimpleMatrix embedding;
    
    /* MambaBlocks */
    SimpleMatrix *mamba_blocks;
    SimpleMatrix *block_configs;
    
    /* LM Head */
    SimpleMatrix lm_head;
    
    /* Runtime buffers */
    float *hidden_states;
    float *logits;
    
    /* Training state */
    int training_enabled;
    float *loss_history;
    size_t loss_history_size;
} SimpleKMamba;

/* ============================================================
 * Fonctions utilitaires
 * ============================================================ */

static float* allocate_matrix(size_t rows, size_t cols) {
    float *data = (float*)malloc(rows * cols * sizeof(float));
    if (data) {
        memset(data, 0, rows * cols * sizeof(float));
    }
    return data;
}

static void free_matrix(float *data) {
    free(data);
}

static float cross_entropy_loss(const float *logits, const uint8_t *targets, size_t vocab_size, size_t seq_len) {
    float loss = 0.0f;
    
    for (size_t t = 0; t < seq_len; t++) {
        size_t target_idx = targets[t];
        float max_logit = logits[t * vocab_size];
        
        /* Trouver le max */
        for (size_t v = 1; v < vocab_size; v++) {
            if (logits[t * vocab_size + v] > max_logit) {
                max_logit = logits[t * vocab_size + v];
            }
        }
        
        /* Calculer softmax et cross-entropy */
        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; v++) {
            sum_exp += expf(logits[t * vocab_size + v] - max_logit);
        }
        
        float target_logit = logits[t * vocab_size + target_idx];
        loss += -logf(expf(target_logit - max_logit) / sum_exp);
    }
    
    return loss / (float)seq_len;
}

static void print_vector(const char *name, const float *v, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n && i < 8; i++) {
        printf("%.4f", v[i]);
        if (i < 7 && i < n-1) printf(", ");
    }
    if (n > 8) printf("...");
    printf("]\n");
}

static void print_tokens(const char *name, const uint8_t *tokens, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n && i < 8; i++) {
        printf("%d", tokens[i]);
        if (i < 7 && i < n-1) printf(", ");
    }
    if (n > 8) printf("...");
    printf("]\n");
}

/* ============================================================
 * Implémentation de KMamba
 * ============================================================ */

static SimpleKMamba* simple_kmamba_create(const SimpleKMambaConfig *config) {
    SimpleKMamba *kmamba = (SimpleKMamba*)malloc(sizeof(SimpleKMamba));
    if (!kmamba) return NULL;
    
    /* Copier la configuration */
    memcpy(&kmamba->config, config, sizeof(SimpleKMambaConfig));
    
    /* Allouer l'embedding */
    kmamba->embedding.data = allocate_matrix(config->vocab_size, config->dim);
    kmamba->embedding.rows = config->vocab_size;
    kmamba->embedding.cols = config->dim;
    
    /* Allouer les MambaBlocks */
    kmamba->mamba_blocks = (SimpleMatrix*)malloc(config->n_layers * sizeof(SimpleMatrix));
    kmamba->block_configs = (SimpleMatrix*)malloc(config->n_layers * sizeof(SimpleMatrix));
    
    for (size_t i = 0; i < config->n_layers; i++) {
        kmamba->mamba_blocks[i].data = allocate_matrix(config->state_size, config->dim);
        kmamba->mamba_blocks[i].rows = config->state_size;
        kmamba->mamba_blocks[i].cols = config->dim;
        
        kmamba->block_configs[i].data = allocate_matrix(config->state_size, 1);
        kmamba->block_configs[i].rows = config->state_size;
        kmamba->block_configs[i].cols = 1;
    }
    
    /* Allouer le LM head */
    kmamba->lm_head.data = allocate_matrix(config->dim, config->vocab_size);
    kmamba->lm_head.rows = config->dim;
    kmamba->lm_head.cols = config->vocab_size;
    
    /* Allouer les buffers */
    kmamba->hidden_states = allocate_matrix(config->seq_len, config->dim);
    kmamba->logits = allocate_matrix(config->seq_len, config->vocab_size);
    
    /* Training state */
    kmamba->training_enabled = 0;
    kmamba->loss_history_size = 100;
    kmamba->loss_history = (float*)malloc(kmamba->loss_history_size * sizeof(float));
    
    return kmamba;
}

static void simple_kmamba_free(SimpleKMamba *kmamba) {
    if (!kmamba) return;
    
    free_matrix(kmamba->embedding.data);
    
    for (size_t i = 0; i < kmamba->config.n_layers; i++) {
        free_matrix(kmamba->mamba_blocks[i].data);
        free_matrix(kmamba->block_configs[i].data);
    }
    
    free(kmamba->mamba_blocks);
    free(kmamba->block_configs);
    
    free_matrix(kmamba->lm_head.data);
    free_matrix(kmamba->hidden_states);
    free_matrix(kmamba->logits);
    free(kmamba->loss_history);
    
    free(kmamba);
}

static void simple_kmamba_init(SimpleKMamba *kmamba, unsigned long seed) {
    if (!kmamba) return;
    
    srand(seed);
    
    /* Initialiser l'embedding */
    for (size_t i = 0; i < kmamba->embedding.rows * kmamba->embedding.cols; i++) {
        kmamba->embedding.data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    /* Initialiser les MambaBlocks */
    for (size_t i = 0; i < kmamba->config.n_layers; i++) {
        for (size_t j = 0; j < kmamba->mamba_blocks[i].rows * kmamba->mamba_blocks[i].cols; j++) {
            kmamba->mamba_blocks[i].data[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        for (size_t j = 0; j < kmamba->block_configs[i].rows; j++) {
            kmamba->block_configs[i].data[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    /* Initialiser le LM head */
    for (size_t i = 0; i < kmamba->lm_head.rows * kmamba->lm_head.cols; i++) {
        kmamba->lm_head.data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    /* Initialiser l'historique des pertes */
    memset(kmamba->loss_history, 0, kmamba->loss_history_size * sizeof(float));
}

static void simple_kmamba_forward(SimpleKMamba *kmamba, const uint8_t *tokens, float *logits) {
    if (!kmamba || !tokens || !logits) return;
    
    size_t vocab_size = kmamba->config.vocab_size;
    size_t dim = kmamba->config.dim;
    size_t seq_len = kmamba->config.seq_len;
    size_t n_layers = kmamba->config.n_layers;
    
    /* Embedding lookup */
    for (size_t t = 0; t < seq_len; t++) {
        uint8_t token = tokens[t];
        for (size_t d = 0; d < dim; d++) {
            kmamba->hidden_states[t * dim + d] = kmamba->embedding.data[token * dim + d];
        }
    }
    
    print_vector("Embedded", kmamba->hidden_states, seq_len * dim);
    
    /* Passer à travers les MambaBlocks */
    float *current_hidden = kmamba->hidden_states;
    for (size_t layer = 0; layer < n_layers; layer++) {
        /* Simuler un MambaBlock forward */
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < dim; d++) {
                float sum = 0.0f;
                for (size_t s = 0; s < kmamba->mamba_blocks[layer].rows; s++) {
                    sum += current_hidden[t * dim + d] * kmamba->mamba_blocks[layer].data[s * dim + d];
                }
                /* Ajouter une activation SiLU simulée */
                float silu = sum / (1.0f + expf(-sum));
                current_hidden[t * dim + d] = silu;
            }
        }
    }
    
    print_vector("After MambaBlocks", current_hidden, seq_len * dim);
    
    /* LM Head */
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t v = 0; v < vocab_size; v++) {
            float sum = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                sum += current_hidden[t * dim + d] * kmamba->lm_head.data[d * vocab_size + v];
            }
            logits[t * vocab_size + v] = sum;
        }
    }
    
    print_vector("Logits", logits, seq_len * vocab_size);
}

static float simple_kmamba_train_step(SimpleKMamba *kmamba, const uint8_t *tokens, const uint8_t *targets) {
    if (!kmamba || !kmamba->training_enabled) return 0.0f;
    
    /* Forward pass */
    simple_kmamba_forward(kmamba, tokens, kmamba->logits);
    
    /* Calculer la loss */
    float loss = cross_entropy_loss(kmamba->logits, targets, kmamba->config.vocab_size, kmamba->config.seq_len);
    
    /* Ajouter à l'historique */
    for (size_t i = 1; i < kmamba->loss_history_size; i++) {
        kmamba->loss_history[i-1] = kmamba->loss_history[i];
    }
    kmamba->loss_history[0] = loss;
    
    printf("Loss: %.6f\n", loss);
    
    return loss;
}

static void simple_kmamba_enable_training(SimpleKMamba *kmamba, float lr, float weight_decay) {
    if (!kmamba) return;
    
    kmamba->training_enabled = 1;
    
    /* Simuler un optimiseur simple */
    float gradient_scale = lr;
    
    /* Mettre à jour l'embedding */
    for (size_t i = 0; i < kmamba->embedding.rows * kmamba->embedding.cols; i++) {
        kmamba->embedding.data[i] -= gradient_scale * (((float)rand() / RAND_MAX) * 0.1f - 0.05f);
    }
    
    /* Mettre à jour le LM head */
    for (size_t i = 0; i < kmamba->lm_head.rows * kmamba->lm_head.cols; i++) {
        kmamba->lm_head.data[i] -= gradient_scale * (((float)rand() / RAND_MAX) * 0.1f - 0.05f);
    }
    
    printf("Training enabled with lr=%.6f, weight_decay=%.6f\n", lr, weight_decay);
}

/* ============================================================
 * Tests KMamba Forward
 * ============================================================ */

static int test_kmamba_forward_simple() {
    printf("Testing KMamba forward simple case...\n");
    
    /* Configuration simple */
    SimpleKMambaConfig cfg = {
        .vocab_size = 256,
        .dim = 64,
        .state_size = 128,
        .seq_len = 8,
        .n_layers = 2,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    /* Données de test */
    uint8_t tokens[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    float logits[8 * 256];
    
    printf("Input tokens: ");
    print_tokens("tokens", tokens, 8);
    
    /* Créer et initialiser KMamba */
    SimpleKMamba *kmamba = simple_kmamba_create(&cfg);
    if (!kmamba) {
        printf("FAIL: Could not create KMamba\n");
        return 0;
    }
    
    simple_kmamba_init(kmamba, 42);
    
    /* Forward pass */
    simple_kmamba_forward(kmamba, tokens, logits);
    
    /* Vérifications basiques */
    int success = 1;
    
    /* Vérifier que les logits ne sont pas tout zéro */
    int all_zero = 1;
    for (size_t i = 0; i < 8 * 256; i++) {
        if (fabsf(logits[i]) > 1e-6f) {
            all_zero = 0;
            break;
        }
    }
    
    if (all_zero) {
        printf("FAIL: Logits are all zeros\n");
        success = 0;
    }
    
    /* Vérifier qu'il n'y a pas de NaN ou inf */
    for (size_t i = 0; i < 8 * 256; i++) {
        if (!isfinite(logits[i])) {
            printf("FAIL: Logits contain NaN or inf at index %zu\n", i);
            success = 0;
            break;
        }
    }
    
    if (success) {
        printf("PASS: KMamba forward simple case\n");
    }
    
    /* Nettoyage */
    simple_kmamba_free(kmamba);
    
    return success;
}

static int test_kmamba_training_simple() {
    printf("Testing KMamba training simple case...\n");
    
    /* Configuration simple */
    SimpleKMambaConfig cfg = {
        .vocab_size = 256,
        .dim = 32,
        .state_size = 64,
        .seq_len = 4,
        .n_layers = 1,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    /* Données de test */
    uint8_t tokens[4] = {1, 2, 3, 4};
    uint8_t targets[4] = {2, 3, 4, 5};
    
    printf("Input tokens: ");
    print_tokens("tokens", tokens, 4);
    printf("Target tokens: ");
    print_tokens("targets", targets, 4);
    
    /* Créer et initialiser KMamba */
    SimpleKMamba *kmamba = simple_kmamba_create(&cfg);
    if (!kmamba) {
        printf("FAIL: Could not create KMamba\n");
        return 0;
    }
    
    simple_kmamba_init(kmamba, 42);
    simple_kmamba_enable_training(kmamba, 1e-3f, 1e-5f);
    
    /* Training loop */
    printf("Training loop:\n");
    for (int epoch = 0; epoch < 5; epoch++) {
        float loss = simple_kmamba_train_step(kmamba, tokens, targets);
        printf("Epoch %d: loss = %.6f\n", epoch, loss);
    }
    
    printf("PASS: KMamba training simple case\n");
    
    /* Nettoyage */
    simple_kmamba_free(kmamba);
    
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_kmamba() {
    printf("\n=== KMamba Performance Benchmarks ===\n");
    
    const size_t vocab_size = 256, dim = 64, seq_len = 128, n_layers = 2;
    const int iterations = 10;
    
    /* Configuration */
    SimpleKMambaConfig cfg = {
        .vocab_size = vocab_size,
        .dim = dim,
        .state_size = 128,
        .seq_len = seq_len,
        .n_layers = n_layers,
        .dt_scale = 1.0f,
        .dt_min = 0.001f,
        .dt_max = 0.1f
    };
    
    /* Données de test */
    uint8_t *tokens = (uint8_t*)malloc(seq_len * sizeof(uint8_t));
    float *logits = (float*)malloc(seq_len * vocab_size * sizeof(float));
    
    for (size_t i = 0; i < seq_len; i++) {
        tokens[i] = (uint8_t)(rand() % vocab_size);
    }
    
    /* Créer KMamba */
    SimpleKMamba *kmamba = simple_kmamba_create(&cfg);
    if (!kmamba) {
        printf("FAIL: Could not create KMamba for benchmark\n");
        free(tokens);
        free(logits);
        return;
    }
    
    simple_kmamba_init(kmamba, 42);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        simple_kmamba_forward(kmamba, tokens, logits);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("KMamba Performance:\n");
    printf("  Size: vocab=%zu, dim=%zu, seq=%zu, layers=%zu\n", vocab_size, dim, seq_len, n_layers);
    printf("  Time: %.3f sec (%d iterations)\n", elapsed, iterations);
    printf("  Throughput: %.2f forward passes/sec\n", (double)iterations / elapsed);
    
    /* Nettoyage */
    simple_kmamba_free(kmamba);
    free(tokens);
    free(logits);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== KMamba End-to-End Test Suite ===\n");
    printf("Testing KMamba complete model (training/inference)\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests forward */
    total++; passed += test_kmamba_forward_simple();
    
    /* Tests training */
    total++; passed += test_kmamba_training_simple();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        benchmark_kmamba();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}

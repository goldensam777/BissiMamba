/*
 * test_loss_functions.c — Tests approfondis des fonctions de loss
 *
 * Tests complets des différentes fonctions de loss utilisées dans k-mamba
 * Cross-entropy, MSE, MAE, Huber, etc.
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
 * Fonctions de loss à tester
 * ============================================================ */

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

static float mse_loss(const float *predictions, const float *targets, size_t n) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    
    return loss / (float)n;
}

static float mae_loss(const float *predictions, const float *targets, size_t n) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(predictions[i] - targets[i]);
        loss += diff;
    }
    
    return loss / (float)n;
}

static float huber_loss(const float *predictions, const float *targets, size_t n, float delta) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        
        if (abs_diff <= delta) {
            loss += 0.5f * diff * diff;
        } else {
            loss += delta * (abs_diff - 0.5f * delta);
        }
    }
    
    return loss / (float)n;
}

static float binary_cross_entropy_loss(const float *predictions, const uint8_t *targets, size_t n) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float pred = fmaxf(EPSILON, fminf(1.0f - EPSILON, predictions[i]));
        float target = (float)targets[i];
        
        loss += -target * logf(pred) - (1.0f - target) * logf(1.0f - pred);
    }
    
    return loss / (float)n;
}

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

static void fill_test_data(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * (max - min) + min;
    }
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

static void print_tokens(const char *name, const uint8_t *v, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n && i < 8; i++) {
        printf("%d", v[i]);
        if (i < 7 && i < n-1) printf(", ");
    }
    if (n > 8) printf("...");
    printf("]\n");
}

/* ============================================================
 * Tests Cross-Entropy Loss
 * ============================================================ */

static int test_cross_entropy_perfect_prediction() {
    printf("Testing Cross-Entropy Loss - Perfect Prediction...\n");
    
    const size_t vocab_size = 4, seq_len = 3;
    
    /* Logits avec forte confiance sur les targets */
    float logits[12] = {
        10.0f, -10.0f, -10.0f, -10.0f,  /* target = 0 */
        -10.0f, 10.0f, -10.0f, -10.0f,  /* target = 1 */
        -10.0f, -10.0f, 10.0f, -10.0f   /* target = 2 */
    };
    
    uint8_t targets[3] = {0, 1, 2};
    
    printf("Logits:\n");
    for (size_t t = 0; t < seq_len; t++) {
        printf("  t=%zu: [", t);
        for (size_t v = 0; v < vocab_size; v++) {
            printf("%.1f", logits[t * vocab_size + v]);
            if (v < vocab_size - 1) printf(", ");
        }
        printf("] target=%d\n", targets[t]);
    }
    
    float loss = cross_entropy_loss(logits, targets, vocab_size, seq_len);
    printf("Cross-Entropy Loss: %.6f\n", loss);
    
    /* Pour prédiction parfaite, loss devrait être proche de 0 */
    if (loss < 0.1f) {
        printf("PASS: Perfect prediction gives low loss\n");
        return 1;
    } else {
        printf("FAIL: Perfect prediction should give low loss\n");
        return 0;
    }
}

static int test_cross_entropy_random_prediction() {
    printf("Testing Cross-Entropy Loss - Random Prediction...\n");
    
    const size_t vocab_size = 10, seq_len = 5;
    
    /* Logits aléatoires */
    float logits[50];
    fill_test_data(logits, vocab_size * seq_len, -2.0f, 2.0f);
    
    /* Targets aléatoires */
    uint8_t targets[5];
    for (size_t i = 0; i < seq_len; i++) {
        targets[i] = (uint8_t)(rand() % vocab_size);
    }
    
    printf("Targets: ");
    print_tokens("targets", targets, seq_len);
    
    float loss = cross_entropy_loss(logits, targets, vocab_size, seq_len);
    printf("Cross-Entropy Loss: %.6f\n", loss);
    
    /* Pour prédiction aléatoire, loss devrait être ~log(vocab_size) */
    float expected_loss = logf((float)vocab_size);
    printf("Expected loss (random): %.6f\n", expected_loss);
    
    if (fabsf(loss - expected_loss) < 1.0f) {
        printf("PASS: Random prediction gives expected loss\n");
        return 1;
    } else {
        printf("FAIL: Random prediction loss not in expected range\n");
        return 0;
    }
}

/* ============================================================
 * Tests MSE Loss
 * ============================================================ */

static int test_mse_perfect_prediction() {
    printf("Testing MSE Loss - Perfect Prediction...\n");
    
    const size_t n = 5;
    
    float predictions[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float targets[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    printf("Predictions: ");
    print_vector("pred", predictions, n);
    printf("Targets: ");
    print_vector("target", targets, n);
    
    float loss = mse_loss(predictions, targets, n);
    printf("MSE Loss: %.6f\n", loss);
    
    /* Pour prédiction parfaite, loss devrait être 0 */
    if (loss < EPSILON) {
        printf("PASS: Perfect prediction gives zero loss\n");
        return 1;
    } else {
        printf("FAIL: Perfect prediction should give zero loss\n");
        return 0;
    }
}

static int test_mse_known_error() {
    printf("Testing MSE Loss - Known Error...\n");
    
    const size_t n = 3;
    
    float predictions[3] = {2.0f, 3.0f, 4.0f};
    float targets[3] = {1.0f, 5.0f, 6.0f};
    
    printf("Predictions: ");
    print_vector("pred", predictions, n);
    printf("Targets: ");
    print_vector("target", targets, n);
    
    float loss = mse_loss(predictions, targets, n);
    printf("MSE Loss: %.6f\n", loss);
    
    /* Calcul manuel : ((2-1)² + (3-5)² + (4-6)²) / 3 = (1 + 4 + 4) / 3 = 3.0 */
    float expected_loss = 3.0f;
    
    if (fabsf(loss - expected_loss) < EPSILON) {
        printf("PASS: MSE calculation correct\n");
        return 1;
    } else {
        printf("FAIL: MSE calculation incorrect, expected %.6f\n", expected_loss);
        return 0;
    }
}

/* ============================================================
 * Tests MAE Loss
 * ============================================================ */

static int test_mae_perfect_prediction() {
    printf("Testing MAE Loss - Perfect Prediction...\n");
    
    const size_t n = 4;
    
    float predictions[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    float targets[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    
    printf("Predictions: ");
    print_vector("pred", predictions, n);
    printf("Targets: ");
    print_vector("target", targets, n);
    
    float loss = mae_loss(predictions, targets, n);
    printf("MAE Loss: %.6f\n", loss);
    
    /* Pour prédiction parfaite, loss devrait être 0 */
    if (loss < EPSILON) {
        printf("PASS: Perfect prediction gives zero loss\n");
        return 1;
    } else {
        printf("FAIL: Perfect prediction should give zero loss\n");
        return 0;
    }
}

/* ============================================================
 * Tests Huber Loss
 * ============================================================ */

static int test_huber_loss_small_errors() {
    printf("Testing Huber Loss - Small Errors...\n");
    
    const size_t n = 3;
    const float delta = 1.0f;
    
    float predictions[3] = {1.1f, 2.2f, 3.3f};
    float targets[3] = {1.0f, 2.0f, 3.0f};
    
    printf("Predictions: ");
    print_vector("pred", predictions, n);
    printf("Targets: ");
    print_vector("target", targets, n);
    printf("Delta: %.1f\n", delta);
    
    float loss = huber_loss(predictions, targets, n, delta);
    printf("Huber Loss: %.6f\n", loss);
    
    /* Pour petites erreurs, Huber ≈ MSE */
    float mse = mse_loss(predictions, targets, n);
    printf("MSE Loss: %.6f\n", mse);
    
    if (fabsf(loss - mse) < 0.1f) {
        printf("PASS: Huber ≈ MSE for small errors\n");
        return 1;
    } else {
        printf("FAIL: Huber should approximate MSE for small errors\n");
        return 0;
    }
}

/* ============================================================
 * Tests Binary Cross-Entropy Loss
 * ============================================================ */

static int test_binary_cross_entropy_perfect() {
    printf("Testing Binary Cross-Entropy Loss - Perfect Prediction...\n");
    
    const size_t n = 4;
    
    float predictions[4] = {0.9f, 0.1f, 0.8f, 0.2f};
    uint8_t targets[4] = {1, 0, 1, 0};
    
    printf("Predictions: ");
    print_vector("pred", predictions, n);
    printf("Targets: ");
    print_tokens("target", targets, n);
    
    float loss = binary_cross_entropy_loss(predictions, targets, n);
    printf("Binary Cross-Entropy Loss: %.6f\n", loss);
    
    /* Pour bonnes prédictions, loss devrait être faible */
    if (loss < 0.5f) {
        printf("PASS: Good predictions give low loss\n");
        return 1;
    } else {
        printf("FAIL: Good predictions should give low loss\n");
        return 0;
    }
}

/* ============================================================
 * Tests de stabilité numérique
 * ============================================================ */

static int test_loss_numerical_stability() {
    printf("Testing Loss Functions - Numerical Stability...\n");
    
    const size_t vocab_size = 1000, seq_len = 1;
    
    /* Test avec des logits très grands */
    float large_logits[1000];
    for (size_t i = 0; i < vocab_size; i++) {
        large_logits[i] = (i == 244) ? 50.0f : -50.0f;  // Valeur plus raisonnable
    }
    
    uint8_t target = 244;  // Valeur valide pour uint8_t
    
    printf("Testing with large logits (±50)...\n");
    float loss_large = cross_entropy_loss(large_logits, &target, vocab_size, seq_len);
    printf("Cross-Entropy Loss (large logits): %.6f\n", loss_large);
    
    /* Test avec des logits très petits */
    float small_logits[1000];
    for (size_t i = 0; i < vocab_size; i++) {
        small_logits[i] = (i == 244) ? 0.001f : -0.001f;
    }
    
    printf("Testing with very small logits (±0.001)...\n");
    float loss_small = cross_entropy_loss(small_logits, &target, vocab_size, seq_len);
    printf("Cross-Entropy Loss (small logits): %.6f\n", loss_small);
    
    /* Vérifier qu'il n'y a pas de NaN ou inf */
    if (isfinite(loss_large) && isfinite(loss_small)) {
        printf("PASS: Loss functions numerically stable\n");
        return 1;
    } else {
        printf("FAIL: Loss functions produce NaN or inf\n");
        return 0;
    }
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_loss_functions() {
    printf("\n=== Loss Functions Performance Benchmarks ===\n");
    
    const size_t vocab_size = 1000, seq_len = 128;
    const int iterations = 1000;
    
    /* Données de test */
    float *logits = (float*)malloc(vocab_size * seq_len * sizeof(float));
    uint8_t *targets = (uint8_t*)malloc(seq_len * sizeof(uint8_t));
    
    fill_test_data(logits, vocab_size * seq_len, -5.0f, 5.0f);
    for (size_t i = 0; i < seq_len; i++) {
        targets[i] = (uint8_t)(rand() % vocab_size);
    }
    
    struct timespec start, end;
    
    /* Benchmark Cross-Entropy */
    clock_gettime(CLOCK_MONOTONIC, &start);
    float ce_loss = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        ce_loss += cross_entropy_loss(logits, targets, vocab_size, seq_len);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double ce_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    /* Benchmark MSE */
    float *predictions = (float*)malloc(seq_len * sizeof(float));
    float *mse_targets = (float*)malloc(seq_len * sizeof(float));
    
    fill_test_data(predictions, seq_len, -1.0f, 1.0f);
    fill_test_data(mse_targets, seq_len, -1.0f, 1.0f);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    float mse_loss_val = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        mse_loss_val += mse_loss(predictions, mse_targets, seq_len);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double mse_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Loss Functions Performance:\n");
    printf("  Size: vocab=%zu, seq=%zu\n", vocab_size, seq_len);
    printf("  Cross-Entropy: %.6f loss, %.3f ms (%.2f ops/sec)\n", 
           ce_loss / iterations, ce_time * 1000.0, (double)iterations / ce_time);
    printf("  MSE: %.6f loss, %.3f ms (%.2f ops/sec)\n", 
           mse_loss_val / iterations, mse_time * 1000.0, (double)iterations / mse_time);
    
    free(logits);
    free(targets);
    free(predictions);
    free(mse_targets);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Loss Functions Test Suite ===\n");
    printf("Testing various loss functions used in k-mamba\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests Cross-Entropy */
    total++; passed += test_cross_entropy_perfect_prediction();
    total++; passed += test_cross_entropy_random_prediction();
    
    /* Tests MSE */
    total++; passed += test_mse_perfect_prediction();
    total++; passed += test_mse_known_error();
    
    /* Tests MAE */
    total++; passed += test_mae_perfect_prediction();
    
    /* Tests Huber */
    total++; passed += test_huber_loss_small_errors();
    
    /* Tests Binary Cross-Entropy */
    total++; passed += test_binary_cross_entropy_perfect();
    
    /* Tests de stabilité numérique */
    total++; passed += test_loss_numerical_stability();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        benchmark_loss_functions();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}

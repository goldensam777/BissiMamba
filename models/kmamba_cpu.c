/* ============================================================================
 * kmamba_cpu.c - High-Performance Training Instance (CPU)
 * 
 * Theory Validation & Benchmarking for PC Local (32K Vocab)
 * Matches PLAN.md: CPU Small (120M params)
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "kmamba.h"
#include "kmamba_ser.h"

/* ═══════════════════════════════════════════════════════════════
 * Configuration (PLAN.md: CPU Small)
 * ═══════════════════════════════════════════════════════════════ */
#define VOCAB_SIZE      32768
#define MODEL_DIM       384
#define STATE_SIZE      512
#define LAYERS          2
#define SEQ_LEN         256
#define D_CONV          4
#define EXPAND          2.0f

#define DEFAULT_DATASET "data/kmamba_tiny_1M.txt"
#define SAVE_PATH       "models/kmamba_cpu.ser"

/* ═══════════════════════════════════════════════════════════════
 * Data Loader (Expertise: Memory Efficient)
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t *tokens;
    size_t    n_tokens;
} DataLoader;

DataLoader dataloader_create(const char *path) {
    DataLoader dl = {NULL, 0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return dl;

    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buf = malloc(fsize + 1);
    if (!buf) { fclose(fp); return dl; }
    
    size_t read_bytes = fread(buf, 1, fsize, fp);
    buf[read_bytes] = '\0';
    fclose(fp);

    /* Use our hybrid tokenizer in 32K mode */
    kmamba_tokenizer_init("bytes");
    dl.tokens = kmamba_encode(buf, &dl.n_tokens);
    free(buf);
    
    return dl;
}

/* ═══════════════════════════════════════════════════════════════
 * Performance Metrics
 * ═══════════════════════════════════════════════════════════════ */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

/* Estimate FLOPs per training step (Forward + Backward) 
 * Mamba complexity ≈ 10 * dim * state * seq_len per layer
 * GEMM complexity ≈ 2 * M * N * K
 */
double estimate_step_flops(const KMambaConfig *c) {
    double f = 0;
    double L = (double)c->seq_len;
    double D = (double)c->dim;
    double N = (double)c->state_size;
    double V = (double)c->vocab_size;

    /* Per layer: SSM + Projections + MLP (if any) */
    double layer_flops = L * (
        12 * D * N +  /* SSM scan & data-dep (Approx) */
        4 * D * D     /* In/Out projections */
    );
    
    f += c->n_layers * layer_flops;
    f += 2 * L * D * V; /* Embedding & Head */
    
    return f * 3; /* x3 for backward pass approx */
}

/* ═══════════════════════════════════════════════════════════════
 * Main Training Instance
 * ═══════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    printf("🧬 K-Mamba CPU Theory Validation — %d Threads\n", omp_get_max_threads());
    printf("══════════════════════════════════════════════════\n\n");

    /* 1. Init Tokenizer */
    kmamba_tokenizer_init("bytes");
    uint32_t vocab_size = (uint32_t)kmamba_vocab_size();

    KMambaConfig cfg = {0};
    cfg.vocab_size = vocab_size;
    cfg.dim = MODEL_DIM;
    cfg.state_size = STATE_SIZE;
    cfg.n_layers = LAYERS;
    cfg.seq_len = SEQ_LEN;
    cfg.d_conv = D_CONV;
    cfg.expand_factor = EXPAND;
    cfg.weight_tying = 1;
    strncpy(cfg.model_name, "kmamba-cpu-120m", 63);

    KMamba *m = kmamba_create(&cfg);
    kmamba_init(m, 1337);

    MBOptimConfig opt = {
        .lr = 6e-4f,
        .mu = 0.9f,      /* Matches 'mu' in kernels/optimizer_f32.c */
        .beta2 = 0.98f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 0.1f
    };
    kmamba_enable_training(m, &opt, 6e-4f, 0.1f);

    /* 2. Load Dataset */
    const char *data_path = (argc > 1) ? argv[1] : DEFAULT_DATASET;
    DataLoader dl = dataloader_create(data_path);
    if (!dl.tokens) {
        printf("⚠ Error: Could not load data from %s\n", data_path);
        return 1;
    }
    printf("📊 Data: %zu tokens loaded (Vocab: %zu)\n", dl.n_tokens, kmamba_vocab_size());

    /* 3. Training Loop */
    int steps = 2000;
    double flops_per_step = estimate_step_flops(&cfg);
    
    printf("⚡ Estimated %.2f GFLOPs per step\n\n", flops_per_step / 1e9);
    printf("%-6s | %-8s | %-10s | %-10s | %-8s | %-6s\n", 
           "Step", "Loss", "Tok/s", "GFLOPS", "GNorm", "Time");
    printf("-------|----------|------------|------------|----------|--------\n");

    double start_time = get_time();
    double last_time = start_time;
    float running_loss = -1.0f;

    for (int step = 0; step < steps; step++) {
        /* Sample random sequence */
        size_t offset = rand() % (dl.n_tokens - cfg.seq_len - 1);
        uint32_t *tokens_plus1 = &dl.tokens[offset];

        double step_start = get_time();
        float loss = kmamba_train_step(m, tokens_plus1);
        double step_end = get_time();

        if (running_loss < 0) running_loss = loss;
        else running_loss = 0.99f * running_loss + 0.01f * loss;

        if (step % 50 == 0 || step == steps - 1) {
            double now = get_time();
            double dt = now - last_time;
            if (step == 0) dt = now - start_time;
            
            double tps = (step == 0) ? cfg.seq_len / dt : (50.0 * cfg.seq_len) / dt;
            double gflops = (flops_per_step * (step == 0 ? 1 : 50)) / (dt * 1e9);
            
            printf("%-6d | %-8.4f | %-10.2f | %-10.2f | %-8.4f | %-6.1fs\n", 
                   step, running_loss, tps, gflops, kmamba_last_grad_norm(m), now - start_time);
            
            last_time = now;
        }

        /* Periodic checkpointing */
        if (step > 0 && step % 1000 == 0) {
            kmamba_save_ser(m, SAVE_PATH, KSER_FP32);
        }
    }

    printf("\n🏁 Training Finished. Final Loss: %.4f\n", running_loss);
    kmamba_save_ser(m, SAVE_PATH, KSER_FP32);

    /* Cleanup */
    kmamba_free_tokens(dl.tokens, dl.n_tokens);
    kmamba_free(m);
    return 0;
}

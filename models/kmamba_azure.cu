/* ============================================================================
 * kmamba_azure.cu - Cloud-Scale Training Instance (Azure)
 * 
 * Matches PLAN.md: GPU Large Azure (7.5B params)
 * 100K Vocab (Tiktoken cl100k_base)
 * Full GPU Training
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "kmamba.h"
#include "kmamba_ser.h"

/* ═══════════════════════════════════════════════════════════════
 * Configuration (PLAN.md: Azure Large 7.5B)
 * ═══════════════════════════════════════════════════════════════ */
#define VOCAB_SIZE      100277
#define MODEL_DIM       3072
#define STATE_SIZE      4096
#define LAYERS          24
#define SEQ_LEN         2048
#define D_CONV          4
#define EXPAND          2.0f

#define DEFAULT_DATASET "data/conversations.tok" /* Pre-tokenized uint32 array */
#define SAVE_PATH       "models/kmamba_azure.ser"

typedef struct {
    uint32_t *tokens;
    size_t    n_tokens;
} DataLoader;

DataLoader dataloader_load_pretokenized(const char *path) {
    DataLoader dl = {NULL, 0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return dl;

    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    dl.tokens = (uint32_t*)malloc(fsize);
    if (!dl.tokens) { fclose(fp); return dl; }
    
    dl.n_tokens = fread(dl.tokens, sizeof(uint32_t), fsize / sizeof(uint32_t), fp);
    fclose(fp);
    
    printf("📦 Loaded %zu pre-tokenized tokens from %s\n", dl.n_tokens, path);
    return dl;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

int main(int argc, char **argv) {
    printf("☁  K-Mamba Azure Large — 7.5B Cloud Integration\n");
    printf("═══════════════════════════════════════════════\n\n");

    /* 1. Init cl100k Tokenizer */
    kmamba_tokenizer_init("cl100k");
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
    strncpy(cfg.model_name, "kmamba-7.5b-azure", 63);

    KMamba *m = kmamba_create(&cfg);
    kmamba_init(m, 777);

    MBOptimConfig opt = {
        .lr = 3e-4f,
        .mu = 0.9f,
        .beta2 = 0.95f,
        .eps = 1e-8f,
        .clip_norm = 1.0f,
        .weight_decay = 0.1f
    };
    
    /* Full GPU Mode: All parameters on GPU */
    kmamba_enable_training(m, &opt, 3e-4f, 0.1f);

    /* 3. Load Data */
    const char *data_path = (argc > 1) ? argv[1] : DEFAULT_DATASET;
    DataLoader dl = dataloader_load_pretokenized(data_path);
    if (!dl.tokens) {
        printf("⚠ Error: Pre-tokenized dataset not found at %s. Please run preparation script.\n", data_path);
        printf("Creating dummy data for GFLOPs test...\n");
        dl.n_tokens = 100000;
        dl.tokens = (uint32_t*)calloc(dl.n_tokens, sizeof(uint32_t));
    }

    /* 4. Training Loop (Full GPU Batch) */
    int steps = 100; /* Cloud runs are long, we test stability here */
    int batch_size = 4;
    
    printf("🏃 Training Full GPU | Batch=%d | Seq=%d\n\n", batch_size, SEQ_LEN);
    printf("%-6s | %-8s | %-10s | %-8s\n", "Step", "Loss", "Tokens/s", "Time");
    printf("-------|----------|------------|--------\n");

    double start_time = get_time();
    double last_time = start_time;

    uint32_t *batch_buf = (uint32_t*)malloc(batch_size * (SEQ_LEN + 1) * sizeof(uint32_t));

    for (int step = 0; step < steps; step++) {
        for (int b = 0; b < batch_size; b++) {
            size_t offset = rand() % (dl.n_tokens - SEQ_LEN - 1);
            memcpy(&batch_buf[b * (SEQ_LEN + 1)], &dl.tokens[offset], (SEQ_LEN + 1) * sizeof(uint32_t));
        }

        /* Full GPU batch training */
        float loss = kmamba_train_batch(m, batch_buf, batch_size);

        if (step % 5 == 0 || step == steps - 1) {
            double now = get_time();
            double dt = now - last_time;
            if (step == 0) dt = now - start_time;
            double tps = (5.0 * batch_size * SEQ_LEN) / dt;
            
            printf("%-6d | %-8.4f | %-10.2f | %-6.1fs\n", 
                   step, loss, tps, now - start_time);
            last_time = now;
        }

        if (step > 0 && step % 50 == 0) {
            kmamba_save_ser(m, SAVE_PATH, KSER_BF16); /* Save in BF16 for Azure */
        }
    }

    kmamba_save_ser(m, SAVE_PATH, KSER_BF16);
    printf("\n✅ Cloud Training Checkpoint Saved to %s\n", SAVE_PATH);

    free(batch_buf);
    free(dl.tokens);
    kmamba_free(m);
    return 0;
}

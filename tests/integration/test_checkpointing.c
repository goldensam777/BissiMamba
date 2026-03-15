/*
 * test_checkpointing.c — Tests de sauvegarde/chargement de checkpoints
 *
 * Phase 4 : Tests de régression et benchmarks
 * Tests de checkpointing pour k-mamba
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

#define KMAMBA_MAGIC 0x4B4D414D  // "KMAMBA" en little-endian
#define KMAMBA_VERSION 1
#define EPSILON 1e-6f

/* ============================================================
 * Structures pour checkpointing
 * ============================================================ */

typedef struct {
    uint32_t magic;        /* "KMAMBA" */
    uint32_t version;       /* Version du format */
    uint32_t vocab_size;   /* Taille du vocabulaire */
    uint32_t dim;          /* Dimension du modèle */
    uint32_t state_size;   /* Taille de l'état */
    uint32_t seq_len;      /* Longueur de séquence */
    uint32_t n_layers;     /* Nombre de couches */
    float dt_scale;        /* Paramètres SSM */
    float dt_min;
    float dt_max;
    /* Suivi des données des paramètres */
    uint64_t data_offset;   /* Offset des données */
    uint64_t data_size;     /* Taille des données */
} KMambaCheckpointHeader;

typedef struct {
    KMambaCheckpointHeader header;
    float *embedding;      /* [vocab_size x dim] */
    float *mamba_blocks;   /* Paramètres des MambaBlocks */
    float *lm_head;        /* Paramètres du LM head */
    float *optimizer_state; /* État de l'optimiseur */
} KMambaCheckpoint;

/* ============================================================
 * Fonctions de checkpointing
 * ============================================================ */

static int kmamba_save_checkpoint(const KMambaCheckpoint *ckpt, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("FAIL: Cannot open file for writing: %s\n", filename);
        return 0;
    }
    
    /* Écrire le header */
    if (fwrite(&ckpt->header, sizeof(KMambaCheckpointHeader), 1, fp) != 1) {
        printf("FAIL: Cannot write header\n");
        fclose(fp);
        return 0;
    }
    
    /* Écrire les données */
    uint64_t total_size = 0;
    
    /* Embedding */
    uint64_t embedding_size = (uint64_t)ckpt->header.vocab_size * ckpt->header.dim * sizeof(float);
    if (fwrite(ckpt->embedding, embedding_size, 1, fp) != 1) {
        printf("FAIL: Cannot write embedding\n");
        fclose(fp);
        return 0;
    }
    total_size += embedding_size;
    
    /* MambaBlocks */
    uint64_t blocks_size = (uint64_t)ckpt->header.n_layers * ckpt->header.dim * ckpt->header.state_size * sizeof(float);
    if (fwrite(ckpt->mamba_blocks, blocks_size, 1, fp) != 1) {
        printf("FAIL: Cannot write mamba blocks\n");
        fclose(fp);
        return 0;
    }
    total_size += blocks_size;
    
    /* LM Head */
    uint64_t lm_head_size = (uint64_t)ckpt->header.dim * ckpt->header.vocab_size * sizeof(float);
    if (fwrite(ckpt->lm_head, lm_head_size, 1, fp) != 1) {
        printf("FAIL: Cannot write lm head\n");
        fclose(fp);
        return 0;
    }
    total_size += lm_head_size;
    
    /* Optimizer state */
    uint64_t optimizer_size = (uint64_t)ckpt->header.n_layers * 10 * sizeof(float); /* Estimation */
    if (fwrite(ckpt->optimizer_state, optimizer_size, 1, fp) != 1) {
        printf("FAIL: Cannot write optimizer state\n");
        fclose(fp);
        return 0;
    }
    total_size += optimizer_size;
    
    /* Mettre à jour le header avec les tailles */
    KMambaCheckpointHeader updated_header = ckpt->header;
    updated_header.data_offset = sizeof(KMambaCheckpointHeader);
    updated_header.data_size = total_size;
    
    fseek(fp, 0, SEEK_SET);
    if (fwrite(&updated_header, sizeof(KMambaCheckpointHeader), 1, fp) != 1) {
        printf("FAIL: Cannot update header\n");
        fclose(fp);
        return 0;
    }
    
    /* Forcer l'écriture des données */
    fflush(fp);
    
    fclose(fp);
    return 1;
}

static int kmamba_load_checkpoint(KMambaCheckpoint *ckpt, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("FAIL: Cannot open file for reading: %s\n", filename);
        return 0;
    }
    
    /* Lire le header */
    if (fread(&ckpt->header, sizeof(KMambaCheckpointHeader), 1, fp) != 1) {
        printf("FAIL: Cannot read header\n");
        fclose(fp);
        return 0;
    }
    
    /* Valider le magic et la version */
    if (ckpt->header.magic != KMAMBA_MAGIC) {
        printf("FAIL: Invalid magic number: 0x%08X\n", ckpt->header.magic);
        fclose(fp);
        return 0;
    }
    
    if (ckpt->header.version != KMAMBA_VERSION) {
        printf("FAIL: Unsupported version: %d\n", ckpt->header.version);
        fclose(fp);
        return 0;
    }
    
    /* Allouer la mémoire */
    ckpt->embedding = (float*)malloc((size_t)ckpt->header.vocab_size * ckpt->header.dim * sizeof(float));
    ckpt->mamba_blocks = (float*)malloc((size_t)ckpt->header.n_layers * ckpt->header.dim * ckpt->header.state_size * sizeof(float));
    ckpt->lm_head = (float*)malloc((size_t)ckpt->header.dim * ckpt->header.vocab_size * sizeof(float));
    ckpt->optimizer_state = (float*)malloc((size_t)ckpt->header.n_layers * 10 * sizeof(float));
    
    if (!ckpt->embedding || !ckpt->mamba_blocks || !ckpt->lm_head || !ckpt->optimizer_state) {
        printf("FAIL: Cannot allocate memory\n");
        fclose(fp);
        return 0;
    }
    
    /* Lire les données */
    if (fread(ckpt->embedding, (size_t)ckpt->header.vocab_size * ckpt->header.dim * sizeof(float), 1, fp) != 1) {
        printf("FAIL: Cannot read embedding\n");
        fclose(fp);
        return 0;
    }
    
    if (fread(ckpt->mamba_blocks, (size_t)ckpt->header.n_layers * ckpt->header.dim * ckpt->header.state_size * sizeof(float), 1, fp) != 1) {
        printf("FAIL: Cannot read mamba blocks\n");
        fclose(fp);
        return 0;
    }
    
    if (fread(ckpt->lm_head, (size_t)ckpt->header.dim * ckpt->header.vocab_size * sizeof(float), 1, fp) != 1) {
        printf("FAIL: Cannot read lm head\n");
        fclose(fp);
        return 0;
    }
    
    if (fread(ckpt->optimizer_state, (size_t)ckpt->header.n_layers * 10 * sizeof(float), 1, fp) != 1) {
        printf("FAIL: Cannot read optimizer state\n");
        fclose(fp);
        return 0;
    }
    
    fclose(fp);
    return 1;
}

static void kmamba_free_checkpoint(KMambaCheckpoint *ckpt) {
    if (!ckpt) return;
    
    free(ckpt->embedding);
    free(ckpt->mamba_blocks);
    free(ckpt->lm_head);
    free(ckpt->optimizer_state);
    
    memset(ckpt, 0, sizeof(KMambaCheckpoint));
}

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

static int compare_checkpoints(const KMambaCheckpoint *ckpt1, const KMambaCheckpoint *ckpt2) {
    /* Comparer les headers */
    if (memcmp(&ckpt1->header, &ckpt2->header, sizeof(KMambaCheckpointHeader)) != 0) {
        printf("FAIL: Headers differ\n");
        return 0;
    }
    
    /* Comparer les données */
    size_t embedding_size = (size_t)ckpt1->header.vocab_size * ckpt1->header.dim;
    if (memcmp(ckpt1->embedding, ckpt2->embedding, embedding_size * sizeof(float)) != 0) {
        printf("FAIL: Embedding data differs\n");
        return 0;
    }
    
    size_t blocks_size = (size_t)ckpt1->header.n_layers * ckpt1->header.dim * ckpt1->header.state_size;
    if (memcmp(ckpt1->mamba_blocks, ckpt2->mamba_blocks, blocks_size * sizeof(float)) != 0) {
        printf("FAIL: Mamba blocks data differs\n");
        return 0;
    }
    
    size_t lm_head_size = (size_t)ckpt1->header.dim * ckpt1->header.vocab_size;
    if (memcmp(ckpt1->lm_head, ckpt2->lm_head, lm_head_size * sizeof(float)) != 0) {
        printf("FAIL: LM head data differs\n");
        return 0;
    }
    
    return 1;
}

static void print_checkpoint_info(const KMambaCheckpoint *ckpt) {
    printf("Checkpoint Information:\n");
    printf("  Magic: 0x%08X\n", ckpt->header.magic);
    printf("  Version: %d\n", ckpt->header.version);
    printf("  Vocab Size: %u\n", ckpt->header.vocab_size);
    printf("  Dim: %u\n", ckpt->header.dim);
    printf("  State Size: %u\n", ckpt->header.state_size);
    printf("  Seq Len: %u\n", ckpt->header.seq_len);
    printf("  N Layers: %u\n", ckpt->header.n_layers);
    printf("  DT Scale: %.3f\n", ckpt->header.dt_scale);
    printf("  DT Min: %.3f\n", ckpt->header.dt_min);
    printf("  DT Max: %.3f\n", ckpt->header.dt_max);
    printf("  Data Offset: %llu\n", (unsigned long long)ckpt->header.data_offset);
    printf("  Data Size: %llu\n", (unsigned long long)ckpt->header.data_size);
}

/* ============================================================
 * Tests de checkpointing
 * ============================================================ */

static int test_checkpoint_save_load() {
    printf("Testing checkpoint save/load...\n");
    
    /* Créer un checkpoint de test */
    KMambaCheckpoint ckpt_orig = {0};
    
    ckpt_orig.header.magic = KMAMBA_MAGIC;
    ckpt_orig.header.version = KMAMBA_VERSION;
    ckpt_orig.header.vocab_size = 256;
    ckpt_orig.header.dim = 64;
    ckpt_orig.header.state_size = 128;
    ckpt_orig.header.seq_len = 128;
    ckpt_orig.header.n_layers = 2;
    ckpt_orig.header.dt_scale = 1.0f;
    ckpt_orig.header.dt_min = 0.001f;
    ckpt_orig.header.dt_max = 0.1f;
    
    /* Allouer et remplir les données */
    size_t embedding_size = (size_t)ckpt_orig.header.vocab_size * ckpt_orig.header.dim;
    ckpt_orig.embedding = (float*)malloc(embedding_size * sizeof(float));
    for (size_t i = 0; i < embedding_size; i++) {
        ckpt_orig.embedding[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    size_t blocks_size = (size_t)ckpt_orig.header.n_layers * ckpt_orig.header.dim * ckpt_orig.header.state_size;
    ckpt_orig.mamba_blocks = (float*)malloc(blocks_size * sizeof(float));
    for (size_t i = 0; i < blocks_size; i++) {
        ckpt_orig.mamba_blocks[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    size_t lm_head_size = (size_t)ckpt_orig.header.dim * ckpt_orig.header.vocab_size;
    ckpt_orig.lm_head = (float*)malloc(lm_head_size * sizeof(float));
    for (size_t i = 0; i < lm_head_size; i++) {
        ckpt_orig.lm_head[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    size_t optimizer_size = (size_t)ckpt_orig.header.n_layers * 10;
    ckpt_orig.optimizer_state = (float*)malloc(optimizer_size * sizeof(float));
    for (size_t i = 0; i < optimizer_size; i++) {
        ckpt_orig.optimizer_state[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    printf("Original checkpoint created\n");
    print_checkpoint_info(&ckpt_orig);
    
    /* Sauvegarder */
    const char *filename = "test_checkpoint.kmamba";
    if (!kmamba_save_checkpoint(&ckpt_orig, filename)) {
        printf("FAIL: Cannot save checkpoint\n");
        kmamba_free_checkpoint(&ckpt_orig);
        return 0;
    }
    
    printf("Checkpoint saved to: %s\n", filename);
    
    /* Charger */
    KMambaCheckpoint ckpt_loaded = {0};
    if (!kmamba_load_checkpoint(&ckpt_loaded, filename)) {
        printf("FAIL: Cannot load checkpoint\n");
        kmamba_free_checkpoint(&ckpt_orig);
        return 0;
    }
    
    printf("Checkpoint loaded from: %s\n", filename);
    print_checkpoint_info(&ckpt_loaded);
    
    /* Comparer */
    if (compare_checkpoints(&ckpt_orig, &ckpt_loaded)) {
        printf("PASS: Checkpoint save/load successful\n");
        kmamba_free_checkpoint(&ckpt_orig);
        kmamba_free_checkpoint(&ckpt_loaded);
        return 1;
    } else {
        printf("FAIL: Checkpoint data differs after save/load\n");
        kmamba_free_checkpoint(&ckpt_orig);
        kmamba_free_checkpoint(&ckpt_loaded);
        return 0;
    }
}

static int test_checkpoint_invalid_magic() {
    printf("Testing checkpoint with invalid magic...\n");
    
    /* Créer un fichier avec magic invalide */
    FILE *fp = fopen("invalid_checkpoint.kmamba", "wb");
    if (!fp) {
        printf("FAIL: Cannot create test file\n");
        return 0;
    }
    
    KMambaCheckpointHeader header = {0};
    header.magic = 0xDEADBEEF;  // Magic invalide
    header.version = KMAMBA_VERSION;
    
    if (fwrite(&header, sizeof(KMambaCheckpointHeader), 1, fp) != 1) {
        printf("FAIL: Cannot write invalid header\n");
        fclose(fp);
        return 0;
    }
    
    fclose(fp);
    
    /* Tenter de charger */
    KMambaCheckpoint ckpt = {0};
    if (kmamba_load_checkpoint(&ckpt, "invalid_checkpoint.kmamba")) {
        printf("FAIL: Should reject invalid magic\n");
        kmamba_free_checkpoint(&ckpt);
        return 0;
    }
    
    printf("PASS: Invalid magic correctly rejected\n");
    kmamba_free_checkpoint(&ckpt);
    return 1;
}

static int test_checkpoint_invalid_version() {
    printf("Testing checkpoint with invalid version...\n");
    
    /* Créer un fichier avec version invalide */
    FILE *fp = fopen("invalid_version.kmamba", "wb");
    if (!fp) {
        printf("FAIL: Cannot create test file\n");
        return 0;
    }
    
    KMambaCheckpointHeader header = {0};
    header.magic = KMAMBA_MAGIC;
    header.version = 999;  // Version invalide
    
    if (fwrite(&header, sizeof(KMambaCheckpointHeader), 1, fp) != 1) {
        printf("FAIL: Cannot write invalid version header\n");
        fclose(fp);
        return 0;
    }
    
    fclose(fp);
    
    /* Tenter de charger */
    KMambaCheckpoint ckpt = {0};
    if (kmamba_load_checkpoint(&ckpt, "invalid_version.kmamba")) {
        printf("FAIL: Should reject invalid version\n");
        kmamba_free_checkpoint(&ckpt);
        return 0;
    }
    
    printf("PASS: Invalid version correctly rejected\n");
    kmamba_free_checkpoint(&ckpt);
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_checkpointing() {
    printf("\n=== Checkpointing Performance Benchmarks ===\n");
    
    const size_t vocab_size = 1000, dim = 256, state_size = 512, n_layers = 4;
    const int iterations = 10;
    
    /* Créer un grand checkpoint */
    KMambaCheckpoint ckpt = {0};
    
    ckpt.header.magic = KMAMBA_MAGIC;
    ckpt.header.version = KMAMBA_VERSION;
    ckpt.header.vocab_size = vocab_size;
    ckpt.header.dim = dim;
    ckpt.header.state_size = state_size;
    ckpt.header.seq_len = 1024;
    ckpt.header.n_layers = n_layers;
    ckpt.header.dt_scale = 1.0f;
    ckpt.header.dt_min = 0.001f;
    ckpt.header.dt_max = 0.1f;
    
    size_t embedding_size = vocab_size * dim;
    size_t blocks_size = n_layers * dim * state_size;
    size_t lm_head_size = dim * vocab_size;
    size_t optimizer_size = n_layers * 10;
    
    ckpt.embedding = (float*)malloc(embedding_size * sizeof(float));
    ckpt.mamba_blocks = (float*)malloc(blocks_size * sizeof(float));
    ckpt.lm_head = (float*)malloc(lm_head_size * sizeof(float));
    ckpt.optimizer_state = (float*)malloc(optimizer_size * sizeof(float));
    
    /* Remplir avec des données */
    for (size_t i = 0; i < embedding_size; i++) {
        ckpt.embedding[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < blocks_size; i++) {
        ckpt.mamba_blocks[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < lm_head_size; i++) {
        ckpt.lm_head[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < optimizer_size; i++) {
        ckpt.optimizer_state[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    printf("Large checkpoint created:\n");
    printf("  Embedding: %zu elements\n", embedding_size);
    printf("  MambaBlocks: %zu elements\n", blocks_size);
    printf("  LM Head: %zu elements\n", lm_head_size);
    printf("  Total data: %zu elements (%.2f MB)\n", 
           embedding_size + blocks_size + lm_head_size + optimizer_size,
           (float)(embedding_size + blocks_size + lm_head_size + optimizer_size) * sizeof(float) / (1024*1024));
    
    /* Benchmark sauvegarde */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        if (!kmamba_save_checkpoint(&ckpt, "benchmark_checkpoint.kmamba")) {
            printf("FAIL: Cannot save during benchmark\n");
            break;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double save_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    /* Benchmark chargement */
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        KMambaCheckpoint loaded = {0};
        if (!kmamba_load_checkpoint(&loaded, "benchmark_checkpoint.kmamba")) {
            printf("FAIL: Cannot load during benchmark\n");
            kmamba_free_checkpoint(&loaded);
            break;
        }
        kmamba_free_checkpoint(&loaded);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double load_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Checkpointing Performance:\n");
    printf("  Save: %.3f sec (%d iterations, %.3f sec per save)\n", 
           save_time, iterations, save_time / iterations);
    printf("  Load: %.3f sec (%d iterations, %.3f sec per load)\n", 
           load_time, iterations, load_time / iterations);
    printf("  Throughput: %.2f saves/sec, %.2f loads/sec\n", 
           (double)iterations / save_time, (double)iterations / load_time);
    
    /* Nettoyage */
    kmamba_free_checkpoint(&ckpt);
    remove("benchmark_checkpoint.kmamba");
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Checkpointing Test Suite ===\n");
    printf("Testing k-mamba checkpoint save/load functionality\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests de base */
    total++; passed += test_checkpoint_save_load();
    total++; passed += test_checkpoint_invalid_magic();
    total++; passed += test_checkpoint_invalid_version();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        benchmark_checkpointing();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}

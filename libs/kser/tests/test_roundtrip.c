/* ============================================================================
 * test_roundtrip.c - Test écriture/lecture .ser
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kser.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s at line %d\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

static int float_eq(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int vocab_count = 0;
static int vocab_cb(uint32_t id, const char* token, uint16_t len, void* userdata) {
    (void)userdata;
    printf("  Vocab[%u]: '%.*s' (len=%u)\n", id, len, token, len);
    vocab_count++;
    return 0;
}

int main(void) {
    printf("=== libkser Roundtrip Test ===\n\n");
    
    /* Test 1: Simple FP32 roundtrip */
    printf("Test 1: FP32 roundtrip\n");
    {
        KSerConfig cfg = {
            .vocab_size = 100,
            .dim = 64,
            .state_size = 128,
            .n_layers = 2,
            .seq_len = 256,
            .d_conv = 4,
            .expand_factor = 2.0f,
            .dtype = KSER_FP32,
            .model_name = "test_model"
        };
        
        /* Create writer */
        KSerWriter* w = kser_writer_create("tests/test1.ser", &cfg);
        TEST_ASSERT(w != NULL, "writer_create failed");
        
        /* Add some vocab */
        kser_writer_add_vocab(w, 0, "<pad>", 5);
        kser_writer_add_vocab(w, 1, "hello", 5);
        kser_writer_add_vocab(w, 2, "world", 5);
        
        /* Create test tensor */
        float tensor_data[64] = {0};
        for (int i = 0; i < 64; i++) tensor_data[i] = (float)i * 0.5f;
        
        uint32_t shape[4] = {64, 1, 1, 1};
        int ret = kser_writer_add_tensor(w, "test_tensor", tensor_data, shape, 
                                         KSER_FP32, KSER_FP32);
        TEST_ASSERT(ret == KSER_OK, "add_tensor failed");
        
        /* Finalize */
        ret = kser_writer_finalize(w);
        TEST_ASSERT(ret == KSER_OK, "finalize failed");
        kser_writer_free(w);
        
        /* Read back */
        KSerReader* r = kser_reader_open("tests/test1.ser");
        TEST_ASSERT(r != NULL, "reader_open failed");
        
        const KSerConfig* rcfg = kser_reader_config(r);
        TEST_ASSERT(rcfg != NULL, "reader_config failed");
        TEST_ASSERT(rcfg->vocab_size == 100, "vocab_size mismatch");
        TEST_ASSERT(rcfg->dim == 64, "dim mismatch");
        TEST_ASSERT(rcfg->n_layers == 2, "n_layers mismatch");
        
        /* Check vocab */
        vocab_count = 0;
        kser_reader_load_vocab(r, vocab_cb, NULL);
        TEST_ASSERT(vocab_count == 3, "vocab count mismatch");
        
        /* Check tensor */
        float* read_data = kser_reader_load_tensor(r, "test_tensor", KSER_FP32);
        TEST_ASSERT(read_data != NULL, "load_tensor failed");
        
        int tensor_ok = 1;
        for (int i = 0; i < 64; i++) {
            if (!float_eq(read_data[i], tensor_data[i], 1e-6f)) {
                tensor_ok = 0;
                break;
            }
        }
        TEST_ASSERT(tensor_ok, "tensor data mismatch");
        
        free(read_data);
        kser_reader_close(r);
        
        printf("[PASS] FP32 roundtrip\n\n");
    }
    
    /* Test 2: Info API */
    printf("Test 2: File info API\n");
    {
        KSerInfo info = kser_file_info("tests/test1.ser");
        TEST_ASSERT(info.valid == 1, "file not valid");
        TEST_ASSERT(info.vocab_size == 100, "info vocab_size mismatch");
        TEST_ASSERT(info.dim == 64, "info dim mismatch");
        printf("[PASS] File info: %s, ~%.1fM params\n", 
               info.model_name, info.n_params / 1e6);
        printf("[PASS] Info API\n\n");
    }
    
    /* Test 3: Quantization FP32 -> FP16 */
    printf("Test 3: FP16 quantization\n");
    {
        float input[4] = {1.0f, 2.5f, -0.5f, 1000.0f};
        uint16_t output[4];
        float recovered[4];
        
        int ret = kser_quantize(input, output, 4, KSER_FP16);
        TEST_ASSERT(ret == KSER_OK, "quantize to FP16 failed");
        
        ret = kser_dequantize(output, recovered, 4, KSER_FP16);
        TEST_ASSERT(ret == KSER_OK, "dequantize from FP16 failed");
        
        /* Check approximate equality */
        TEST_ASSERT(float_eq(recovered[0], 1.0f, 0.001f), "FP16 decode[0] mismatch");
        TEST_ASSERT(float_eq(recovered[1], 2.5f, 0.001f), "FP16 decode[1] mismatch");
        
        printf("[PASS] FP16 quantization\n\n");
    }
    
    /* Test 4: Quantization FP32 -> BF16 */
    printf("Test 4: BF16 quantization\n");
    {
        float input[4] = {1.0f, 2.5f, -0.5f, 1000.0f};
        uint16_t output[4];
        float recovered[4];
        
        int ret = kser_quantize(input, output, 4, KSER_BF16);
        TEST_ASSERT(ret == KSER_OK, "quantize to BF16 failed");
        
        ret = kser_dequantize(output, recovered, 4, KSER_BF16);
        TEST_ASSERT(ret == KSER_OK, "dequantize from BF16 failed");
        
        TEST_ASSERT(float_eq(recovered[0], 1.0f, 0.01f), "BF16 decode[0] mismatch");
        
        printf("[PASS] BF16 quantization\n\n");
    }
    
    /* Test 5: Quantization FP32 -> INT8 */
    printf("Test 5: INT8 quantization\n");
    {
        float input[4] = {0.0f, 0.25f, 0.5f, 1.0f};
        uint8_t output[4 + sizeof(float) * 2]; /* Space for quantization params */
        float recovered[4];
        
        int ret = kser_quantize(input, output, 4, KSER_INT8);
        TEST_ASSERT(ret == KSER_OK, "quantize to INT8 failed");
        
        ret = kser_dequantize(output, recovered, 4, KSER_INT8);
        TEST_ASSERT(ret == KSER_OK, "dequantize from INT8 failed");
        
        TEST_ASSERT(float_eq(recovered[0], 0.0f, 0.01f), "INT8 decode[0] mismatch");
        TEST_ASSERT(float_eq(recovered[3], 1.0f, 0.01f), "INT8 decode[3] mismatch");
        
        printf("[PASS] INT8 quantization\n\n");
    }
    
    printf("=== All tests passed ===\n");
    return 0;
}

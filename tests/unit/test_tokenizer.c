#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kmamba.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("[FAIL] %s\n", msg); \
        return 1; \
    } \
} while(0)

int main() {
    printf("=== Testing Hybrid Tokenizer ===\n\n");

    /* 1. Test Default / Byte-level (32K) */
    printf("Mode: bytes (32K)\n");
    kmamba_tokenizer_init("bytes");
    
    size_t vocab_size = kmamba_vocab_size();
    printf("  Vocab size: %zu\n", vocab_size);
    TEST_ASSERT(vocab_size == 32768, "Vocab size should be 32768 for bytes");

    const char* text = "Hello η"; // η is 2 bytes in UTF-8
    size_t len = 0;
    uint32_t* tokens = kmamba_encode(text, &len);
    
    printf("  Encoding '%s' -> %zu tokens\n", text, len);
    // 'H','e','l','l','o',' ','\xCE','\xB7'
    TEST_ASSERT(len == 8, "Byte-level should produce 8 tokens for 'Hello η'");
    TEST_ASSERT(tokens[6] == 0xCE, "Token 6 should be 0xCE");
    TEST_ASSERT(tokens[7] == 0xB7, "Token 7 should be 0xB7");

    char* decoded = kmamba_decode(tokens, len);
    printf("  Decoded: '%s'\n", decoded);
    TEST_ASSERT(strcmp(decoded, text) == 0, "Roundtrip failed for bytes");

    kmamba_free_tokens(tokens, len);
    kmamba_free_string(decoded);
    printf("[PASS] Byte-level tests\n\n");

    /* 2. Test Tiktoken (100K) */
    printf("Mode: cl100k (100K)\n");
    kmamba_tokenizer_init("cl100k");
    
    vocab_size = kmamba_vocab_size();
    printf("  Vocab size: %zu\n", vocab_size);
    TEST_ASSERT(vocab_size == 100277, "Vocab size should be 100277 for cl100k");

    tokens = kmamba_encode(text, &len);
    printf("  Encoding '%s' -> %zu tokens\n", text, len);
    // Tiktoken usually merges 'Hello' into one or two tokens
    TEST_ASSERT(len < 8, "Tiktoken should be more efficient than byte-level");

    decoded = kmamba_decode(tokens, len);
    printf("  Decoded: '%s'\n", decoded);
    TEST_ASSERT(strcmp(decoded, text) == 0, "Roundtrip failed for cl100k");

    kmamba_free_tokens(tokens, len);
    kmamba_free_string(decoded);
    printf("[PASS] Tiktoken tests\n\n");

    printf("=== All Tokenizer Tests Passed ===\n");
    return 0;
}

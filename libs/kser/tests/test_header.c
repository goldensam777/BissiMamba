/* Test header values */
#include <stdio.h>
#include <stdint.h>
#include "kser.h"

int main() {
    printf("KSER_MAGIC_BYTES:\n");
    for (int i = 0; i < KSER_HEADER_SIZE; i++) {
        printf("  [%d] = 0x%02x ('%c')\n", i, KSER_MAGIC_BYTES[i], 
               (KSER_MAGIC_BYTES[i] >= 32 && KSER_MAGIC_BYTES[i] < 127) ? KSER_MAGIC_BYTES[i] : '?');
    }
    
    printf("\nExpected:\n");
    printf("  [0-7] = SERENITY\n");
    printf("  [8-9] = 0xCE 0xB7 (eta)\n");
    printf("  [10]  = 0x01 (version)\n");
    printf("  [11-15] = 0x00 (reserved)\n");
    
    /* Write test file */
    FILE* fp = fopen("tests/test_header.ser", "wb");
    if (fp) {
        fwrite(KSER_MAGIC_BYTES, 1, KSER_HEADER_SIZE, fp);
        fclose(fp);
        printf("\nWrote tests/test_header.ser\n");
    }
    
    return 0;
}

/* Minimal test for file writing */
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>

#define HEADER_SIZE 16

static const uint8_t MAGIC_BYTES[HEADER_SIZE] = {
    'S', 'E', 'R', 'E', 'N', 'I', 'T', 'Y',
    0xCE, 0xB7,
    0x01,
    0x00, 0x00, 0x00, 0x00, 0x00
};

int main() {
    const char* path = "tests/minimal.ser";
    
    /* Test 1: Using write() */
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    printf("Writing %d bytes with write():\n", HEADER_SIZE);
    for (int i = 0; i < HEADER_SIZE; i++) {
        printf("%02x ", MAGIC_BYTES[i]);
    }
    printf("\n");
    
    ssize_t n = write(fd, MAGIC_BYTES, HEADER_SIZE);
    printf("write() returned %zd\n", n);
    
    close(fd);
    
    /* Read back and verify */
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen for read");
        return 1;
    }
    
    uint8_t buf[HEADER_SIZE];
    size_t r = fread(buf, 1, HEADER_SIZE, fp);
    fclose(fp);
    
    printf("Read %zu bytes back:\n", r);
    for (int i = 0; i < (int)r; i++) {
        printf("%02x ", buf[i]);
    }
    printf("\n");
    
    /* Compare */
    int match = (r == HEADER_SIZE && memcmp(buf, MAGIC_BYTES, HEADER_SIZE) == 0);
    printf("Match: %s\n", match ? "YES" : "NO");
    
    return match ? 0 : 1;
}

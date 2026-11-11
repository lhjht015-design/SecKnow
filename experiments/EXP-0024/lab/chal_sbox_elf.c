/* EXP-0024 V1：编译版逐字节 S 盒（建表路径）。
 *
 * 编译：
 *   gcc chal_sbox_elf.c -o chal_sbox_elf -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>

static unsigned char TABLES[4][256];

static void build(void) {
    for (int k = 0; k < 4; k++) {
        for (int b = 0; b < 256; b++) {
            TABLES[k][b] = (unsigned char)((((b * 181 + 0x3B + k * 97) & 0xFF)
                                           ^ ((k * 0x5D) & 0xFF)) & 0xFF);
        }
    }
}

int main(void) {
    build();
    static unsigned char buf[4096];
    size_t total = 0;
    ssize_t got;
    while (total < sizeof(buf) &&
           (got = read(0, buf + total, sizeof(buf) - total)) > 0) {
        total += (size_t)got;
    }
    for (size_t i = 0; i < total; i++) {
        buf[i] = (unsigned char)(TABLES[i & 3][buf[i]]
                                 ^ ((i * 0x5D + 0x11) & 0xFF));
    }
    fwrite(buf, 1, total, stdout);
    fflush(stdout);
    return 0;
}

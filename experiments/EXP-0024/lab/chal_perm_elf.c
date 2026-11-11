/* EXP-0024 V3：编译版位置置换 + 常量异或（置换路径）。
 *
 * 编译：
 *   gcc chal_perm_elf.c -o chal_perm_elf -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>

int main(void) {
    static unsigned char buf[4096];
    static unsigned char out[4096];
    size_t total = 0;
    ssize_t got;
    while (total < sizeof(buf) &&
           (got = read(0, buf + total, sizeof(buf) - total)) > 0) {
        total += (size_t)got;
    }
    for (size_t j = 0; j < total; j++) {
        size_t src = total ? (j + total / 2) % total : 0;
        out[j] = (unsigned char)(buf[src] ^ ((j * 0x33 + 0x11) & 0xFF));
    }
    fwrite(out, 1, total, stdout);
    fflush(stdout);
    return 0;
}

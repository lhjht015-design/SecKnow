/* EXP-0024 V4：编译版 GF(2) 线性变换（稠密路径）。
 *
 * out[j] = buf[j] ^ buf[(j+1)%n] ^ buf[(j+3)%n] ^ K[j]
 * 循环多项式 1+z+z^3 与 z^n-1=(z+1)^n 互素（f(1)=1），因此对任意 n 可逆。
 *
 * 编译：
 *   gcc chal_gf2_elf.c -o chal_gf2_elf -O2 -s
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
        unsigned char value = buf[j];
        if (total > 1) value ^= buf[(j + 1) % total];
        if (total > 3) value ^= buf[(j + 3) % total];
        out[j] = (unsigned char)(value ^ ((j * 0x5B + 0x2D) & 0xFF));
    }
    fwrite(out, 1, total, stdout);
    fflush(stdout);
    return 0;
}

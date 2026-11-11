/* EXP-0020 靶场：编译后的"主办方加密程序"。
 *
 * 读 stdin 直到 EOF，逐字节做仿射变换，把结果写到 stdout。
 * 与 P2 的 Python 脚本等价，用来验证"附件是 ELF 时也能自动建立 oracle 入口"。
 *
 * 编译：
 *   gcc chal_transform.c -o chal_transform -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>

int main(void) {
    static unsigned char buf[4096];
    size_t total = 0;
    ssize_t got;
    while (total < sizeof(buf) &&
           (got = read(0, buf + total, sizeof(buf) - total)) > 0) {
        total += (size_t)got;
    }
    for (size_t i = 0; i < total; i++) {
        unsigned char b = buf[i];
        unsigned int key = ((i * 0x2B + 0x07) & 0xFF) ^ ((i << 3) & 0xFF);
        buf[i] = (unsigned char)((0x9D * b + key) & 0xFF);
    }
    fwrite(buf, 1, total, stdout);
    fflush(stdout);
    return 0;
}

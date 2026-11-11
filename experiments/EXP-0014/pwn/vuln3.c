/* v016 用的靶场：ret2libc
 *   - 无 canary、无 PIE，先隔离出"libc 版本识别 + 两阶段利用"这一个变量
 *   - vuln() 用 read() 溢出
 *   - 需要一个能泄漏 libc 的调用：puts@plt(puts@got)
 *
 * 编译：
 *   gcc vuln3.c -o vuln3 -fno-stack-protector -no-pie -O0
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* gadget 真值 */
__asm__(
    ".global g3_pop_rdi\n"
    "g3_pop_rdi:\n"
    "  pop %rdi\n"
    "  ret\n"
    ".global g3_ret\n"
    "g3_ret:\n"
    "  ret\n"
);

void vuln(void) {
    char buf[64];
    read(0, buf, 512);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    puts("input:");
    vuln();
    puts("bye");
    return 0;
}

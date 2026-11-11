/* EXP-0016 负样本：canary + PIE 都开，但没有可控的格式化字符串泄漏。
 *
 * 仍然调用 printf，但格式串是常量，缓冲区不会被当作格式串解析——
 * 因此拿不到 canary / PIE 泄漏，系统应当返回 leak_failed 并拒绝，
 * 而不是硬猜 canary 继续打。
 *
 * 编译：
 *   gcc vuln4_noleak.c -o vuln4_noleak -fstack-protector-all -pie -fPIE -O0
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__asm__(
    ".global g4_pop_rdi\n"
    "g4_pop_rdi:\n"
    "  pop %rdi\n"
    "  ret\n"
    ".global g4_ret\n"
    "g4_ret:\n"
    "  ret\n"
);

void vuln(void) {
    char buf[128];
    read(0, buf, 0x200);
    printf("hello %d\n", 1);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    while (1) {
        puts("input:");
        vuln();
    }
}

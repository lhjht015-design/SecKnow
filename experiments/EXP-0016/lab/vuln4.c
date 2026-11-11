/* EXP-0016 靶场：canary + PIE + ret2libc 三保护组合
 *
 * 结构：
 *     main: while(1){ puts("input:"); vuln(); }
 *     vuln: char buf[128]; read(0, buf, 0x200); printf(buf);
 *
 * 三个保护同时开启：
 *     -fstack-protector-all   每个函数都有 canary
 *     -pie -fPIE             代码地址随机化
 *     系统 libc                ret2libc 需要题目附带的 libc 偏移
 *
 * 泄漏与利用必须在同一进程（ASLR 每次运行都变），所以 main 必须循环，
 * 三段 payload 依次喂入：
 *     1) 格式化字符串：泄漏 canary + PIE 基址（以及缓冲区在 varargs 中的位置）
 *     2) 格式化字符串 %s：通过 PIE 基址算出 puts@got，泄漏 libc 地址
 *     3) 溢出：canary + saved rbp + ROP → system("/bin/sh")
 *
 * 编译：
 *   gcc vuln4.c -o vuln4 -fstack-protector-all -pie -fPIE -O0
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* 固定 gadget，避免依赖编译器碰巧生成 pop rdi; ret */
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
    printf(buf);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    while (1) {
        puts("input:");
        vuln();
    }
}

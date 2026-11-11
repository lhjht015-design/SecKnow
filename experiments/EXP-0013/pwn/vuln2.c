/* v015 用的靶场：两阶段
 *   stage1: printf(buf) 格式化字符串漏洞 —— 用来泄漏 canary / PIE 基址
 *   stage2: read() 栈溢出 —— 泄漏之后打 ret2win
 *
 * 编译变体：
 *   gcc vuln2.c -o vuln2_canary -fstack-protector-all -no-pie -O0
 *   gcc vuln2.c -o vuln2_pie    -fno-stack-protector -pie -fPIE -O0
 *   gcc vuln2.c -o vuln2_both   -fstack-protector-all -pie -fPIE -O0
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void win(void) {
    FILE *fp = fopen("/tmp/flag.txt", "r");
    char buf[256] = {0};
    if (fp) {
        if (fgets(buf, sizeof(buf), fp)) printf("%s", buf);
        fclose(fp);
    } else {
        printf("flag{leak_bypass_fallback}\n");
    }
    fflush(stdout);
    _exit(0);
}

void win_arg(long x) {
    if (x == 0x1234) win();
    puts("nope");
    fflush(stdout);
    _exit(1);
}

/* 显式 gadget，作为"找 gadget"这一步的真值 */
__asm__(
    ".global g2_pop_rdi\n"
    "g2_pop_rdi:\n"
    "  pop %rdi\n"
    "  ret\n"
    ".global g2_ret\n"
    "g2_ret:\n"
    "  ret\n"
);

void stage1(void) {
    char buf[256];
    if (!fgets(buf, sizeof(buf), stdin)) return;
    printf(buf);            /* 格式化字符串漏洞 */
    fflush(stdout);
}

void stage2(void) {
    char buf[64];
    read(0, buf, 256);      /* 栈溢出 */
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    puts("stage1:");
    stage1();
    puts("stage2:");
    stage2();
    puts("done");
    return 0;
}

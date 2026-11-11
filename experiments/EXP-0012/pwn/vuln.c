/* v014 用的靶场程序：一个典型的、可被 ret2win 的栈溢出
 * 编译：gcc vuln.c -o vuln -fno-stack-protector -no-pie -O0
 * 特征：无 canary、无 PIE、NX 开启；gets() 无边界检查；存在 win() 后门。
 * 额外注册 SIGSEGV 处理器，把崩溃地址打印出来——这样"偏移量"可以由真实崩溃推导，
 * 而不是靠题面直接给。
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <ucontext.h>
#include <string.h>
#include <unistd.h>

void win(void) {
    FILE *fp = fopen("/tmp/flag.txt", "r");
    char buf[256] = {0};
    if (fp) {
        if (fgets(buf, sizeof(buf), fp)) printf("%s", buf);
        fclose(fp);
    } else {
        printf("flag{ret2win_local_fallback}\n");
    }
    fflush(stdout);
    _exit(0);
}

/* 需要参数的后门：只有 x == 0x1234 才给 flag。
 * 这样 exploit 必须真的找到 `pop rdi; ret` 并传参，而不是只跳地址。 */
void win_arg(long x) {
    if (x == 0x1234) {
        win();
    }
    puts("nope");
    fflush(stdout);
    _exit(1);
}

static void crash_handler(int sig, siginfo_t *info, void *uctx) {
    ucontext_t *ctx = (ucontext_t *)uctx;
    fprintf(stderr, "CRASH sig=%d addr=%p rip=%p\n",
            sig, info->si_addr, (void *)ctx->uc_mcontext.gregs[REG_RIP]);
    _exit(139);
}

/* 显式放几个 gadget，作为"找 gadget"这一步的真值。
 * 真实二进制里这些通常在 libc 或 CRT 里；这里内联是为了让测试自包含、
 * 并且能直接拿符号地址当标准答案。 */
__asm__(
    ".global gadget_pop_rdi\n"
    "gadget_pop_rdi:\n"
    "  pop %rdi\n"
    "  ret\n"
    ".global gadget_pop_rsi_r15\n"
    "gadget_pop_rsi_r15:\n"
    "  pop %rsi\n"
    "  pop %r15\n"
    "  ret\n"
    ".global gadget_ret\n"
    "gadget_ret:\n"
    "  ret\n"
);

void vulnerable(void) {
    char buf[64];
    gets(buf);          /* 无边界检查 */
    puts(buf);
}

int main(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = crash_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    setvbuf(stdout, NULL, _IONBF, 0);
    puts("input:");
    vulnerable();
    puts("done");
    return 0;
}

/* EXP-0015 靶场：把后门函数改名。
 *
 * 目的：模拟"主办方换函数名"。符号表里不再是 win，符号启发式会失效；
 * 正确做法只能靠**字符串引用**找到它。
 *
 * 编译：
 *   gcc vuln_renamed.c -o vuln_renamed_o2 -fno-stack-protector -no-pie -O2
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <ucontext.h>
#include <string.h>
#include <unistd.h>

void helper_a3f9(void) {
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

void h9_arg(long x) {
    if (x == 0x1234) {
        helper_a3f9();
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

__asm__(
    ".global gadget_pop_rdi\n"
    "gadget_pop_rdi:\n"
    "  pop %rdi\n"
    "  ret\n"
    ".global gadget_ret\n"
    "gadget_ret:\n"
    "  ret\n"
);

void vulnerable(void) {
    char buf[64];
    gets(buf);
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

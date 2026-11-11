/* EXP-0015 负样本 N1：没有任何后门，也没有 flag 相关字符串。
 *
 * 期望：系统返回 win_not_found 并拒绝，而不是随便挑一个函数跳过去。
 *
 * 编译：
 *   gcc vuln_nowin.c -o vuln_nowin_o2s -fno-stack-protector -no-pie -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <ucontext.h>
#include <string.h>
#include <unistd.h>

static void crash_handler(int sig, siginfo_t *info, void *uctx) {
    ucontext_t *ctx = (ucontext_t *)uctx;
    fprintf(stderr, "CRASH sig=%d addr=%p rip=%p\n",
            sig, info->si_addr, (void *)ctx->uc_mcontext.gregs[REG_RIP]);
    _exit(139);
}

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

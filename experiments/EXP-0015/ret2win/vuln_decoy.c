/* EXP-0015 负样本 N2：有看起来像后门的函数，但它给的是假 flag。
 *
 * 目的：验证"候选必须靠执行验证"。字符串启发式会把 decoy 当成候选，
 * 但真实执行后拿到的不是期望 flag，系统必须继续找或如实失败，
 * 不能看到一个 flag{...} 就宣布成功。
 *
 * 编译：
 *   gcc vuln_decoy.c -o vuln_decoy_o2s -fno-stack-protector -no-pie -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <ucontext.h>
#include <string.h>
#include <unistd.h>

/* 诱导项：引用 flag 字样，但输出的是假答案。 */
void decoy(void) {
    puts("/tmp/flag.txt");
    printf("flag{decoy_not_the_answer}\n");
    fflush(stdout);
    _exit(1);
}

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

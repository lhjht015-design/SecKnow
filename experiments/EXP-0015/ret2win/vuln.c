/* EXP-0015 靶场：ret2win 复测（与 EXP-0012 同源，便于对照）
 *
 * 编译（见 versions/v017_ret2win_retest.src）：
 *   gcc vuln.c -o vuln_o0 -fno-stack-protector -no-pie -O0
 *   gcc vuln.c -o vuln_o2 -fno-stack-protector -no-pie -O2
 *   gcc vuln.c -o vuln_o2s -fno-stack-protector -no-pie -O2 -s
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

/* 需要参数的后门：只有 x == 0x1234 才给 flag。 */
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

/* 显式 gadget，作为"找 gadget"的真值。 */
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

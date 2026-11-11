/* EXP-0022 N2：每匹配一个字节只 sleep 0.2ms，信号埋在进程启动噪声里。
 *
 * 期望：检测/爬坡判定信号太弱，返回 timing_signal_weak（或 no_prefix_signal）。
 *
 * 编译：
 *   gcc chal_timing_faint.c -o chal_timing_faint -O2 -s
 */
#define _GNU_SOURCE
#include <string.h>
#include <time.h>
#include <unistd.h>

static const char SECRET[] = "flag{faint_timing_signal}";

int main(void) {
    unsigned char buf[512];
    size_t total = 0;
    ssize_t got;
    while (total < sizeof(buf) &&
           (got = read(0, buf + total, sizeof(buf) - total)) > 0) {
        total += (size_t)got;
    }
    size_t limit = total < strlen(SECRET) ? total : strlen(SECRET);
    size_t i = 0;
    while (i < limit && buf[i] == (unsigned char)SECRET[i]) {
        i++;
        struct timespec ts = {0, 200000L};
        nanosleep(&ts, 0);
    }
    (void)write(1, "wrong\n", 6);
    return 1;
}

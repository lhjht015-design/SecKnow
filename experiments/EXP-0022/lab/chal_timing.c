/* EXP-0022 T1：时序侧信道校验器。
 *
 * 逐字节比较，每匹配一个字节 sleep 8ms，然后输出 "wrong" 并退出。
 * stdout 与退出码都恒定，唯一的信号是**耗时**。
 *
 * 编译：
 *   gcc chal_timing.c -o chal_timing -O2 -s
 */
#define _GNU_SOURCE
#include <string.h>
#include <time.h>
#include <unistd.h>

static const char SECRET[] = "flag{t1ming}";

static void nap_ms(long ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, 0);
}

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
        nap_ms(6);
    }
    (void)write(1, "wrong\n", 6);
    return 1;
}

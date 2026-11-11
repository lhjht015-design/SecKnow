/* EXP-0022 N1：每次执行都恒定 sleep 5ms，与输入无关。
 *
 * 期望：检测不到"耗时随输入变化"，返回 no_prefix_signal。
 *
 * 编译：
 *   gcc chal_timing_const.c -o chal_timing_const -O2 -s
 */
#define _GNU_SOURCE
#include <time.h>
#include <unistd.h>

int main(void) {
    unsigned char buf[256];
    (void)read(0, buf, sizeof(buf));
    struct timespec ts = {0, 5 * 1000000L};
    nanosleep(&ts, 0);
    (void)write(1, "wrong\n", 6);
    return 1;
}

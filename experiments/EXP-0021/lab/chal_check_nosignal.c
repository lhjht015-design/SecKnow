/* EXP-0021 N5：ELF 校验器，只回 "wrong"，不泄露任何进度。
 *
 * 期望：接口检测返回 no_prefix_signal，拒绝。
 *
 * 编译：
 *   gcc chal_check_nosignal.c -o chal_check_nosignal -O2 -s
 */
#define _GNU_SOURCE
#include <unistd.h>

int main(void) {
    static unsigned char buf[256];
    (void)read(0, buf, sizeof(buf));
    (void)write(1, "wrong\n", 6);
    return 1;
}

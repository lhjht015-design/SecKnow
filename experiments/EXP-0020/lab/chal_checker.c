/* EXP-0020 负样本 N4：看起来是"程序附件"，但不是可逆变换。
 *
 * 无论输入什么都打印 "wrong" 并以 1 退出。黑盒入口能建立，
 * 但 probe_facts 会发现输出长度不随输入长度变化，应如实拒绝。
 *
 * 编译：
 *   gcc chal_checker.c -o chal_checker -O2 -s
 */
#define _GNU_SOURCE
#include <unistd.h>

int main(void) {
    static unsigned char buf[256];
    (void)read(0, buf, sizeof(buf));
    (void)write(1, "wrong\n", 6);
    return 1;
}

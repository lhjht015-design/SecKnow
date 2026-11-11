/* EXP-0021 N6：ELF 程序，输出一个数字但与输入无关（恒为 0）。
 *
 * 用来验证检测器要求"分数随输入变化"，而不是看到数字就当前缀接口。
 *
 * 编译：
 *   gcc chal_check_constant.c -o chal_check_constant -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>

int main(void) {
    static unsigned char buf[256];
    (void)read(0, buf, sizeof(buf));
    printf("0\n");
    fflush(stdout);
    return 0;
}

/* EXP-0021 E3：ELF 校验器，把前缀长度编码在退出码里（0..255）。
 *
 * 编译：
 *   gcc chal_check_exit.c -o chal_check_exit -O2 -s
 */
#define _GNU_SOURCE
#include <string.h>
#include <unistd.h>

static const char SECRET[] = "flag{elf_prefix_exit}";

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
    }
    (void)write(1, "wrong\n", 6);
    return (int)i;
}

/* EXP-0021 E2：ELF 校验器，输出带标签的前缀长度（prefix=N）。
 *
 * 编译：
 *   gcc chal_check_fmt.c -o chal_check_fmt -O2 -s
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const char SECRET[] = "flag{elf_prefix_fmt}";

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
    printf("prefix=%zu\n", i);
    fflush(stdout);
    return 0;
}

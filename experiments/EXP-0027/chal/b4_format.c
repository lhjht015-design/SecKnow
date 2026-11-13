#include <stdio.h>

int main(void) {
    char buf[128];
    if (!fgets(buf, sizeof(buf), stdin)) return 1;
    printf(buf);
    return 0;
}

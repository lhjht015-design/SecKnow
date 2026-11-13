#include <stdio.h>

static void banner(void) {
    puts("welcome");
}

int main(void) {
    char buf[64];
    banner();
    gets(buf);
    return 0;
}

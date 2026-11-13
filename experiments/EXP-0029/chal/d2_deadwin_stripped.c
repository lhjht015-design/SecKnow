#include <stdio.h>
#include <unistd.h>

static void fn_kkrx0y(void) {
    unsigned char data[16] = {0x3c, 0x36, 0x3b, 0x3d, 0x21, 0x69, 0x39, 0x62, 0x3f, 0x6a, 0x6b, 0x68, 0x6b, 0x6b, 0x68, 0x27};
    char out[17];
    for (int i = 0; i < 16; i++) out[i] = data[i] ^ 0x5A;
    out[16] = 0;
    puts(out);
    fflush(stdout);
    _exit(0);
}

int main(void) {
    char buf[64];
    gets(buf);
    return 0;
}

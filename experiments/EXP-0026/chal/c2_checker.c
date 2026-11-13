#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const unsigned char cipher[16] = {0xfc, 0xf8, 0x9e, 0x4c, 0xca, 0x89, 0x8e, 0xa1, 0x58, 0xa9, 0xdd, 0x45, 0x29, 0x19, 0x32, 0x6b};
static const unsigned char perm[16] = {8, 3, 9, 11, 15, 1, 10, 2, 14, 5, 6, 7, 0, 12, 4, 13};
static const unsigned char key[16] = {138, 202, 160, 25, 225, 213, 180, 238, 50, 146, 140, 13, 0, 80, 83, 64};

int main(int argc, char **argv) {
    unsigned char in[16];
    if (read(0, in, 16) != 16) return 1;
    unsigned char t[16];
    for (int j = 0; j < 16; j++) t[j] = in[perm[j]] ^ key[j];
    if (argc > 1 && strcmp(argv[1], "-t") == 0) {
        fwrite(t, 1, 16, stdout);
        return 0;
    }
    if (memcmp(t, cipher, 16) == 0) {
        FILE *f = fopen("/mnt/d/SecKnow/experiments/EXP-0026/chal/flag2.txt", "r");
        char out[256] = {0};
        if (f && fgets(out, sizeof(out), f)) printf("%s", out);
    } else {
        puts("nope");
    }
    return 0;
}

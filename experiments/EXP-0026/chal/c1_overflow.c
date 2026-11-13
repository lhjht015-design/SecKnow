#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void grant_g4nwac(void) {
    FILE *f = fopen("/mnt/d/SecKnow/experiments/EXP-0026/chal/flag.txt", "r");
    char out[256] = {0};
    if (f && fgets(out, sizeof(out), f)) printf("%s", out);
    fflush(stdout);
    _exit(0);
}

int main(void) {
    char buf[64];
    gets(buf);
    puts("done");
    return 0;
}

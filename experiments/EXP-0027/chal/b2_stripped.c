#include <stdio.h>
#include <unistd.h>

void grant_kkrx0y(void) {
    FILE *f = fopen("/mnt/d/SecKnow/experiments/EXP-0027/chal/flag.txt", "r");
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char buf[64];
    char cmd[128];
    gets(buf);
    sprintf(cmd, "echo pinging %s", buf);
    system(cmd);
    return 0;
}

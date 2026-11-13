static long sys3(long n, long a, long b, long c) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret)
                     : "a"(n), "D"(a), "S"(b), "d"(c)
                     : "rcx", "r11", "memory");
    return ret;
}
void win(void) {
    unsigned char data[16] = {0x3c, 0x36, 0x3b, 0x3d, 0x21, 0x69, 0x39, 0x62, 0x3f, 0x6a, 0x6b, 0x68, 0x6b, 0x6b, 0x68, 0x27};
    char out[16];
    for (int i = 0; i < 16; i++) out[i] = data[i] ^ 0x5A;
    sys3(1, 1, (long)out, 16);
    sys3(60, 0, 0, 0);
    __builtin_unreachable();
}
void win_arg(long x) {
    if (x == 0x1234) win();
    sys3(60, 1, 0, 0);
    __builtin_unreachable();
}
__asm__(".global g_pop_rdi\n"
        "g_pop_rdi:\n"
        "  pop %rdi\n"
        "  ret\n");
void vuln(void) {
    char buf[64];
    sys3(0, 0, (long)buf, 256);
}

void _start(void) {
    vuln();
    sys3(60, 0, 0, 0);
}

# EXP-0037 — 第五轮真实赛题

## 汇总

```json
{
  "cases": 3,
  "solved": 2,
  "unsupported": 1,
  "failed": 0
}
```

| 题 | 判定 | 链 | 说明 | 耗时 |
| --- | --- | --- | --- | --- |
| echo | solved | format-string | flag{fla | 3224.9 ms |
| ret2csu | solved | command-injection, pwn-ret2win | ird argument (rdx) must be 0xdeadcafebabebeef

> flag{flag} | 377.1 ms |
| greeting | unsupported | format-string, pwn-ret2win | fmt_buffer_not_on_stack | 5.8 ms |

## 说明

- echo：flag 在栈上，位置参数 %N$lx 重建；
- ret2csu：csu 设置 rdi/rsi/rdx 后跳到 ret2win，解出 flag{flag}；
- greeting：getline 把格式串放在堆上，栈参数扫不到地址，%n 不可达。

## 稳定性

连跑 2 次结果一致：解出 2、不支持 1、失败 0。

## 边界

- echo：flag 读入栈后循环 printf(input)，位置参数 %N$lx 重建；
  4 字节窗口能取到 `flag{...}` 前缀，题面 flag 为 `flag{flag}`。
- ret2csu：用 __libc_csu 的 pop/call 两段设置 rdi/rsi/rdx，
  再跳到 ret2win；rdx=0xdeadcafebabebeef 是题目要求的第三个参数。
  注意不同二进制的 csu 寄存器映射不同（这题 rdx←r15、rsi←r14、edi←r13d），
  helper 必须按映射填值。
- greeting：getline 把格式串放在堆上，栈参数扫不到地址，%n 不可达；
  参考 exploit 依赖其运行环境的栈布局，当前单元不自动覆盖。

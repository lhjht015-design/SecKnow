# EXP-0032 — 真实赛题验证

## 汇总

```json
{
  "cases": 10,
  "solved": 9,
  "refused": 0,
  "unsupported": 1,
  "failed": 0
}
```

## 逐题

| 题 | 位数 | 保护 | 入口 | 链 | 判定 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| ret2text | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 1476.7 ms |
| ret2libc1 | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 1318.7 ms |
| rop | 32 | canary=False pie=False | unsafe_copy | syscall-execve, pwn-ret2win | solved（—） | 1931.7 ms |
| contacts | 32 | canary=True pie=False | format_string | format-string | solved（—） | 5198.2 ms |
| pwn3 | 32 | canary=False pie=False | format_string, unsafe_copy | format-string, pwn-ret2win | unsupported（challenge_specific_logic） | 5.9 ms |
| leakmemory | 32 | canary=False pie=False | format_string | format-string | solved（—） | 31.7 ms |
| overwrite | 32 | canary=False pie=False | format_string | format-string | solved（—） | 33.3 ms |
| ret2libc2 | 32 | canary=False pie=False | unsafe_copy | staged-call, pwn-ret2win | solved（—） | 4074.5 ms |
| ret2libc3 | 32 | canary=False pie=False | unsafe_copy | ret2libc-leak, pwn-ret2win | solved（—） | 2383.0 ms |
| ret2shellcode | 32 | canary=False pie=False | unsafe_copy | staged-printf-leak, pwn-ret2win | solved（—） | 60689.5 ms |

## 说明

- 题目与附件来自 ctf-wiki 收录的真实赛题（NCTU bamboofox / CSAW 2015 / CCTF 2016）；
- 分析用产品代码，执行用适配层（pwntools 常驻 worker）；
- libc/syscall 链的判据是真拿到 shell；fmtstr 写用程序的成功输出判定；
- 题目特有口令逻辑（pwn3）与需要多阶段堆布局的题（contacts）如实标为不支持/拒绝。

## 稳定性

10 道真题连跑 3 次，结果一致：解出 9、不支持 1、失败 0。

## 本轮新补的两类链

- **contacts（格式化字符串 + 栈迁移）**：
  `%31$p` 泄漏 libc 返回地址 → 换算 system；`%6$p%11$p` 取 EBP 指针与堆指针；
  `%6$n` 把 saved EBP 改成堆地址-4；退出时 `leave; ret` 迁移到堆上的
  `[system, 'bbbb', /bin/sh]` 链。canary 与这条链无关。
- **staged-printf-leak（NX 关闭时的替代路径）**：
  `gets(buf2)` 写入 `%s` → `printf(buf2, printf@got)` 泄漏 libc → 二次溢出调 system。
  ret2shellcode 因此被覆盖，但没有走“栈地址 shellcode”的老路。

## 边界与代价

- contacts 的 `%6$n` 需要写入约 1.36 亿字符（堆地址很大），
  实际耗时约 5–6 秒；输出必须被完整读取，否则会卡住。
- libc 返回地址的偏移通过一次校准进程得到（读 `/proc/pid/maps` 求 `leak-base`），
  之后只用泄漏值 + 常量偏移；跨 libc 版本仍需重新校准。
- pwn3 仍不支持：口令由硬编码字符串逐位减一推导，属于题目特有逻辑。
- 真正的“栈地址”泄漏仍没有通用单元；当前是用 libc 泄漏替代 shellcode 路径。

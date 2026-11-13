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
| ret2text | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 1513.6 ms |
| ret2libc1 | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 1314.7 ms |
| rop | 32 | canary=False pie=False | unsafe_copy, unsafe_copy | syscall-execve, pwn-ret2win, syscall-execve, pwn-ret2win | solved（—） | 1916.7 ms |
| contacts | 32 | canary=True pie=False | format_string | format-string | solved（—） | 5574.7 ms |
| pwn3 | 32 | canary=False pie=False | format_string, unsafe_copy, unsafe_copy | format-string, ret2libc-leak, pwn-ret2win, ret2libc-leak, pwn-ret2win | unsupported（challenge_specific_logic） | 6.7 ms |
| leakmemory | 32 | canary=False pie=False | format_string | format-string | solved（—） | 36.7 ms |
| overwrite | 32 | canary=False pie=False | format_string | format-string | solved（—） | 45.8 ms |
| ret2libc2 | 32 | canary=False pie=False | unsafe_copy | staged-call, pwn-ret2win | solved（—） | 4078.1 ms |
| ret2libc3 | 32 | canary=False pie=False | unsafe_copy | ret2libc-leak, pwn-ret2win | solved（—） | 2358.9 ms |
| ret2shellcode | 32 | canary=False pie=False | unsafe_copy | staged-printf-leak, pwn-ret2win | solved（—） | 60674.5 ms |

## 说明

- 题目与附件来自 ctf-wiki 收录的真实赛题（NCTU bamboofox / CSAW 2015 / CCTF 2016）；
- 分析用产品代码，执行用适配层（pwntools 常驻 worker）；
- libc/syscall 链的判据是真拿到 shell；fmtstr 写用程序的成功输出判定；
- 题目特有口令逻辑（pwn3）与需要多阶段堆布局的题（contacts）如实标为不支持/拒绝。

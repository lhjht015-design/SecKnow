# EXP-0032 — 真实赛题验证

## 汇总

```json
{
  "cases": 7,
  "solved": 5,
  "refused": 1,
  "unsupported": 1,
  "failed": 0
}
```

## 逐题

| 题 | 位数 | 保护 | 入口 | 链 | 判定 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| ret2text | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 485.2 ms |
| ret2libc1 | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 326.6 ms |
| rop | 32 | canary=False pie=False | unsafe_copy | syscall-execve, pwn-ret2win | solved（—） | 930.4 ms |
| contacts | 32 | canary=True pie=False | 无 | — | refused（canary_present） | 5.5 ms |
| pwn3 | 32 | canary=False pie=False | format_string, unsafe_copy | format-string, pwn-ret2win | unsupported（challenge_specific_logic） | 5.6 ms |
| leakmemory | 32 | canary=False pie=False | 无 | — | solved（—） | 27.9 ms |
| overwrite | 32 | canary=False pie=False | 无 | — | solved（—） | 29.2 ms |

## 说明

- 题目与附件来自 ctf-wiki 收录的真实赛题（NCTU bamboofox / CSAW 2015 / CCTF 2016）；
- 分析用产品代码，执行用适配层（pwntools 常驻 worker）；
- libc/syscall 链的判据是真拿到 shell；fmtstr 写用程序的成功输出判定；
- 题目特有口令逻辑（pwn3）与需要多阶段堆布局的题（contacts）如实标为不支持/拒绝。

## 稳定性

扩展与新增的真题连跑 3 次，结果一致：解出 5/7、拒绝 1、不支持 1、失败 0。

## 本轮新增的两类链

- **syscall-execve**：没有 system 时，用 `pop eax/ebx/edx:ecx:ebx + int 0x80`
  构造 `execve("/bin/sh",0,0)`；在 bamboofox ret2syscall 上真拿到 shell（偏移 112）。
- **format-string**：偏移探测（标记 0x41414141）→ 泄漏 / 写；
  在 leakmemory 上泄漏 8 个地址，在 overwrite 上写入 `c=16` 并触发
  “modified c.” 输出。

## 边界

- contacts 仍是 canary + 多阶段堆布局，未实现完整链，拒绝。
- pwn3 需要从二进制里反推题目特有的口令变换（`sysbdmin` 每位 -1），
  属于 LAW-0017 的“题目特有逻辑”，标为不支持。
- leakmemory 的“解出”判据是成功泄漏多个地址，不是拿 flag（该题本身没有 flag）。

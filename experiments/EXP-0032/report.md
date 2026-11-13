# EXP-0032 — 真实赛题验证

## 汇总

```json
{
  "cases": 5,
  "solved": 2,
  "refused": 1,
  "unsupported": 2,
  "failed": 0
}
```

## 逐题

| 题 | 位数 | 保护 | 入口 | 链 | 判定 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| ret2text | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 642.3 ms |
| ret2libc1 | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 330.9 ms |
| rop | 32 | canary=False pie=False | unsafe_copy | pwn-ret2win | unsupported（no_chain_for_this_class） | 464.1 ms |
| contacts | 32 | canary=True pie=False | 无 | — | refused（canary_present） | 5.0 ms |
| pwn3 | 32 | canary=False pie=False | format_string, unsafe_copy | format-string, pwn-ret2win | unsupported（needs_leak_chain） | 5.4 ms |

## 说明

- 题目与附件来自 ctf-wiki 收录的真实赛题（NCTU bamboofox / CSAW 2015 / CCTF 2016）；
- 分析用产品代码，执行用适配层（pwntools 常驻 worker），成功判据是真拿到 shell；
- 没有 system/字符串的题（ret2syscall）与需要泄漏的格式化字符串题如实标为不支持。

## 稳定性

同一批真实附件连跑 3 次，结果一致：解出 2/5、拒绝 1、不支持 2、失败 0。

## 来源与边界

- 附件来自 ctf-wiki 收录的真实赛题（NCTU bamboofox / CSAW 2015 / CCTF 2016），
  清单见 `real/SOURCES.md`。
- 真实题暴露的问题不是“执行能力”，而是 ABI 与题目映射：
  32 位反汇编、`.plt` 的 16 字节桩、32 位 `mov [esp], imm` 字面量、
  “system 参数是否常量”的判断，以及“没有 win 函数、要直接调 system”。
- 未解出的 2 题：ret2syscall 需要 execve 系统调用链；格式化字符串需要泄漏链。
  两者都在结果里标为 `unsupported`，没有硬凑。

# EXP-0032 — 真实赛题验证

## 汇总

```json
{
  "cases": 10,
  "solved": 7,
  "refused": 1,
  "unsupported": 2,
  "failed": 0
}
```

## 逐题

| 题 | 位数 | 保护 | 入口 | 链 | 判定 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| ret2text | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 1475.0 ms |
| ret2libc1 | 32 | canary=False pie=False | unsafe_copy | libc-call, pwn-ret2win | solved（—） | 1310.2 ms |
| rop | 32 | canary=False pie=False | unsafe_copy | syscall-execve, pwn-ret2win | solved（—） | 1906.5 ms |
| contacts | 32 | canary=True pie=False | 无 | — | refused（canary_present） | 6.4 ms |
| pwn3 | 32 | canary=False pie=False | format_string, unsafe_copy | format-string, pwn-ret2win | unsupported（challenge_specific_logic） | 6.1 ms |
| leakmemory | 32 | canary=False pie=False | 无 | — | solved（—） | 31.3 ms |
| overwrite | 32 | canary=False pie=False | 无 | — | solved（—） | 34.6 ms |
| ret2libc2 | 32 | canary=False pie=False | unsafe_copy | staged-call, pwn-ret2win | solved（—） | 4033.5 ms |
| ret2libc3 | 32 | canary=False pie=False | unsafe_copy | ret2libc-leak, pwn-ret2win | solved（—） | 2329.0 ms |
| ret2shellcode | 32 | canary=False pie=False | unsafe_copy | ret2shellcode, pwn-ret2win | unsupported（needs_stack_address） | 5.1 ms |

## 说明

- 题目与附件来自 ctf-wiki 收录的真实赛题（NCTU bamboofox / CSAW 2015 / CCTF 2016）；
- 分析用产品代码，执行用适配层（pwntools 常驻 worker）；
- libc/syscall 链的判据是真拿到 shell；fmtstr 写用程序的成功输出判定；
- 题目特有口令逻辑（pwn3）与需要多阶段堆布局的题（contacts）如实标为不支持/拒绝。

## 稳定性

10 道真题连跑 3 次，结果一致：解出 7、拒绝 1、不支持 2、失败 0。

## 工具层（WSL 内常驻 worker）

| 工具 | 版本 | 用途 |
| --- | --- | --- |
| pwntools | 4.15.0 | ELF/ROP/process/shellcraft/fmtstr_payload |
| ROPgadget | 7.7 | gadget 搜索（pwntools 依赖） |
| capstone | 6.0.0a11 | 反汇编 |
| pyelftools | 0.33 | ELF 解析 |
| LibcSearcher | 1.1.5 | libc 候选查找（多候选需二次校验，暂作参考） |

## 本轮新增单元与教训

- **staged-call**（ret2libc2）：先 `gets(buf2)` 写 `/bin/sh`，再 `system(buf2)`。
- **ret2libc-leak**（ret2libc3）：`puts@got` 泄漏 → 计算 system → 二次溢出。
- 教训：
  1. 泄漏与第二阶段必须在同一进程；偏移也不一定相同
     （ret2libc3 泄漏偏移 112、第二阶段偏移 104，因为 main 重入后栈对齐变化）。
  2. stdio 会把多余输入缓冲在父进程里，链里若有第二次 `gets`，
     后续 shell 命令必须等它消费后再发，否则 shell 读不到。
  3. 泄漏候选要按“libc base 页对齐”筛选；直接用首个 `0xf7...` 会选错 GOT 表项。
  4. 题目自带的 libc 与本机运行时不一定是同一个；本地跑用 `elf.libc`，
     远程/跨环境才需要 libc 数据库。

## 边界

- ret2shellcode：NX 关闭，但需要栈地址（ASLR），当前没有栈泄漏单元，标为不支持。
- contacts：canary + 多阶段堆布局；pwn3：题目特有口令变换。

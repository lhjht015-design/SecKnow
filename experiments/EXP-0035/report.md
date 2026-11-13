# EXP-0035 — 第四轮真实赛题（深水区）

## 汇总

```json
{
  "cases": 4,
  "solved": 2,
  "unsupported": 2,
  "failed": 0
}
```

| 题 | 位数 | 判定 | 说明 | 耗时 |
| --- | --- | --- | --- | --- |
| goodluck | 64 | solved | flag{11111111111111111 | 234.3 ms |
| srop-smallest | 64 | solved | 0x7ffe8e6053f0 | 2059.8 ms |
| partial-gets | 64 | unsupported | partial_target_challenge_specific | 4.2 ms |
| dlresolve-gets | 32 | unsupported | fake_reloc_rejected_by_loader | 4.6 ms |

## 说明

- goodluck：flag 读入栈后 printf(input)，位置参数 %N$lx 重建栈取出；题目只读 22 字节，flag 末尾被截断是题面行为；
- srop-smallest：需要多阶段 sigreturn 帧 + 栈泄漏，未实现；
- partial-gets：partial overwrite 的目标是题目特有隐藏地址，未自动化；
- dlresolve-gets：pwntools fake relocation 被 loader 拒绝，参考解法要改写 DT_STRTAB 指向伪 dynstr，未实现。

## SROP 补齐（srop-smallest）

- 二进制只有 `read(0,rsp,0x400); ret`。利用步骤：
  1. 连发三个 `start` 地址，让程序停留在 read；
  2. 发一个 `\xb3` 改低字节，把返回地址改成 `start+3`（跳过 `xor rax`），
     read 返回 1 → rax=1，触发 `write(1, rsp, 0x400)` 泄漏栈；
  3. 从泄漏里取栈地址，构造 read 的 sigreturn 帧（rax=15 触发 rt_sigreturn）；
  4. 第二个帧设置 execve("/bin/sh") 参数，再次触发；
- worker 的 `srop` op 固化这套流程，2.0 秒拿到 shell；
- 产品链：极小 64 位 syscall 二进制给出 `srop`（needs-execution）。

## 稳定性

连跑 2 次结果一致：解出 2（goodluck、srop-smallest）、不支持 2（partial、dlresolve）。

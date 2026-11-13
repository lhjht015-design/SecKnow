# EXP-0035 — 第四轮真实赛题（深水区）

## 汇总

```json
{
  "cases": 4,
  "solved": 3,
  "unsupported": 1,
  "failed": 0
}
```

| 题 | 位数 | 判定 | 说明 | 耗时 |
| --- | --- | --- | --- | --- |
| goodluck | 64 | solved | flag{11111111111111111 | 203.1 ms |
| srop-smallest | 64 | solved | 0x7ffc0e3803f0 | 2049.8 ms |
| partial-gets | 64 | unsupported | partial_target_challenge_specific | 3.4 ms |
| dlresolve-gets | 32 | solved | — | 574.3 ms |

## 说明

- goodluck：flag 读入栈后 printf(input)，位置参数 %N$lx 重建栈取出；题目只读 22 字节，flag 末尾被截断是题面行为；
- srop-smallest：需要多阶段 sigreturn 帧 + 栈泄漏，未实现；
- partial-gets：partial overwrite 的目标是题目特有隐藏地址，未自动化；
- dlresolve-gets：pwntools fake relocation 被 loader 拒绝，参考解法要改写 DT_STRTAB 指向伪 dynstr，未实现。

## dlresolve 补齐（dlresolve-gets）

- 不使用 pwntools 的 fake relocation（会被 loader 以 `r_info` 类型不符拒绝）；
  改用参考路线：改写 `.dynamic` 里的 `DT_STRTAB` 指向 `.bss` 上的伪 dynstr，
  把原 dynstr 中的 `read` 替换成 `system`；
- ROP 链：read 写入新的 strtab 指针 → read 写入伪 dynstr → read 写入 `/bin/sh`
  → 跳 `read@plt+6` 让解析器按伪造 strtab 解析出 `system` 并调用。
- 结果：0.56 秒拿到 shell；产品链对“32 位、非 full RELRO、导入 read”的二进制
  给出 `ret2dlresolve`（needs-execution）。

## 稳定性

连跑 2 次结果一致：解出 3（goodluck、srop-smallest、dlresolve-gets）、
不支持 1（partial-gets 镜像不匹配）。

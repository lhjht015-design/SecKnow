# EXP-0035 — 第四轮真实赛题（深水区）

## 汇总

```json
{
  "cases": 4,
  "solved": 1,
  "unsupported": 3,
  "failed": 0
}
```

| 题 | 位数 | 判定 | 说明 | 耗时 |
| --- | --- | --- | --- | --- |
| goodluck | 64 | solved | flag{11111111111111111 | 218.5 ms |
| srop-smallest | 64 | unsupported | srop_multistage_not_implemented | 1.8 ms |
| partial-gets | 64 | unsupported | partial_target_challenge_specific | 3.3 ms |
| dlresolve-gets | 32 | unsupported | fake_reloc_rejected_by_loader | 4.7 ms |

## 说明

- goodluck：flag 读入栈后 printf(input)，位置参数 %N$lx 重建栈取出；题目只读 22 字节，flag 末尾被截断是题面行为；
- srop-smallest：需要多阶段 sigreturn 帧 + 栈泄漏，未实现；
- partial-gets：partial overwrite 的目标是题目特有隐藏地址，未自动化；
- dlresolve-gets：pwntools fake relocation 被 loader 拒绝，参考解法要改写 DT_STRTAB 指向伪 dynstr，未实现。

## 稳定性

连跑 2 次结果一致：解出 1、不支持 3、失败 0。

## 边界

- goodluck：`%31$p` 风格的位置参数重建栈，直接读到栈上的 flag；
  题目只读 22 字节，flag 末尾的 `}` 被截断是题面行为，不是漏读。
- srop-smallest：`read(0,rsp,0x400); ret`，需要多阶段 sigreturn 帧
  （先泄漏栈地址、再改 rax=15 触发 rt_sigreturn），当前未实现。
- partial-gets：partial overwrite 的落点是题目特有二阶段布局，
  参考解法还依赖隐藏地址；没有通用单元。
- dlresolve-gets：pwntools 的 fake relocation 被 loader 以
  `r_info` 类型不符拒绝；参考解法改写 DT_STRTAB 指向伪 dynstr，未实现。

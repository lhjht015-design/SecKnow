# EXP-0033 — angr 接入真实逆向题

## 汇总

```json
{
  "cases": 4,
  "solved": 2,
  "unsolved": 2,
  "failed": 0
}
```

| 题 | 判定 | 输入 | 耗时 |
| --- | --- | --- | --- |
| ais3_crackme | solved（—） | `ais3{I_tak3_g00d_n0t3s}` | 4358.8 ms |
| defcamp_r100 | solved（—） | `Code_Talkers` | 2008.0 ms |
| csaw_wyvern | unsolved（timeout_or_no_path） | `—` | 29571.6 ms |
| rev100 | unsolved（timeout_or_no_path） | `—` | 62084.3 ms |

## 说明

- 附件来自 angr-doc 示例（真实 CTF 赛题：AIS3 2015 / DefCamp 2014 / CSAW 2015 / Volga CTF 2014）；
- 输入由 angr 求解，随后喂回原程序确认成功输出；
- 产品链选择在没有 pwn 链时给出 symbolic-execution（needs-execution）。

## 稳定性

连跑 2 次结果一致：解出 2、未解 2、失败 0。

## 工具层

| 工具 | 版本 | 用途 |
| --- | --- | --- |
| angr | 9.2.213 | 符号执行求解输入 |
| unicorn | angr 内置 | 加速 full_init_state（wyvern 用） |

## 边界

- ais3_crackme：argv 符号化 + 目标地址 0x400602，约 4.4 秒解出。
- defcamp_r100：stdin 符号化 + 目标地址 0x400844，约 2.2 秒解出。
- csaw_wyvern：已知需要约 15 分钟；本实验给 60 秒上限，标为未解。
- rev100：1.2MB 静态二进制，60 秒内无解。
- angr 只用于逆向题；pwn 题仍走 libc/syscall/fmtstr 链。

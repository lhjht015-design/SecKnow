# EXP-0036 — partial overwrite 真实赛题

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
| pwn2 | solved | partial-overwrite | TAMUctf 2019：PIE 低字节覆盖到 printFlag | 336.0 ms |
| vuln-chat2.0 | solved | command-injection, format-string | TUCTF 2017：用户名后低字节覆盖 | 1225.1 ms |
| partial-gets | unsupported | pwn-ret2win | mirror_mismatch_0x400216_not_in_text | 3.6 ms |

## 说明

- 两道题都是真实 PIE/低字节覆盖：只改返回地址低字节，命中 printFlag；
- 产品链：PIE + 命中后门候选时给出 partial-overwrite（needs-execution）；
- partial-gets 的镜像里 .text 从 0x400420 开始，参考脚本的 0x400216 不存在，属于镜像/版本不一致，标不支持。

## 稳定性

连跑 2 次结果一致：解出 2、不支持 1、失败 0。

## 边界

- pwn2：32 位 PIE，gets 溢出偏移 0x1e，只改写返回地址低字节 0xd8，
  命中 printFlag 输出 `flag{g0ttem_b0yz}`；flag.txt 必须放在二进制同目录。
- vuln-chat2.0：先输入用户名，再在第二段输入里用偏移 0x2b、低字节 0x72
  命中 printFlag（`/bin/cat ./flag.txt`）。
- partial-gets：镜像的 `.text` 从 0x400420 开始，参考脚本的落点 0x400216
  在该二进制中不存在，属于镜像/版本不一致，不能算能力缺口。

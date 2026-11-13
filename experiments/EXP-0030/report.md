# EXP-0030 — 自研单元 vs pwntools 的 A/B

## 汇总

```json
{
  "cases": 4,
  "self_ok": 3,
  "pwntools_ok": 4,
  "self_ms": 927.9,
  "pwntools_ms": 259.1
}
```

## 逐题

| 题 | 自研 | pwntools | 自研错误 | 工具 gadget |
| --- | --- | --- | --- | --- |
| P1 | solved | solved | — | ret |
| P2 | solved | solved | — | pop rdi;ret@0x4010f3 |
| P3 | failed | solved | gadget_missing:pop rsi; ret | pop rsi;pop rbp;ret@0x4010f7 |
| N1 | refused | refused | protection_gate | — |

## 说明

- 两臂使用同一份偏移候选和同一套保护闸门；只比较 gadget 发现与利用构造。
- P3 故意只提供 `pop rsi; pop rbp; ret`；朴素扫描只认 `pop X; ret`，会漏。
- 工具臂的目标地址仍由同一份符号/交叉引用分析给出，避免把“找后门”算进工具能力。

## 稳定性

三组随机种子各跑一遍，结果一致：

| 种子 | 自研 | pwntools |
| --- | --- | --- |
| 1 | 3/4 | 4/4 |
| 2 | 3/4 | 4/4 |
| 3 | 3/4 | 4/4 |

## 边界与解读

- 简单题（P1/P2）两臂打平：gadget 是 `pop X; ret`，自研扫描足够。
- P3 只提供 `pop rsi; pop rbp; ret`：自研扫描只认相邻的 `pop X; ret`，漏检；
  pwntools 的 `ROP.find_gadget` 能找到多指令 gadget，这是工具的实际优势。
- 自研每次尝试都起一个 `wsl -e` 进程（约 140 ms），工具在 WSL 内一个进程里完成
  全部尝试；P1/P2 的耗时差主要来自这里，而不是 gadget 搜索本身。
- 工具的“强”是有限的：目标函数（后门）仍由我们的入口/交叉引用/调用图分析给出；
  pwntools 不会替你做“这题该打哪个函数”的判断。
- 环境：WSL 里 `pip install --user pwntools`（4.15.0），不在离线依赖清单内。

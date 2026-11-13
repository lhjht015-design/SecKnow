# EXP-0027 — 二进制-only 入口发现 + 单元链执行验证

## 汇总

```json
{
  "attack_total": 3,
  "attack_solved": 3,
  "negative_total": 3,
  "negative_correctly_refused": 3,
  "entries_detected": 6,
  "format_entry_detected_no_chain": true,
  "elapsed_ms": 2861.1
}
```

## 逐题

| 题 | 入口 | 判定 | 链 | 耗时 |
| --- | --- | --- | --- | --- |
| B1 | unsafe_copy | solved | entry → elf_info → find_win → find_gadgets → offset_candidates → execute | 1473.9 ms |
| B2 | unsafe_copy | solved | entry → elf_info → find_win → find_gadgets → offset_candidates → execute | 1216.9 ms |
| B3 | command_injection | solved | entry → craft_payload → execute | 157.6 ms |
| B4 | format_string | refused（needs_leak_chain_not_implemented） | entry → chain_lookup | 3.1 ms |
| N1 | 无 | refused（no_entry） |  | 2.9 ms |
| N2 | unsafe_copy | refused（canary_present） | entry → elf_info → protection_gate | 3.4 ms |
| N3 | unsafe_copy | refused（pie_enabled） | entry → elf_info → protection_gate | 3.3 ms |

## 说明

- 入口全部来自二进制：导入表、printf 调用点、字符串交叉引用、ELF 保护；
- B2 用 `-s` 去掉符号，后门只能靠 flag 字符串的引用定位；
- B4 只验证“能识别格式化字符串入口”，利用链不在本实验；
- 负样本必须如实拒绝。

## 稳定性

三组随机种子（后门名、flag 全变）各跑一遍，结果一致：

| 种子 | 攻击题 | 负样本 | 入口识别 |
| --- | --- | --- | --- |
| 1 | 3/3 | 3/3 | 6/7 |
| 2 | 3/3 | 3/3 | 6/7 |
| 3 | 3/3 | 3/3 | 6/7 |

## 边界（如实记录）

- 入口发现是静态特征：导入表、调用点、字符串交叉引用、ELF 保护，不做数据流。
- B2 用 `-s` 去掉符号，后门靠 flag 字符串的 `lea` 交叉引用定位，与 B1 得到同一地址。
- B4 只证明“能识别 printf(buf) 入口”，利用链（泄漏 → 覆写）本实验未实现。
- 后门若不引用任何字符串、也不在符号表里，这套方法会漏；真实二进制需要更多特征。
- 偏移探测是有限候选 + 执行验证，不依赖崩溃转储。

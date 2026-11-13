# EXP-0026 — 粘贴源码 → 入口识别 → 单元链 → 真实执行验证

## 汇总

```json
{
  "positive_total": 3,
  "positive_solved": 3,
  "negative_total": 2,
  "negative_correctly_refused": 2,
  "detected_entries": 4,
  "elapsed_ms": 4639.1
}
```

## 逐题

| 题 | 入口 | 判定 | 链 | 耗时 |
| --- | --- | --- | --- | --- |
| C1 | unsafe_copy | solved | entry → elf_info → find_win → find_gadgets → offset_candidates → execute | 1346.5 ms |
| C2 | blackbox_check | solved | entry → probe_determinism → classify → invert_permutation → verify_execute | 3119.8 ms |
| C3 | command_injection | solved | entry → craft_payload → execute | 169.5 ms |
| N1 | 无 | refused（no_entry） |  | 0.0 ms |
| N2 | unsafe_copy | refused（canary_present） | entry → elf_info → protection_gate | 3.3 ms |

## 说明

- 入口来自源码规则匹配，不读取题面答案；
- 溢出链的偏移由执行验证筛出，不靠猜；
- 黑盒链只做输入输出探测，不读源码里的表；
- 负样本必须如实拒绝。

## 稳定性

换三组随机种子（后门名、置换表、密钥、flag 全变）各跑一遍：

| 种子 | 正样本 | 负样本 |
| --- | --- | --- |
| 1 | 3/3 | 2/2 |
| 2 | 3/3 | 2/2 |
| 3 | 3/3 | 2/2 |

## 边界（如实记录）

- 题目提供可编译源码并产出二进制；真实比赛常见只给二进制或反编译代码。
- 入口识别是“单行规则 + 局部窗口”，不做完整数据流；`fgets` 之后再 `strcpy` 这类跨函数路径会漏检。
- C2 依赖题目提供的变换接口（`-t`）。如果校验器只回成功/失败、不给变换结果，
  16 字节空间无法在不爆破的情况下反解。
- canary / PIE 一律拒绝，不假装能打；这正是 v015–v018 另做泄漏链的原因。

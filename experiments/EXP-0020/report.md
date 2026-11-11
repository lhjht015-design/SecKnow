# EXP-0020 — 按附件自动建立 oracle 入口

入口由附件类型/魔数决定，不依赖题面出现技术词；策略由结构分类选择。

## 汇总

```json
{
  "positive_total": 6,
  "positive_solved": 6,
  "negative_total": 4,
  "negative_correctly_refused": 4,
  "total_probes": 4910,
  "mean_elapsed_ms": 270.0
}
```

## 逐题

| 题 | 类型 | 识别 | 结果 | 路径 / 原因 | 探测数 |
| --- | --- | --- | --- | --- | --- |
| A1 | 正样本（后缀 .src 的 Python 脚本） | python | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 153 |
| A2 | 正样本（改名 .dat，按内容识别） | python | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 178 |
| A3 | 正样本（zip 内的 Python 脚本） | python | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_table → verify_plain | 4450 |
| A4 | 正样本（ELF 变换程序） | elf | ✓ 解出 | probe_facts → classify_structure → invert_bytewise_affine → verify_plain | 117 |
| A5 | 正样本（文本附件：hex 密文） | text | ✓ 解出 | xor_single_byte | 0 |
| A6 | 正样本（文本附件：多层 Base64） | text | ✓ 解出 | base64_chain | 0 |
| N1 | 负样本（随机二进制） | unknown | ✓ 如实拒绝 | attachment_unknown | 0 |
| N2 | 负样本（脚本无 enc/check） | python | ✓ 如实拒绝 | no_enc_or_check | 0 |
| N3 | 负样本（zip 内无可用成员） | unknown | ✓ 如实拒绝 | zip_no_usable_member | 0 |
| N4 | 负样本（ELF 校验器（非变换）） | elf | ✓ 如实拒绝 | not_length_preserving | 12 |
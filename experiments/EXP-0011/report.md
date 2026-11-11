# EXP-0011 — 真实攻击链覆盖率（1156 条）

数据来源：`Des-CTF-Knowledge/CTF大赛WP集合/articles/*.meta.md` 的 `attack_chain`。

> 注意：该仓库提供的是**真实题目的解题链**，不含可执行附件。
> 因此这里测的是「单元库覆盖广度」，不是「能否真的解出」。

## 汇总

```json
{
  "chains": 726,
  "steps": 6795,
  "unclassified_steps": 4191,
  "bespoke_step_rate": 0.617,
  "classified_steps": 2604,
  "covered_families": [
    "archive",
    "binary",
    "classical",
    "crc",
    "encoding",
    "image",
    "pcap",
    "rsa",
    "xor"
  ],
  "chains_fully_covered": 81,
  "chains_partly_covered": 337,
  "chains_uncovered": 231,
  "fully_covered_rate": 0.112,
  "steps_in_covered_families": 1035,
  "steps_in_uncovered_families": 2161,
  "covered_step_share": 0.324
}
```

## 各技术族在真实步骤中出现的次数（按次数排序）

| 技术族 | 步骤数 | 我们是否已覆盖 |
| --- | --- | --- |
| pwn | 635 | ❌ |
| reverse | 304 | ❌ |
| encoding | 287 | ✅ |
| hash | 190 | ❌ |
| binary | 187 | ✅ |
| block_cipher | 174 | ❌ |
| sql | 162 | ❌ |
| xor | 145 | ✅ |
| image | 126 | ✅ |
| rce | 125 | ❌ |
| rsa | 97 | ✅ |
| deserialize | 94 | ❌ |
| pcap | 93 | ✅ |
| crypto_prng | 82 | ❌ |
| archive | 72 | ✅ |
| lattice | 71 | ❌ |
| upload | 70 | ❌ |
| ssti | 38 | ❌ |
| sidechannel | 32 | ❌ |
| lfi | 30 | ❌ |
| web_logic | 27 | ❌ |
| jwt | 26 | ❌ |
| ssrf | 23 | ❌ |
| xss | 22 | ❌ |
| blockchain | 21 | ❌ |
| forensics | 20 | ❌ |
| crc | 17 | ✅ |
| jndi | 15 | ❌ |
| classical | 11 | ✅ |

## 结论怎么读

- **61.7%** 的真实步骤无法归入任何标准技术族，是题目特有的实现细节（自定义 VM、驱动内部结构、模型文件格式、常识检索等）；
- 在能归类的步骤里，**32.4%** 落在我们已实现的单元族；
- 能由现有单元**完整覆盖**的整条攻击链只有 **81/726（11.2%）**。

这不是检索或规划算法的问题，而是**单元库的边界**：
真实 CTF 的主体是主办方自己写的抽象逻辑，标准技术只占少数。
上表里标 ❌ 的族，就是下一步该补的单元方向（按步骤数从多到少排）。
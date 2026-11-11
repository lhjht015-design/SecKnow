# EXP-0002 — 带确定性验证的离线回答

模型：qwen2.5:3b-instruct-q4_K_M

## 判据

- `fabricated_citation_rate`：引用了不存在证据编号的比例（模型在编引用）
- `unsupported_artifact_rate`：声称的关键命令/载荷在**它自己引用的**证据里找不到的比例
- `verdict`：UNSUPPORTED 表示闸门拒绝了输出完整方案

## 汇总

```json
{
  "model": "qwen2.5:3b-instruct-q4_K_M",
  "queries": 1,
  "verdict_counts": {
    "SUPPORTED": 0,
    "PARTIAL": 0,
    "UNSUPPORTED": 1
  },
  "fabricated_citation_rate": 0.0,
  "unsupported_artifact_rate": 0.467,
  "mean_latency_s": 13.9
}
```

## 逐题渲染结果

### PHP 命令执行漏洞怎么利用？给出具体 payload

判定：**UNSUPPORTED**

```
## 结论：证据不足，**不提供**确定性方案

问题：PHP 命令执行漏洞怎么利用？给出具体 payload

- 步骤 1（无据）：[E1]
    - 证据中找不到：`http://example.com/ssrf.php? url=expect://id`, `expect://`, `/ssrf.php`
- 步骤 2（有据）：[E2]
- 步骤 3（有据）：[E3]
- 步骤 4（有据）：[E4]
- 步骤 5（有据）：[E5]
- 步骤 6（无据）：[E6]
    - 证据中找不到：`? file=shell.jpg%ff`
- 步骤 7（无据）：[E1]
    - 证据中找不到：`http://example.com/ssrf.php? url=expect://id`, `expect://`, `/ssrf.php`

**已拒绝输出完整方案**：核心步骤无法在证据中核实。
请补充相关材料后重试，或查阅下列已核实的信息。
```

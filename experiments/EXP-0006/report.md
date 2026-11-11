# EXP-0006 — 分布漂移测试

同一批技术，三档陌生度：L1 自编题 / L2 改名换格式 / L3 只描述行为（不提技术名）。

| 召回策略 | L1 | L2 | L3 | 合计 |
| --- | --- | --- | --- | --- |
| keyword | 5/5 | 1/5 | 0/5 | 6/15 |
| shape | 5/5 | 5/5 | 5/5 | 15/15 |
| hybrid | 5/5 | 5/5 | 5/5 | 15/15 |

## 逐题明细

### router=keyword

| 题 | 召回 | 结果 | 失败原因 |
| --- | --- | --- | --- |
| RSA-L1 | rsa_small_e | ✓ |  |
| RSA-L2 | rsa_small_e | ✓ |  |
| RSA-L3 |  | ✗ |  |
| B64-L1 | base64_chain | ✓ |  |
| B64-L2 |  | ✗ |  |
| B64-L3 |  | ✗ |  |
| XOR-L1 | xor_single_byte,crc32_brute | ✓ |  |
| XOR-L2 |  | ✗ |  |
| XOR-L3 |  | ✗ |  |
| CRC-L1 | crc32_brute | ✓ |  |
| CRC-L2 |  | ✗ |  |
| CRC-L3 |  | ✗ |  |
| CAE-L1 | caesar_brute | ✓ |  |
| CAE-L2 |  | ✗ |  |
| CAE-L3 |  | ✗ |  |

### router=shape

| 题 | 召回 | 结果 | 失败原因 |
| --- | --- | --- | --- |
| RSA-L1 | base64_chain,rsa_small_e | ✓ |  |
| RSA-L2 | base64_chain,rsa_small_e | ✓ |  |
| RSA-L3 | base64_chain,rsa_small_e | ✓ |  |
| B64-L1 | base64_chain | ✓ |  |
| B64-L2 | base64_chain | ✓ |  |
| B64-L3 | base64_chain | ✓ |  |
| XOR-L1 | base64_chain,xor_single_byte | ✓ |  |
| XOR-L2 | base64_chain,xor_single_byte | ✓ |  |
| XOR-L3 | base64_chain,xor_single_byte | ✓ |  |
| CRC-L1 | crc32_brute | ✓ |  |
| CRC-L2 | crc32_brute | ✓ |  |
| CRC-L3 | crc32_brute | ✓ |  |
| CAE-L1 | caesar_brute | ✓ |  |
| CAE-L2 | caesar_brute | ✓ |  |
| CAE-L3 | caesar_brute | ✓ |  |

### router=hybrid

| 题 | 召回 | 结果 | 失败原因 |
| --- | --- | --- | --- |
| RSA-L1 | rsa_small_e,base64_chain | ✓ |  |
| RSA-L2 | rsa_small_e,base64_chain | ✓ |  |
| RSA-L3 | base64_chain,rsa_small_e | ✓ |  |
| B64-L1 | base64_chain | ✓ |  |
| B64-L2 | base64_chain | ✓ |  |
| B64-L3 | base64_chain | ✓ |  |
| XOR-L1 | xor_single_byte,crc32_brute | ✓ |  |
| XOR-L2 | base64_chain,xor_single_byte | ✓ |  |
| XOR-L3 | base64_chain,xor_single_byte | ✓ |  |
| CRC-L1 | crc32_brute | ✓ |  |
| CRC-L2 | crc32_brute | ✓ |  |
| CRC-L3 | crc32_brute | ✓ |  |
| CAE-L1 | caesar_brute | ✓ |  |
| CAE-L2 | caesar_brute | ✓ |  |
| CAE-L3 | caesar_brute | ✓ |  |

# EXP-0003 — 基于 meta WP 真值的检索质量评测

生成时间：2026-11-11 10:34:19
样本：120 题  Top-K：8  模式：competition

## 判据

- `tag_recall_at_k`：检索到的文本覆盖了真值 tags 的比例。**这是「有没有找对技术」的主指标。**
- `similar_wp_rate`：Top-K 里是否命中「同类型且标签重叠」的**其它** WP，即「找相似题」能力。
- `similar_wp_rate_conditional`：只在**确实存在**近似题的子集上算这个比例。
  单独给出原始值是因为实测 67% 的题在全部 1156 篇里根本没有近似题——
  那种情况下指标低是语料的性质，不是检索的失败，两者必须分开看。
- `top_k_composition`：Top-K 各命中位来自 meta WP / 大部头 PDF / 其它 的比例，
  用来判断检索是否被长文档淹没。
- `vuln_type_hit_rate`：Top-K 是否出现同漏洞类型的 WP。
- `self_leak_rate`：是否检索到题目自己的 meta 文件。对真人用户这有用，对评测是泄漏，因此单列。

## 汇总

```json
{
  "queries": 120,
  "top_k": 8,
  "mode": "competition",
  "tag_recall_at_k": 0.5925,
  "tag_recall_excl_self": 0.1602,
  "self_leak_rate": 0.825,
  "similar_wp_rate": 0.0667,
  "similar_wp_ground_truth_exists_rate": 0.3333,
  "similar_wp_rate_conditional": 0.2,
  "similar_wp_subset_size": 40,
  "vuln_type_hit_rate": 0.1333,
  "elapsed_s": 422.8,
  "top_k_composition": {
    "meta_WP": 0.83,
    "book_PDF": 0.086,
    "other": 0.084
  },
  "mean_meta_wp_per_query": 5.0083,
  "queries_with_zero_wp": 0,
  "most_missed_tags": [
    [
      "puppeteer",
      2
    ],
    [
      "XXTEA",
      2
    ],
    [
      "车联网-ransomware",
      1
    ],
    [
      "Sepolia-Etherscan",
      1
    ],
    [
      "calldata-extract",
      1
    ],
    [
      "XOR-decrypt-3-9",
      1
    ],
    [
      "RSA-public-key-PEM",
      1
    ],
    [
      "encrypted-seed",
      1
    ],
    [
      "java.perform",
      1
    ],
    [
      "caesar-cipher",
      1
    ],
    [
      "get_random",
      1
    ],
    [
      "seccomp只允许ptrace/read/wait4",
      1
    ],
    [
      "ptrace注入",
      1
    ],
    [
      "shellcode注入1号进程",
      1
    ],
    [
      "PTRACE_POKETEXT",
      1
    ],
    [
      "PTRACE_KILL sleep",
      1
    ],
    [
      "mmap rwx 0x1000",
      1
    ],
    [
      "libc可执行段",
      1
    ],
    [
      "pem",
      1
    ],
    [
      "openssl",
      1
    ]
  ]
}
```

## 分类型拆解

| 类型 | 题数 | tag_recall | similar_wp | self_leak |
| --- | --- | --- | --- | --- |
| misc | 5 | 0.8 | 0.4 | 1.0 |
| web | 5 | 0.6883 | 0.2 | 1.0 |
| reverse | 5 | 0.3476 | 0.2 | 0.8 |
| pwn | 5 | 0.0844 | 0.0 | 0.8 |
| crypto | 5 | 0.6 | 0.0 | 0.8 |
| forensic | 5 | 1.0 | 0.2 | 1.0 |
| heap | 5 | 0.4 | 0.2 | 0.6 |
| rop | 5 | 1.0 | 0.2 | 0.8 |
| rce | 5 | 0.4939 | 0.0 | 0.6 |
| deserialize | 5 | 0.7769 | 0.0 | 1.0 |
| sqli | 5 | 0.8857 | 0.0 | 1.0 |
| stego | 5 | 0.2615 | 0.0 | 0.6 |
| auth | 5 | 0.552 | 0.0 | 0.8 |
| lfi | 5 | 0.65 | 0.0 | 1.0 |
| lattice | 5 | 0.3352 | 0.0 | 1.0 |
| ssrf | 5 | 0.4417 | 0.0 | 0.4 |
| block | 5 | 0.4833 | 0.0 | 0.8 |
| ssti | 5 | 0.3956 | 0.2 | 0.6 |
| fmt | 4 | 0.5278 | 0.0 | 0.5 |
| xss | 4 | 0.1339 | 0.0 | 0.5 |
| logic | 4 | 0.9722 | 0.0 | 1.0 |
| ret2libc | 3 | 0.6905 | 0.0 | 1.0 |
| ecdsa | 2 | 0.5625 | 0.0 | 1.0 |
| xxe | 2 | 1.0 | 0.0 | 1.0 |
| srop | 2 | 0.9091 | 0.0 | 1.0 |
| jwt | 2 | 1.0 | 0.0 | 1.0 |
| hash | 1 | 0.2857 | 0.0 | 1.0 |
| race | 1 | 1.0 | 0.0 | 1.0 |
| vm | 1 | 0.0 | 0.0 | 1.0 |
| stack | 1 | 0.2857 | 0.0 | 1.0 |
| upload | 1 | 1.0 | 0.0 | 1.0 |
| uaf | 1 | 1.0 | 0.0 | 1.0 |
| kernel | 1 | 1.0 | 0.0 | 1.0 |

## 检索最差的 15 题（优先排查对象）

| 题目 | 类型 | tag_recall | 漏掉的 tags |
| --- | --- | --- | --- |
| 【WriteUP·上篇】2024WIDC 世界智能驾驶挑战赛"天融信杯"信息安全攻防赛 | misc_unknown | 0.0 | 车联网-ransomware, Sepolia-Etherscan, calldata-extract, XOR-decrypt-3-9, RSA-public-key-PEM, encrypted-seed |
| 第二届"长城杯"信息安全铁人三项赛(防护赛)总决赛-Pwn方向部分题解 | heap_exploit | 0.0 | Pwn-Heap, libc2.31, Protobuf, 菜单题, free_hook覆盖 |
| 2024 国城杯 writeup by Mini-Venom | stego_image/stego_traffic/crypto_unknown/misc_unknown/reverse | 0.0 | .264 视频隐写, np.random seed 爆破, DataMatrix 二维码, PGP wordlist 编码, shiro 反序列化, AES-ECB |
| 2024 网鼎杯 半决赛 & 决赛 | lattice/stream_cipher/crypto_rsa/ecdsa/finite_field | 0.0 | 已知 p 高位 Coppersmith, 多模数共享素数, GF(p^n) 有限域同构, LFSR NFSR NOISE 序列密码, noekeon 或同等密, 全部爆破 7 比特 |
| 数据安全大赛 2024 数信杯 东部赛区 | fmt_string/pwn_unknown/stego_image/block_cipher/stream_cipher | 0.0 | 非栈上格式化字符串, RBP 多级指针改返回地址, %hn 写 short, one_gadget, base64 循环移位, RC4 + XOR 双加密 |
| 2025 黄鹤杯 PWN Virtual_Vehicle（VM 逆向+VM PWN+seccom | vm/shellcode/rop/seccomp | 0.0 | 黄鹤杯 2025 PWN Virtual_Vehicle, 自定义 VM 逆向+VM PWN, vtable 8 个函数指针 (push_from_reg/push_from_imm/pop_to_reg/add_reg2_reg3_to_reg1/sub/mul/load/print), heap 0x1000 ope 字节码, hashtable mmap RWX 0x1000, seccomp-tools 沙箱 |
| 赛题全解|第四届 RWCTF 体验赛 WriteUp | pwn_unknown | 0.0 | Java-Processor, Annotation-Processing, SNMP-aarch64, kernel-UAF, Docker-escape, lvm-vm |
| 2024 能源网络安全大赛预赛【金盾检测】 | stego_traffic/crypto_rsa/reverse/misc_unknown/web_unknown/deserialize | 0.0 | GOOSE 协议提取, item4 < 128 ASCII, DCS 双网卡 IP, Modbus 流量, FINS 协议响应构造, RSA 小模数 |
| 2026 DesCTF wp（Misc 全 - bkcrack 已知明文攻击 + zigzag  | block_cipher/stego_image/ai/pwn_unknown/crypto_unknown/forensic_disk | 0.0 | DesCTF 2026 Misc, bkcrack 已知明文攻击 challenge.png 89504E47 IHDR, 7z 头 37 7A BC AF 27 1C, 2D zigzag 还原 PIL, torch 加载 model.pth 提 eval_cache embedding 反演, 红外遥控器 cmd:00 15=OK/16=U/17=D/18=R/19=L 6x6 网格 ABCDEF/GHIJKL/MNOPQR/STUVWX/YZ1234/567890 |
| 2024 第四届"网鼎杯"青龙组 writeup | xss/ret2libc/reverse/block_cipher/web_unknown/pwn_unknown | 0.0 | 网鼎杯, 青龙组, 存储型XSS, admin-bot, fetch-exfil, 栈迁移 |
| 2024 第四届"网鼎杯"朱雀组 writeup（SMC XOR 0x42 + base24 + | reverse/misc_unknown/crypto_unknown | 0.0 | Re1 SMC XOR 0x42, idapython PatchByte, base24 自定义码表 24 字符, decode_base24 libnum.n2s, Re2 VMP 壳, svchost.exe 字符串大法 |
| idek 2022: Pwn && Reverse Writeup by r3kapig | pwn_unknown | 0.0 | fmt-string, canary-leak, stack-leak, ret2dlresolve, fmt-arg-overlap, glibc-2.31 |
| web选手入门pwn(31.5) 强网杯bph stack-leak 变种 | heap_exploit | 0.0 | glibc-2.39, _IO_FILE, _IO_write_base-leak, environ, stack-read, House-of-Apple |
| 祥云杯 2022/Misc super_electric | stego_traffic/reverse/block_cipher | 0.0 | MMS 协议, 工控流量分析, tshark 提取, MZ 头识别, 反调试, 自解压壳 |
| Realworld CTF 2023 The_cult_of_8_bit 详解 | xss | 0.0 | jsonp, same-origin-method-execution, iframe-focus, post-id-leak, csrf, char-by-char-extract |
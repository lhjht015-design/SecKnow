# EXP-0034 — 第三轮真实赛题回归

## 汇总

```json
{
  "cases": 3,
  "solved": 1,
  "unsupported": 2,
  "failed": 0
}
```

| 题 | 位数 | 判据 | 判定 | 耗时 |
| --- | --- | --- | --- | --- |
| blind | 64 | flag{this_is_flag} | solved | 387.0 ms |
| b0verfl0w | 32 | fgets_limit_blocks_jmp_esp | unsupported | 4.7 ms |
| pwn100 | 64 | ret2csu_not_implemented | unsupported | 3.5 ms |

## 说明

- blind：位置参数 `%N$lx` 重建栈内容，解出 `flag{this_is_flag}`；
- b0verfl0w：fgets 最多 49 字节，jmp-esp shellcode 放不下；需要 fgets 版 staged 泄漏，未实现；
- pwn100：需要 ret2csu 链，当前单元库未实现，如实标记。

## 稳定性

连跑 2 次结果一致：解出 1、不支持 2、失败 0。

## 边界

- blind：64 位盲格式化字符串，用位置参数 `%N$lx` 依次取出栈值、
  按小端拼回字节流，直接找到 `flag{this_is_flag}`。
- b0verfl0w：`fgets` 最多 49 字节，`jmp esp` + 栈上 shellcode 放不下；
  可行替代是 fgets 版 staged printf 泄漏 libc，当前单元只实现了 gets 版本。
- pwn100：64 位 stripped，需要 ret2csu 链才能先泄漏 libc；该单元未实现。

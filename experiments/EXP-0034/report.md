# EXP-0034 — 第三轮真实赛题回归

## 汇总

```json
{
  "cases": 3,
  "solved": 3,
  "unsupported": 0,
  "failed": 0
}
```

| 题 | 位数 | 判据 | 判定 | 耗时 |
| --- | --- | --- | --- | --- |
| blind | 64 | flag{this_is_flag} | solved | 191.1 ms |
| b0verfl0w | 32 | — | solved | 372.9 ms |
| pwn100 | 64 | — | solved | 20455.5 ms |

## 说明

- blind：位置参数 `%N$lx` 重建栈内容，解出 `flag{this_is_flag}`；
- b0verfl0w：fgets 最多 49 字节，jmp-esp shellcode 放不下；需要 fgets 版 staged 泄漏，未实现；
- pwn100：需要 ret2csu 链，当前单元库未实现，如实标记。

## 稳定性

连跑 2 次结果一致：解出 3、不支持 0、失败 0。

## 本轮补齐的两条链

- **ret2csu（pwn100）**：自动发现 `pop rbx/rbp/r12-r15; ret` 与
  `mov rdx,..; mov rsi,..; mov edi,..; call [r12+rbx*8]` 两段；
  先用 csu 调 `puts(puts@got)` 泄漏 libc，再用 libc 的 `pop rdi; ret` 调 `system`。
  细节：题目按字节读满 200 次，第一阶段必须正好喂 200 字节，否则残留字节会让
  第二阶段错位；`puts` 只打印到 NUL，泄漏值要用 4–8 字节窗口 + 页对齐重建。
- **fgets 版泄漏（b0verfl0w）**：32 位 `fgets(50)` 溢出无法容纳 jmp-esp shellcode，
  改为 `puts(puts@got)` 泄漏后返回 `vul` 重新输入，第二阶段直接
  `system('/bin/sh')`；偏移 36，同步串是 `What's your name?`。

## 回归发现

- 给 ret2csu 加调试输出时，误改了 staged-printf 的共享片段，导致 ret2shellcode
  在 v032 回归里短暂掉到 8/10；定位后恢复为 9/10。
  教训：每个 op 的 `open_and_leak` 是独立副本，改动必须带上下文。

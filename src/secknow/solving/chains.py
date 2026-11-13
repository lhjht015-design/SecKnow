"""把入口映射成候选验证链；产品内只做静态分析，不执行目标程序。"""

from __future__ import annotations

from dataclasses import dataclass

from secknow.solving.binary import BinaryProfile


@dataclass(frozen=True)
class ChainStep:
    unit: str
    title: str
    status: str  # ok / blocked / requires-execution / unknown
    detail: str


@dataclass(frozen=True)
class Chain:
    chain_id: str
    title: str
    category: str
    status: str  # static / needs-binary / needs-execution / needs-leak-chain
    #             needs-oracle / refused / unsupported
    reason: str
    steps: tuple[ChainStep, ...]


def build_chains(
    categories: list[str],
    profile: BinaryProfile | None,
) -> list[Chain]:
    chains: list[Chain] = []
    for category in categories:
        if category == "unsafe_copy" and profile is not None:
            libc = _libc_call(profile)
            if libc is not None:
                chains.append(libc)
        if category == "command_injection":
            chains.append(_command_injection(profile))
        elif category == "format_string":
            chains.append(_format_string(profile))
        elif category == "unsafe_copy":
            chains.append(_overflow(profile))
        elif category == "blackbox_check":
            chains.append(_blackbox(profile))
    return chains


def _protection_step(profile: BinaryProfile | None) -> ChainStep:
    if profile is None:
        return ChainStep("protection_gate", "保护检查", "blocked",
                         "需要二进制附件：canary / PIE / NX / RELRO")
    protections = (
        f"PIE={'开' if profile.is_pie else '关'}，"
        f"canary={'开' if profile.has_canary else '关'}，"
        f"NX={'开' if profile.has_nx else '关'}，RELRO={profile.relro}"
    )
    return ChainStep("protection_gate", "保护检查", "ok", protections)


def _libc_call(profile: BinaryProfile) -> Chain | None:
    plt = profile.plt_names
    shell = profile.strings.get("/bin/sh") or profile.strings.get("/bin/bash")
    if "system" not in plt or not shell:
        return None
    return Chain(
        "libc-call", "调用 system('/bin/sh')", "unsafe_copy", "needs-execution",
        "导入 system 且存在 /bin/sh 字符串；调用与偏移需执行验证",
        (
            ChainStep("entry", "入口确认", "ok", "导入表中存在无边界拷贝函数"),
            _protection_step(profile),
            ChainStep("string", "字符串定位", "ok",
                      f"/bin/sh @ {hex(shell[0])}"),
            ChainStep("system_call", "调用 system", "requires-execution",
                      f"system @ {hex(plt['system'])}；按 ABI 构造返回地址与参数"),
            ChainStep("execute", "执行验证", "requires-execution",
                      "产品内不执行目标二进制；请在靶场环境复核"),
        ))


def _overflow(profile: BinaryProfile | None) -> Chain:
    if profile is None:
        return Chain(
            "pwn-ret2win", "栈溢出 → ret2win", "unsafe_copy", "needs-binary",
            "需要目标二进制才能检查保护、定位后门与估算偏移",
            (
                ChainStep("entry", "入口确认", "ok", "源码中出现无边界拷贝"),
                _protection_step(None),
                ChainStep("find_win", "后门定位", "requires-execution",
                          "在二进制中按符号或字符串交叉引用定位"),
                ChainStep("probe_offset", "偏移探测", "requires-execution",
                          "候选偏移逐个执行，以崩溃/输出现象验证"),
                ChainStep("execute", "执行验证", "requires-execution",
                          "重定向 stdin 投放 payload，确认拿到 flag"),
            ))

    if profile.has_canary or profile.is_pie:
        reason = "canary_present" if profile.has_canary else "pie_enabled"
        label = "存在 canary，需先泄漏" if profile.has_canary else "PIE 开启，地址随机化"
        return Chain(
            "pwn-ret2win", "栈溢出 → ret2win", "unsafe_copy", "refused", reason,
            (
                ChainStep("entry", "入口确认", "ok", "导入表中存在无边界拷贝函数"),
                _protection_step(profile),
                ChainStep("protection_gate", "保护闸门", "blocked", label),
            ))

    candidates = profile.win_candidates
    win = candidates[0] if candidates else None
    if win is None:
        return Chain(
            "pwn-ret2win", "栈溢出 → ret2win", "unsafe_copy", "blocked",
            "win_not_found",
            (
                ChainStep("entry", "入口确认", "ok", "导入表中存在无边界拷贝函数"),
                _protection_step(profile),
                ChainStep("find_win", "后门定位", "blocked",
                          "未找到引用 flag 字符串或名称可疑的函数"),
            ))
    symbol = win.get("symbol") or "未命名函数"
    hints = profile.stack_hints or []
    steps = [
        ChainStep("entry", "入口确认", "ok", "导入表中存在无边界拷贝函数"),
        _protection_step(profile),
    ]
    graph = profile.call_graph
    if graph:
        steps.append(ChainStep(
            "call_graph", "调用图", "ok",
            f"{graph.get('functions', 0)} 个函数 / {graph.get('edges', 0)} 条边；"
            f"主流程不可达 {graph.get('dead', 0)} 个，其中会输出 {graph.get('dead_output', 0)} 个"))
    steps.extend([
        ChainStep("find_win", "后门定位", "ok",
                  f"{symbol} @ {hex(win['address'])}（{win['source']}）"),
        ChainStep("offset_candidates", "偏移候选", "requires-execution",
                  "静态提示：" + (", ".join(str(h) for h in hints) or "无")),
        ChainStep("execute", "执行验证", "requires-execution",
                  "产品内不执行目标二进制；请在靶场环境复核"),
    ])
    reason = "静态条件满足；偏移与利用需在目标环境执行验证"
    if win.get("source") == "call-graph-dead-output":
        reason = "后门来自调用图（主流程不可达且会输出）；偏移与利用需执行验证"
    return Chain("pwn-ret2win", "栈溢出 → ret2win", "unsafe_copy", "needs-execution",
                 reason, tuple(steps))


def _command_injection(profile: BinaryProfile | None) -> Chain:
    detail = "源码中出现 os.system / shell=True 拼接"
    if profile is not None:
        shells = sorted(profile.imports & {"system", "popen", "execve", "execv", "execvp"})
        detail = "导入 " + ", ".join(shells) if shells else detail
    return Chain(
        "command-injection", "命令注入 → 读取 flag", "command_injection",
        "needs-execution", "payload 需在目标环境执行确认",
        (
            ChainStep("entry", "入口确认", "ok", detail),
            ChainStep("craft_payload", "构造 payload", "ok",
                      "候选：`127.0.0.1; cat flag.txt`"),
            ChainStep("execute", "执行验证", "requires-execution",
                      "产品内不执行目标程序；请在靶场环境复核"),
        ))


def _format_string(profile: BinaryProfile | None) -> Chain:
    detail = "printf 的格式参数不是字面量"
    if profile is not None and profile.printf_entry is not None:
        detail = f"非字面量 printf 调用 @ {hex(profile.printf_entry)}"
    return Chain(
        "format-string", "格式化字符串 → 泄漏 / 覆写", "format_string",
        "needs-leak-chain", "需要泄漏链（栈偏移 → 读 → 写），当前单元库未实现",
        (
            ChainStep("entry", "入口确认", "ok", detail),
            ChainStep("leak_chain", "泄漏链", "unknown", "未实现"),
            ChainStep("overwrite", "覆写 / 执行", "unknown", "未实现"),
        ))


def _blackbox(profile: BinaryProfile | None) -> Chain:
    return Chain(
        "blackbox-invert", "黑盒变换 → 探测 / 反解 / 复核", "blackbox_check",
        "needs-oracle", "需要可交互的变换接口（或把目标接到本地 oracle）",
        (
            ChainStep("entry", "入口确认", "ok", "源码出现逐字节变换 + memcmp 校验"),
            ChainStep("probe", "探测", "requires-execution", "确定性 / 长度 / 依赖矩阵"),
            ChainStep("invert", "反解", "requires-execution", "按结构选择反解单元"),
            ChainStep("verify", "复核", "requires-execution", "重新加密等于密文"),
        ))

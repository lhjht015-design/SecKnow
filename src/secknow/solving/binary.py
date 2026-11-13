"""只读 ELF 分析：保护、导入、调用点、字符串交叉引用。不执行目标程序。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SHELLS = {"system", "popen", "execl", "execlp", "execv", "execvp"}
_UNSAFE_READS = {"gets", "strcpy", "strcat", "sprintf", "scanf"}
_SYMBOL_HINTS = re.compile(r"(win|flag|grant|backdoor|secret)", re.I)


def dependencies_available() -> bool:
    try:
        import capstone  # noqa: F401
        from elftools.elf.elffile import ELFFile  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class BinaryFinding:
    category: str
    technique: str
    evidence: str
    address: int | None = None


@dataclass
class BinaryProfile:
    path: Path
    is_pie: bool
    has_canary: bool
    has_nx: bool
    relro: str
    imports: set[str]
    symbols: dict[str, int]
    plt: dict[int, str]
    bits: int = 64
    strings: dict[str, list[int]] = field(default_factory=dict)
    flag_string: str | None = None
    flag_string_addr: int | None = None
    win_candidates: list[dict[str, Any]] = field(default_factory=list)
    stack_hints: list[int] = field(default_factory=list)
    instruction_count: int = 0
    printf_entry: int | None = None
    command_input_entry: int | None = None
    call_graph: dict[str, Any] = field(default_factory=dict)

    @property
    def plt_names(self) -> dict[str, int]:
        return {name: address for address, name in self.plt.items()}

    def findings(self) -> list[BinaryFinding]:
        found: list[BinaryFinding] = []
        shells = sorted(self.imports & _SHELLS)
        reads = self.imports & (_UNSAFE_READS | {"fgets", "read"})
        if shells and reads and self.command_input_entry is not None:
            found.append(BinaryFinding(
                "command_injection", "命令注入（system 家族 + 非常量参数）",
                f"导入 {', '.join(shells)}；调用点 {hex(self.command_input_entry)}"))
        if self.printf_entry is not None:
            found.append(BinaryFinding(
                "format_string", "格式化字符串（非字面量格式参数）",
                f"call printf at {hex(self.printf_entry)}", self.printf_entry))
        unsafe = sorted(self.imports & _UNSAFE_READS)
        if unsafe:
            found.append(BinaryFinding(
                "unsafe_copy", "栈溢出 / 无边界拷贝", f"导入 {', '.join(unsafe)}"))
        return found


def _disassemble(profile_path: Path, bits: int = 64) -> list[Any]:
    from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
    from elftools.elf.elffile import ELFFile

    with open(profile_path, "rb") as handle:
        elf = ELFFile(handle)
        text = elf.get_section_by_name(".text")
        if text is None:
            return []
        engine = Cs(CS_ARCH_X86, CS_MODE_32 if bits == 32 else CS_MODE_64)
        return list(engine.disasm(text.data(), text["sh_addr"]))


def _function_starts(instructions: list[Any]) -> list[int]:
    starts = set()
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic == "endbr64":
            starts.add(instruction.address)
        elif instruction.mnemonic == "push" and instruction.op_str in {"rbp", "ebp"}:
            if index > 0 and instructions[index - 1].mnemonic == "endbr64":
                continue
            starts.add(instruction.address)
    return sorted(starts)


def _xref_function(instructions: list[Any], starts: list[int], target: int) -> int | None:
    for instruction in instructions:
        if instruction.mnemonic != "lea":
            continue
        parts = [item.strip() for item in instruction.op_str.split(",")]
        if len(parts) != 2 or "rip" not in parts[1]:
            continue
        match = re.search(r"rip ([+-]) (0x[0-9a-fA-F]+)", parts[1])
        if not match:
            continue
        offset = int(match.group(2), 16)
        if match.group(1) == "-":
            offset = -offset
        if instruction.address + instruction.size + offset != target:
            continue
        enclosing = None
        for start in starts:
            if start <= instruction.address:
                enclosing = start
            else:
                break
        return enclosing
    return None


def _sink_input_entry(instructions: list[Any], plt: dict[int, str],
                      rodata_addr: int, rodata_len: int, bits: int) -> int | None:
    """找参数不是常量的 system/popen 调用点；常量串（如 system("/bin/sh")）不算注入。"""
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic != "call" or not instruction.op_str.startswith("0x"):
            continue
        try:
            target = int(instruction.op_str, 16)
        except ValueError:
            continue
        if plt.get(target) not in {"system", "popen"}:
            continue
        constant = False
        for back in range(1, 9):
            if index - back < 0:
                break
            previous = instructions[index - back]
            if bits == 32:
                if previous.mnemonic == "push" and previous.op_str.startswith("0x"):
                    constant = True
                elif previous.mnemonic == "mov":
                    parts = [item.strip() for item in previous.op_str.split(",")]
                    if len(parts) == 2 and "[esp]" in parts[0] and parts[1].startswith("0x"):
                        constant = True
            else:
                if (previous.mnemonic == "lea" and "rdi" in previous.op_str
                        and "rip" in previous.op_str) or (previous.mnemonic == "mov"
                      and previous.op_str.startswith("rdi,")
                      and previous.op_str.split(",", 1)[1].strip().startswith("0x")):
                    constant = True
            if constant:
                break
        if not constant:
            return int(instruction.address)
    return None


def _in_rodata(address: int, rodata_addr: int, size: int) -> bool:
    return 0 <= address - rodata_addr < size


def _printf_entry(instructions: list[Any], plt: dict[int, str],
                  rodata_addr: int, rodata_data: bytes, bits: int = 64) -> int | None:
    size = len(rodata_data)
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic != "call" or not instruction.op_str.startswith("0x"):
            continue
        try:
            target = int(instruction.op_str, 16)
        except ValueError:
            continue
        if plt.get(target) != "printf":
            continue
        literal = False
        found_source = False
        for back in range(1, 12):
            if index - back < 0:
                break
            previous = instructions[index - back]
            if previous.mnemonic == "call":
                break
            if bits == 32:
                if previous.mnemonic == "push":
                    found_source = True
                    if previous.op_str.startswith("0x"):
                        try:
                            literal = _in_rodata(int(previous.op_str, 16),
                                                 rodata_addr, size)
                        except ValueError:
                            literal = False
                    break
                if previous.mnemonic == "mov":
                    parts = [item.strip() for item in previous.op_str.split(",")]
                    if len(parts) == 2 and "[esp]" in parts[0]:
                        found_source = True
                        if parts[1].startswith("0x"):
                            try:
                                literal = _in_rodata(int(parts[1], 16),
                                                     rodata_addr, size)
                            except ValueError:
                                literal = False
                        break
            else:
                if previous.mnemonic == "lea" and "rdi" in previous.op_str.split(",", 1)[0]:
                    found_source = True
                    match = re.search(r"rip ([+-]) (0x[0-9a-fA-F]+)", previous.op_str)
                    if match:
                        offset = int(match.group(2), 16)
                        if match.group(1) == "-":
                            offset = -offset
                        literal = _in_rodata(previous.address + previous.size + offset,
                                             rodata_addr, size)
                    break
                if previous.mnemonic == "mov" and previous.op_str.split(",", 1)[0].strip() \
                        in {"rdi", "edi"}:
                    found_source = True
                    source = previous.op_str.split(",", 1)[1].strip()
                    if source.startswith("0x"):
                        try:
                            literal = _in_rodata(int(source, 16), rodata_addr, size)
                        except ValueError:
                            literal = False
                    break
        if not found_source or not literal:
            return int(instruction.address)
    return None


def _enclosing(starts: list[int], address: int) -> int | None:
    found = None
    for start in starts:
        if start <= address:
            found = start
        else:
            break
    return found


_OUTPUT_FUNCTIONS = {"puts", "printf", "fwrite", "write", "system", "fprintf"}


def _call_graph(instructions: list[Any], starts: list[int],
                plt: dict[int, str]) -> tuple[dict[int, set[int]], dict[int, set[str]]]:
    """函数级调用图；同时记录每个函数调用了哪些库函数。"""
    known = set(starts)
    edges: dict[int, set[int]] = {start: set() for start in starts}
    library_calls: dict[int, set[str]] = {start: set() for start in starts}
    for instruction in instructions:
        if instruction.mnemonic != "call" or not instruction.op_str.startswith("0x"):
            continue
        try:
            target = int(instruction.op_str, 16)
        except ValueError:
            continue
        source = _enclosing(starts, instruction.address)
        if source is None:
            continue
        if target in known:
            edges.setdefault(source, set()).add(target)
        elif target in plt:
            library_calls.setdefault(source, set()).add(plt[target])
    return edges, library_calls


def _reachable(roots: set[int], edges: dict[int, set[int]]) -> set[int]:
    seen = set(roots)
    queue = list(roots)
    while queue:
        node = queue.pop()
        for target in edges.get(node, ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def _main_address(instructions: list[Any], starts: list[int], plt: dict[int, str],
                  entry: int) -> int | None:
    """没有符号表时，从 `_start` 里找传给 __libc_start_main 的函数地址。"""
    known = set(starts)
    entry_function = _enclosing(starts, entry)
    target_index = None
    for index, instruction in enumerate(instructions):
        if _enclosing(starts, instruction.address) != entry_function:
            continue
        if instruction.mnemonic != "call" or not instruction.op_str.startswith("0x"):
            continue
        try:
            target = int(instruction.op_str, 16)
        except ValueError:
            continue
        if plt.get(target) in {"__libc_start_main", "__libc_start_call_main"}:
            target_index = index
            break
    if target_index is None:
        return None
    for back in range(1, 16):
        index = target_index - back
        if index < 0:
            break
        instruction = instructions[index]
        candidate = None
        if instruction.mnemonic == "lea" and "rip" in instruction.op_str:
            match = re.search(r"rip ([+-]) (0x[0-9a-fA-F]+)", instruction.op_str)
            if match:
                offset = int(match.group(2), 16)
                if match.group(1) == "-":
                    offset = -offset
                candidate = instruction.address + instruction.size + offset
        elif instruction.mnemonic == "mov" and instruction.op_str.startswith(("edi,", "rdi,")):
            text = instruction.op_str.split(",", 1)[1].strip()
            if text.startswith("0x"):
                try:
                    candidate = int(text, 16)
                except ValueError:
                    candidate = None
        elif instruction.mnemonic == "push" and instruction.op_str.startswith("0x"):
            try:
                candidate = int(instruction.op_str, 16)
            except ValueError:
                candidate = None
        if candidate in known:
            return int(candidate)
    return None


def analyze(path: Path) -> BinaryProfile:
    from elftools.elf.elffile import ELFFile

    with open(path, "rb") as handle:
        elf = ELFFile(handle)
        is_pie = elf["e_type"] == "ET_DYN"
        entry = int(elf["e_entry"])
        imports: set[str] = set()
        symbols: dict[str, int] = {}
        for section_name in (".symtab", ".dynsym"):
            section = elf.get_section_by_name(section_name)
            if section is None:
                continue
            for symbol in section.iter_symbols():  # type: ignore[attr-defined]
                if not symbol.name:
                    continue
                imports.add(symbol.name)
                if symbol["st_value"]:
                    symbols.setdefault(symbol.name, symbol["st_value"])
        has_canary = "__stack_chk_fail" in imports or "__stack_chk_guard" in imports
        has_nx, relro = False, "partial"
        for segment in elf.iter_segments():
            if segment["p_type"] == "PT_GNU_STACK":
                has_nx = not bool(segment["p_flags"] & 0x1)
            if segment["p_type"] == "PT_GNU_RELRO":
                relro = "full" if elf.get_section_by_name(".got.plt") is None else "partial"

        plt_map: dict[int, str] = {}
        plt = elf.get_section_by_name(".plt")
        plt_sec = elf.get_section_by_name(".plt.sec")
        rela = (elf.get_section_by_name(".rela.plt")
                or elf.get_section_by_name(".rel.plt"))
        dynsym = elf.get_section_by_name(".dynsym")
        if plt is not None and rela is not None and dynsym is not None:
            # 32 位 .plt 的 sh_entsize 是 4（指令长度），不是桩长度；桩固定 16 字节
            stride = 16
            base, bias = (int(plt_sec["sh_addr"]), 0) if plt_sec is not None \
                else (int(plt["sh_addr"]), 1)
            for index, relocation in enumerate(rela.iter_relocations()):  # type: ignore[attr-defined]
                symbol = dynsym.get_symbol(relocation["r_info_sym"])  # type: ignore[attr-defined]
                plt_map[base + (index + bias) * stride] = symbol.name

        rodata = elf.get_section_by_name(".rodata")
        rodata_addr = rodata["sh_addr"] if rodata else 0
        rodata_data = rodata.data() if rodata else b""
        bits = 64 if elf.elfclass == 64 else 32

    strings: dict[str, list[int]] = {}
    for needle in (b"/bin/sh", b"/bin/bash"):
        found: list[int] = []
        start = rodata_data.find(needle)
        while start != -1:
            found.append(rodata_addr + start)
            start = rodata_data.find(needle, start + 1)
        if found:
            strings[needle.decode()] = found

    profile = BinaryProfile(path=path, is_pie=is_pie, has_canary=has_canary,
                            has_nx=has_nx, relro=relro, imports=imports,
                            symbols=symbols, plt=plt_map, bits=bits, strings=strings)

    for match in re.finditer(rb"[ -~]{3,200}", rodata_data):
        text = match.group(0).decode("ascii", "ignore")
        if "flag" in text.lower() and ("/" in text or text.lower().endswith(".txt")):
            profile.flag_string = text
            profile.flag_string_addr = rodata_addr + match.start()
            break

    instructions = _disassemble(path, bits)
    profile.instruction_count = len(instructions)
    profile.printf_entry = _printf_entry(instructions, plt_map, rodata_addr, rodata_data, bits)
    profile.command_input_entry = _sink_input_entry(instructions, plt_map, rodata_addr,
                                                   len(rodata_data), bits)
    starts = _function_starts(instructions)

    if profile.flag_string_addr is not None:
        target = _xref_function(instructions, starts, profile.flag_string_addr)
        if target is not None:
            name = next((symbol for symbol, address in symbols.items() if address == target), None)
            profile.win_candidates.append({"address": target, "symbol": name,
                                           "source": "flag-string-xref"})
    for name, address in symbols.items():
        if _SYMBOL_HINTS.search(name) and name not in {"main"}:
            profile.win_candidates.append({"address": address, "symbol": name,
                                           "source": "symbol-name"})
    # 调用图：主流程（entry / main / 输入读取函数）到不了的“死函数”，
    # 如果还会输出，就是 ret2win 的后门候选。
    edges, library_calls = _call_graph(instructions, starts, plt_map)
    main_address = symbols.get("main") or _main_address(instructions, starts, plt_map, entry)
    entry_function = _enclosing(starts, entry)
    roots: set[int] = set()
    if entry_function is not None:
        roots.add(entry_function)
    if main_address is not None and main_address in set(starts):
        roots.add(main_address)
    for start in starts:
        if library_calls.get(start, set()) & {"gets", "fgets", "read"}:
            roots.add(start)
    reachable = _reachable(roots, edges)

    def function_size(start: int) -> int:
        later = [value for value in starts if value > start]
        end = later[0] if later else (
            instructions[-1].address + instructions[-1].size if instructions else start)
        return end - start

    dead_output = [
        start for start in starts
        if start not in reachable
        and library_calls.get(start, set()) & _OUTPUT_FUNCTIONS
        and function_size(start) >= 32
    ]
    dead_output.sort(key=lambda start: -len(library_calls.get(start, set())))
    for start in dead_output[:3]:
        name = next((symbol for symbol, address in symbols.items() if address == start), None)
        profile.win_candidates.append({"address": start, "symbol": name,
                                       "source": "call-graph-dead-output"})
    profile.call_graph = {
        "functions": len(starts),
        "edges": sum(len(targets) for targets in edges.values()),
        "dead": len([start for start in starts if start not in reachable]),
        "dead_output": len(dead_output),
        "main": hex(main_address) if main_address else None,
    }

    unique: list[dict[str, Any]] = []
    seen_addresses: set[int] = set()
    for candidate in profile.win_candidates:
        if candidate["address"] in seen_addresses:
            continue
        seen_addresses.add(candidate["address"])
        unique.append(candidate)
    profile.win_candidates = unique

    for instruction in instructions:
        if instruction.mnemonic == "sub" and instruction.op_str.startswith(("rsp,", "esp,")):
            try:
                size = int(instruction.op_str.split(",")[1].strip(), 0)
            except ValueError:
                continue
            for extra in (8, 16):
                if size + extra not in profile.stack_hints:
                    profile.stack_hints.append(size + extra)
    profile.stack_hints = profile.stack_hints[:4]
    return profile

"""只读 ELF 分析：保护、导入、调用点、字符串交叉引用。不执行目标程序。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SHELLS = {"system", "popen", "execl", "execlp", "execv", "execvp"}
_UNSAFE_READS = {"gets", "strcpy", "strcat", "sprintf", "scanf"}
_SYMBOL_HINTS = re.compile(r"(win|flag|grant|backdoor|secret)", re.I)


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
    flag_string: str | None = None
    flag_string_addr: int | None = None
    win_candidates: list[dict[str, Any]] = field(default_factory=list)
    stack_hints: list[int] = field(default_factory=list)
    instruction_count: int = 0
    printf_entry: int | None = None

    def findings(self) -> list[BinaryFinding]:
        found: list[BinaryFinding] = []
        shells = sorted(self.imports & _SHELLS)
        reads = self.imports & (_UNSAFE_READS | {"fgets", "read"})
        if shells and reads:
            found.append(BinaryFinding(
                "command_injection", "命令注入（system 家族 + 输入读取）",
                f"导入 {', '.join(shells)}"))
        if self.printf_entry is not None:
            found.append(BinaryFinding(
                "format_string", "格式化字符串（非字面量格式参数）",
                f"call printf at {hex(self.printf_entry)}", self.printf_entry))
        unsafe = sorted(self.imports & _UNSAFE_READS)
        if unsafe:
            found.append(BinaryFinding(
                "unsafe_copy", "栈溢出 / 无边界拷贝", f"导入 {', '.join(unsafe)}"))
        return found


def _disassemble(profile_path: Path) -> list[Any]:
    from capstone import CS_ARCH_X86, CS_MODE_64, Cs
    from elftools.elf.elffile import ELFFile

    with open(profile_path, "rb") as handle:
        elf = ELFFile(handle)
        text = elf.get_section_by_name(".text")
        if text is None:
            return []
        engine = Cs(CS_ARCH_X86, CS_MODE_64)
        return list(engine.disasm(text.data(), text["sh_addr"]))


def _function_starts(instructions: list[Any]) -> list[int]:
    starts = set()
    for index, instruction in enumerate(instructions):
        if instruction.mnemonic == "endbr64":
            starts.add(instruction.address)
        elif instruction.mnemonic == "push" and instruction.op_str == "rbp":
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


def _printf_entry(instructions: list[Any], plt: dict[int, str],
                  rodata_addr: int, rodata_data: bytes) -> int | None:
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
        for back in range(1, 11):
            if index - back < 0:
                break
            previous = instructions[index - back]
            if previous.mnemonic != "lea" or "rip" not in previous.op_str:
                continue
            parts = [item.strip() for item in previous.op_str.split(",")]
            match = re.search(r"rip ([+-]) (0x[0-9a-fA-F]+)", parts[-1])
            if not match:
                continue
            offset = int(match.group(2), 16)
            if match.group(1) == "-":
                offset = -offset
            address = previous.address + previous.size + offset
            relative = address - rodata_addr
            if 0 <= relative < len(rodata_data):
                chunk = rodata_data[relative:relative + 200].split(b"\0")[0]
                if b"%" in chunk:
                    literal = True
                    break
        if not literal:
            return int(instruction.address)
    return None


def analyze(path: Path) -> BinaryProfile:
    from elftools.elf.elffile import ELFFile

    with open(path, "rb") as handle:
        elf = ELFFile(handle)
        is_pie = elf["e_type"] == "ET_DYN"
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
        rela = elf.get_section_by_name(".rela.plt")
        dynsym = elf.get_section_by_name(".dynsym")
        if plt is not None and rela is not None and dynsym is not None:
            stride = int(plt["sh_entsize"] or 16)
            base, bias = (int(plt_sec["sh_addr"]), 0) if plt_sec is not None \
                else (int(plt["sh_addr"]), 1)
            for index, relocation in enumerate(rela.iter_relocations()):  # type: ignore[attr-defined]
                symbol = dynsym.get_symbol(relocation["r_info_sym"])  # type: ignore[attr-defined]
                plt_map[base + (index + bias) * stride] = symbol.name

        rodata = elf.get_section_by_name(".rodata")
        rodata_addr = rodata["sh_addr"] if rodata else 0
        rodata_data = rodata.data() if rodata else b""

    profile = BinaryProfile(path=path, is_pie=is_pie, has_canary=has_canary,
                            has_nx=has_nx, relro=relro, imports=imports,
                            symbols=symbols, plt=plt_map)

    for match in re.finditer(rb"[ -~]{3,200}", rodata_data):
        text = match.group(0).decode("ascii", "ignore")
        if "flag" in text.lower() and ("/" in text or text.lower().endswith(".txt")):
            profile.flag_string = text
            profile.flag_string_addr = rodata_addr + match.start()
            break

    instructions = _disassemble(path)
    profile.instruction_count = len(instructions)
    profile.printf_entry = _printf_entry(instructions, plt_map, rodata_addr, rodata_data)
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
    unique: list[dict[str, Any]] = []
    seen_addresses: set[int] = set()
    for candidate in profile.win_candidates:
        if candidate["address"] in seen_addresses:
            continue
        seen_addresses.add(candidate["address"])
        unique.append(candidate)
    profile.win_candidates = unique

    for instruction in instructions:
        if instruction.mnemonic == "sub" and instruction.op_str.startswith("rsp,"):
            try:
                size = int(instruction.op_str.split(",")[1].strip(), 0)
            except ValueError:
                continue
            for extra in (8, 16):
                if size + extra not in profile.stack_hints:
                    profile.stack_hints.append(size + extra)
    profile.stack_hints = profile.stack_hints[:4]
    return profile

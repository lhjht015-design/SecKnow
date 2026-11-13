from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from secknow.api.app import create_app
from secknow.config import RuntimeMode, Settings
from secknow.solving.binary import BinaryProfile
from secknow.solving.chains import build_chains
from secknow.solving.entries import detect_source_entries


def settings_for(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    data = tmp_path / "data"
    corpus.mkdir()
    return Settings(
        mode=RuntimeMode.STRICT_LEXICAL,
        corpus_root=corpus,
        data_root=data,
        database_path=data / "secknow.db",
        snapshot_root=data / "snapshots",
        model_root=tmp_path / "models",
        frontend_dist=tmp_path / "dist",
        source_manifest=tmp_path / "sources" / "sources.yaml",
        bge_model_path=tmp_path / "models" / "bge-m3",
        reranker_model_path=tmp_path / "models" / "reranker",
    )


def test_detects_unbounded_copy_and_command_injection() -> None:
    source = "int main(){char b[64]; gets(b);}\nvoid f(char *h){system(h);}\n"

    categories = [entry.category for entry in detect_source_entries(source)]

    assert "unsafe_copy" in categories
    assert "command_injection" in categories
    assert categories.index("command_injection") < categories.index("unsafe_copy")


def test_detects_format_string_but_not_literal() -> None:
    unsafe = "printf(buf);\n"
    safe = 'printf("%s", buf);\n'

    assert [entry.category for entry in detect_source_entries(unsafe)] == ["format_string"]
    assert detect_source_entries(safe) == []


def test_safe_source_has_no_entry() -> None:
    source = "fgets(buf, sizeof(buf), stdin);\nprintf(\"%s\", buf);\n"

    assert detect_source_entries(source) == []


def test_source_only_chain_requires_binary() -> None:
    chains = build_chains(["unsafe_copy"], None)

    assert len(chains) == 1
    assert chains[0].status == "needs-binary"
    assert chains[0].steps[1].status == "blocked"


def test_binary_chain_refuses_canary_and_accepts_plain() -> None:
    base = {
        "path": Path("target"),
        "is_pie": False,
        "has_nx": True,
        "relro": "partial",
        "imports": {"gets"},
        "symbols": {"grant": 0x4011F6},
        "plt": {},
    }
    canary = BinaryProfile(has_canary=True, **base)
    plain = BinaryProfile(
        has_canary=False,
        flag_string="/flag",
        flag_string_addr=0x402010,
        win_candidates=[{"address": 0x4011F6, "symbol": "grant", "source": "flag-string-xref"}],
        stack_hints=[72],
        **base,
    )

    refused = build_chains(["unsafe_copy"], canary)[0]
    usable = build_chains(["unsafe_copy"], plain)[0]

    assert refused.status == "refused"
    assert refused.reason == "canary_present"
    assert usable.status == "needs-execution"
    assert any(step.title == "后门定位" and step.status == "ok" for step in usable.steps)


def test_chain_api_returns_entries_and_steps(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/chain",
            json={"source": "int main(){char b[64]; gets(b);}", "filename": "t.c"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["entries"][0]["category"] == "unsafe_copy"
    assert body["chains"][0]["status"] == "needs-binary"
    assert body["chains"][0]["steps"][0]["title"] == "入口确认"
    assert body["binary"] is None


def test_chain_api_rejects_empty_request(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/chain", json={})

    assert response.status_code == 422


def test_binary_profile_offers_libc_call_chain() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"gets", "system"},
        symbols={},
        plt={0x8048490: "system"},
        bits=32,
        strings={"/bin/sh": [0x8048763]},
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "libc-call"
    assert chains[0].status == "needs-execution"
    assert any(step.unit == "string" and "/bin/sh" in step.detail for step in chains[0].steps)


def test_binary_profile_offers_syscall_chain_without_system() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"gets"},
        symbols={},
        plt={},
        bits=32,
        strings={"/bin/sh": [0x80BE408]},
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "syscall-execve"
    assert chains[0].status == "needs-execution"


def test_format_string_chain_requires_execution() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"printf"},
        symbols={},
        plt={},
        bits=32,
        printf_entry=0x80486BA,
    )

    chains = build_chains(["format_string"], profile)

    assert chains[0].status == "needs-execution"
    assert any(step.unit == "fmt_offset" for step in chains[0].steps)


def test_staged_call_chain_when_system_without_binsh() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"gets", "system"},
        symbols={},
        plt={0x8048460: "gets", 0x8048490: "system"},
        bits=32,
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "staged-call"
    assert chains[0].status == "needs-execution"


def test_ret2libc_leak_chain_without_system() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"gets", "puts"},
        symbols={},
        plt={0x8048440: "gets", 0x8048460: "puts"},
        bits=32,
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "ret2libc-leak"
    assert chains[0].status == "needs-execution"


def test_shellcode_chain_is_unsupported_without_stack_leak() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=False,
        relro="partial",
        imports={"gets"},
        symbols={},
        plt={0x8048460: "gets"},
        bits=32,
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "ret2shellcode"
    assert chains[0].status == "unsupported"


def test_nx_off_with_printf_offers_staged_printf_chain() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=False,
        relro="partial",
        imports={"gets", "printf"},
        symbols={},
        plt={0x80483C0: "printf", 0x80483D0: "gets"},
        bits=32,
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "staged-printf-leak"
    assert chains[0].status == "needs-execution"


def test_canary_does_not_block_format_string_chain() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=True,
        has_nx=True,
        relro="partial",
        imports={"printf"},
        symbols={},
        plt={0x80484F0: "printf"},
        bits=32,
        printf_entry=0x804873A,
    )

    chains = build_chains(["format_string"], profile)

    assert chains[0].chain_id == "format-string"
    assert chains[0].status == "needs-execution"


def test_profile_without_pwn_entry_offers_symbolic_chain() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"scanf", "puts"},
        symbols={},
        plt={},
        bits=64,
        instruction_count=1000,
    )

    chains = build_chains([], profile)

    assert chains[0].chain_id == "symbolic-execution"
    assert chains[0].status == "needs-execution"


def test_ret2csu_chain_for_64bit_without_system() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports={"read", "puts"},
        symbols={},
        plt={0x400500: "puts", 0x400520: "read"},
        bits=64,
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "ret2csu-leak"
    assert chains[0].status == "needs-execution"


def test_ret2libc_leak_chain_works_with_fgets_binary() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=False,
        relro="partial",
        imports={"fgets", "puts"},
        symbols={},
        plt={0x80483C0: "fgets", 0x80483D0: "puts"},
        bits=32,
    )

    chains = build_chains(["unsafe_copy"], profile)

    assert chains[0].chain_id == "ret2libc-leak"
    assert chains[0].status == "needs-execution"


def test_tiny_syscall_binary_offers_srop_chain() -> None:
    profile = BinaryProfile(
        path=Path("target"),
        is_pie=False,
        has_canary=False,
        has_nx=True,
        relro="partial",
        imports=set(),
        symbols={},
        plt={},
        bits=64,
        instruction_count=6,
    )

    chains = build_chains([], profile)

    assert chains[0].chain_id == "srop"
    assert chains[0].status == "needs-execution"

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

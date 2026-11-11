from __future__ import annotations

from secknow.ingestion.code_parser import CodeParser


def test_python_ast_extracts_symbols_without_execution() -> None:
    source = '''
import os

class Exploit:
    """Demonstration only."""
    def build(self, payload):
        return helper(payload)
'''
    result = CodeParser().parse_python("element:1", source)
    assert not result.diagnostics
    names = {symbol.qualified_name for symbol in result.symbols}
    assert names == {"Exploit", "Exploit.build"}
    method = next(symbol for symbol in result.symbols if symbol.name == "build")
    parent = next(symbol for symbol in result.symbols if symbol.name == "Exploit")
    assert method.parent_symbol_id == parent.symbol_id
    assert "helper" in method.calls


def test_invalid_python_is_preserved_as_diagnostic() -> None:
    result = CodeParser().parse_python("element:2", "def broken(")
    assert result.symbols == ()
    assert result.diagnostics[0].startswith("python-syntax:")

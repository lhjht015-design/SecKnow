from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from secknow.domain.ids import stable_id
from secknow.domain.models import CodeSymbol

FENCE_PATTERN = re.compile(
    r"```(?P<language>[A-Za-z0-9_+.-]*)\s*\n(?P<code>.*?)```",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class CodeParseResult:
    symbols: tuple[CodeSymbol, ...]
    diagnostics: tuple[str, ...] = ()


class PythonStructureVisitor(ast.NodeVisitor):
    def __init__(self, element_id: str) -> None:
        self.element_id = element_id
        self.parents: list[str] = []
        self.parent_ids: list[str] = []
        self.symbols: list[CodeSymbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node, "async_function")

    def _visit_symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        qualified = ".".join([*self.parents, node.name])
        symbol_id = stable_id("symbol", self.element_id, qualified, node.lineno)
        calls = tuple(
            sorted(
                {
                    self._call_name(child.func)
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call) and self._call_name(child.func)
                }
            )
        )
        imports = tuple(
            sorted(
                {
                    alias.name
                    for child in ast.walk(node)
                    if isinstance(child, (ast.Import, ast.ImportFrom))
                    for alias in child.names
                }
            )
        )
        signature = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = self._signature(node)
        self.symbols.append(
            CodeSymbol(
                symbol_id=symbol_id,
                element_id=self.element_id,
                name=node.name,
                qualified_name=qualified,
                kind=kind,
                language="python",
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", node.lineno),
                parent_symbol_id=self.parent_ids[-1] if self.parent_ids else None,
                signature=signature,
                docstring=ast.get_docstring(node, clean=True),
                imports=imports,
                calls=calls,
            )
        )
        self.parents.append(node.name)
        self.parent_ids.append(symbol_id)
        self.generic_visit(node)
        self.parent_ids.pop()
        self.parents.pop()

    @staticmethod
    def _call_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    @staticmethod
    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        arguments = [argument.arg for argument in node.args.posonlyargs]
        arguments.extend(argument.arg for argument in node.args.args)
        if node.args.vararg:
            arguments.append(f"*{node.args.vararg.arg}")
        arguments.extend(argument.arg for argument in node.args.kwonlyargs)
        if node.args.kwarg:
            arguments.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(arguments)})"


class CodeParser:
    def parse_python(self, element_id: str, text: str) -> CodeParseResult:
        try:
            tree = ast.parse(text, mode="exec", type_comments=True)
        except SyntaxError as error:
            diagnostic = f"python-syntax:{error.lineno}:{error.offset}:{error.msg}"
            return CodeParseResult((), (diagnostic,))
        visitor = PythonStructureVisitor(element_id)
        visitor.visit(tree)
        return CodeParseResult(tuple(visitor.symbols))

    def parse_tree_sitter(self, element_id: str, language: str, text: str) -> CodeParseResult:
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            return CodeParseResult((), ("tree-sitter-unavailable",))
        try:
            parser_factory: Any = get_parser
            tree = parser_factory(language).parse(text.encode("utf-8", errors="replace"))
        except (LookupError, ValueError) as error:
            return CodeParseResult((), (f"tree-sitter-language:{error}",))
        diagnostics = ("tree-sitter-parse-errors",) if tree.root_node.has_error else ()
        symbols: list[CodeSymbol] = []
        symbol_types = {
            "function_definition": "function",
            "class_definition": "class",
            "function_declaration": "function",
            "method_declaration": "method",
        }
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            kind = symbol_types.get(node.type)
            if kind:
                name_node = node.child_by_field_name("name")
                name = (
                    text.encode("utf-8")[name_node.start_byte : name_node.end_byte].decode("utf-8")
                    if name_node
                    else f"anonymous@{node.start_point[0] + 1}"
                )
                symbols.append(
                    CodeSymbol(
                        symbol_id=stable_id(
                            "symbol", element_id, language, name, node.start_point[0] + 1
                        ),
                        element_id=element_id,
                        name=name,
                        qualified_name=name,
                        kind=kind,
                        language=language,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    )
                )
            stack.extend(reversed(node.children))
        return CodeParseResult(tuple(symbols), diagnostics)

    def fenced_blocks(self, text: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (match.group("language").casefold() or "text", match.group("code"))
            for match in FENCE_PATTERN.finditer(text)
        )

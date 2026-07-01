"""Python-based DSL interpreter — walks IR nodes and executes them.

Phase 0: the interpreter prints the IR structure and simulates basic
evaluation (state tracking, event dispatching).  No code generation yet.
"""

from __future__ import annotations

import sys
from .ast import (
    ArrayExpr,
    AssignmentStmt,
    BinaryExpr,
    ComponentDef,
    EnumDef,
    EventHandler,
    ExprStmt,
    File,
    ForStmt,
    IdentExpr,
    IfStmt,
    ImportStmt,
    InterpolatedExpr,
    LetStmt,
    LiteralExpr,
    MemberExpr,
    ParamDef,
    PlatformBlock,
    Property,
    RawBlock,
    Span,
    StateDecl,
    TagNode,
    TryStmt,
    TypeAnnotation,
    UnaryExpr,
)
from .parser import ExprCall


class Interpreter:
    """Walks the IR and produces output (print or execute).

    In Phase 0 the interpreter runs in two modes:
      - 'print': dump the IR tree as structured text (default)
      - 'eval':  simulate evaluation, state, and event dispatching
    """

    def __init__(self, file_ir: File, mode: str = "print"):
        self.file = file_ir
        self.mode = mode
        self._indent = 0

    def run(self) -> str:
        """Execute the file and return output text."""
        lines: list[str] = []
        self._output = lines
        self._visit_file(self.file)
        return "\n".join(lines)

    # ── Output helpers ─────────────────────────────────────────────

    def _write(self, line: str = "") -> None:
        self._output.append("  " * self._indent + line)

    def _enter(self, title: str) -> None:
        self._write(title)
        self._indent += 1

    def _leave(self) -> None:
        self._indent -= 1

    def _write_kv(self, key: str, value: object) -> None:
        self._write(f"{key}: {value}")

    # ── Visitor ────────────────────────────────────────────────────

    def _visit_file(self, node: File) -> None:
        self._enter("File")
        for imp in node.imports:
            self._visit_import(imp)
        for decl in node.declarations:
            if isinstance(decl, ComponentDef):
                self._visit_component(decl)
            elif isinstance(decl, EnumDef):
                self._visit_enum(decl)
        self._leave()

    def _visit_import(self, node: ImportStmt) -> None:
        names = f" {{{', '.join(node.names)}}}" if node.names else ""
        self._write(f"import{names} {node.path}")

    def _visit_enum(self, node: EnumDef) -> None:
        prefix = "export " if node.exported else ""
        self._enter(f"{prefix}enum {node.name}")
        for v in node.variants:
            self._write(v)
        self._leave()

    def _visit_component(self, node: ComponentDef) -> None:
        prefix = "export " if node.exported else ""
        params = self._fmt_params(node.params)
        self._enter(f"{prefix}component {node.name}{params}")
        for item in node.body:
            if isinstance(item, Property):
                self._visit_property(item)
            elif isinstance(item, StateDecl):
                self._visit_state(item)
            elif isinstance(item, TagNode):
                self._visit_tag(item)
            elif isinstance(item, EventHandler):
                self._visit_event(item)
            elif isinstance(item, PlatformBlock):
                self._visit_platform(item)
            elif isinstance(item, RawBlock):
                self._visit_raw(item)
        self._leave()

    def _visit_tag(self, node: TagNode) -> None:
        label = node.name
        if node.instance_name:
            label += f" {node.instance_name}"
        props = " " + " ".join(
            f"{p.name}={self._fmt_expr(p.value)}" for p in node.properties
        ) if node.properties else ""
        self._enter(f"<{label}{props}>")
        for child in node.children:
            if isinstance(child, TagNode):
                self._visit_tag(child)
            elif isinstance(child, PlatformBlock):
                self._visit_platform(child)
            elif isinstance(child, RawBlock):
                self._visit_raw(child)
        self._leave()

    def _visit_property(self, node: Property) -> None:
        self._write(f"{node.name} = {self._fmt_expr(node.value)}")

    def _visit_state(self, node: StateDecl) -> None:
        typ = f": {node.type.name}" if node.type else ""
        self._write(f"state {node.name}{typ} = {self._fmt_expr(node.value)}")

    def _visit_event(self, node: EventHandler) -> None:
        params = self._fmt_params(node.params) if node.params else ""
        self._enter(f"on {node.event}{params}")
        for stmt in node.body:
            self._visit_stmt(stmt)
        self._leave()

    def _visit_platform(self, node: PlatformBlock) -> None:
        self._enter(f"@platform({node.platform})")
        for item in node.body:
            if isinstance(item, Property):
                self._visit_property(item)
            elif isinstance(item, TagNode):
                self._visit_tag(item)
        self._leave()

    def _visit_raw(self, node: RawBlock) -> None:
        self._write(f"{{{{{node.target}}}}} {node.code[:60]}...")

    # ── Statements ─────────────────────────────────────────────────

    def _visit_stmt(self, node) -> None:
        name = type(node).__name__
        handler = getattr(self, f"_visit_stmt_{name}", self._visit_stmt_default)
        handler(node)

    def _visit_stmt_default(self, node) -> None:
        self._write(f";; {node}")

    def _visit_stmt_ExprStmt(self, node: ExprStmt) -> None:
        self._write(self._fmt_expr(node.expr))

    def _visit_stmt_AssignmentStmt(self, node: AssignmentStmt) -> None:
        self._write(f"{self._fmt_expr(node.target)} = {self._fmt_expr(node.value)}")

    def _visit_stmt_IfStmt(self, node: IfStmt) -> None:
        self._enter(f"if {self._fmt_expr(node.condition)}")
        for s in node.then_body:
            self._visit_stmt(s)
        if node.else_body:
            if isinstance(node.else_body, IfStmt):
                self._leave()
                self._write("else")
                self._visit_stmt(node.else_body)
                return
            self._leave()
            self._enter("else")
            for s in node.else_body:
                self._visit_stmt(s)
        self._leave()

    def _visit_stmt_ForStmt(self, node: ForStmt) -> None:
        self._enter(f"for {node.variable} in {self._fmt_expr(node.iterable)}")
        for s in node.body:
            self._visit_stmt(s)
        self._leave()

    def _visit_stmt_LetStmt(self, node: LetStmt) -> None:
        typ = f": {node.type_annotation.name}" if node.type_annotation else ""
        val = f" = {self._fmt_expr(node.value)}" if node.value else ""
        self._write(f"let {node.name}{typ}{val}")

    def _visit_stmt_TryStmt(self, node: TryStmt) -> None:
        self._enter("try")
        for s in node.try_body:
            self._visit_stmt(s)
        self._leave()
        self._enter(f"catch {node.catch_var}")
        for s in node.catch_body:
            self._visit_stmt(s)
        self._leave()

    # ── Expression formatting ──────────────────────────────────────

    def _fmt_expr(self, expr) -> str:
        name = type(expr).__name__
        handler = getattr(self, f"_fmt_{name}", None)
        if handler:
            return handler(expr)
        return str(expr)

    def _fmt_LiteralExpr(self, e: LiteralExpr) -> str:
        if e.kind == "string":
            return f'"{e.value}"'
        if e.kind == "color":
            return f"#{e.value}"
        if e.kind == "null":
            return "null"
        if e.kind == "bool":
            return "true" if e.value else "false"
        return str(e.value)

    def _fmt_IdentExpr(self, e: IdentExpr) -> str:
        return e.name

    def _fmt_MemberExpr(self, e: MemberExpr) -> str:
        return f"{self._fmt_expr(e.obj)}.{e.member}"

    def _fmt_InterpolatedExpr(self, e: InterpolatedExpr) -> str:
        parts = []
        for p in e.parts:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(f"${{{self._fmt_expr(p)}}}")
        return f'"{"".join(parts)}"'

    def _fmt_BinaryExpr(self, e: BinaryExpr) -> str:
        return f"({self._fmt_expr(e.left)} {e.op} {self._fmt_expr(e.right)})"

    def _fmt_UnaryExpr(self, e: UnaryExpr) -> str:
        return f"{e.op}{self._fmt_expr(e.operand)}"

    def _fmt_ArrayExpr(self, e: ArrayExpr) -> str:
        elements = ", ".join(self._fmt_expr(el) for el in e.elements)
        return f"[{elements}]"

    def _fmt_ExprCall(self, e: ExprCall) -> str:
        args = ", ".join(self._fmt_expr(a) for a in e.args)
        return f"{self._fmt_expr(e.fn)}({args})"

    @staticmethod
    def _fmt_params(params: list[ParamDef]) -> str:
        if not params:
            return ""
        items = []
        for p in params:
            default = f" = {p.default}" if p.default is not None else ""
            items.append(f"{p.name}: {p.type.name}{default}")
        return f"({', '.join(items)})"

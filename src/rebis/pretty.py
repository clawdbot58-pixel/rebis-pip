"""Pretty printer — renders IR nodes as colorized, readable text.

Used by the REPL and accessible via `rebis eval --pretty`.
ANSI color codes make structure and token types visually distinct.
"""

from __future__ import annotations

import sys
from typing import Any

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
    StateDecl,
    TagNode,
    TryStmt,
    TypeAnnotation,
    UnaryExpr,
)
from .parser import ExprCall


# ── ANSI color/style codes ────────────────────────────────────────────────

class Style:
    """Named ANSI escape sequences for syntax highlighting."""
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"
    ITALIC      = "\033[3m"
    UNDERLINE   = "\033[4m"

    # Foreground
    BLACK       = "\033[30m"
    RED         = "\033[31m"
    GREEN       = "\033[32m"
    YELLOW      = "\033[33m"
    BLUE        = "\033[34m"
    MAGENTA     = "\033[35m"
    CYAN        = "\033[36m"
    WHITE       = "\033[37m"
    BRIGHT_BLACK  = "\033[90m"
    BRIGHT_RED    = "\033[91m"
    BRIGHT_GREEN  = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE   = "\033[94m"
    BRIGHT_MAGENTA= "\033[95m"
    BRIGHT_CYAN   = "\033[96m"
    BRIGHT_WHITE  = "\033[97m"

NO_COLOR = not sys.stdout.isatty()


def _s(*codes: str) -> str:
    """Return an ANSI escape sequence from the given codes, or empty if not a tty."""
    if NO_COLOR:
        return ""
    return "".join(codes)


def _st(tag: str, text: str, *modifiers: str) -> str:
    """Wrap *text* in ANSI *tag* style, plus optional *modifiers*."""
    code = getattr(Style, tag.upper(), "")
    return f"{_s(code, *modifiers)}{text}{_s(Style.RESET)}"


# ── Convenience wrappers ───────────────────────────────────────────────────

def keyword(text: str) -> str:
    return _st("CYAN", text)

def name(text: str) -> str:
    return _st("YELLOW", text)

def string(text: str) -> str:
    return _st("GREEN", text)

def integer(text: str) -> str:
    return _st("MAGENTA", text)

def float_(text: str) -> str:
    return _st("MAGENTA", text, Style.BOLD)

def boolean(text: str) -> str:
    return _st("BLUE", text)

def null(text: str) -> str:
    return _st("BRIGHT_BLACK", text, Style.ITALIC)

def color(text: str) -> str:
    return _st("YELLOW", text, Style.UNDERLINE)

def tag(text: str) -> str:
    return _st("WHITE", text, Style.BOLD)

def punctuation(text: str) -> str:
    return _st("BRIGHT_BLACK", text)

def type_(text: str) -> str:
    return _st("BRIGHT_BLUE", text)

def operator(text: str) -> str:
    return _st("BRIGHT_CYAN", text)

def comment(text: str) -> str:
    return _st("BRIGHT_BLACK", text, Style.ITALIC)

def error(text: str) -> str:
    return _st("RED", text, Style.BOLD)

def dim(text: str) -> str:
    return _st("BRIGHT_BLACK", text)


# ── PrettyPrinter ──────────────────────────────────────────────────────────

class PrettyPrinter:
    """Renders a Rebis IR tree as colorized, indented text.

    Usage::
        pp = PrettyPrinter()
        print(pp.format(file_ir))
    """

    def __init__(self, color: bool | None = None):
        self._color = color if color is not None else sys.stdout.isatty()
        self._indent = 0
        self._lines: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    def format(self, node: File) -> str:
        """Format a File IR node as a string."""
        self._lines = []
        self._indent = 0
        self._visit(node)
        return "\n".join(self._lines)

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _visit(self, node: Any) -> None:
        """Visit an IR node and append formatted lines."""
        name = type(node).__name__
        handler = getattr(self, f"_visit_{name}", self._default)
        handler(node)

    def _default(self, node: Any) -> None:
        self._write(f"<{type(node).__name__}> {node}")

    # ── Output helpers ────────────────────────────────────────────────────

    def _w(self, text: str = "") -> None:
        """Write a line at the current indent level."""
        indent_str = "  " * self._indent
        self._lines.append(indent_str + text)

    def _enter(self) -> None:
        self._indent += 1

    def _leave(self) -> None:
        self._indent -= 1

    def _fmt(self, text: str, color_fn=None) -> str:
        """Apply color function if color is enabled."""
        if self._color and color_fn:
            return color_fn(text)
        return text

    # ── File ──────────────────────────────────────────────────────────────

    def _visit_File(self, node: File) -> None:
        self._w(self._fmt("File", dim))
        self._enter()
        for imp in node.imports:
            self._visit(imp)
        for decl in node.declarations:
            self._visit(decl)
        if not node.imports and not node.declarations:
            self._w(self._fmt("(empty)", dim))
        self._leave()

    # ── Imports ───────────────────────────────────────────────────────────

    def _visit_ImportStmt(self, node: ImportStmt) -> None:
        parts = [self._fmt("import", keyword)]
        if node.names:
            joined = ", ".join(self._fmt(n, name) for n in node.names)
            parts.append(f"{{{joined}}}")
        parts.append(self._fmt('"' + node.path + '"', string))
        self._w(" ".join(parts))

    # ── Component ─────────────────────────────────────────────────────────

    def _visit_ComponentDef(self, node: ComponentDef) -> None:
        prefix = f"{self._fmt('export', keyword)} " if node.exported else ""
        sig = f"{self._fmt('component', keyword)} {self._fmt(node.name, name)}{self._fmt_params(node.params)}"
        self._w(f"{prefix}{sig}")
        self._enter()
        for item in node.body:
            self._visit(item)
        self._leave()

    # ── Enum ──────────────────────────────────────────────────────────────

    def _visit_EnumDef(self, node: EnumDef) -> None:
        prefix = f"{self._fmt('export', keyword)} " if node.exported else ""
        self._w(f"{prefix}{self._fmt('enum', keyword)} {self._fmt(node.name, name)}")
        self._enter()
        for v in node.variants:
            self._w(self._fmt(v, name))
        self._leave()

    # ── Tags (the core UI tree) ──────────────────────────────────────────

    def _visit_TagNode(self, node: TagNode) -> None:
        label = node.name
        if node.instance_name:
            label += f" {node.instance_name}"

        props = self._fmt_props(node.properties)

        tag_open = self._fmt(f"<{label}{props}>", tag)
        self._w(tag_open)

        if node.children:
            self._enter()
            for child in node.children:
                self._visit(child)
            self._leave()

    # ── Property ──────────────────────────────────────────────────────────

    def _visit_Property(self, node: Property) -> None:
        val = self._fmt_expr(node.value)
        self._w(f"{self._fmt(node.name, name)} = {val}")

    def _fmt_props(self, props: list[Property]) -> str:
        """Format an inline property list (for tag headers)."""
        if not props:
            return ""
        parts = []
        for p in props:
            val = self._fmt_expr(p.value)
            parts.append(f"{self._fmt(p.name, name)}={val}")
        return " " + " ".join(parts)

    # ── State declaration ─────────────────────────────────────────────────

    def _visit_StateDecl(self, node: StateDecl) -> None:
        parts = [self._fmt("state", keyword)]
        name_part = self._fmt(node.name, name)
        if node.type:
            name_part += f"{self._fmt(':', punctuation)} {self._visit_type(node.type)}"
        parts.append(name_part)
        if node.value is not None:
            parts.append(f"{self._fmt('=', operator)} {self._fmt_expr(node.value)}")
        self._w(" ".join(parts))

    # ── Event handler ─────────────────────────────────────────────────────

    def _visit_EventHandler(self, node: EventHandler) -> None:
        params = self._fmt_params(node.params) if node.params else ""
        self._w(f"{self._fmt('on', keyword)} {self._fmt(node.event, name)}{params}")
        if node.body:
            self._enter()
            for stmt in node.body:
                self._visit(stmt)
            self._leave()

    # ── Platform block ────────────────────────────────────────────────────

    def _visit_PlatformBlock(self, node: PlatformBlock) -> None:
        self._w(f"{self._fmt('@platform', keyword)}({self._fmt(node.platform, name)})")
        if node.body:
            self._enter()
            for item in node.body:
                self._visit(item)
            self._leave()

    # ── Raw block ─────────────────────────────────────────────────────────

    def _visit_RawBlock(self, node: RawBlock) -> None:
        code = node.code[:60] + ("..." if len(node.code) > 60 else "")
        tag_content = self._fmt(node.target, name)
        self._w(f"{{{{{tag_content}}}}} {code}")

    # ── Statements (event handler / control flow) ─────────────────────────

    def _visit_ExprStmt(self, node: ExprStmt) -> None:
        self._w(self._fmt_expr(node.expr))

    def _visit_AssignmentStmt(self, node: AssignmentStmt) -> None:
        self._w(f"{self._fmt_expr(node.target)} = {self._fmt_expr(node.value)}")

    def _visit_IfStmt(self, node: IfStmt) -> None:
        self._w(f"{self._fmt('if', keyword)} {self._fmt_expr(node.condition)}")
        self._enter()
        for s in node.then_body:
            self._visit(s)
        self._leave()
        if node.else_body:
            if isinstance(node.else_body, IfStmt):
                self._w(self._fmt("else", keyword))
                self._visit_IfStmt(node.else_body)
            else:
                self._w(self._fmt("else", keyword))
                self._enter()
                for s in node.else_body:
                    self._visit(s)
                self._leave()

    def _visit_ForStmt(self, node: ForStmt) -> None:
        self._w(f"{self._fmt('for', keyword)} {self._fmt(node.variable, name)} {self._fmt('in', keyword)} {self._fmt_expr(node.iterable)}")
        self._enter()
        for s in node.body:
            self._visit(s)
        self._leave()

    def _visit_LetStmt(self, node: LetStmt) -> None:
        parts = [self._fmt("let", keyword), self._fmt(node.name, name)]
        if node.type_annotation:
            parts.append(f": {self._visit_type(node.type_annotation)}")
        if node.value is not None:
            parts.append(f"= {self._fmt_expr(node.value)}")
        self._w(" ".join(parts))

    def _visit_TryStmt(self, node: TryStmt) -> None:
        self._w(self._fmt("try", keyword))
        self._enter()
        for s in node.try_body:
            self._visit(s)
        self._leave()
        self._w(f"{self._fmt('catch', keyword)} {self._fmt(node.catch_var, name)}")
        self._enter()
        for s in node.catch_body:
            self._visit(s)
        self._leave()

    # ── Expression formatting ─────────────────────────────────────────────

    def _fmt_expr(self, expr: Any) -> str:
        name = type(expr).__name__
        handler = getattr(self, f"_fmt_{name}", None)
        if handler:
            return handler(expr)
        return str(expr)

    def _fmt_LiteralExpr(self, e: LiteralExpr) -> str:
        if e.kind == "string":
            return self._fmt(f'"{e.value}"', string)
        if e.kind == "color":
            return self._fmt(f"#{e.value}", color)
        if e.kind == "null":
            return self._fmt("null", null)
        if e.kind == "bool":
            return self._fmt("true" if e.value else "false", boolean)
        if e.kind == "int":
            return self._fmt(str(e.value), integer)
        if e.kind == "float":
            return self._fmt(str(e.value), float_)
        return str(e.value)

    def _fmt_IdentExpr(self, e: IdentExpr) -> str:
        return self._fmt(e.name, name)

    def _fmt_MemberExpr(self, e: MemberExpr) -> str:
        return f"{self._fmt_expr(e.obj)}.{self._fmt(e.member, name)}"

    def _fmt_InterpolatedExpr(self, e: InterpolatedExpr) -> str:
        parts = []
        for p in e.parts:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(f"${{{self._fmt_expr(p)}}}")
        return self._fmt(f'"{"".join(parts)}"', string)

    def _fmt_BinaryExpr(self, e: BinaryExpr) -> str:
        return f"({self._fmt_expr(e.left)} {self._fmt(e.op, operator)} {self._fmt_expr(e.right)})"

    def _fmt_UnaryExpr(self, e: UnaryExpr) -> str:
        return f"{self._fmt(e.op, operator)}{self._fmt_expr(e.operand)}"

    def _fmt_ArrayExpr(self, e: ArrayExpr) -> str:
        elements = ", ".join(self._fmt_expr(el) for el in e.elements)
        return f"[{elements}]"

    def _fmt_ExprCall(self, e: ExprCall) -> str:
        args = ", ".join(self._fmt_expr(a) for a in e.args)
        return f"{self._fmt_expr(e.fn)}({args})"

    # ── Type annotation ───────────────────────────────────────────────────

    def _visit_type(self, t: TypeAnnotation) -> str:
        if t.name == "array":
            return f"{self._fmt(t.inner.name, type_)}[]"
        if t.name == "optional":
            return f"{self._fmt(t.inner.name, type_)}?"
        if t.state_inner:
            return f"State<{self._fmt(t.state_inner.name, type_)}>"
        return self._fmt(t.name, type_)

    # ── Params ────────────────────────────────────────────────────────────

    def _fmt_params(self, params: list[ParamDef]) -> str:
        if not params:
            return ""
        items = []
        for p in params:
            default = ""
            if p.default is not None:
                default = f" = {self._fmt_expr(p.default)}"
            items.append(f"{self._fmt(p.name, name)}: {self._visit_type(p.type)}{default}")
        return f"({', '.join(items)})"

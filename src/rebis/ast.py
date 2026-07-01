"""Intermediate Representation (IR) node definitions.

These dataclasses form the language-agnostic IR that sits between parsing
and every code-generation backend.  All Phase-0+ passes operate on IR nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Source location ──────────────────────────────────────────────────────

@dataclass
class Span:
    """A span of source text (1-indexed line/column, inclusive end)."""
    file: str = "<unknown>"
    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.file}:{self.start_line}:{self.start_col}"
        return f"{self.file}:{self.start_line}-{self.end_line}"


# ── Type annotations ─────────────────────────────────────────────────────

@dataclass
class TypeAnnotation:
    name: str                              # "String", "Int", "Bool", etc.
    inner: TypeAnnotation | None = None    # for T[] (inner=T), T?
    state_inner: TypeAnnotation | None = None  # for State<T>
    is_optional: bool = False
    span: Span | None = None


# ── Expression kinds ─────────────────────────────────────────────────────

@dataclass
class LiteralExpr:
    kind: str       # "string" | "int" | "float" | "bool" | "null" | "color"
    value: Any
    span: Span | None = None


@dataclass
class IdentExpr:
    name: str
    span: Span | None = None


@dataclass
class MemberExpr:
    obj: IdentExpr | MemberExpr
    member: str
    span: Span | None = None


@dataclass
class InterpolatedExpr:
    parts: list[str | IdentExpr | MemberExpr]
    span: Span | None = None


@dataclass
class UnaryExpr:
    op: str            # "!" | "-"
    operand: Expr
    span: Span | None = None


@dataclass
class BinaryExpr:
    op: str            # "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||"
    left: Expr
    right: Expr
    span: Span | None = None


@dataclass
class ArrayExpr:
    elements: list[Expr]
    span: Span | None = None


Expr = (
    LiteralExpr
    | IdentExpr
    | MemberExpr
    | InterpolatedExpr
    | UnaryExpr
    | BinaryExpr
    | ArrayExpr
)


# ── Property ─────────────────────────────────────────────────────────────

@dataclass
class Property:
    name: str
    value: Expr
    span: Span | None = None


# ── Statements (event-handler bodies) ────────────────────────────────────

@dataclass
class AssignmentStmt:
    target: MemberExpr | IdentExpr
    value: Expr
    span: Span | None = None


@dataclass
class ExprStmt:
    expr: Expr
    span: Span | None = None


@dataclass
class IfStmt:
    condition: Expr
    then_body: list[Stmt]
    else_body: IfStmt | list[Stmt] | None = None
    span: Span | None = None


@dataclass
class ForStmt:
    variable: str
    iterable: Expr
    body: list[Stmt]
    span: Span | None = None


@dataclass
class LetStmt:
    name: str
    type_annotation: TypeAnnotation | None = None
    value: Expr | None = None
    span: Span | None = None


@dataclass
class TryStmt:
    try_body: list[Stmt]
    catch_var: str
    catch_body: list[Stmt]
    span: Span | None = None


Stmt = (
    AssignmentStmt
    | ExprStmt
    | IfStmt
    | ForStmt
    | LetStmt
    | TryStmt
)


# ── Top-level declarations ───────────────────────────────────────────────

@dataclass
class ParamDef:
    name: str
    type: TypeAnnotation
    default: Expr | None = None
    span: Span | None = None


@dataclass
class ImportStmt:
    path: str
    names: list[str] | None = None  # None = wildcard import
    span: Span | None = None


@dataclass
class StateDecl:
    name: str
    type: TypeAnnotation | None = None
    value: Expr | None = None
    span: Span | None = None


@dataclass
class EventHandler:
    event: str
    params: list[ParamDef] | None = None
    body: list[Stmt] = field(default_factory=list)
    span: Span | None = None


@dataclass
class TagNode:
    """A UI tag / component instantiation — the core IR node."""
    name: str
    instance_name: str | None = None  # optional label like `text_input username`
    properties: list[Property] = field(default_factory=list)
    children: list[TagNode | PlatformBlock | RawBlock] = field(default_factory=list)
    span: Span | None = None


@dataclass
class PlatformBlock:
    platform: str
    body: list[Property | TagNode] = field(default_factory=list)
    span: Span | None = None


@dataclass
class RawBlock:
    target: str
    code: str
    span: Span | None = None


@dataclass
class EnumDef:
    name: str
    variants: list[str] = field(default_factory=list)
    exported: bool = False
    span: Span | None = None


@dataclass
class ComponentDef:
    name: str
    params: list[ParamDef] = field(default_factory=list)
    body: list[
        Property | StateDecl | TagNode | EventHandler | PlatformBlock | RawBlock
    ] = field(default_factory=list)
    exported: bool = False
    span: Span | None = None


@dataclass
class File:
    """Root IR node — one per source module."""
    imports: list[ImportStmt] = field(default_factory=list)
    declarations: list[ComponentDef | EnumDef] = field(default_factory=list)
    span: Span | None = None


# ── Visitor base ─────────────────────────────────────────────────────────

class Visitor:
    """Base visitor with default recursive walk.

    Subclasses override `visit_*` methods.  Each returns None by default
    (in-place transforms can mutate nodes; pure analyses collect data on
    the side).
    """

    def visit(self, node: Any) -> None:
        name = type(node).__name__
        handler = getattr(self, f"visit_{name}", self._default)
        handler(node)

    def _default(self, node: Any) -> None:
        for attr_name in dir(node):
            val = getattr(node, attr_name)
            if isinstance(val, list):
                for item in val:
                    if hasattr(item, "accept"):
                        item.accept(self)
            elif hasattr(val, "accept"):
                val.accept(self)


# ── Accept mixin ──────────────────────────────────────────────────────────

def _accept(self: Any, visitor: Visitor) -> None:
    visitor.visit(self)

# Attach accept to every IR node type.
for _t in (
    File,
    ImportStmt,
    ComponentDef,
    EnumDef,
    ParamDef,
    StateDecl,
    EventHandler,
    TagNode,
    PlatformBlock,
    RawBlock,
    Property,
    AssignmentStmt,
    ExprStmt,
    IfStmt,
    ForStmt,
    LetStmt,
    TryStmt,
    LiteralExpr,
    IdentExpr,
    MemberExpr,
    InterpolatedExpr,
    UnaryExpr,
    BinaryExpr,
    ArrayExpr,
):
    _t.accept = _accept

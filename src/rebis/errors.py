"""Structured error types with source-location information."""

from __future__ import annotations

from .ast import Span


# ── Base ─────────────────────────────────────────────────────────────────

class RebisError(Exception):
    """Base for all Rebis errors."""

    def __init__(
        self,
        message: str,
        span: Span | None = None,
        *,
        hint: str | None = None,
    ):
        self.message = message
        self.span = span
        self.hint = hint
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = []
        if self.span:
            parts.append(str(self.span))
        parts.append(self.message)
        s = ": ".join(parts)
        if self.hint:
            s += f"\n  hint: {self.hint}"
        return s


# ── Specific error types ─────────────────────────────────────────────────

class ParseError(RebisError):
    """The input could not be parsed (syntax error)."""

    def __init__(
        self,
        message: str,
        span: Span | None = None,
        *,
        token: str | None = None,
        expected: list[str] | None = None,
        hint: str | None = None,
    ):
        self.token = token
        self.expected = expected or []
        msg = message
        if expected:
            msg += f" (expected: {' | '.join(expected)})"
        if token:
            msg += f" at token '{token}'"
        super().__init__(msg, span, hint=hint)


class ImportError(RebisError):
    """A module import could not be resolved."""

    def __init__(
        self,
        message: str,
        span: Span | None = None,
        *,
        path: str | None = None,
        hint: str | None = None,
    ):
        self.path = path
        super().__init__(message, span, hint=hint)


class TypeError(RebisError):
    """A type mismatch or invalid type annotation."""

    def __init__(
        self,
        message: str,
        span: Span | None = None,
        *,
        expected: str | None = None,
        got: str | None = None,
        hint: str | None = None,
    ):
        self.expected = expected
        self.got = got
        msg = message
        if expected and got:
            msg += f" (expected {expected}, got {got})"
        super().__init__(msg, span, hint=hint)


class TemplateError(RebisError):
    """A template expansion or code-generation error."""
    pass


# ── Diagnostic helpers ──────────────────────────────────────────────────

def format_parse_error(
    source: str,
    error: ParseError,
    *,
    context_lines: int = 2,
) -> str:
    """Format a ParseError with source snippet for human readers."""
    if not error.span or error.span.start_line < 1:
        return str(error)

    lines = source.split("\n")
    line_idx = error.span.start_line - 1
    if line_idx >= len(lines):
        return str(error)

    parts = [str(error), ""]
    start = max(0, line_idx - context_lines)
    end = min(len(lines), line_idx + context_lines + 1)

    for i in range(start, end):
        line_no = i + 1
        marker = ">" if i == line_idx else " "
        parts.append(f" {marker} {line_no:4d} | {lines[i]}")

        if i == line_idx:
            # underline the error
            col = max(0, error.span.start_col - 1) if error.span else 0
            parts.append(f"      | {' ' * col}{'^' * 3}")

    return "\n".join(parts)

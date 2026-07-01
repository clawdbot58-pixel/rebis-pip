"""Line-based tokenizer with tab-indent tracking for the Rebis DSL."""

from __future__ import annotations

from dataclasses import dataclass
from .errors import ParseError, Span


# ── Token type constants ─────────────────────────────────────────────────

class TokenType:
    NEWLINE        = "NEWLINE"
    INDENT         = "INDENT"
    DEDENT         = "DEDENT"
    EOF            = "EOF"

    IDENTIFIER     = "IDENTIFIER"
    KEYWORD        = "KEYWORD"
    STRING         = "STRING"
    INT            = "INT"
    FLOAT          = "FLOAT"
    BOOL           = "BOOL"
    NULL           = "NULL"
    COLOR          = "COLOR"

    EQUALS         = "EQUALS"          # =
    DOT            = "DOT"             # .
    COMMA          = "COMMA"           # ,
    COLON          = "COLON"           # :
    ARROW          = "ARROW"           # ->

    OPEN_BRACE     = "OPEN_BRACE"      # {
    CLOSE_BRACE    = "CLOSE_BRACE"     # }
    OPEN_PAREN     = "OPEN_PAREN"      # (
    CLOSE_PAREN    = "CLOSE_PAREN"     # )
    OPEN_BRACKET   = "OPEN_BRACKET"    # [
    CLOSE_BRACKET  = "CLOSE_BRACKET"   # ]

    PIPE           = "PIPE"            # |

    # Operators
    EQ_EQ          = "EQ_EQ"           # ==
    NOT_EQ         = "NOT_EQ"          # !=
    LTE            = "LTE"             # <=
    GTE            = "GTE"             # >=
    LT             = "LT"              # <
    GT             = "GT"              # >
    AND_AND        = "AND_AND"         # &&
    OR_OR          = "OR_OR"           # ||
    BANG           = "BANG"            # !
    PLUS           = "PLUS"            # +
    MINUS          = "MINUS"           # -
    STAR           = "STAR"            # *
    SLASH          = "SLASH"           # /
    QUESTION       = "QUESTION"        # ?

    # Directives
    AT_PLATFORM    = "AT_PLATFORM"     # @platform
    RAW_OPEN       = "RAW_OPEN"        # {{
    RAW_CLOSE      = "RAW_CLOSE"       # }}
    RAW_CLOSE_TAG  = "RAW_CLOSE_TAG"   # {{/


KEYWORDS: dict[str, str] = {
    "import":    "import",
    "export":    "export",
    "component": "component",
    "enum":      "enum",
    "state":     "state",
    "on":        "on",
    "if":        "if",
    "else":      "else",
    "for":       "for",
    "in":        "in",
    "let":       "let",
    "try":       "try",
    "catch":     "catch",
    "true":      "true",
    "false":     "false",
    "null":      "null",
}

# Keywords that are also reserved — parser uses these tokens:
RESERVED_KEYWORDS = {
    "import", "export", "component", "enum", "state",
    "on", "if", "else", "for", "in", "let", "try", "catch",
    "true", "false", "null",
}


# ── Token ────────────────────────────────────────────────────────────────

class Token:
    """A single token from the tokenizer."""

    __slots__ = ("type", "value", "line", "col")

    def __init__(
        self,
        type: str,
        value: str | int | float | bool | None = None,
        line: int = 0,
        col: int = 0,
    ):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        val = f"={self.value!r}" if self.value is not None else ""
        return f"Token({self.type}{val} @{self.line}:{self.col})"


# ── Tokenizer ────────────────────────────────────────────────────────────

class Tokenizer:
    """Line-based tokenizer producing a flat token stream.

    Handles tab-based INDENT/DEDENT, line comments, all literal types,
    and emits NEWLINE at the end of each logical line.
    """

    def __init__(self, source: str, file: str = "<unknown>"):
        self.source = source
        self.file = file
        self.tokens: list[Token] = []
        self._pos = 0

    def tokenize(self) -> list[Token]:
        """Run tokenization and return the token list."""
        self._tokenize_lines()
        self.tokens.append(Token(TokenType.EOF, line=0, col=0))
        return self.tokens

    # ── Helpers ──────────────────────────────────────────────────────

    def _peek(self) -> Token | None:
        if self._pos < len(self.tokens):
            return self.tokens[self._pos]
        return None

    def _pop(self) -> Token:
        tok = self.tokens[self._pos]
        self._pos += 1
        return tok

    # ── Line processing ──────────────────────────────────────────────

    def _tokenize_lines(self) -> None:
        """Main tokenization — split into lines, track indent, emit tokens."""
        lines = self.source.split("\n")
        indent_stack: list[int] = [0]  # tracks tab-count per level

        for line_no, raw_line in enumerate(lines, start=1):
            # Skip pure comment lines (first non-tab is //)
            stripped = raw_line.lstrip("\t")
            stripped_no_spaces = stripped.lstrip()
            pure_comment = stripped_no_spaces.startswith("//")

            stripped_line = raw_line.rstrip("\n\r")  # keep trailing tabs for lex? no, we strip tabs already counted
            tabs = 0
            while tabs < len(raw_line) and raw_line[tabs] == "\t":
                tabs += 1

            # Line content after leading tabs
            content = raw_line[tabs:]

            # Blank line — emit NEWLINE but don't change indent level
            if not content.strip() or pure_comment:
                self._emit(Token(TokenType.NEWLINE, line=line_no, col=1))
                continue

            # Strip trailing comment (first // not inside a string)
            content = self._strip_trailing_comment(content)

            # Handle indent / dedent
            if tabs > indent_stack[-1]:
                indent_stack.append(tabs)
                self._emit(Token(TokenType.INDENT, line=line_no, col=1))
            elif tabs < indent_stack[-1]:
                while indent_stack and tabs < indent_stack[-1]:
                    indent_stack.pop()
                    self._emit(Token(TokenType.DEDENT, line=line_no, col=1))
                if indent_stack and tabs != indent_stack[-1]:
                    self._error(line_no, 1,
                        f"Inconsistent indentation (expected {indent_stack[-1]} tabs, got {tabs})")

            # Lex the line
            self._lex_line(content, line_no)

            # End-of-line
            self._emit(Token(TokenType.NEWLINE, line=line_no, col=len(content) + 1))

        # Close all remaining indent levels
        while len(indent_stack) > 1:
            indent_stack.pop()
            self._emit(Token(TokenType.DEDENT, line=len(lines), col=1))

    @staticmethod
    def _strip_trailing_comment(line: str) -> str:
        """Remove a trailing // comment, respecting strings."""
        in_string = False
        for i, ch in enumerate(line):
            if ch == '"':
                in_string = not in_string
            if not in_string and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                return line[:i].rstrip()
        return line.rstrip()

    # ── Line lexer ───────────────────────────────────────────────────

    def _lex_line(self, s: str, line_no: int) -> None:
        """Tokenize the content of a single line (no indentation prefix)."""
        col = 1  # 1-indexed column within the line
        i = 0
        while i < len(s):
            ch = s[i]

            # Whitespace (spaces between tokens)
            if ch == " " or ch == "\t":
                i += 1
                col += 1
                continue

            # String literal
            if ch == '"':
                tok, consumed = self._lex_string(s, i, line_no, col)
                self._emit(tok)
                i += consumed
                col += consumed
                continue

            # Color literal
            if ch == "#" and i + 1 < len(s) and _is_hex(s[i + 1]):
                tok, consumed = self._lex_color(s, i, line_no, col)
                self._emit(tok)
                i += consumed
                col += consumed
                continue

            # Number
            if ch.isdigit() or (ch == "-" and i + 1 < len(s) and s[i + 1].isdigit()):
                tok, consumed = self._lex_number(s, i, line_no, col)
                self._emit(tok)
                i += consumed
                col += consumed
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == "_":
                tok, consumed = self._lex_ident(s, i, line_no, col)
                self._emit(tok)
                i += consumed
                col += consumed
                continue

            # @platform
            if ch == "@":
                tok, consumed = self._lex_directive(s, i, line_no, col)
                self._emit(tok)
                i += consumed
                col += consumed
                continue

            # Raw blocks {{, }}, {{/
            if ch == "{" and i + 1 < len(s) and s[i + 1] == "{":
                self._emit(Token(TokenType.RAW_OPEN, line=line_no, col=col))
                i += 2
                col += 2
                continue
            if ch == "}" and i + 1 < len(s) and s[i + 1] == "}":
                if i + 2 < len(s) and s[i + 2] == "}":
                    # {{{ is not a token — just single }}
                    pass
                self._emit(Token(TokenType.RAW_CLOSE, line=line_no, col=col))
                i += 2
                col += 2
                continue
            if (ch == "{" and i + 2 < len(s)
                    and s[i + 1] == "{" and s[i + 2] == "/"):
                self._emit(Token(TokenType.RAW_CLOSE_TAG, line=line_no, col=col))
                i += 3
                col += 3
                continue

            # Multi-char operators
            two = s[i:i+2] if i + 1 < len(s) else ""
            if two == "==":
                self._emit(Token(TokenType.EQ_EQ, line=line_no, col=col))
                i += 2; col += 2; continue
            if two == "!=":
                self._emit(Token(TokenType.NOT_EQ, line=line_no, col=col))
                i += 2; col += 2; continue
            if two == "<=":
                self._emit(Token(TokenType.LTE, line=line_no, col=col))
                i += 2; col += 2; continue
            if two == ">=":
                self._emit(Token(TokenType.GTE, line=line_no, col=col))
                i += 2; col += 2; continue
            if two == "&&":
                self._emit(Token(TokenType.AND_AND, line=line_no, col=col))
                i += 2; col += 2; continue
            if two == "||":
                self._emit(Token(TokenType.OR_OR, line=line_no, col=col))
                i += 2; col += 2; continue
            if two == "/*":
                # Block comment start — skip to */
                i += 2; col += 2
                while i < len(s):
                    if s[i:i+2] == "*/":
                        i += 2; col += 2
                        break
                    i += 1; col += 1
                continue

            # Single-char operators / punctuation
            if ch == "=":
                self._emit(Token(TokenType.EQUALS, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == ".":
                self._emit(Token(TokenType.DOT, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == ",":
                self._emit(Token(TokenType.COMMA, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == ":":
                self._emit(Token(TokenType.COLON, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "{":
                self._emit(Token(TokenType.OPEN_BRACE, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "}":
                self._emit(Token(TokenType.CLOSE_BRACE, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "(":
                self._emit(Token(TokenType.OPEN_PAREN, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == ")":
                self._emit(Token(TokenType.CLOSE_PAREN, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "[":
                self._emit(Token(TokenType.OPEN_BRACKET, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "]":
                self._emit(Token(TokenType.CLOSE_BRACKET, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "|":
                self._emit(Token(TokenType.PIPE, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "!":
                self._emit(Token(TokenType.BANG, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "+":
                self._emit(Token(TokenType.PLUS, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "-":
                self._emit(Token(TokenType.MINUS, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "*":
                self._emit(Token(TokenType.STAR, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "/":
                self._emit(Token(TokenType.SLASH, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "<":
                self._emit(Token(TokenType.LT, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == ">":
                self._emit(Token(TokenType.GT, line=line_no, col=col))
                i += 1; col += 1; continue
            if ch == "?":
                self._emit(Token(TokenType.QUESTION, line=line_no, col=col))
                i += 1; col += 1; continue

            # Unknown character
            self._error(line_no, col, f"Unexpected character: {ch!r}")

        # end while

    # ── Sub-lexers ───────────────────────────────────────────────────

    def _lex_string(self, s: str, start: int, line_no: int, col: int) -> tuple[Token, int]:
        """Lex a double-quoted string. Returns (token, chars_consumed)."""
        assert s[start] == '"'
        i = start + 1
        chars = []
        while i < len(s):
            ch = s[i]
            if ch == '"':
                i += 1
                value = "".join(chars)
                return (Token(TokenType.STRING, value, line_no, col), i - start)
            if ch == "\\" and i + 1 < len(s):
                esc = s[i + 1]
                if esc == '"' or esc == "\\":
                    chars.append(esc)
                    i += 2
                elif esc == "n":
                    chars.append("\n"); i += 2
                elif esc == "t":
                    chars.append("\t"); i += 2
                elif esc == "${":
                    chars.append("${"); i += 3  # wait, this is \$ not \$
                    # Actually, \$ is just "$", then { follows as normal
                    # Let me handle this differently
                else:
                    chars.append(esc)
                    i += 2
                continue
            if ch == "$" and i + 1 < len(s) and s[i + 1] == "{":
                # Interpolation marker — keep as raw "${..." in the value
                chars.append("$")
                chars.append("{")
                i += 2
                depth = 1  # we already consumed the opening "{"
                while i < len(s) and depth > 0:
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                    chars.append(s[i])
                    i += 1
                continue
            chars.append(ch)
            i += 1

        self._error(line_no, col, "Unterminated string literal")
        # unreachable

    def _lex_color(self, s: str, start: int, line_no: int, col: int) -> tuple[Token, int]:
        """Lex a hex color literal."""
        i = start + 1
        while i < len(s) and _is_hex(s[i]):
            i += 1
        value = s[start:i]
        return (Token(TokenType.COLOR, value, line_no, col), i - start)

    def _lex_number(self, s: str, start: int, line_no: int, col: int) -> tuple[Token, int]:
        """Lex an integer or float literal."""
        i = start
        if s[i] == "-":
            i += 1
        # Allow leading zeros, but detect float by decimal point
        is_float = False
        while i < len(s) and s[i].isdigit():
            i += 1
        if i < len(s) and s[i] == "." and i + 1 < len(s) and s[i + 1].isdigit():
            is_float = True
            i += 1
            while i < len(s) and s[i].isdigit():
                i += 1
        raw = s[start:i]
        if is_float:
            return (Token(TokenType.FLOAT, float(raw), line_no, col), i - start)
        return (Token(TokenType.INT, int(raw), line_no, col), i - start)

    def _lex_ident(self, s: str, start: int, line_no: int, col: int) -> tuple[Token, int]:
        """Lex an identifier or keyword."""
        i = start
        while i < len(s) and (s[i].isalnum() or s[i] == "_"):
            i += 1
        word = s[start:i]
        if word in RESERVED_KEYWORDS:
            tok_type = TokenType.KEYWORD
        else:
            tok_type = TokenType.IDENTIFIER
        return (Token(tok_type, word, line_no, col), i - start)

    def _lex_directive(self, s: str, start: int, line_no: int, col: int) -> tuple[Token, int]:
        """Lex a @-directive."""
        i = start + 1
        while i < len(s) and (s[i].isalpha() or s[i] == "_"):
            i += 1
        word = s[start:i]
        if word == "@platform":
            return (Token(TokenType.AT_PLATFORM, line=line_no, col=col), i - start)
        self._error(line_no, col, f"Unknown directive: {word}")

    # ── Emit / error ─────────────────────────────────────────────────

    def _emit(self, token: Token) -> None:
        self.tokens.append(token)

    def _error(self, line: int, col: int, msg: str) -> None:
        span = Span(self.file, line, col, line, col + 1)
        raise ParseError(msg, span)


# ── Helpers ─────────────────────────────────────────────────────────────

def _is_hex(ch: str) -> bool:
    return ch.isdigit() or ("a" <= ch <= "f") or ("A" <= ch <= "F")

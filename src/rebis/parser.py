"""Recursive-descent parser for the Rebis DSL.

Produces an IR tree (from .ast) by consuming the token stream produced
by the Tokenizer.  Handles the indent-based two-position rule for tags,
component definitions, event handlers, expressions, and statements.
"""

from __future__ import annotations

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
    Stmt,
    TagNode,
    TryStmt,
    TypeAnnotation,
    UnaryExpr,
)
from .errors import ParseError
from .tokenizer import Token, TokenType


# ── Simple expression node for function calls (inline definition) ──
# Avoids circular import / separate file during Phase 0.

class ExprCall:
    """A function-call expression: fn(arg, arg, ...)."""
    def __init__(self, fn: IdentExpr | MemberExpr, args: list, span: Span | None = None):
        self.fn = fn
        self.args = args
        self.span = span
    def __repr__(self):
        return f"call({self.fn}, args={self.args})"


# ── Parser ─────────────────────────────────────────────────────────

class Parser:
    """Consumes tokens and produces a File IR node."""

    def __init__(self, tokens: list[Token], file: str = "<unknown>"):
        self.tokens = tokens
        self.file = file
        self.pos = 0

    # ── Public entry ────────────────────────────────────────────────

    def parse(self) -> File:
        result = self._parse_file()
        remaining = self.peek()
        if remaining.type != TokenType.EOF:
            raise ParseError(
                f"Unexpected token after top-level declarations: {remaining.type}",
                self._span_at(remaining),
                token=remaining.value,
            )
        return result

    # ── Low-level helpers ───────────────────────────────────────────

    def _at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def peek_type(self, *types: str) -> bool:
        if self._at_end():
            return False
        return self.tokens[self.pos].type in types

    def peek_value(self) -> str | int | float | bool | None:
        if self._at_end():
            return None
        return self.tokens[self.pos].value

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, *types: str) -> Token:
        if self._at_end():
            expected = " | ".join(types) if types else "token"
            raise ParseError(f"Unexpected end of input (expected {expected})")
        tok = self.tokens[self.pos]
        if types and tok.type not in types:
            span = self._span_at(tok)
            raise ParseError(
                f"Expected {' | '.join(types)}, got {tok.type}",
                span,
                token=tok.value,
                expected=list(types),
            )
        return self.advance()

    def _match(self, *types: str) -> Token | None:
        if not self._at_end() and self.tokens[self.pos].type in types:
            return self.advance()
        return None

    def _match_keyword(self, keyword: str) -> Token | None:
        tok = self.peek()
        if tok.type == TokenType.KEYWORD and tok.value == keyword:
            return self.advance()
        return None

    def _consume_newlines(self) -> None:
        while self.peek_type(TokenType.NEWLINE):
            self.advance()

    def _span_at(self, tok: Token) -> Span:
        return Span(self.file, tok.line, tok.col, tok.line, tok.col + 1)

    def _span_between(self, start: Token, end: Token) -> Span:
        return Span(
            self.file,
            start.line, start.col,
            end.line, end.col + (1 if isinstance(end.value, str) else 1),
        )

    # ── File ────────────────────────────────────────────────────────

    def _parse_file(self) -> File:
        imports: list[ImportStmt] = []
        declarations: list[ComponentDef | EnumDef] = []
        while not self._at_end():
            self._consume_newlines()
            if self._at_end() or self.peek().type == TokenType.EOF:
                break
            tok = self.peek()
            if tok.type == TokenType.KEYWORD and tok.value == "import":
                imports.append(self._parse_import())
            elif (tok.type == TokenType.KEYWORD
                  and tok.value in ("component", "export", "enum")):
                declarations.append(self._parse_top_level_decl())
            else:
                span = self._span_at(tok)
                raise ParseError(
                    f"Expected top-level declaration, got {tok.value}",
                    span,
                    token=tok.value,
                )
        return File(imports, declarations)

    def _parse_top_level_decl(self) -> ComponentDef | EnumDef:
        exported = False
        if self._match_keyword("export"):
            exported = True
        tok = self.peek()
        if tok.type == TokenType.KEYWORD:
            if tok.value == "component":
                return self._parse_component_def(exported)
            elif tok.value == "enum":
                return self._parse_enum_def(exported)
        span = self._span_at(tok)
        raise ParseError(
            f"Expected 'component' or 'enum' after 'export', got {tok.value}",
            span,
            token=tok.value,
        )

    # ── Import ──────────────────────────────────────────────────────

    def _parse_import(self) -> ImportStmt:
        start = self.advance()  # consume 'import'
        names: list[str] | None = None

        if self._match(TokenType.OPEN_BRACE):
            # { Name, Name }
            names = []
            first = self._expect(TokenType.IDENTIFIER)
            names.append(str(first.value))
            while self._match(TokenType.COMMA):
                ident = self._expect(TokenType.IDENTIFIER)
                names.append(str(ident.value))
            self._expect(TokenType.CLOSE_BRACE)
        elif (self.peek_type(TokenType.IDENTIFIER)
              and self.peek_value() not in ("from", "export", "component")):
            # Single name without braces: import Name "./path"
            name_tok = self.advance()
            names = [str(name_tok.value)]

        # Optional 'from' keyword (syntactic sugar)
        if self.peek_type(TokenType.IDENTIFIER) and self.peek_value() == "from":
            self.advance()
        path_tok = self._expect(TokenType.STRING)
        return ImportStmt(
            path=str(path_tok.value),
            names=names,
            span=self._span_between(start, path_tok),
        )

    # ── Component definition ────────────────────────────────────────

    def _parse_component_def(self, exported: bool = False) -> ComponentDef:
        start = self.advance()  # consume 'component'
        name_tok = self._expect(TokenType.IDENTIFIER)
        name = str(name_tok.value)
        params = self._parse_params()
        self._expect(TokenType.OPEN_BRACE)
        self._consume_newlines()
        body: list = []
        if self.peek_type(TokenType.INDENT):
            body = self._parse_body_block()
        end = self._expect(TokenType.CLOSE_BRACE)
        self._consume_newlines()
        return ComponentDef(
            name=name, params=params, body=body,
            exported=exported, span=self._span_between(start, end),
        )

    def _parse_enum_def(self, exported: bool = False) -> EnumDef:
        start = self.advance()  # consume 'enum'
        name_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.OPEN_BRACE)
        self._consume_newlines()
        variants: list[str] = []
        if self.peek_type(TokenType.INDENT):
            self.advance()
            while not self.peek_type(TokenType.DEDENT, TokenType.EOF):
                if self.peek_type(TokenType.NEWLINE):
                    self.advance()
                    continue
                if self.peek_type(TokenType.CLOSE_BRACE):
                    break
                ident = self._expect(TokenType.IDENTIFIER)
                variants.append(str(ident.value))
            self._expect(TokenType.DEDENT)
        else:
            # Inline variants: enum X { a b c }
            while not self.peek_type(TokenType.CLOSE_BRACE, TokenType.EOF):
                if self.peek_type(TokenType.NEWLINE):
                    self.advance()
                    continue
                ident = self._expect(TokenType.IDENTIFIER)
                variants.append(str(ident.value))
        self._expect(TokenType.CLOSE_BRACE)
        self._consume_newlines()
        return EnumDef(
            name=str(name_tok.value), variants=variants,
            exported=exported, span=self._span_between(start, name_tok),
        )

    def _parse_params(self) -> list[ParamDef]:
        if not self.peek_type(TokenType.OPEN_PAREN):
            return []
        self.advance()
        params: list[ParamDef] = []
        while not self.peek_type(TokenType.CLOSE_PAREN):
            if params:
                self._expect(TokenType.COMMA)
            params.append(self._parse_param())
        self._expect(TokenType.CLOSE_PAREN)
        return params

    def _parse_param(self) -> ParamDef:
        name_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.COLON)
        typ = self._parse_type()
        default = None
        if self._match(TokenType.EQUALS):
            default = self._parse_expr()
        return ParamDef(
            name=str(name_tok.value), type=typ, default=default,
            span=self._span_at(name_tok),
        )

    # ── Body block (inside component { }) ─────────────────────────

    def _parse_body_block(self) -> list:
        """Parse an INDENT … DEDENT block (component body)."""
        items: list = []
        self._expect(TokenType.INDENT)
        while not self.peek_type(TokenType.DEDENT, TokenType.EOF):
            if self.peek_type(TokenType.NEWLINE):
                self.advance()
                continue
            items.append(self._parse_body_item())
        self._expect(TokenType.DEDENT)
        return items

    def _parse_body_item(self):
        """One item inside a component body."""
        tok = self.peek()

        if tok.type == TokenType.KEYWORD and tok.value == "state":
            return self._parse_state_decl()
        if tok.type == TokenType.KEYWORD and tok.value == "on":
            return self._parse_event_handler()
        if tok.type == TokenType.AT_PLATFORM:
            return self._parse_platform_block()
        if tok.type == TokenType.RAW_OPEN:
            return self._parse_raw_block()
        if tok.type in (TokenType.IDENTIFIER, TokenType.KEYWORD):
            if self._is_property_start():
                return self._parse_property()
            return self._parse_tag()

        span = self._span_at(tok)
        raise ParseError(f"Unexpected in component body: {tok}", span)

    def _is_property_start(self) -> bool:
        """Check if current ident starts a property (ident = ...) or a tag."""
        if self._at_end() or self.pos + 1 >= len(self.tokens):
            return False
        return self.tokens[self.pos + 1].type == TokenType.EQUALS

    # ── Tag (core of the UI tree) ──────────────────────────────────

    def _parse_tag(self) -> TagNode:
        """Parse a tag line and optional indented children.

        Two-position rule:
          tag prop=val  → leaf (no children)
          tag           → may have children on next indented lines
        """
        start = self._expect(TokenType.IDENTIFIER, TokenType.KEYWORD)
        name = str(start.value)

        # Optional instance name (second bare ident before any `=`)
        instance_name: str | None = None
        if (self.peek_type(TokenType.IDENTIFIER)
                and not self._is_property_start()):
            instance_name = str(self.advance().value)

        props: list[Property] = []
        while self.peek_type(TokenType.IDENTIFIER):
            props.append(self._parse_property())

        self._expect(TokenType.NEWLINE)

        children: list[TagNode | PlatformBlock | RawBlock] = []
        if self.peek_type(TokenType.INDENT):
            self.advance()
            while not self.peek_type(TokenType.DEDENT, TokenType.EOF):
                if self.peek_type(TokenType.NEWLINE):
                    self.advance()
                    continue
                tok = self.peek()
                if tok.type == TokenType.AT_PLATFORM:
                    children.append(self._parse_platform_block())
                elif tok.type == TokenType.RAW_OPEN:
                    children.append(self._parse_raw_block())
                elif self._is_property_start():
                    # Indented property (e.g. visible=... under text)
                    props.append(self._parse_property())
                else:
                    children.append(self._parse_tag())
            self._expect(TokenType.DEDENT)

        return TagNode(
            name=name, instance_name=instance_name,
            properties=props, children=children,
            span=self._span_between(start, start),
        )

    # ── Property ────────────────────────────────────────────────────

    def _parse_property(self) -> Property:
        start = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.EQUALS)
        value = self._parse_expr()
        return Property(
            name=str(start.value), value=value,
            span=self._span_between(start, start),
        )

    # ── State declaration ───────────────────────────────────────────

    def _parse_state_decl(self) -> StateDecl:
        start = self.advance()
        name_tok = self._expect(TokenType.IDENTIFIER)
        name = str(name_tok.value)
        typ: TypeAnnotation | None = None
        if self._match(TokenType.COLON):
            typ = self._parse_type()
        self._expect(TokenType.EQUALS)
        value = self._parse_expr()
        self._consume_newlines()
        return StateDecl(
            name=name, type=typ, value=value,
            span=self._span_between(start, name_tok),
        )

    # ── Event handler ───────────────────────────────────────────────

    def _parse_event_handler(self) -> EventHandler:
        start = self.advance()
        event_tok = self._expect(TokenType.IDENTIFIER)
        event = str(event_tok.value)
        params: list[ParamDef] | None = None
        if self.peek_type(TokenType.OPEN_PAREN):
            params = self._parse_params()
        body: list[Stmt] = []
        if self.peek_type(TokenType.OPEN_BRACE):
            body = self._parse_brace_block()
        else:
            self._consume_newlines()
            if self.peek_type(TokenType.INDENT):
                body = self._parse_body_block()
            elif self.peek_type(TokenType.OPEN_BRACE):
                body = self._parse_brace_block()
        return EventHandler(
            event=event, params=params, body=body,
            span=self._span_between(start, event_tok),
        )

    # ── Platform block ──────────────────────────────────────────────

    def _parse_platform_block(self) -> PlatformBlock:
        start = self.advance()
        self._expect(TokenType.OPEN_PAREN)
        plat_tok = self._expect(TokenType.IDENTIFIER)
        platform = str(plat_tok.value)
        self._expect(TokenType.CLOSE_PAREN)
        body: list[Property | TagNode] = []
        if self.peek_type(TokenType.OPEN_BRACE):
            self.advance()
            self._consume_newlines()
            if self.peek_type(TokenType.INDENT):
                self.advance()
                while not self.peek_type(TokenType.DEDENT, TokenType.CLOSE_BRACE, TokenType.EOF):
                    if self.peek_type(TokenType.NEWLINE):
                        self.advance()
                        continue
                    body.append(self._parse_body_item())
                self._match(TokenType.DEDENT)
            self._expect(TokenType.CLOSE_BRACE)
        self._consume_newlines()
        return PlatformBlock(
            platform=platform, body=body,
            span=self._span_between(start, plat_tok),
        )

    # ── Raw block ───────────────────────────────────────────────────

    def _parse_raw_block(self) -> RawBlock:
        start = self.advance()
        target_tok = self._expect(TokenType.IDENTIFIER)
        target = str(target_tok.value)
        self._consume_newlines()
        raw_tokens: list[Token] = []
        while not self.peek_type(TokenType.RAW_CLOSE_TAG, TokenType.EOF):
            if self.peek_type(TokenType.RAW_CLOSE):
                break
            raw_tokens.append(self.advance())
        self._expect(TokenType.RAW_CLOSE_TAG)
        self._match(TokenType.IDENTIFIER)
        code = " ".join(str(t.value or t.type) for t in raw_tokens)
        return RawBlock(
            target=target, code=code,
            span=self._span_between(start, target_tok),
        )

    # ── Brace block (event handler bodies) ──────────────────────────

    def _parse_brace_block(self) -> list[Stmt]:
        """Parse { statement* }, handling internal INDENT/DEDENT."""
        self._expect(TokenType.OPEN_BRACE)
        self._consume_newlines()
        stmts: list[Stmt] = []

        # Inside braces content is indented in practice — handle INDENT
        if self.peek_type(TokenType.INDENT):
            self.advance()
            while not self.peek_type(TokenType.DEDENT, TokenType.EOF):
                if self.peek_type(TokenType.NEWLINE):
                    self.advance()
                    continue
                if self.peek_type(TokenType.CLOSE_BRACE):
                    break
                stmts.append(self._parse_stmt())
            self._expect(TokenType.DEDENT)
        else:
            # No indent — parse until close brace
            while not self.peek_type(TokenType.CLOSE_BRACE, TokenType.EOF):
                if self.peek_type(TokenType.NEWLINE):
                    self.advance()
                    continue
                stmts.append(self._parse_stmt())

        self._expect(TokenType.CLOSE_BRACE)
        self._consume_newlines()
        return stmts

    # ── Type helpers ───────────────────────────────────────────────

    def _parse_type(self) -> TypeAnnotation:
        tok = self._expect(TokenType.IDENTIFIER)
        name = str(tok.value)
        inner: TypeAnnotation | None = None
        state_inner: TypeAnnotation | None = None
        is_optional = False

        if name == "State" and self.peek_type(TokenType.LT):
            self.advance()
            state_inner = self._parse_type()
            self._expect(TokenType.GT)

        if self._match(TokenType.OPEN_BRACKET):
            self._expect(TokenType.CLOSE_BRACKET)
            inner = TypeAnnotation(name=name)
            name = "array"
        elif self._match(TokenType.QUESTION):
            is_optional = True
            inner = TypeAnnotation(name=name)
            name = "optional"

        return TypeAnnotation(
            name=name, inner=inner, state_inner=state_inner,
            is_optional=is_optional, span=self._span_at(tok),
        )

    # ── Statements (event handler body) ─────────────────────────────

    def _parse_stmt(self) -> Stmt:
        tok = self.peek()
        if tok.type == TokenType.KEYWORD:
            kw = str(tok.value) if tok.value else ""
            if kw == "if":
                return self._parse_if_stmt()
            elif kw == "for":
                return self._parse_for_stmt()
            elif kw == "let":
                return self._parse_let_stmt()
            elif kw == "try":
                return self._parse_try_stmt()

        expr = self._parse_expr()
        if self._match(TokenType.EQUALS):
            if not isinstance(expr, (IdentExpr, MemberExpr)):
                raise ParseError("Invalid assignment target", getattr(expr, "span", None))
            value = self._parse_expr()
            self._consume_newlines()
            return AssignmentStmt(target=expr, value=value, span=getattr(expr, "span", None))

        self._consume_newlines()
        return ExprStmt(expr=expr, span=getattr(expr, "span", None))

    def _parse_if_stmt(self) -> IfStmt:
        start = self.advance()
        condition = self._parse_expr()
        then_body = self._parse_brace_block()
        else_body: IfStmt | list[Stmt] | None = None
        if self._match_keyword("else"):
            if self._match_keyword("if"):
                else_body = self._parse_if_stmt()
            else:
                else_body = self._parse_brace_block()
        return IfStmt(
            condition=condition, then_body=then_body, else_body=else_body,
            span=self._span_between(start, start),
        )

    def _parse_for_stmt(self) -> ForStmt:
        start = self.advance()
        var_tok = self._expect(TokenType.IDENTIFIER)
        self._match_keyword("in") or self._expect(TokenType.IDENTIFIER)
        iterable = self._parse_expr()
        body = self._parse_brace_block()
        return ForStmt(
            variable=str(var_tok.value), iterable=iterable, body=body,
            span=self._span_between(start, var_tok),
        )

    def _parse_let_stmt(self) -> LetStmt:
        start = self.advance()
        name_tok = self._expect(TokenType.IDENTIFIER)
        name = str(name_tok.value)
        typ: TypeAnnotation | None = None
        if self._match(TokenType.COLON):
            typ = self._parse_type()
        value = None
        if self._match(TokenType.EQUALS):
            value = self._parse_expr()
        self._consume_newlines()
        return LetStmt(
            name=name, type_annotation=typ, value=value,
            span=self._span_between(start, name_tok),
        )

    def _parse_try_stmt(self) -> TryStmt:
        start = self.advance()
        try_body = self._parse_brace_block()
        self._match_keyword("catch")
        catch_var_tok = self._expect(TokenType.IDENTIFIER)
        catch_body = self._parse_brace_block()
        return TryStmt(
            try_body=try_body, catch_var=str(catch_var_tok.value),
            catch_body=catch_body, span=self._span_between(start, catch_var_tok),
        )

    # ── Expressions ─────────────────────────────────────────────────

    def _parse_expr(self):
        """Expression parser.  Precedence: lowest first."""
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self._match(TokenType.OR_OR):
            op_tok = self.tokens[self.pos - 1]
            right = self._parse_and()
            left = BinaryExpr("||", left, right, span=self._span_between(op_tok, op_tok))
        return left

    def _parse_and(self):
        left = self._parse_comparison()
        while self._match(TokenType.AND_AND):
            op_tok = self.tokens[self.pos - 1]
            right = self._parse_comparison()
            left = BinaryExpr("&&", left, right, span=self._span_between(op_tok, op_tok))
        return left

    def _parse_comparison(self):
        left = self._parse_additive()
        for op_type in (TokenType.EQ_EQ, TokenType.NOT_EQ,
                        TokenType.LT, TokenType.GT,
                        TokenType.LTE, TokenType.GTE):
            if self._match(op_type):
                op_tok = self.tokens[self.pos - 1]
                right = self._parse_additive()
                return BinaryExpr(
                    self._op_str(op_tok.type), left, right,
                    span=self._span_between(op_tok, op_tok),
                )
        return left

    def _parse_additive(self):
        """+ and − (binary)."""
        left = self._parse_unary()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op_tok = self.tokens[self.pos - 1]
            right = self._parse_unary()
            left = BinaryExpr(
                self._op_str(op_tok.type), left, right,
                span=self._span_between(op_tok, op_tok),
            )
        return left

    def _parse_unary(self):
        if self._match(TokenType.BANG, TokenType.MINUS):
            op_tok = self.tokens[self.pos - 1]
            operand = self._parse_unary()
            return UnaryExpr(
                self._op_str(op_tok.type), operand,
                span=self._span_between(op_tok, op_tok),
            )
        return self._parse_primary()

    def _parse_primary(self):
        """Literals, identifiers, member access, arrays, parens, calls."""
        tok = self.peek()

        if self._match(TokenType.OPEN_PAREN):
            inner = self._parse_expr()
            self._expect(TokenType.CLOSE_PAREN)
            return inner

        if self._match(TokenType.OPEN_BRACKET):
            elements: list = []
            if not self.peek_type(TokenType.CLOSE_BRACKET):
                elements.append(self._parse_expr())
                while self._match(TokenType.COMMA):
                    elements.append(self._parse_expr())
            self._expect(TokenType.CLOSE_BRACKET)
            return ArrayExpr(elements=elements)

        if tok.type == TokenType.STRING:
            self.advance()
            raw = str(tok.value) if tok.value else ""
            parts = self._parse_interpolation(raw)
            if len(parts) == 1 and isinstance(parts[0], str):
                return LiteralExpr("string", raw, span=self._span_at(tok))
            return InterpolatedExpr(parts, span=self._span_at(tok))

        if tok.type == TokenType.KEYWORD and tok.value == "true":
            self.advance()
            return LiteralExpr("bool", True, span=self._span_at(tok))
        if tok.type == TokenType.KEYWORD and tok.value == "false":
            self.advance()
            return LiteralExpr("bool", False, span=self._span_at(tok))
        if tok.type == TokenType.KEYWORD and tok.value == "null":
            self.advance()
            return LiteralExpr("null", None, span=self._span_at(tok))

        if tok.type == TokenType.INT:
            self.advance()
            return LiteralExpr("int", int(tok.value) if tok.value is not None else 0,
                               span=self._span_at(tok))
        if tok.type == TokenType.FLOAT:
            self.advance()
            return LiteralExpr("float", float(tok.value) if tok.value is not None else 0.0,
                               span=self._span_at(tok))

        if tok.type == TokenType.COLOR:
            self.advance()
            return LiteralExpr("color", str(tok.value) if tok.value else "",
                               span=self._span_at(tok))

        # Identifier → possibly member expr → possibly function call
        if tok.type in (TokenType.IDENTIFIER, TokenType.KEYWORD):
            self.advance()
            expr: IdentExpr | MemberExpr = IdentExpr(
                str(tok.value) if tok.value else "",
                span=self._span_at(tok),
            )
            while self._match(TokenType.DOT):
                member_tok = self._expect(TokenType.IDENTIFIER)
                expr = MemberExpr(
                    expr, str(member_tok.value) if member_tok.value else "",
                    span=self._span_between(tok, member_tok),
                )

            # Function call: ident(...)  or  expr.method(...)
            if self.peek_type(TokenType.OPEN_PAREN):
                self.advance()
                args: list = []
                if not self.peek_type(TokenType.CLOSE_PAREN):
                    args.append(self._parse_expr())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokenType.CLOSE_PAREN)
                return ExprCall(expr, args, span=getattr(expr, "span", None))

            return expr

        span = self._span_at(tok)
        raise ParseError(f"Unexpected token in expression: {tok}", span)

    @staticmethod
    def _op_str(ttype: str) -> str:
        mapping = {
            TokenType.BANG: "!",
            TokenType.MINUS: "-",
            TokenType.PLUS: "+",
            TokenType.EQ_EQ: "==",
            TokenType.NOT_EQ: "!=",
            TokenType.LT: "<",
            TokenType.GT: ">",
            TokenType.LTE: "<=",
            TokenType.GTE: ">=",
            TokenType.AND_AND: "&&",
            TokenType.OR_OR: "||",
        }
        return mapping.get(ttype, ttype)

    # ── String interpolation ────────────────────────────────────────

    @staticmethod
    def _parse_interpolation(raw: str) -> list:
        parts: list = []
        i = 0
        while i < len(raw):
            if raw[i:i+2] == "${":
                j = i + 2
                depth = 1
                while j < len(raw) and depth > 0:
                    if raw[j:j+2] == "${":
                        depth += 1
                        j += 2
                    elif raw[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                        j += 1
                    else:
                        j += 1
                expr_raw = raw[i+2:j]
                parts.append(_parse_simple_expr(expr_raw))
                i = j + 1
            else:
                if raw[i] == "\\" and i + 1 < len(raw):
                    ec = raw[i+1]
                    if ec == "n": parts.append("\n")
                    elif ec == "t": parts.append("\t")
                    else: parts.append(ec)
                    i += 2
                else:
                    parts.append(raw[i])
                    i += 1

        merged: list = []
        for p in parts:
            if isinstance(p, str) and merged and isinstance(merged[-1], str):
                merged[-1] += p
            else:
                merged.append(p)
        return merged


# ── Simple expression parser for interpolated strings ──────────────

def _parse_simple_expr(raw: str) -> IdentExpr | MemberExpr:
    raw = raw.strip()
    if not raw:
        return IdentExpr("", Span())
    parts_list = raw.split(".")
    expr: IdentExpr | MemberExpr = IdentExpr(parts_list[0])
    for p in parts_list[1:]:
        expr = MemberExpr(expr, p)
    return expr

"""Tests for the Rebis tokenizer and parser."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rebis.tokenizer import Tokenizer, TokenType
from rebis.parser import Parser
from rebis.ast import (
    ComponentDef, TagNode, Property, StateDecl, EventHandler,
    EnumDef, ImportStmt, LiteralExpr, IdentExpr, MemberExpr,
    InterpolatedExpr, File,
)
from rebis.interpreter import Interpreter


# ── Helpers ────────────────────────────────────────────────────────

def parse(source: str, file: str = "<test>") -> File:
    tokens = Tokenizer(source, file=file).tokenize()
    parser = Parser(tokens, file=file)
    return parser.parse()


def interpret(source: str) -> str:
    tokens = Tokenizer(source, file="<test>").tokenize()
    parser = Parser(tokens, file="<test>")
    file_ir = parser.parse()
    return Interpreter(file_ir).run()


def tokens(source: str):
    return Tokenizer(source).tokenize()


# ── Tokenizer tests ────────────────────────────────────────────────

class TestTokenizer:
    def test_empty(self):
        t = tokens("")
        # Tokenizer always emits at least NEWLINE + EOF for empty/blank input
        assert t[-1].type == TokenType.EOF

    def test_identifiers(self):
        t = tokens("hello world _private")
        types = [tok.type for tok in t]
        assert TokenType.IDENTIFIER in types
        assert t[0].value == "hello"
        assert t[1].value == "world"
        assert t[2].value == "_private"

    def test_keywords(self):
        t = tokens("component import export")
        assert t[0].type == TokenType.KEYWORD
        assert t[0].value == "component"
        assert t[1].type == TokenType.KEYWORD
        assert t[2].type == TokenType.KEYWORD

    def test_strings(self):
        t = tokens('"hello world"')
        string_toks = [tok for tok in t if tok.type == TokenType.STRING]
        assert len(string_toks) == 1
        assert string_toks[0].value == "hello world"

    def test_string_interpolation(self):
        t = tokens('"Hello ${name}"')
        string_toks = [tok for tok in t if tok.type == TokenType.STRING]
        assert len(string_toks) == 1
        assert "${" in (string_toks[0].value or "")

    def test_numbers(self):
        t = tokens("42 0 3.14 -5")
        assert t[0].type == TokenType.INT
        assert t[0].value == 42
        assert t[2].type == TokenType.FLOAT
        assert t[2].value == 3.14

    def test_colors(self):
        t = tokens("#ff6600 #abc")
        colors = [tok for tok in t if tok.type == TokenType.COLOR]
        assert len(colors) == 2
        assert colors[0].value == "#ff6600"

    def test_operators(self):
        t = tokens("== != <= >= < > && || ! + - = . , :")
        types = [tok.type for tok in t]
        assert TokenType.EQ_EQ in types
        assert TokenType.NOT_EQ in types
        assert TokenType.LTE in types
        assert TokenType.GTE in types
        assert TokenType.BANG in types
        assert TokenType.PLUS in types
        assert TokenType.MINUS in types

    def test_braces_and_parens(self):
        t = tokens("{ } ( ) [ ]")
        assert TokenType.OPEN_BRACE in [tok.type for tok in t]
        assert TokenType.CLOSE_PAREN in [tok.type for tok in t]
        assert TokenType.OPEN_BRACKET in [tok.type for tok in t]

    def test_indent_dedent(self):
        source = "row\n\tbutton\n\t\ttext"
        t = tokens(source)
        types = [tok.type for tok in t]
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types
        # Should have 2 INDENTs (row→button, button→text) and 2 DEDENTs
        indents = types.count(TokenType.INDENT)
        dedents = types.count(TokenType.DEDENT)
        assert indents == 2, f"Expected 2 INDENTs, got {indents}"
        assert dedents == 2, f"Expected 2 DEDENTs, got {dedents}"

    def test_comments(self):
        source = "// comment\nhello"
        t = tokens(source)
        types = [tok.type for tok in t]
        assert TokenType.IDENTIFIER in types
        # Find the IDENTIFIER token
        idents = [tok for tok in t if tok.type == TokenType.IDENTIFIER]
        assert len(idents) == 1
        assert idents[0].value == "hello"

    def test_trailing_comment(self):
        source = "hello // trailing comment"
        t = tokens(source)
        assert t[0].value == "hello"
        # Only one real token + NEWLINE + EOF
        real = [tok for tok in t if tok.type not in (TokenType.NEWLINE, TokenType.EOF)]
        assert len(real) == 1


# ── Parser tests ───────────────────────────────────────────────────

class TestParser:
    def test_empty_file(self):
        ir = parse("")
        assert isinstance(ir, File)
        assert len(ir.imports) == 0
        assert len(ir.declarations) == 0

    def test_minimal_component(self):
        source = "component foo {\n}"
        ir = parse(source)
        assert len(ir.declarations) == 1
        c = ir.declarations[0]
        assert isinstance(c, ComponentDef)
        assert c.name == "foo"
        assert not c.exported

    def test_exported_component(self):
        source = "export component bar {\n}"
        ir = parse(source)
        c = ir.declarations[0]
        assert c.exported
        assert c.name == "bar"

    def test_component_with_params(self):
        source = "component btn(label: String, count: Int = 0) {\n}"
        ir = parse(source)
        c = ir.declarations[0]
        assert len(c.params) == 2
        assert c.params[0].name == "label"
        assert c.params[0].type.name == "String"
        assert c.params[1].name == "count"
        assert c.params[1].default is not None

    def test_tag_with_properties(self):
        source = "component ui {\n\tbutton label=\"Submit\" width=200\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        assert isinstance(tag, TagNode)
        assert tag.name == "button"
        assert len(tag.properties) == 2
        assert tag.properties[0].name == "label"
        assert tag.properties[1].name == "width"

    def test_tag_with_children(self):
        source = "component ui {\n\trow\n\t\ttext value=\"A\"\n\t\ttext value=\"B\"\n}"
        ir = parse(source)
        c = ir.declarations[0]
        row = c.body[0]
        assert isinstance(row, TagNode)
        assert row.name == "row"
        assert len(row.properties) == 0  # no props ⇒ children
        assert len(row.children) == 2
        assert row.children[0].name == "text"
        assert row.children[1].name == "text"

    def test_two_position_rule(self):
        """Tag with props = leaf, tag without = children allowed."""
        source = "component ui {\n\trow\n\t\tbutton label=\"A\"\n\t\tspacer\n}"
        ir = parse(source)
        c = ir.declarations[0]
        row = c.body[0]
        assert row.name == "row"
        # button has props → leaf
        assert row.children[0].properties  # has props = leaf
        assert len(row.children[0].children) == 0  # no children
        # spacer has no props → empty tag
        assert len(row.children[1].properties) == 0
        assert len(row.children[1].children) == 0

    def test_state_declaration(self):
        source = "component ui {\n\tstate name: String = \"World\"\n}"
        ir = parse(source)
        c = ir.declarations[0]
        s = c.body[0]
        assert isinstance(s, StateDecl)
        assert s.name == "name"
        assert s.type is not None
        assert s.type.name == "String"

    def test_state_inferred_type(self):
        source = "component ui {\n\tstate count = 0\n}"
        ir = parse(source)
        c = ir.declarations[0]
        s = c.body[0]
        assert s.name == "count"
        assert s.type is None

    def test_event_handler(self):
        source = "component ui {\n\ton click {\n\t\tstate.count = 1\n\t}\n}"
        ir = parse(source)
        c = ir.declarations[0]
        handler = c.body[0]
        assert isinstance(handler, EventHandler)
        assert handler.event == "click"
        assert len(handler.body) == 1

    def test_empty_tag(self):
        source = "component ui {\n\tspacer\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        assert isinstance(tag, TagNode)
        assert tag.name == "spacer"
        assert len(tag.properties) == 0
        assert len(tag.children) == 0

    def test_nested_children(self):
        source = """component ui {
\tcolumn
\t\trow
\t\t\ttext value="Hi"
\t\t\ttext value="Bye"
\t\tbutton label="OK"
}"""
        ir = parse(source)
        c = ir.declarations[0]
        col = c.body[0]
        assert col.name == "column"
        assert len(col.properties) == 0  # no props ⇒ children follow
        assert len(col.children) == 2
        assert col.children[0].name == "row"
        assert col.children[1].name == "button"
        assert col.children[0].children[0].name == "text"

    def test_property_vs_tag_distinction(self):
        source = "component box {\n\twidth=200\n\theight=300\n}"
        ir = parse(source)
        c = ir.declarations[0]
        assert len(c.body) == 2
        p1 = c.body[0]
        assert isinstance(p1, Property)
        assert p1.name == "width"

    def test_boolean_literals(self):
        source = "component ui {\n\tbutton visible=true disabled=false\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        assert tag.properties[0].value.kind == "bool"
        assert tag.properties[0].value.value is True
        assert tag.properties[1].value.value is False

    def test_null_literal(self):
        source = "component ui {\n\ttext value=null\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        assert tag.properties[0].value.kind == "null"

    def test_color_literal(self):
        source = "component ui {\n\tbutton background=#ff6600\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        assert tag.properties[0].value.kind == "color"
        assert tag.properties[0].value.value == "#ff6600"

    def test_enum_definition(self):
        source = "export enum Theme {\n\tlight\n\tdark\n\tsystem\n}"
        ir = parse(source)
        assert len(ir.declarations) == 1
        e = ir.declarations[0]
        assert isinstance(e, EnumDef)
        assert e.name == "Theme"
        assert e.exported
        assert e.variants == ["light", "dark", "system"]

    def test_import(self):
        source = 'import Button from "./components/button"'
        ir = parse(source)
        assert len(ir.imports) == 1
        imp = ir.imports[0]
        assert imp.path == "./components/button"
        assert imp.names == ["Button"]  # single named import

    def test_wildcard_import(self):
        source = 'import "./styles"'
        ir = parse(source)
        assert len(ir.imports) == 1
        imp = ir.imports[0]
        assert imp.path == "./styles"
        assert imp.names is None

    def test_named_import(self):
        source = 'import { Dialog, Alert } from "./dialogs"'
        ir = parse(source)
        imp = ir.imports[0]
        assert imp.names == ["Dialog", "Alert"]

    def test_identifier_values(self):
        source = "component ui {\n\tbutton action=handle_submit\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        val = tag.properties[0].value
        assert isinstance(val, IdentExpr)
        assert val.name == "handle_submit"

    def test_member_expression(self):
        source = "component ui {\n\tbutton theme=Theme.dark\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        val = tag.properties[0].value
        assert isinstance(val, MemberExpr)
        assert val.member == "dark"

    def test_string_interpolation(self):
        source = 'component ui {\n\ttext value="Hello ${name}"\n}'
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        val = tag.properties[0].value
        assert isinstance(val, InterpolatedExpr)

    def test_float_literal(self):
        source = "component ui {\n\tslider value=0.5\n}"
        ir = parse(source)
        c = ir.declarations[0]
        tag = c.body[0]
        assert tag.properties[0].value.kind == "float"


# ── Interpreter tests ──────────────────────────────────────────────

class TestInterpreter:
    def test_empty_file(self):
        out = interpret("")
        assert "File" in out

    def test_component_output(self):
        out = interpret("component foo {\n}")
        assert "component foo" in out

    def test_exported_component_output(self):
        out = interpret("export component bar {\n}")
        assert "export component bar" in out

    def test_tag_output(self):
        out = interpret("component ui {\n\tbutton width=200\n}")
        assert "<button" in out
        assert "width" in out

    def test_event_handler_output(self):
        source = "component ui {\n\ton click {\n\t\tx = 1\n\t}\n}"
        out = interpret(source)
        assert "on click" in out

    def test_enum_output(self):
        source = "enum Theme {\n\tlight\n\tdark\n}"
        out = interpret(source)
        assert "enum Theme" in out
        assert "light" in out

    def test_interpolation_output(self):
        source = 'component ui {\n\ttext value="Hi ${name}"\n}'
        out = interpret(source)
        assert "Hi" in out
        assert "${" in out or "name" in out


# ── Integration: hello.rebis ───────────────────────────────────────

class TestHelloExample:
    def test_example_parses(self):
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "hello.rebis")
        with open(path) as f:
            source = f.read()
        ir = parse(source)
        assert len(ir.declarations) == 1
        c = ir.declarations[0]
        assert c.name == "greeting_card"
        # Should have state + tag tree
        assert any(isinstance(item, StateDecl) for item in c.body)
        assert any(isinstance(item, TagNode) for item in c.body)

    def test_example_interprets(self):
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "hello.rebis")
        with open(path) as f:
            source = f.read()
        out = interpret(source)
        assert "greeting_card" in out
        assert "Hello" in out

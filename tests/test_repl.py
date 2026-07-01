"""Tests for the REPL, pretty printer, and related utilities."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rebis.repl import (
    Completer,
    FileWatcher,
    ReplState,
    count_braces,
    is_balanced,
)
from rebis.pretty import PrettyPrinter, dim, keyword, name, string
from rebis.repl import _strip_ansi
from rebis.tokenizer import Tokenizer
from rebis.parser import Parser
from rebis.ast import (
    ComponentDef,
    EnumDef,
    File,
    ImportStmt,
    TagNode,
    Property,
    StateDecl,
    LiteralExpr,
    IdentExpr,
    MemberExpr,
)


# ── Pair import path ───────────────────────────────────────────────────────

def _parse(source: str) -> File:
    tokens = Tokenizer(source, file="<test>").tokenize()
    parser = Parser(tokens, file="<test>")
    return parser.parse()


# ─── Brace balance ─────────────────────────────────────────────────────────

class TestBraceBalance:
    def test_empty(self):
        assert is_balanced("")
        assert count_braces("") == 0

    def test_no_braces(self):
        assert is_balanced("hello world")
        assert count_braces("hello world") == 0

    def test_single_open(self):
        assert not is_balanced("component foo {")
        assert count_braces("component foo {") == 1

    def test_balanced(self):
        assert is_balanced("component foo {}")
        assert count_braces("component foo {}") == 0

    def test_nested_balanced(self):
        assert is_balanced("outer { inner {} }")
        assert count_braces("outer { inner {} }") == 0

    def test_nested_unbalanced(self):
        assert not is_balanced("outer { inner {}")
        assert count_braces("outer { inner {}") == 1

    def test_braces_in_string(self):
        assert is_balanced('component foo { text value="hello {world}" }')
        assert count_braces('component foo { text value="hello {world}" }') == 0

    def test_unbalanced_string_does_not_confuse(self):
        """A { inside a string shouldn't affect depth tracking."""
        assert is_balanced('component foo { text value="{ } }" }')
        assert count_braces('component foo { text value="{ } }" }') == 0

    def test_comment_brace(self):
        """A { in a comment should not be counted."""
        assert is_balanced("component foo {\n\t// just a { brace\n}")

    def test_more_close_than_open(self):
        assert count_braces("}") == -1
        assert is_balanced("}")  # depth <= 0

    def test_multiline_component(self):
        src = "component foo {\n\tstate x: Int = 1\n\tbar label=\"hi\"\n}"
        assert is_balanced(src)

    def test_multiline_unclosed(self):
        src = "component foo {\n\tstate x: Int = 1\n\tbar label=\"hi\""
        assert not is_balanced(src)
        assert count_braces(src) == 1


# ─── ReplState ─────────────────────────────────────────────────────────────

class TestReplState:
    def test_empty_state(self):
        state = ReplState()
        assert state.summary() == "empty"
        assert len(state.imports) == 0
        assert len(state.declarations) == 0

    def test_add_file_with_component(self):
        state = ReplState()
        file_ir = _parse("component foo {\n\tbutton label=\"Click\"\n}")
        state.add_file(file_ir)
        assert len(state.declarations) == 1
        assert state.declarations[0].name == "foo"
        assert "component" in state.summary()
        assert "1 declaration" in state.summary()

    def test_add_file_with_enum(self):
        state = ReplState()
        file_ir = _parse("enum Color { red green blue }")
        state.add_file(file_ir)
        assert len(state.declarations) == 1
        assert isinstance(state.declarations[0], EnumDef)

    def test_add_file_with_import(self):
        state = ReplState()
        file_ir = _parse('import "./styles"')
        state.add_file(file_ir)
        assert len(state.imports) == 1
        assert state.imports[0].path == "./styles"
        assert "import" in state.summary()

    def test_accumulate_multiple(self):
        state = ReplState()
        state.add_file(_parse("component A {}"))
        state.add_file(_parse("component B {}"))
        assert len(state.declarations) == 2
        assert "2 declarations" in state.summary()

    def test_to_file(self):
        state = ReplState()
        state.add_file(_parse("component A {}"))
        state.add_file(_parse("component B {}"))
        file_ir = state.to_file()
        assert len(file_ir.declarations) == 2
        assert len(file_ir.imports) == 0

    def test_clear(self):
        state = ReplState()
        state.add_file(_parse("component A {}"))
        state.clear()
        assert len(state.declarations) == 0
        assert state.summary() == "empty"

    def test_source_history(self):
        state = ReplState()
        state.source_history.append("component A {}")
        state.source_history.append("component B {}")
        assert len(state.source_history) == 2


# ─── Completer ─────────────────────────────────────────────────────────────

class TestCompleter:
    def setup_method(self):
        self.c = Completer()

    def test_complete_keyword(self):
        result = self.c.complete("comp", 0)
        assert result is not None
        assert result.startswith("component")

    def test_complete_command(self):
        result = self.c.complete(":q", 0)
        assert result is not None and ":quit" in result

    def test_no_match(self):
        result = self.c.complete("xyzzy", 0)
        assert result is None

    def test_multiple_matches_state(self):
        r0 = self.c.complete(":h", 0)
        r1 = self.c.complete(":h", 1)
        assert r0 is not None
        assert r1 is not None or r0 == r1  # may or may not have 2 matches


# ─── Pretty Printer ────────────────────────────────────────────────────────

class TestPrettyPrinter:
    def test_empty_file(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("")
        out = pp.format(file_ir)
        assert "File" in out

    def test_component_basic(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component foo {}")
        out = pp.format(file_ir)
        assert "component" in out
        assert "foo" in out

    def test_exported_component(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("export component bar {}")
        out = pp.format(file_ir)
        assert "export" in out

    def test_component_with_params(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component btn(label: String, count: Int = 0) {}")
        out = pp.format(file_ir)
        assert "btn" in out
        assert "String" in out
        assert "0" in out

    def test_tag_with_properties(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component ui {\n\tbutton label=\"Submit\" width=200\n}")
        out = pp.format(file_ir)
        assert "<button" in out
        assert "label" in out
        assert "Submit" in out
        assert "200" in out

    def test_tag_with_children(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component ui {\n\trow\n\t\ttext value=\"A\"\n\t\ttext value=\"B\"\n}")
        out = pp.format(file_ir)
        assert "<row>" in out
        assert "<text" in out
        assert '"A"' in out
        assert '"B"' in out
        assert out.index("<row>") < out.index("<text")  # row before children

    def test_state_declaration(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component ui {\n\tstate name: String = \"World\"\n}")
        out = pp.format(file_ir)
        assert "state name: String = \"World\"" in out or "state name: String" in out

    def test_enum(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("enum Theme { light dark }")
        out = pp.format(file_ir)
        assert "enum Theme" in out

    def test_import(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse('import Button from "./components/button"')
        out = pp.format(file_ir)
        assert "import" in out
        assert "Button" in out
        assert "components/button" in out

    def test_literals(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component ui {\n\tbutton visible=true count=42 ratio=0.5 bg=null\n}")
        out = pp.format(file_ir)
        assert "true" in out
        assert "42" in out
        assert "0.5" in out
        assert "null" in out

    def test_member_expr(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component ui {\n\tbutton action=Theme.dark\n}")
        out = pp.format(file_ir)
        assert "Theme.dark" in out or "Theme" in out

    def test_event_handler_output(self):
        pp = PrettyPrinter(color=False)
        source = "component ui {\n\ton click {\n\t\tcount = 1\n\t}\n}"
        out = pp.format(_parse(source))
        assert "on click" in out
        assert "count = 1" in out

    def test_color_output(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse("component ui {\n\tbutton bg=#ff6600\n}")
        out = pp.format(file_ir)
        assert "#ff6600" in out

    def test_interpolation_output(self):
        pp = PrettyPrinter(color=False)
        file_ir = _parse('component ui {\n\ttext value="Hello ${name}"\n}')
        out = pp.format(file_ir)
        assert "Hello" in out
        assert "${" in out or "name" in out

    def test_ansi_color_enabled_looks_different(self):
        """When color=True, the output should contain ANSI escape codes."""
        pp_color = PrettyPrinter(color=True)
        pp_plain = PrettyPrinter(color=False)
        out_color = pp_color.format(_parse("component foo {}"))
        out_plain = pp_plain.format(_parse("component foo {}"))
        stripped = _strip_ansi(out_color)
        assert stripped == out_plain  # same content, different encoding


# ─── File Watcher ──────────────────────────────────────────────────────────

class TestFileWatcher:
    def test_start_stop(self):
        """FileWatcher can be started and stopped cleanly."""
        fired = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rebis", delete=False) as f:
            f.write("component A {}")
            f.flush()
            path = f.name

        try:
            def callback():
                fired.append(True)

            w = FileWatcher(path, callback, interval=0.1)
            w.start()
            assert w.is_running

            # Modify the file
            with open(path, "w") as f:
                f.write("component B {}")

            import time
            time.sleep(0.3)
            w.stop()
            assert not w.is_running
            # Should have fired at least once
            assert len(fired) >= 1
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """Should not crash when watching a missing file."""
        fired = []
        w = FileWatcher("/nonexistent/file.rebis", lambda: fired.append(True), interval=0.1)
        w.start()
        import time
        time.sleep(0.25)
        w.stop()
        # No crash is the pass condition


# ─── CLI integration ───────────────────────────────────────────────────────

class TestCLIIntegration:
    def test_help_shows_repl(self):
        """`rebis --help` should mention the repl command."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "rebis", "--help"],
            capture_output=True, text=True
        )
        assert "repl" in result.stderr

    def test_repl_accepts_file_arg(self):
        """`rebis repl <file>` should load and interpret the file."""
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "hello.rebis")
        result = subprocess.run(
            [sys.executable, "-m", "rebis", "repl", path],
            input=":summary\n:quit\n",
            capture_output=True, text=True
        )
        assert "greeting_card" in result.stdout
        assert "Hello" in result.stdout

    def test_repl_eval_component(self):
        """Piping valid DSL to the REPL should parse and print it."""
        import subprocess
        input_data = "component test {}\n:quit\n"
        result = subprocess.run(
            [sys.executable, "-m", "rebis", "repl"],
            input=input_data,
            capture_output=True, text=True
        )
        assert "component test" in result.stdout

    def test_repl_parse_error_shows_context(self):
        """A parse error should display the error context."""
        import subprocess
        # Use input that is brace-balanced but has a syntax error:
        # "component foo" is missing its opening brace
        input_data = "component foo\n:quit\n"
        result = subprocess.run(
            [sys.executable, "-m", "rebis", "repl"],
            input=input_data,
            capture_output=True, text=True
        )
        # ParseError messages use "Expected" not "Error"
        assert "Expected" in result.stdout


# ─── Hello example integration ─────────────────────────────────────────────

class TestPrettyHello:
    def test_pretty_hello_matches_interpret(self):
        """Pretty printer output should contain the same key content as Interpreter."""
        from rebis.interpreter import Interpreter

        path = os.path.join(os.path.dirname(__file__), "..", "examples", "hello.rebis")
        with open(path) as f:
            source = f.read()
        file_ir = _parse(source)

        pp = PrettyPrinter(color=False)
        pretty_out = pp.format(file_ir)

        interp = Interpreter(file_ir)
        interp_out = interp.run()

        # Both should mention the same key identifiers
        for key in ("greeting_card", "name", "column", "Hello"):
            assert key in pretty_out, f"'{key}' missing from pretty output"
            assert key in interp_out, f"'{key}' missing from interpreter output"

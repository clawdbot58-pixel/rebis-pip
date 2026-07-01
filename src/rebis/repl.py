"""Interactive REPL for the Rebis DSL.

Provides an interactive shell for writing, testing, and inspecting
Rebis DSL snippets.  Supports multiline input, history, tab completion,
and special commands.

Usage::
    python -m rebis repl          # start the REPL
    python -m rebis repl <file>   # load + watch a file
"""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from typing import NoReturn

from . import __version__
from .ast import (
    ComponentDef,
    EnumDef,
    File,
    ImportStmt,
)
from .errors import ParseError, format_parse_error
from .interpreter import Interpreter
from .parser import Parser
from .pretty import PrettyPrinter, dim, error as err_style, comment, keyword
from .tokenizer import Tokenizer


# ── ANSI helpers ───────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes (for tests and non-tty output)."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


# ── File Watcher ───────────────────────────────────────────────────────────

class FileWatcher:
    """Polls a file for modification and calls a callback on change.

    Runs in a daemon thread so the REPL stays responsive.
    """

    def __init__(self, path: str, callback, interval: float = 0.5):
        self._path = Path(path)
        self._callback = callback
        self._interval = interval
        self._mtime: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            self._mtime = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _poll(self) -> None:
        while self._running:
            try:
                mtime = self._path.stat().st_mtime
                if mtime != self._mtime:
                    self._mtime = mtime
                    self._callback()
            except OSError:
                pass
            time.sleep(self._interval)


# ── Brace tracker ──────────────────────────────────────────────────────────

def count_braces(source: str) -> int:
    """Return the brace depth of *source* (positive = unclosed '{')."""
    depth = 0
    in_string = False
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == '/' and i + 1 < len(source) and source[i + 1] == '/':
                # Line comment — skip to end of line
                while i < len(source) and source[i] != '\n':
                    i += 1
                continue
        i += 1
    return depth


def is_balanced(source: str) -> bool:
    """Return True when braces are balanced (depth == 0) and source is non-empty."""
    if not source.strip():
        return True  # empty is technically balanced
    return count_braces(source) <= 0


# ── REPL State ─────────────────────────────────────────────────────────────

class ReplState:
    """Accumulated declarations and imports across REPL interactions."""

    def __init__(self):
        self.imports: list[ImportStmt] = []
        self.declarations: list[ComponentDef | EnumDef] = []
        self.source_history: list[str] = []

    def add_file(self, file_ir: File) -> None:
        """Merge a parsed file into the accumulated state."""
        self.imports.extend(file_ir.imports)
        self.declarations.extend(file_ir.declarations)

    def to_file(self) -> File:
        """Build a synthetic File node from accumulated state."""
        return File(imports=self.imports, declarations=self.declarations)

    def clear(self) -> None:
        """Reset all accumulated state."""
        self.imports.clear()
        self.declarations.clear()
        self.source_history.clear()

    def summary(self) -> str:
        """Return a one-line summary of accumulated state."""
        parts = []
        if self.declarations:
            n = len(self.declarations)
            kinds = {}
            for d in self.declarations:
                kind = type(d).__name__.replace("Def", "").lower()
                kinds[kind] = kinds.get(kind, 0) + 1
            desc = ", ".join(f"{n} {k}" for k, n in kinds.items())
            parts.append(f"{n} declaration{'s' if n != 1 else ''} ({desc})")
        if self.imports:
            parts.append(f"{len(self.imports)} import{'s' if len(self.imports) != 1 else ''}")
        return "; ".join(parts) if parts else "empty"


# ── Tab completion ─────────────────────────────────────────────────────────

REPL_KEYWORDS = [
    "component", "export", "import", "enum", "state", "on",
    "if", "else", "for", "in", "let", "try", "catch",
    "true", "false", "null",
]

REPL_COMMANDS = [
    ":quit", ":exit", ":help", ":inspect", ":clear",
    ":load", ":watch", ":unwatch", ":version", ":summary",
    ":history", ":edit",
]


class Completer:
    """Readline tab-completion for the REPL."""

    def __init__(self):
        self.words = REPL_KEYWORDS + REPL_COMMANDS

    def complete(self, text: str, state: int) -> str | None:
        candidates = [w for w in self.words if w.startswith(text)]
        if state < len(candidates):
            return candidates[state] + " "
        return None


# ── REPL ───────────────────────────────────────────────────────────────────

class Repl:
    """Interactive Rebis REPL.

    Reads Rebis DSL input, parses it, and displays the resulting IR tree
    using the pretty printer.  Maintains accumulated state across inputs.
    """

    @staticmethod
    def _banner() -> str:
        v = f"REPL v{__version__}"
        return f"""\
{dim("╭────────────────────────────────────────────╮")}
{dim("│")}  {keyword("Rebis")} {dim(v)}                     {dim("│")}
{dim("│")}  {comment("Type :help for commands")}         {dim("│")}
{dim("╰────────────────────────────────────────────╯")}
"""

    def __init__(self, file_path: str | None = None):
        self.file_path = file_path
        self.state = ReplState()
        self.watcher: FileWatcher | None = None
        self._watch_path: str | None = None
        self._input_buffer: list[str] = []
        self._active = True

        # Load initial file if given
        if file_path:
            self._load_file(file_path)

        # Readline setup
        self._setup_readline()

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> int:
        """Run the REPL loop. Returns exit code."""
        try:
            self._say(self._banner())
            if self.state.declarations:
                self._say(self.state.summary())

            while self._active:
                try:
                    self._handle_one_input()
                except EOFError:
                    self._say("")
                    break
                except KeyboardInterrupt:
                    self._say("")
                    self._input_buffer = []
                    continue
        finally:
            self._save_history()

        return 0

    # ── Input handling ─────────────────────────────────────────────────────

    def _handle_one_input(self) -> None:
        """Read one complete Rebis input (possibly multiline) and evaluate it."""
        self._input_buffer = []

        while True:
            prompt = f"{dim('. ')}" if self._input_buffer else f"{dim('> ')}"
            raw = input(prompt)

            # Handle commands on first line only
            if not self._input_buffer and raw.startswith(":"):
                self._run_command(raw)
                return

            stripped = raw.strip()
            if not stripped:
                if self._input_buffer:
                    # Empty line submits the buffer
                    break
                continue  # ignore leading blank lines

            self._input_buffer.append(raw)

            # If braces are balanced and we have at least one line, submit
            full_source = "\n".join(self._input_buffer)
            if is_balanced(full_source):
                break

        source = "\n".join(self._input_buffer)
        if source.strip():
            self._evaluate(source)

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _evaluate(self, source: str) -> None:
        """Parse *source* as Rebis DSL and print the IR tree."""
        try:
            tokens = Tokenizer(source, file="<repl>").tokenize()
            parser = Parser(tokens, file="<repl>")
            file_ir = parser.parse()
        except ParseError as e:
            self._show_parse_error(source, e)
            return
        except Exception as e:
            self._say(f"  {err_style('Error:')} {e}")
            return

        # Merge parsed declarations into accumulated state
        self.state.add_file(file_ir)
        self.state.source_history.append(source)

        # Print the result
        pp = PrettyPrinter(color=sys.stdout.isatty())
        output = pp.format(file_ir)
        for line in output.split("\n"):
            self._say(f"  {line}")

    # ── File loading ───────────────────────────────────────────────────────

    def _load_file(self, path: str) -> bool:
        """Load and evaluate a .rebis file. Returns True on success."""
        if not os.path.exists(path):
            self._say(f"  {err_style('Error:')} file not found: {path}")
            return False
        try:
            with open(path) as f:
                source = f.read()
        except OSError as e:
            self._say(f"  {err_style('Error:')} cannot read file: {e}")
            return False

        try:
            tokens = Tokenizer(source, file=path).tokenize()
            parser = Parser(tokens, file=path)
            file_ir = parser.parse()
        except ParseError as e:
            self._show_parse_error(source, e)
            return False
        except Exception as e:
            self._say(f"  {err_style('Error:')} {e}")
            return False

        self.state.add_file(file_ir)
        self.state.source_history.append(f"# load {path}")

        pp = PrettyPrinter(color=sys.stdout.isatty())
        output = pp.format(file_ir)
        for line in output.split("\n"):
            self._say(f"  {line}")
        return True

    # ── Commands ───────────────────────────────────────────────────────────

    COMMANDS: dict[str, tuple[str, str]] = {
        "quit":    ("Exit the REPL",        ""),
        "exit":    ("Exit the REPL",        ""),
        "help":    ("Show this help",       "[:help <command>]"),
        "inspect": ("Dump accumulated IR",  ""),
        "summary": ("Show state summary",   ""),
        "clear":   ("Clear accumulated declarations", ""),
        "load":    ("Load a .rebis file",   ":load <path>"),
        "watch":   ("Watch a file for changes", ":watch <path>"),
        "unwatch": ("Stop watching current file", ""),
        "edit":    ("Load and watch a file", ":edit <path>"),
        "version": ("Show version info",    ""),
        "history": ("Show input history (last 20)", ":history [n]"),
    }

    def _run_command(self, raw: str) -> None:
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0].lstrip(":").lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            self._active = False
        elif cmd == "help":
            self._cmd_help(arg)
        elif cmd == "inspect":
            self._cmd_inspect()
        elif cmd == "summary":
            self._say(f"  {dim('State:')} {self.state.summary()}")
        elif cmd == "clear":
            n = len(self.state.declarations)
            self.state.clear()
            self._say(f"  {comment('Cleared')} {n} declaration{'s' if n != 1 else ''}")
        elif cmd == "load":
            if arg:
                self._load_file(arg)
            else:
                self._say(f"  {err_style('Usage:')} :load <path>")
        elif cmd in ("watch", "edit"):
            if cmd == "edit" and arg:
                self._load_file(arg)
            self._cmd_watch(arg)
        elif cmd == "unwatch":
            self._cmd_unwatch()
        elif cmd == "version":
            self._say(f"  Rebis v{__version__}")
        elif cmd == "history":
            self._cmd_history(arg)
        else:
            self._say(f"  {err_style('Unknown command:')} :{cmd}")
            self._say(f"  {comment('Type :help for available commands')}")

    def _cmd_help(self, topic: str) -> None:
        if topic:
            if topic in self.COMMANDS:
                desc, usage = self.COMMANDS[topic]
                self._say(f"  {keyword(':' + topic)}")
                self._say(f"    {desc}")
                if usage:
                    self._say(f"    {comment(usage)}")
            else:
                self._say(f"  {err_style('Unknown command:')} {topic}")
            return

        self._say(f"  {keyword('Available commands')}")
        for name, (desc, usage) in sorted(self.COMMANDS.items()):
            line = f"    :{name:<12} {desc}"
            self._say(line)

    def _cmd_inspect(self) -> None:
        """Pretty-print the entire accumulated state."""
        if not self.state.declarations and not self.state.imports:
            self._say(f"  {dim('(no accumulated state)')}")
            return
        file_ir = self.state.to_file()
        pp = PrettyPrinter(color=sys.stdout.isatty())
        output = pp.format(file_ir)
        for line in output.split("\n"):
            self._say(f"  {line}")
        self._say(f"  {dim('──')}")
        self._say(f"  {self.state.summary()}")

    def _cmd_watch(self, path: str) -> None:
        if not path and not self._watch_path:
            self._say(f"  {err_style('Usage:')} :watch <path>")
            return
        path = path or self._watch_path
        if not os.path.exists(path):
            self._say(f"  {err_style('Error:')} file not found: {path}")
            return

        # Stop existing watcher
        if self.watcher:
            self.watcher.stop()

        def on_change():
            self._say(f"\n  {dim('[file changed]')}")
            self.state.clear()
            self._load_file(path)
            self._say(f"  {dim('─' * 30)}")

        self.watcher = FileWatcher(path, on_change)
        self.watcher.start()
        self._watch_path = path
        self._say(f"  {comment('Watching')} {path} {comment('(Ctrl-C to stop watching)')}")

    def _cmd_unwatch(self) -> None:
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
            self._say(f"  {comment('Stopped watching')} {self._watch_path}")
            self._watch_path = None
        else:
            self._say(f"  {dim('No file is being watched')}")

    def _cmd_history(self, arg: str) -> None:
        try:
            n = max(1, min(int(arg) if arg else 20, 100))
        except ValueError:
            n = 20
        history = self.state.source_history
        if not history:
            self._say(f"  {dim('(no history)')}")
            return
        start = max(0, len(history) - n)
        for i, src in enumerate(history[start:], start=start + 1):
            # Show first line of each entry
            first_line = src.split("\n")[0][:60]
            self._say(f"  {dim(f'{i:3d}.')} {first_line}")

    # ── Error display ──────────────────────────────────────────────────────

    def _show_parse_error(self, source: str, error: ParseError) -> None:
        """Display a formatted parse error with source context."""
        formatted = format_parse_error(source, error, context_lines=1)
        for line in formatted.split("\n"):
            if line.startswith("Error") or line.startswith("<repl>"):
                self._say(f"  {err_style(line)}")
            else:
                self._say(f"  {line}")

    # ── Output ─────────────────────────────────────────────────────────────

    @staticmethod
    def _say(text: str = "") -> None:
        """Print text to stdout."""
        print(text, flush=True)

    # ── Readline setup ─────────────────────────────────────────────────────

    HISTORY_FILE = os.path.expanduser("~/.rebis_history")

    def _setup_readline(self) -> None:
        try:
            import readline

            completer = Completer()
            readline.set_completer(completer.complete)
            readline.parse_and_bind("tab: complete")

            # Load history
            if os.path.exists(self.HISTORY_FILE):
                try:
                    readline.read_history_file(self.HISTORY_FILE)
                except OSError:
                    pass
        except ImportError:
            pass  # readline not available (Windows)

    def _save_history(self) -> None:
        try:
            import readline
            try:
                readline.write_history_file(self.HISTORY_FILE)
            except OSError:
                pass
        except ImportError:
            pass


# ── Standalone entry point ─────────────────────────────────────────────────

def run_repl(args: list[str]) -> int:
    """Run the REPL with the given CLI args. Called from __main__.py."""
    file_path = args[0] if args else None
    repl = Repl(file_path=file_path)
    return repl.run()

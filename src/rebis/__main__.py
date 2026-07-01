"""CLI entrypoint for Rebis.

Usage:
  python -m rebis repl [<file>]     Start the interactive REPL
  python -m rebis eval <file>       Parse and interpret a .rebis file
  python -m rebis eval --stdin       Read from stdin
  python -m rebis tokenize <file>   Show token stream (debug)
"""

from __future__ import annotations

import sys
import os


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_help()
        return 0

    command = argv[0]

    if command == "repl":
        return _cmd_repl(argv[1:])
    elif command == "eval":
        return _cmd_eval(argv[1:])
    elif command == "tokenize":
        return _cmd_tokenize(argv[1:])
    elif command in ("-h", "--help", "help"):
        _print_help()
        return 0
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_help()
        return 1


def _print_help() -> None:
    print(__doc__.strip(), file=sys.stderr)


def _cmd_eval(args: list[str]) -> int:
    from .tokenizer import Tokenizer
    from .parser import Parser
    from .interpreter import Interpreter

    stdin_mode = "--stdin" in args
    filenames = [a for a in args if not a.startswith("--")]

    source: str | None = None
    file_path = "<stdin>"

    if stdin_mode or not filenames:
        source = sys.stdin.read()
    else:
        file_path = filenames[0]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}", file=sys.stderr)
            return 1
        with open(file_path, "r") as f:
            source = f.read()

    try:
        tokens = Tokenizer(source, file=file_path).tokenize()
        parser = Parser(tokens, file=file_path)
        file_ir = parser.parse()
        output = Interpreter(file_ir).run()
        print(output)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_repl(args: list[str]) -> int:
    """Start the interactive REPL."""
    from .repl import run_repl
    return run_repl(args)


def _cmd_tokenize(args: list[str]) -> int:
    from .tokenizer import Tokenizer

    filenames = [a for a in args if not a.startswith("--")]
    source: str | None = None
    file_path = "<stdin>"

    if not filenames:
        source = sys.stdin.read()
    else:
        file_path = filenames[0]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}", file=sys.stderr)
            return 1
        with open(file_path, "r") as f:
            source = f.read()

    tokens = Tokenizer(source, file=file_path).tokenize()
    for tok in tokens:
        val = f"={tok.value!r}" if tok.value is not None else ""
        print(f"  {tok.type:20s} {val}  @{tok.line}:{tok.col}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

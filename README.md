# rebis-dsl

A cross-platform UI specification language — one DSL, many native backends.

Write UI layouts once in a compact, indentation-based DSL.  Transpile to
idiomatic C++, TypeScript, Swift, Kotlin, or Python.

> **Status:** Alpha — the DSL foundation and REPL are working. Transpiler
> backends are being built.

## Quick start

```bash
pip install rebis-dsl
```

```bash
# REPL — interactive DSL sketching
rebis repl

# Evaluate a .rebis file
rebis eval hello.rebis

# Debug the token stream
rebis tokenize hello.rebis
```

## Example

```rebis
component greeting_card {
    state name: String = "World"

    column padding=24 spacing=16
        text value="Hello, ${name}" font="bold 24"
        button label="Tap Me" width=200 height=48
}
```

## What works now

- **Tokenizer** — tab-indent tracking, all literal types, string interpolation, `{{raw}}` blocks, block comments
- **Parser** — full recursive-descent parser producing a language-agnostic IR
- **Interpreter** — IR tree pretty-printer with ANSI color output
- **REPL** — interactive shell with multiline input, history, tab completion, file watching
- **Module resolver** — import resolution with cycle detection
- **Error reporting** — structured errors with source snippets

## CLI

```
python -m rebis repl [<file>]     Start the interactive REPL
python -m rebis eval <file>       Parse and interpret a .rebis file
python -m rebis eval --stdin       Read from stdin
python -m rebis tokenize <file>   Show token stream (debug)
```

## Python API

```python
from rebis.tokenizer import Tokenizer
from rebis.parser import Parser
from rebis.interpreter import Interpreter
from rebis.pretty import PrettyPrinter

source = 'button width=200 label="Click"'
tokens = Tokenizer(source).tokenize()
file_ir = Parser(tokens).parse()

# Plain text
print(Interpreter(file_ir).run())

# Colorized
print(PrettyPrinter().format(file_ir))
```

## License

MIT

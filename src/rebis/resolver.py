"""Module resolver — import resolution, cycle detection, symbol tables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .ast import ComponentDef, EnumDef, File, ImportStmt, Visitor
from .errors import ImportError as RebisImportError


@dataclass
class Module:
    """A resolved module with its IR and symbol table."""
    path: str                     # canonical path (no extension)
    file: File
    symbols: dict[str, Any] = field(default_factory=dict)
    imported: bool = False


class SymbolTable:
    """Per-module symbol table."""

    def __init__(self):
        self._symbols: dict[str, ComponentDef | EnumDef] = {}

    def define(self, name: str, node: ComponentDef | EnumDef) -> None:
        if name in self._symbols:
            raise RebisImportError(f"Duplicate symbol: {name}")
        self._symbols[name] = node

    def lookup(self, name: str) -> ComponentDef | EnumDef | None:
        return self._symbols.get(name)


class ModuleResolver:
    """Resolves imports and builds symbol tables across a set of files."""

    def __init__(self, search_paths: list[str] | None = None):
        self.search_paths = search_paths or [os.getcwd()]
        self.modules: dict[str, Module] = {}
        self._resolving: set[str] = set()  # cycle detection

    def resolve_file(self, path: str, source: str) -> Module:
        """Resolve a main source file (not imported)."""
        canon = self._canonical(path)
        if canon in self.modules:
            return self.modules[canon]
        mod = self._build_module(canon, source)
        self.modules[canon] = mod
        # Resolve imports
        for imp in mod.file.imports:
            self._resolve_import(imp, mod)
        return mod

    def _resolve_import(self, imp: ImportStmt, from_mod: Module) -> None:
        """Resolve a single import statement."""
        imp_path = imp.path
        if not imp_path.endswith(".rebis"):
            imp_path += ".rebis"

        # Resolve path relative to importing module's directory
        base_dir = os.path.dirname(from_mod.path) if from_mod.path != "<unknown>" else "."
        full_path = os.path.normpath(os.path.join(base_dir, imp_path))

        if not os.path.exists(full_path):
            # Try search paths
            for sp in self.search_paths:
                candidate = os.path.normpath(os.path.join(sp, imp_path))
                if os.path.exists(candidate):
                    full_path = candidate
                    break
            else:
                raise RebisImportError(
                    f"Module not found: {imp.path}",
                    span=imp.span,
                    path=imp.path,
                    hint=f"Searched in {base_dir} and search paths",
                )

        canon = self._canonical(full_path)
        if canon in self.modules:
            return  # already resolved

        if canon in self._resolving:
            raise RebisImportError(
                f"Cyclic import detected: {canon}",
                span=imp.span,
                path=imp.path,
            )

        self._resolving.add(canon)
        try:
            with open(canon, "r") as f:
                src = f.read()
            mod = self._build_module(canon, src)
            self.modules[canon] = mod
            # Recursively resolve imports of the imported module
            for nested_imp in mod.file.imports:
                self._resolve_import(nested_imp, mod)
        finally:
            self._resolving.discard(canon)

    def _build_module(self, path: str, source: str) -> Module:
        """Build a Module from parsed source.  Imported separately to avoid
        circular dependency on Parser at import time."""
        # Deferred import to avoid circular dependency
        from .parser import Parser
        from .tokenizer import Tokenizer

        tokens = Tokenizer(source, file=path).tokenize()
        parser = Parser(tokens, file=path)
        file_ir = parser.parse()
        mod = Module(path=path, file=file_ir)
        self._build_symbols(mod)
        return mod

    def _build_symbols(self, mod: Module) -> None:
        """Walk the IR and populate the module's symbol table."""
        for decl in mod.file.declarations:
            if isinstance(decl, ComponentDef):
                mod.symbols[decl.name] = decl
            elif isinstance(decl, EnumDef):
                mod.symbols[decl.name] = decl

    @staticmethod
    def _canonical(path: str) -> str:
        return os.path.abspath(path)

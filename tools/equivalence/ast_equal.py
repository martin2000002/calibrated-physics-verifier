"""AST equality of every Python module between two revisions, for pure-move and comment/docstring
commits.

    ast_equal.py <repo> <old-rev> <new-rev> [--allow allowed.tsv] [--renames renames.tsv]
                 [--paths src scripts tests] [--sort-imports] [--strip-bare-strings]

Modules are paired by git rename detection, plus the explicit `<old path>\t<new path>` lines of
renames.tsv (needed for small files whose similarity git cannot see). Before comparing, docstrings
are stripped, the old side's imports are rewritten through the module-rename map derived from
the paired files, and the allowed substitutions are applied to the old side. A string constant
follows the module-rename map only when it names a paired module or a dotted descendant of one,
and never when the old revision assigns that value at module level to a name like `*_SCHEMA`,
`*_FILE`, `*_VERSION`, `LEDGER_LABEL*` or `GUARDED` (a persisted id that must not move with the
code). allowed.tsv lines:

    <new path or *>\t<kind>\t<old>\t<new>        kind: name (identifiers) | literal (str constants)
        | alias (an imported name -> "<name> as <asname>")
        | variable (like name, but only variables and imported names, never attributes)

With --sort-imports, each run of consecutive import statements and each import's names are
sorted on both sides, so the reordering a linter makes after a rename does not count.
With --strip-bare-strings, every string-constant statement is removed on both sides, attribute
docstrings included, not only the docstrings of modules, classes and functions.

Exit 0 when every paired module is equal, no module was added or deleted and every allowed.tsv
line was applied at least once.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("name", "literal", "alias", "variable")
PERSISTED_NAME = re.compile(r"_(SCHEMA|FILE|VERSION)$|^LEDGER_LABEL|^GUARDED$")

Line = tuple[str, str, str]


@dataclass
class Allowed:
    tables: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=lambda: {kind: {} for kind in KINDS}
    )

    def for_path(self, kind: str, path: str) -> dict[str, str]:
        table = self.tables[kind]
        return {**table.get("*", {}), **table.get(path, {})}

    def lines(self) -> set[Line]:
        return {
            (where, kind, old)
            for kind, table in self.tables.items()
            for where, entries in table.items()
            for old in entries
        }

    def line_of(self, kind: str, old: str, path: str) -> Line:
        return (path if old in self.tables[kind].get(path, {}) else "*", kind, old)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _python_files(repo: Path, rev: str, roots: Sequence[str]) -> set[str]:
    listing = _git(repo, "ls-tree", "-r", "--name-only", rev, "--", *roots)
    return {line for line in listing.splitlines() if line.endswith(".py")}


def _renames(repo: Path, old: str, new: str, roots: Sequence[str]) -> dict[str, str]:
    out = _git(repo, "diff", "--name-status", "-M30%", old, new, "--", *roots)
    pairs = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if parts[0].startswith("R") and parts[1].endswith(".py"):
            pairs[parts[1]] = parts[2]
    return pairs


def module_name(path: str) -> str | None:
    parts = Path(path).with_suffix("").parts
    if parts[:1] == ("src",):
        parts = parts[1:]
    elif parts[:2] == ("tests", "fixtures"):
        parts = parts[2:]
    else:
        return None
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _module_map(pairs: Mapping[str, str]) -> dict[str, str]:
    mapping = {}
    for old, new in pairs.items():
        before, after = module_name(old), module_name(new)
        if before and after and before != after:
            mapping[before] = after
    return mapping


def _rename_module(name: str, mapping: Mapping[str, str]) -> str:
    for old in sorted(mapping, key=len, reverse=True):
        if name == old or name.startswith(old + "."):
            return mapping[old] + name[len(old) :]
    return name


def persisted_values(sources: Iterable[str]) -> frozenset[str]:
    values = set()
    for source in sources:
        for node in ast.parse(source).body:
            targets = node.targets if isinstance(node, ast.Assign) else []
            if isinstance(node, ast.AnnAssign):
                targets = [node.target]
            value = getattr(node, "value", None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                names = (target.id for target in targets if isinstance(target, ast.Name))
                if any(PERSISTED_NAME.search(name) for name in names):
                    values.add(value.value)
    return frozenset(values)


class _Rewrite(ast.NodeTransformer):
    def __init__(
        self,
        modules: Mapping[str, str],
        allowed: Mapping[str, Mapping[str, str]],
        persisted: frozenset[str] = frozenset(),
    ) -> None:
        self.modules, self.persisted = modules, persisted
        self.allowed = {kind: allowed.get(kind, {}) for kind in KINDS}
        self.used: set[tuple[str, str]] = set()

    def _lookup(self, kind: str, value: str) -> str | None:
        if value not in self.allowed[kind]:
            return None
        self.used.add((kind, value))
        return self.allowed[kind][value]

    def _variable(self, value: str) -> str:
        found = self._lookup("variable", value)
        return self._name(value) if found is None else found

    def _name(self, value: str) -> str:
        found = self._lookup("name", value)
        return value if found is None else found

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module and node.level == 0:
            node.module = _rename_module(node.module, self.modules)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        alias = self._lookup("alias", node.name) if node.asname is None else None
        if alias is not None:
            node.name, node.asname = alias.split(" as ")
            return node
        node.name = _rename_module(self._variable(node.name), self.modules)
        if node.asname:
            node.asname = self._variable(node.asname)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._variable(node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.attr = self._name(node.attr)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._name(node.arg)
        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        if node.arg:
            node.arg = self._name(node.arg)
        return self.generic_visit(node)

    def _definition(self, node: ast.AST) -> ast.AST:
        node.name = self._name(node.name)  # type: ignore[attr-defined]
        return self.generic_visit(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _definition  # noqa: N815

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            found = self._lookup("literal", node.value)
            value = node.value if found is None else found
            movable = value.startswith("physverify") and value not in self.persisted
            node.value = _rename_module(value, self.modules) if movable else value
        return node


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


def _strip_bare_strings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            statements = getattr(node, name, None)
            if not isinstance(statements, list) or not statements:
                continue
            kept = [statement for statement in statements if not _is_bare_string(statement)]
            setattr(node, name, kept or ([ast.Pass()] if name == "body" else []))


def _is_bare_string(statement: object) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _sorted_run(run: list[ast.stmt]) -> list[ast.stmt]:
    for node in run:
        node.names.sort(key=ast.dump)  # type: ignore[attr-defined]
    return sorted(run, key=ast.dump)


def _sort_imports(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        ordered: list[ast.stmt] = []
        run: list[ast.stmt] = []
        for statement in [*body, None]:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                run.append(statement)
                continue
            ordered.extend(_sorted_run(run))
            run = []
            if statement is not None:
                ordered.append(statement)
        node.body = ordered  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Options:
    sort_imports: bool = False
    strip_bare_strings: bool = False


def _dump(source: str, rewrite: _Rewrite | None, options: Options) -> str:
    tree = ast.parse(source)
    _strip_docstrings(tree)
    if options.strip_bare_strings:
        _strip_bare_strings(tree)
    if rewrite is not None:
        tree = rewrite.visit(tree)
    if options.sort_imports:
        _sort_imports(tree)
    return ast.dump(tree, indent=1)


def _load_allowed(path: Path | None) -> Allowed:
    allowed = Allowed()
    if path is None:
        return allowed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        where, kind, old, new = line.split("\t")
        allowed.tables[kind].setdefault(where, {})[old] = new
    return allowed


def compare(
    repo: Path,
    old: str,
    new: str,
    roots: Sequence[str],
    allowed: Allowed,
    explicit: Mapping[str, str],
    options: Options | None = None,
) -> int:
    options = options or Options()
    before, after = _python_files(repo, old, roots), _python_files(repo, new, roots)
    pairs = {path: path for path in before & after}
    pairs.update({o: n for o, n in _renames(repo, old, new, roots).items() if o in before})
    pairs.update({o: n for o, n in explicit.items() if o in before and n in after})
    modules = _module_map(pairs)
    sources = {path: _git(repo, "show", f"{old}:{path}") for path in sorted(pairs)}
    persisted = persisted_values(sources.values())
    used: set[Line] = set()
    problems = 0
    for path in sorted(before - set(pairs)):
        print(f"deleted: {path}")
        problems += 1
    for path in sorted(after - set(pairs.values())):
        print(f"added: {path}")
        problems += 1
    for old_path, new_path in sorted(pairs.items()):
        tables = {kind: allowed.for_path(kind, new_path) for kind in KINDS}
        rewrite = _Rewrite(modules, tables, persisted)
        left = _dump(sources[old_path], rewrite, options)
        right = _dump(_git(repo, "show", f"{new}:{new_path}"), None, options)
        used |= {allowed.line_of(kind, value, new_path) for kind, value in rewrite.used}
        if left != right:
            problems += 1
            label = old_path if old_path == new_path else f"{old_path} -> {new_path}"
            delta = difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm="", n=1)
            print(f"differs: {label}")
            print("\n".join(list(delta)[2:22]))
    for where, kind, value in sorted(allowed.lines() - used):
        print(f"unused allow line: {where}\t{kind}\t{value}")
        problems += 1
    print(f"{len(pairs)} modules paired, {problems} problems")
    return 1 if problems else 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", type=Path)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--allow", type=Path)
    parser.add_argument("--renames", type=Path)
    parser.add_argument("--paths", nargs="+", default=["src", "scripts", "tests"])
    parser.add_argument("--sort-imports", action="store_true")
    parser.add_argument("--strip-bare-strings", action="store_true")
    args = parser.parse_args(argv)
    explicit = {}
    if args.renames is not None:
        for line in args.renames.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                source, target = line.split("\t")
                explicit[source] = target
    allowed = _load_allowed(args.allow)
    options = Options(args.sort_imports, args.strip_bare_strings)
    return compare(args.repo, args.old, args.new, args.paths, allowed, explicit, options)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

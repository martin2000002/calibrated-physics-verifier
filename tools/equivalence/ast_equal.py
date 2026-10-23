"""AST equality of every Python module between two revisions, for pure-move and comment/docstring
commits.

    ast_equal.py <repo> <old-rev> <new-rev> [--allow allowed.tsv] [--renames renames.tsv]
                 [--paths src scripts tests] [--sort-imports]

Modules are paired by git rename detection, plus the explicit `<old path>\t<new path>` lines of
renames.tsv (needed for small files whose similarity git cannot see). Before comparing, docstrings
are stripped, the old side's imports are rewritten through the module-rename map derived from
the paired files, and the allowed substitutions are applied to the old side. allowed.tsv lines:

    <new path or *>\t<kind>\t<old>\t<new>        kind: name (identifiers) | literal (str constants)
        | alias (an imported name -> "<name> as <asname>")
        | variable (like name, but only variables and imported names, never attributes)

With --sort-imports, each run of consecutive import statements and each import's names are
sorted on both sides, so the reordering a linter makes after a rename does not count.

Exit 0 when every paired module is equal and no module was added or deleted.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Allowed:
    names: dict[str, dict[str, str]] = field(default_factory=dict)
    literals: dict[str, dict[str, str]] = field(default_factory=dict)
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    variables: dict[str, dict[str, str]] = field(default_factory=dict)

    def for_path(self, table: Mapping[str, dict[str, str]], path: str) -> dict[str, str]:
        return {**table.get("*", {}), **table.get(path, {})}


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


class _Rewrite(ast.NodeTransformer):
    def __init__(
        self,
        modules: Mapping[str, str],
        names: Mapping[str, str],
        literals: Mapping[str, str],
        aliases: Mapping[str, str] | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> None:
        self.modules, self.names, self.literals = modules, names, literals
        self.aliases = aliases or {}
        self.variables = variables or {}

    def _variable(self, value: str) -> str:
        return self.variables.get(value, self._name(value))

    def _name(self, value: str) -> str:
        return self.names.get(value, value)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module and node.level == 0:
            node.module = _rename_module(node.module, self.modules)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname is None and node.name in self.aliases:
            node.name, node.asname = self.aliases[node.name].split(" as ")
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
            value = self.literals.get(node.value, node.value)
            node.value = (
                _rename_module(value, self.modules) if value.startswith("physverify") else value
            )
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


def _dump(source: str, rewrite: _Rewrite | None, sort_imports: bool = False) -> str:
    tree = ast.parse(source)
    _strip_docstrings(tree)
    if rewrite is not None:
        tree = rewrite.visit(tree)
    if sort_imports:
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
        tables = {
            "name": allowed.names,
            "literal": allowed.literals,
            "alias": allowed.aliases,
            "variable": allowed.variables,
        }
        table = tables[kind]
        table.setdefault(where, {})[old] = new
    return allowed


def compare(
    repo: Path,
    old: str,
    new: str,
    roots: Sequence[str],
    allowed: Allowed,
    explicit: Mapping[str, str],
    sort_imports: bool = False,
) -> int:
    before, after = _python_files(repo, old, roots), _python_files(repo, new, roots)
    pairs = {path: path for path in before & after}
    pairs.update({o: n for o, n in _renames(repo, old, new, roots).items() if o in before})
    pairs.update({o: n for o, n in explicit.items() if o in before and n in after})
    modules = _module_map(pairs)
    problems = 0
    for path in sorted(before - set(pairs)):
        print(f"deleted: {path}")
        problems += 1
    for path in sorted(after - set(pairs.values())):
        print(f"added: {path}")
        problems += 1
    for old_path, new_path in sorted(pairs.items()):
        rewrite = _Rewrite(
            modules,
            allowed.for_path(allowed.names, new_path),
            allowed.for_path(allowed.literals, new_path),
            allowed.for_path(allowed.aliases, new_path),
            allowed.for_path(allowed.variables, new_path),
        )
        left = _dump(_git(repo, "show", f"{old}:{old_path}"), rewrite, sort_imports)
        right = _dump(_git(repo, "show", f"{new}:{new_path}"), None, sort_imports)
        if left != right:
            problems += 1
            label = old_path if old_path == new_path else f"{old_path} -> {new_path}"
            delta = difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm="", n=1)
            print(f"differs: {label}")
            print("\n".join(list(delta)[2:22]))
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
    args = parser.parse_args(argv)
    explicit = {}
    if args.renames is not None:
        for line in args.renames.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                source, target = line.split("\t")
                explicit[source] = target
    allowed = _load_allowed(args.allow)
    return compare(args.repo, args.old, args.new, args.paths, allowed, explicit, args.sort_imports)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""A public plugin module imports no other public plugin module of its package: shared code
lives in a private sibling (``_name.py``), so loading one plugin never loads another."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from physverify.app.plugins import PLUGIN_KINDS, PLUGIN_PACKAGES
from physverify.cli.context import COMMANDS_PACKAGE

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGES = (*PLUGIN_PACKAGES.values(), COMMANDS_PACKAGE)


def _public_modules(package: str) -> dict[str, Path]:
    folder = SOURCE.joinpath(*package.split("."))
    return {path.stem: path for path in folder.glob("*.py") if not path.stem.startswith("_")}


def _imported_modules(package: str, node: ast.AST) -> Iterator[str]:
    """First component under ``package`` of every module ``node`` imports."""
    if isinstance(node, ast.Import):
        targets = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        base = package if node.level else node.module or ""
        if node.level and node.module:
            base = f"{package}.{node.module}"
        targets = [base] if base != package else [f"{base}.{a.name}" for a in node.names]
    else:
        return
    for target in targets:
        if target.startswith(package + "."):
            yield target[len(package) + 1 :].split(".")[0]


def _sibling_imports(package: str) -> list[str]:
    public = _public_modules(package)
    found: list[str] = []
    for name, path in sorted(public.items()):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            found += [
                f"{name}:{getattr(node, 'lineno', 0)} -> {sibling}"
                for sibling in _imported_modules(package, node)
                if sibling in public and sibling != name
            ]
    return found


def test_every_plugin_kind_names_its_package() -> None:
    assert list(PLUGIN_PACKAGES) == list(PLUGIN_KINDS)


@pytest.mark.parametrize("package", PACKAGES)
def test_public_plugin_modules_import_no_public_sibling(package: str) -> None:
    assert _public_modules(package)
    assert _sibling_imports(package) == []

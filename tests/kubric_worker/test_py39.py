"""The synthetic Kubric dataset's container modules run on the Kubric image (Python 3.9, numpy
1.21): 3.9 syntax, and at module level only the stdlib, numpy and the container-safe physverify
packages."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "src" / "physverify" / "kubric_worker"
CONTAINER_MODULES = ("build_scene.py", "tasks/simulate.py", "tasks/contacts.py", "tasks/render.py")
ALLOWED_PREFIXES = ("numpy", "physverify.schema", "physverify.kubric_worker", "__future__")


def _module_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def _allowed(name: str) -> bool:
    root = name.split(".")[0]
    return root in sys.stdlib_module_names or name.startswith(ALLOWED_PREFIXES)


@pytest.mark.parametrize("relative", CONTAINER_MODULES)
def test_module_parses_as_python_3_9(relative: str) -> None:
    source = (WORKER / relative).read_text(encoding="utf-8")
    ast.parse(source, filename=relative, feature_version=(3, 9))
    assert "from __future__ import annotations" in source


@pytest.mark.parametrize("relative", CONTAINER_MODULES)
def test_heavy_libraries_are_imported_inside_functions(relative: str) -> None:
    tree = ast.parse((WORKER / relative).read_text(encoding="utf-8"))
    rejected = [n for n in _module_level_imports(tree) if not _allowed(n)]
    assert rejected == []

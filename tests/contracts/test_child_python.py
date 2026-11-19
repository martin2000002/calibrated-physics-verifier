"""Tests start the test interpreter only through ``fixtures/child_python.py``, which runs it on the
``physverify`` under test. Elsewhere ``sys.executable`` may only be the ``REAL_PYTHON`` a shell
sandbox's fake python execs: the sandbox and its scripts set that child's PYTHONPATH themselves."""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
HELPER = TESTS / "fixtures" / "child_python.py"
SANDBOX_KEY = "REAL_PYTHON"


def _names_executable(node: ast.Attribute | ast.ImportFrom) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == "sys" and any(alias.name == "executable" for alias in node.names)
    return node.attr == "executable" and isinstance(node.value, ast.Name) and node.value.id == "sys"


def executable_lines(source: str) -> list[int]:
    """Lines that name ``sys.executable``, except as the value of a ``REAL_PYTHON`` key."""
    tree = ast.parse(source)
    handed = {
        id(value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == SANDBOX_KEY
    }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute | ast.ImportFrom)
        and _names_executable(node)
        and id(node) not in handed
    )


def test_the_test_interpreter_starts_only_through_the_helper() -> None:
    offenders = [
        f"{path.relative_to(TESTS)}:{line}"
        for path in sorted(TESTS.rglob("*.py"))
        if path != HELPER
        for line in executable_lines(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "start the test interpreter with child_python.run_python"


def test_the_scan_sees_a_spawn_and_spares_a_sandbox_variable() -> None:
    assert executable_lines("import sys\nsubprocess.run([sys.executable, '-c', 'pass'])\n") == [2]
    assert executable_lines("from sys import executable\n") == [1]
    assert executable_lines("import sys\nenv.update({'REAL_PYTHON': sys.executable})\n") == []

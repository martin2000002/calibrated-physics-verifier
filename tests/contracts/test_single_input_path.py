"""Check inputs are built from stage plans in one place: ``routed_inputs``
(``verification/check_routes.py``), so no consumer reads the main plan for a routed check.

Every call to an inputs builder in ``src/physverify`` must be one of ``ALLOWED``: the builders
themselves and the plan sources ``routed_inputs`` is handed. ``OracleSource`` is not guarded: at the
oracle level every stage is ground truth, so every route resolves to the main plan."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "physverify"
BUILDERS = frozenset(
    {
        "CheckInputs",
        "build_inputs",
        "perceived_inputs",
        "PerceivedSource",
        "PipelineOutputLoader",
        "RoutedSource",
        "routed_source",
        "_PlanSource",
    }
)
ALLOWED = frozenset(
    {
        ("verification/inputs.py", "build_inputs", "CheckInputs"),
        ("verification/inputs.py", "perceived_inputs", "build_inputs"),
        ("verification/inputs.py", "_prepared", "build_inputs"),
        ("verification/inputs.py", "PerceivedSource._one", "perceived_inputs"),
        ("verification/check_routes.py", "routed_source", "RoutedSource"),
        ("verification/check_routes.py", "routed_inputs", "routed_source"),
        ("verification/check_routes.py", "perceived_routed_source.source_of", "PerceivedSource"),
        (
            "verification/check_routes.py",
            "perceived_routed_source.source_of",
            "PipelineOutputLoader",
        ),
        ("diagnosis/verification_adapter.py", "_PlanSource.load", "perceived_inputs"),
        ("diagnosis/verification_adapter.py", "variant_inputs.source_of", "_PlanSource"),
    }
)

Call = tuple[str, str, str]


class _Calls(ast.NodeVisitor):
    """Calls to a builder, under any import alias, with the dotted scope they are made in."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.aliases = {name: name for name in BUILDERS}
        self.found: list[Call] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in BUILDERS and alias.asname is not None:
                self.aliases[alias.asname] = alias.name

    def visit(self, node: ast.AST) -> None:
        scoped = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        if scoped:
            self.scope.append(node.name)
        super().visit(node)
        if scoped:
            self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in self.aliases:
            self.found.append((self.module, ".".join(self.scope), self.aliases[name]))
        self.generic_visit(node)


def builder_calls() -> Iterator[Call]:
    for path in sorted(SRC.rglob("*.py")):
        visitor = _Calls(path.relative_to(SRC).as_posix())
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        yield from visitor.found


def test_inputs_are_built_only_through_routed_inputs() -> None:
    calls = set(builder_calls())
    assert calls - ALLOWED == set(), "build check inputs through routed_inputs"
    assert calls >= ALLOWED, "an allowed builder call is gone; drop it from ALLOWED"


def test_the_scan_sees_an_aliased_bypass(tmp_path: Path) -> None:
    source = "from x import perceived_inputs as p\n\ndef consumer():\n    return p(1)\n"
    visitor = _Calls("app/consumer.py")
    visitor.visit(ast.parse(source))
    assert visitor.found == [("app/consumer.py", "consumer", "perceived_inputs")]

"""Check inputs are built from stage plans in one place: ``routed_inputs``
(``verification/check_routes.py``), so no consumer reads the main plan for a routed check.

Every reference to an inputs builder in ``src/physverify`` must be one of ``ALLOWED``: the builders
themselves and the plan sources ``routed_inputs`` is handed. A reference is any use of the name,
not only a call, so a ``partial``, a rebinding or a subclass is seen too, and so is the name as a
string (``getattr``, ``vars()[...]``); annotations and type tests are not references, since they
cannot build. ``OracleSource`` is not guarded: at the oracle level every stage is ground truth, so
every route resolves to the main plan."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

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
        "_prepared",
        "RawInputs",
    }
)
TYPE_TESTS = frozenset({"isinstance", "issubclass"})
ALLOWED = frozenset(
    {
        ("verification/inputs.py", "build_inputs", "CheckInputs"),
        ("verification/inputs.py", "perceived_inputs", "build_inputs"),
        ("verification/inputs.py", "_prepared", "build_inputs"),
        ("verification/inputs.py", "perceived_inputs", "RawInputs"),
        ("verification/inputs.py", "OracleSource._one", "RawInputs"),
        ("verification/inputs.py", "OracleSource._one", "_prepared"),
        ("checks/_base.py", "validate_check", "CheckInputs"),
        ("verification/inputs.py", "PerceivedSource._one", "perceived_inputs"),
        ("verification/check_routes.py", "routed_source", "RoutedSource"),
        ("verification/check_routes.py", "routed_inputs", "routed_source"),
        ("verification/check_routes.py", "perceived_routed_source.source_of", "PerceivedSource"),
        (
            "verification/check_routes.py",
            "perceived_routed_source.source_of",
            "PipelineOutputLoader",
        ),
        ("diagnosis/verification_adapter.py", "_PlanSource._one", "perceived_inputs"),
        ("diagnosis/verification_adapter.py", "variant_inputs.source_of", "_PlanSource"),
    }
)

Use = tuple[str, str, str]


class _Uses(ast.NodeVisitor):
    """References to a builder, under any import alias, with the dotted scope they are made in."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.aliases = {name: name for name in BUILDERS}
        self.found: list[Use] = []

    def visit(self, node: ast.AST) -> None:
        scoped = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        if scoped:
            self.scope.append(node.name)
        super().visit(node)
        if scoped:
            self.scope.pop()

    def _found(self, name: str) -> None:
        self.found.append((self.module, ".".join(self.scope), self.aliases[name]))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in BUILDERS and alias.asname is not None:
                self.aliases[alias.asname] = alias.name

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        defaults = [d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None]
        for child in (*node.decorator_list, *defaults, *node.body):
            self.visit(child)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in TYPE_TESTS:
            self.visit(node.args[0])
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.aliases:
            self._found(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.aliases:
            self._found(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value in BUILDERS:
            self._found(node.value)


def builder_uses() -> Iterator[Use]:
    for path in sorted(SRC.rglob("*.py")):
        yield from uses_in(path.relative_to(SRC).as_posix(), path.read_text(encoding="utf-8"))


def uses_in(module: str, source: str) -> list[Use]:
    visitor = _Uses(module)
    visitor.visit(ast.parse(source))
    return visitor.found


def test_inputs_are_built_only_through_routed_inputs() -> None:
    uses = set(builder_uses())
    assert uses - ALLOWED == set(), "build check inputs through routed_inputs"
    assert uses >= ALLOWED, "an allowed builder use is gone; drop it from ALLOWED"


BYPASSES = {
    "an aliased call": (
        "from x import perceived_inputs as p\n\ndef consumer():\n    return p(1)\n",
        "perceived_inputs",
    ),
    "a partial": ("def consumer():\n    return partial(perceived_inputs, 1)\n", "perceived_inputs"),
    "a rebinding": (
        "def consumer():\n    make = inputs.PerceivedSource\n    return make(1)\n",
        "PerceivedSource",
    ),
    "a getattr": (
        "def consumer():\n    return getattr(inputs, 'PerceivedSource')(1)\n",
        "PerceivedSource",
    ),
    "a module namespace": (
        "def consumer():\n    return vars(inputs)['build_inputs'](1)\n",
        "build_inputs",
    ),
    "the prepared step": ("def consumer():\n    return inputs._prepared(raw)\n", "_prepared"),
    "raw inputs": ("def consumer():\n    return RawInputs(1)\n", "RawInputs"),
    "a subclass": ("class Mine(PerceivedSource):\n    pass\n", "PerceivedSource"),
    "a default argument": (
        "def consumer(make=perceived_inputs):\n    return make(1)\n",
        "perceived_inputs",
    ),
}


@pytest.mark.parametrize(("source", "builder"), BYPASSES.values(), ids=BYPASSES.keys())
def test_the_scan_sees_a_bypass(source: str, builder: str) -> None:
    assert [found for _, _, found in uses_in("app/consumer.py", source)] == [builder]


def test_annotations_and_type_tests_are_not_uses() -> None:
    source = (
        "def consumer(item: CheckInputs) -> RawInputs:\n"
        "    found: PerceivedSource = item\n"
        "    return isinstance(item, CheckInputs)\n"
    )
    assert uses_in("app/consumer.py", source) == []

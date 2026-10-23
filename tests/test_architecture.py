"""Layer contract from ``[tool.importlinter]`` in pyproject.toml, checked with stdlib ``ast``.

Every import counts, including imports inside functions and under ``TYPE_CHECKING``.
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (REPO / "src", REPO)
SCANNED_DIRS = (REPO / "src" / "physverify", REPO / "scripts")

# Current package names that the contract lists under their target name.
LEGACY_PACKAGES: Mapping[str, str] = {
    "physverify.agreement": "physverify.human_agreement",
    "physverify.evaluation": "physverify.generator_comparison",
}

# Pairs "module -> imported" present when the guard was added; fixes shrink this set.
KNOWN_VIOLATIONS: frozenset[str] = frozenset(
    {
        "physverify.datasets.kubric_fresh -> physverify.calibration.scenes",
        "physverify.datasets.pisa_real -> physverify.extraction.providers._common",
        "physverify.datasets.pisa_real -> physverify.extraction.providers.scale_size_prior",
        "physverify.demo.settings -> physverify.timing.settings",
        "physverify.diagnosis.scene_stats -> physverify.calibration.table",
        "physverify.finetune.budget -> physverify.timing.budget",
        "physverify.finetune.generate -> physverify.timing.generation",
        "physverify.finetune.generate -> physverify.timing.mp4",
        "physverify.finetune.generate -> physverify.timing.settings",
        "physverify.finetune.generate -> physverify.timing.targets.wan21_generate",
        "physverify.finetune.settings -> physverify.timing.settings",
        "physverify.finetune.wan_lora -> physverify.timing.settings",
        "physverify.sizing.model -> physverify.extraction.providers._common",
        "physverify.sizing.model -> physverify.extraction.providers.scale_size_prior",
        "physverify.verification.scope -> physverify.calibration.exclusion_record",
        "physverify.verification.selective_risk -> physverify.calibration.table",
        "physverify.verification.selective_risk -> physverify.diagnosis.pairing",
    }
)


@dataclass(frozen=True, order=True)
class Violation:
    module: str
    line: int
    imported: str

    @property
    def pair(self) -> str:
        return f"{self.module} -> {self.imported}"

    def __str__(self) -> str:
        return f"{self.module}:{self.line} -> {self.imported}"


@dataclass(frozen=True)
class ImportEdge:
    module: str
    line: int
    imported: str


def _contracts() -> list[dict[str, Any]]:
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    contracts: list[dict[str, Any]] = config["tool"]["importlinter"]["contracts"]
    return contracts


def _contract(kind: str) -> list[dict[str, Any]]:
    return [contract for contract in _contracts() if contract["type"] == kind]


def _within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _module_name(path: Path) -> str:
    root = next(root for root in SOURCE_ROOTS if path.is_relative_to(root))
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    base = module.split(".")
    if not is_package:
        base.pop()
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _imported_names(module: str, is_package: bool, node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        target = _resolve_relative(module, is_package, node) if node.level else node.module or ""
        if "." in target:
            yield target
        else:
            yield from (f"{target}.{alias.name}" for alias in node.names)


def _edges(path: Path) -> Iterator[ImportEdge]:
    module = _module_name(path)
    is_package = path.name == "__init__.py"
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        for imported in _imported_names(module, is_package, node):
            yield ImportEdge(module, getattr(node, "lineno", 0), imported)


def _all_edges() -> list[ImportEdge]:
    paths = sorted(path for root in SCANNED_DIRS for path in root.rglob("*.py"))
    return [edge for path in paths for edge in _edges(path)]


def _canonical(module: str) -> str:
    for legacy, target in LEGACY_PACKAGES.items():
        if _within(module, legacy):
            return target + module[len(legacy) :]
    return module


def _layer_ranks(layers: Sequence[str]) -> dict[str, int]:
    return {
        package.strip(): len(layers) - index
        for index, entry in enumerate(layers)
        for package in entry.split("|")
    }


def _package_of(module: str, ranks: Mapping[str, int]) -> str | None:
    canonical = _canonical(module)
    return next((package for package in ranks if _within(canonical, package)), None)


def _breaks_layers(edge: ImportEdge, ranks: Mapping[str, int]) -> bool:
    source = _package_of(edge.module, ranks)
    target = _package_of(edge.imported, ranks)
    if source is None or target is None or source == target:
        return False
    return ranks[target] >= ranks[source]


def _breaks_forbidden(edge: ImportEdge, contract: Mapping[str, Any]) -> bool:
    sources = contract["source_modules"]
    if not any(_within(edge.module, source) for source in sources):
        return False
    exempt = [*sources, *contract.get("allowed_modules", [])]
    if any(_within(edge.imported, module) for module in exempt):
        return False
    return any(_within(edge.imported, module) for module in contract["forbidden_modules"])


def _violations() -> list[Violation]:
    edges = _all_edges()
    found: set[Violation] = set()
    for contract in _contract("layers"):
        ranks = _layer_ranks(contract["layers"])
        found.update(_as_violation(edge) for edge in edges if _breaks_layers(edge, ranks))
    for contract in _contract("forbidden"):
        found.update(_as_violation(edge) for edge in edges if _breaks_forbidden(edge, contract))
    return sorted(found)


def _as_violation(edge: ImportEdge) -> Violation:
    return Violation(edge.module, edge.line, edge.imported)


def _top_level_packages() -> set[str]:
    root = REPO / "src" / "physverify"
    return {f"physverify.{path.parent.name}" for path in root.glob("*/__init__.py")}


def test_every_package_is_covered_by_a_contract() -> None:
    covered = {
        package.strip()
        for contract in _contract("layers")
        for entry in contract["layers"]
        for package in entry.split("|")
    }
    covered.update(source for c in _contract("forbidden") for source in c["source_modules"])
    missing = sorted(p for p in _top_level_packages() if _canonical(p) not in covered)
    assert not missing, f"packages missing from [tool.importlinter]: {missing}"


def test_no_new_import_violations() -> None:
    new = [str(v) for v in _violations() if v.pair not in KNOWN_VIOLATIONS]
    assert not new, "imports breaking the layer contract:\n" + "\n".join(new)


def test_known_violations_still_occur() -> None:
    fixed = sorted(KNOWN_VIOLATIONS - {violation.pair for violation in _violations()})
    assert not fixed, "fixed; remove from KNOWN_VIOLATIONS:\n" + "\n".join(fixed)

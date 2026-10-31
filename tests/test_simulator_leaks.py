"""The verifier does not know the simulator's materials: the floor friction Kubric renders with
is not a value of any verifier source or configuration, and R2 samples mu from an interval."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

from physverify.checks.r2_static_equilibrium import EquilibriumParams

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "physverify"
SIMULATOR_PACKAGES = ("synth", "kubric_worker", "generation")
SIMULATOR_FILES = (SOURCE / "schema" / "scene.py",)
VERIFIER_CONFIGS = REPO / "configs" / "verifier"
NUMBER = re.compile(r"(?<![\w.])\d*\.\d+|(?<![\w.])\d+(?![\w.])")


def simulator_friction() -> float:
    with (REPO / "configs" / "datasets" / "kubric_synthetic.toml").open("rb") as handle:
        friction: float = tomllib.load(handle)["dataset"]["floor"]["friction"]
    return friction


def verifier_files() -> Iterator[Path]:
    for path in sorted(SOURCE.rglob("*.py")):
        package = path.relative_to(SOURCE).parts[0]
        if package not in SIMULATOR_PACKAGES and path not in SIMULATOR_FILES:
            yield path
    yield from sorted(VERIFIER_CONFIGS.glob("*.toml"))


def friction_values(path: Path) -> Iterator[tuple[int, float]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "friction" in line.lower():
            yield from ((number, float(token)) for token in NUMBER.findall(line))


def test_no_verifier_file_holds_the_simulator_friction() -> None:
    leaked = simulator_friction()
    found = [
        f"{path.relative_to(REPO)}:{line}"
        for path in verifier_files()
        for line, value in friction_values(path)
        if value == leaked
    ]
    assert not found, f"the simulator's friction {leaked} in the verifier: {found}"


def test_r2_draws_mu_from_an_interval() -> None:
    low, high = EquilibriumParams().friction_range
    assert low < high

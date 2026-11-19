"""The verifier does not know the simulator: no constant the synthetic scenes are made with (floor
and object materials, camera framing, object sizes, render settings, perturbation windows and
levels) is a value of a verifier source or configuration line that names it, and R2 samples mu
from an interval.

Every constant is read from the simulator itself (each registered family's and perturbation's
parameters under every profile, as ``synth/generate.py`` builds them), never copied here."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks.r2_static_equilibrium import EquilibriumParams
from physverify.config.loader import known_profiles, load_settings, read_config_files
from physverify.synth.family_toolkit import gravity_vector
from physverify.synth.placement import CameraFraming
from physverify.synth.settings import (
    DatasetSettings,
    family_params,
    load_dataset_settings,
    load_size_distribution,
    perturbation_params,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
SOURCE = REPO / "src" / "physverify"
SIMULATOR_PACKAGES = ("synth", "kubric_worker", "generation")
SIMULATOR_FILES = (SOURCE / "schema" / "scene.py",)
VERIFIER_CONFIGS = (CONFIGS / "verifier", CONFIGS / "studies")
NUMBER = re.compile(r"(?<![\w.])\d*\.\d+|(?<![\w.])\d+(?![\w.])")

FIELD_KEYWORDS: Mapping[str, str] = {
    "friction": "friction",
    "restitution": "restitution",
    "elevation": "elevation_deg",
    "focal": "focal_length_mm",
    "sensor": "sensor_width_mm",
    "fill": "fill",
    "distance": "max_distance_m",
    "density": "density",
    "mean_m": "mean_m",
    "rel_spread": "rel_spread",
    "aspect": "aspect_range",
    "fps": "fps",
    "n_frames": "n_frames",
    "step_rate": "step_rate",
    "resolution": "resolution",
    "spp": "spp",
}
"""Keyword a verifier line naming the constant contains -> the simulator fields it covers (every
family field whose name contains it, e.g. ``restitution`` covers ``cube_restitution``)."""
FOV = "fov"
GRAVITY = "gravity"
UNPERTURBED = "none"

EXEMPT: Mapping[str, str] = {
    GRAVITY: "g is physics, not a simulator choice: R5b measures departures from it",
    "fps": "the calibration clip's rate: windows are tuned on calib clips and restated in seconds "
    "(physics/row_time.py), tied to that clip by tests/contracts/test_reference_clip.py",
    "n_frames": "the calibration clip's length, tied likewise by test_reference_clip.py",
}
"""Constants listed but not searched for, and why."""


@dataclass(frozen=True)
class LineException:
    path: str
    fragment: str
    reason: str


LINE_EXCEPTIONS = (
    LineException(
        "src/physverify/checks/r2_static_equilibrium.py",
        "0.5 * (self.friction_range[0]",
        "the midpoint of R2's own friction interval, not a friction value",
    ),
    LineException(
        "src/physverify/checks/r3a_bounce_restitution.py",
        "restitution e - 1",
        "the elastic bound e = 1 of mechanics, not the floor's restitution",
    ),
    LineException(
        "src/physverify/checks/variants/_detrended_size.py",
        "The shortest r4_resize bump",
        "R4_size v3's windows are set on the r4_resize bumps (3 to 8 rows) and on simulator data",
    ),
    LineException(
        "configs/verifier/checks.toml",
        "R4_size = 0.10",
        "provisional threshold, r4_resize level 2; the calibration replaces it",
    ),
    LineException(
        "configs/verifier/checks.toml",
        "R1 = 0.05",
        "provisional threshold, r1_penetration level 1; the calibration replaces it",
    ),
    LineException(
        "configs/verifier/checks.toml",
        "R3a = 0.05",
        "provisional threshold, r3_restitution_gain level 1; the calibration replaces it",
    ),
    LineException(
        "configs/verifier/calibration.toml",
        "level-1 effects are >= 0.05",
        "epsilon_c is an upper bound kept below the smallest level-1 effect, not a level",
    ),
    LineException(
        "configs/verifier/checks.toml",
        "(0.4 % on the local f1 clip",
        "a measured share of g, not an r5b_gravity_scale level",
    ),
)


@dataclass(frozen=True)
class SimulatorConstant:
    """``approximate``: a derived value, matched at the precision a line writes it."""

    keyword: str
    values: frozenset[float]
    approximate: bool = False

    def matches(self, token: str) -> bool:
        if not self.approximate:
            return float(token) in self.values
        decimals = len(token.partition(".")[2])
        return any(round(value, decimals) == float(token) for value in self.values)


def numeric_leaves(value: Any, name: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from numeric_leaves(item, str(key))
    elif isinstance(value, list | tuple):
        for item in value:
            yield from numeric_leaves(item, name)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield name, float(value)


def profile_settings() -> Iterator[Any]:
    base = read_config_files(CONFIGS)[0]
    for profile in sorted(known_profiles(base)):
        yield load_settings(CONFIGS, environ={}, profile=profile)


def family_parameters() -> list[Any]:
    """Every registered family's parameters under every profile."""
    found = []
    for settings in profile_settings():
        dataset, priors = load_dataset_settings(settings), load_size_distribution(settings)
        found.extend(
            family_params(family, dataset, priors) for family in PLUGIN_KINDS["families"]().values()
        )
    return found


def perturbation_constants() -> list[SimulatorConstant]:
    """Each perturbation's parameters, on lines naming the perturbation, and its fractional
    levels, on lines naming a check it targets. Whole-number levels (frames, multiples) are not
    searched there: every index and formula line of a check would match them."""
    named: dict[str, set[float]] = {}
    targeted: dict[str, set[float]] = {}
    datasets: list[DatasetSettings] = [load_dataset_settings(s) for s in profile_settings()]
    for name, plugin in PLUGIN_KINDS["perturbations"]().items():
        if name == UNPERTURBED:
            continue
        for dataset in datasets:
            params = perturbation_params(plugin, dataset)
            named.setdefault(name, set()).update(
                v for _, v in numeric_leaves(params.model_dump(mode="json"))
            )
            for check in plugin.target_checks:
                targeted.setdefault(check.lower(), set()).update(
                    v for v in params.levels if not float(v).is_integer()
                )
    return [SimulatorConstant(k, frozenset(v)) for k, v in sorted({**named, **targeted}.items())]


def field_of_view_deg(framing: CameraFraming) -> float:
    return math.degrees(2.0 * math.atan(framing.sensor_width_mm / (2.0 * framing.focal_length_mm)))


def simulator_constants() -> list[SimulatorConstant]:
    params = family_parameters()
    leaves = [leaf for p in params for leaf in numeric_leaves(p.model_dump(mode="json"))]
    named = [
        SimulatorConstant(keyword, frozenset(v for name, v in leaves if field in name))
        for keyword, field in FIELD_KEYWORDS.items()
    ]
    fov = frozenset(field_of_view_deg(p.framing) for p in params)
    gravity = frozenset({float(np.linalg.norm(gravity_vector()))})
    return [
        *named,
        SimulatorConstant(FOV, fov, approximate=True),
        SimulatorConstant(GRAVITY, gravity),
        *perturbation_constants(),
    ]


def verifier_files() -> Iterator[Path]:
    for path in sorted(SOURCE.rglob("*.py")):
        package = path.relative_to(SOURCE).parts[0]
        if package not in SIMULATOR_PACKAGES and path not in SIMULATOR_FILES:
            yield path
    for root in VERIFIER_CONFIGS:
        yield from sorted(root.rglob("*.toml"))
    yield CONFIGS / "default.toml"


def leaks(
    files: Iterable[Path], constants: Iterable[SimulatorConstant]
) -> list[tuple[str, int, str, str]]:
    """``(path, line, keyword, value)`` of every simulator value on a line naming it."""
    searched = [c for c in constants if c.keyword not in EXEMPT]
    found = []
    for path in files:
        shown = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            found.extend(
                (shown, number, constant.keyword, token)
                for constant in searched
                if constant.keyword in lowered
                for token in NUMBER.findall(line)
                if constant.matches(token)
            )
    return found


def excused(path: str, line: str) -> bool:
    return any(e.path == path and e.fragment in line for e in LINE_EXCEPTIONS)


def line_text(path: str, number: int) -> str:
    return (REPO / path).read_text(encoding="utf-8").splitlines()[number - 1]


@pytest.fixture(scope="module")
def constants() -> list[SimulatorConstant]:
    return simulator_constants()


def test_no_verifier_line_holds_a_simulator_constant(constants: list[SimulatorConstant]) -> None:
    found = [
        f"{path}:{number} {keyword} = {value}"
        for path, number, keyword, value in leaks(verifier_files(), constants)
        if not excused(path, line_text(path, number))
    ]
    assert not found, f"simulator constants in the verifier: {found}"


def test_every_exception_still_excuses_a_line(constants: list[SimulatorConstant]) -> None:
    hits = [(p, line_text(p, n)) for p, n, _, _ in leaks(verifier_files(), constants)]
    unused = [e for e in LINE_EXCEPTIONS if not any(excused(p, line) for p, line in hits)]
    assert unused == []


def test_every_constant_is_read_and_every_exemption_is_one(
    constants: list[SimulatorConstant],
) -> None:
    assert all(c.values for c in constants), [c.keyword for c in constants if not c.values]
    assert set(EXEMPT) <= {c.keyword for c in constants}


@pytest.mark.parametrize(
    "planted",
    ["[camera]\nfov = 49\n", "[camera]\nfov_deg = 49.1\n", "[camera]\nelevation_deg = [15, 35]\n"],
)
def test_a_camera_prior_from_the_simulator_is_caught(
    tmp_path: Path, constants: list[SimulatorConstant], planted: str
) -> None:
    config = tmp_path / "camera.toml"
    config.write_text(planted, encoding="utf-8")
    assert leaks([config], constants)


@pytest.mark.parametrize(
    "planted",
    ["[verification.thresholds]\nR5b = 0.2\n", "# r4_resize bumps last 8 rows\nwindow = 17\n"],
)
def test_a_perturbation_level_or_window_is_caught(
    tmp_path: Path, constants: list[SimulatorConstant], planted: str
) -> None:
    config = tmp_path / "checks.toml"
    config.write_text(planted, encoding="utf-8")
    assert leaks([config], constants)


def test_another_field_of_view_is_not_caught(
    tmp_path: Path, constants: list[SimulatorConstant]
) -> None:
    config = tmp_path / "camera.toml"
    config.write_text("[camera]\nfov = 60\n", encoding="utf-8")
    assert leaks([config], constants) == []


def test_r2_draws_mu_from_an_interval() -> None:
    low, high = EquilibriumParams().friction_range
    assert low < high

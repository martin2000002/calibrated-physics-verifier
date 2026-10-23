"""Ground-truth magnitudes of every check on the committed Bullet clips of ``tests/synth/data``:
each clean clip, every level of every trajectory perturbation on it and the committed
gravity-scale re-simulations. They are pinned as ``float.hex`` and compared within float
tolerance, since the last bits differ across platforms."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import record_cuts
from physverify.synth.perturb import (
    PerturbationNotApplicableError,
    SeedKey,
    instantiate,
    perturb_trajectory,
)
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.settings import DatasetSettings, perturbation_params
from physverify.synth.truth_measures._base import measure_truth

DATA = Path(__file__).parent / "data"
PINS = Path(__file__).with_name("truth_measures_pins.json")
NOT_APPLICABLE = "not_applicable"
REL_TOL = 1e-9
ABS_TOL = 1e-12

type Pin = dict[str, str] | str


def measures(
    truth: Any,
    scene: SceneSpec,
    states: ObjectStateTable,
    contacts: ContactTable,
    tol: float,
    cuts: Sequence[int] = (),
) -> dict[str, str]:
    values = measure_truth(truth, scene, states, contacts, tol, cuts)
    return {key: float.hex(value) for key, value in values.items()}


def families() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if (p / "spec.json").is_file())


def computed(bank: Any, ds: DatasetSettings, registry: Any, truth: Any) -> dict[str, Any]:
    """Pins per case ``<family>/<perturbation>/<level>`` (``<family>/none`` for the clean clip)."""
    tol = ds.contacts.tol_m
    plugins = instantiate(registry, ds)
    pins: dict[str, Any] = {}
    for family in families():
        clip = bank.recorded(family)
        pins[f"{family}/none"] = measures(truth, clip.scene, clip.states, clip.contacts, tol)
        for name, plugin in sorted(plugins.items()):
            if name == "none" or plugin.target is not PerturbationTarget.TRAJECTORY:
                continue
            targets = perturbation_params(type(plugin), ds).levels
            for level, target in enumerate(targets, start=1):
                key = SeedKey(ds.seed, family, 0, level)
                try:
                    states, contacts, record = perturb_trajectory(
                        plugin, clip.scene, clip.states, clip.contacts, target, key
                    )
                except PerturbationNotApplicableError:
                    pins[f"{family}/{name}/{level}"] = NOT_APPLICABLE
                    continue
                cuts = record_cuts(record)
                pins[f"{family}/{name}/{level}"] = measures(
                    truth, clip.scene, states, contacts, tol, cuts
                )
    for directory in sorted((DATA / "r5b").iterdir()):
        clip, record = bank.recorded_gravity_scale(directory.name)
        case = f"{directory.name}/{record.name}/{record.level}"
        cuts = record_cuts(record)
        pins[case] = measures(truth, clip.scene, clip.states, clip.contacts, tol, cuts)
    return pins


def same_value(got: str, pinned: str) -> bool:
    x, y = float.fromhex(got), float.fromhex(pinned)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def same_case(got: Pin, pinned: Pin) -> bool:
    if isinstance(got, str) or isinstance(pinned, str):
        return got == pinned
    return got.keys() == pinned.keys() and all(same_value(got[k], pinned[k]) for k in pinned)


def test_truth_measures_match_the_pins(
    bank: Any, clean_ds: DatasetSettings, perturbations: Any, truth_measures: Any
) -> None:
    expected: dict[str, Pin] = json.loads(PINS.read_text(encoding="utf-8"))
    got = computed(bank, clean_ds, perturbations, truth_measures)
    assert got.keys() == expected.keys()
    assert [case for case in expected if not same_case(got[case], expected[case])] == []

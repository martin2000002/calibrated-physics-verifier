"""Every perturbation plugin on clean clips of every family, recorded (Bullet) and analytic: the
recorded magnitude is the one measured on the edited truth, it reaches each level target and the
edit moves only its own truth measure (ADR 0004 §4 and §9.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.loader import load_settings
from physverify.physics.geometry import Plane, signed_distance
from physverify.schema.scene import ObjectSpec, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.magnitudes import record_cuts
from physverify.synth.perturb import (
    MeasuredPerturbation,
    PerturbationNotApplicableError,
    SeedKey,
    instantiate,
    measured,
    perturb_scene,
    perturb_trajectory,
)
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._frame_windows import solid_at
from physverify.synth.settings import DatasetSettings, load_dataset_settings, perturbation_params
from physverify.synth.truth_measures._base import measure_clip

OWN_KEY = {
    "r1_penetration": "R1",
    "r2_unsupported_rest": "R2",
    "r3_restitution_gain": "R3a",
    "r3_energy_gain": "R3b",
    "r4_resize": "R4_size",
    "r4_teleport": "R4_jump",
    "r4_vanish": "R4_missing",
    "r5a_gravity_switch": "R5a",
    "r5a_object_gravity": "R5a",
    "r5b_gravity_scale": "R5b",
}
APPLICABLE = {
    "f1_projectile": {"r4_resize", "r4_vanish", "r5a_gravity_switch", "r5a_object_gravity"},
    "f2_bounce": {"r3_restitution_gain", "r4_resize", "r4_teleport", "r4_vanish"},
    "f3_collision": {"r3_energy_gain", "r4_vanish"},
    "f4_stack": {"r2_unsupported_rest", "r4_teleport", "r4_vanish"},
    "f5_contact": {"r1_penetration", "r4_teleport", "r4_vanish"},
}
GRAVITY_FAMILIES = ("f1_projectile", "f2_bounce")
# Entries a Bullet clip moves besides the plugin's own, with the cause measured in ADR 0004 §9.2:
# the analytic edit replaces the clean bounce and with it Bullet's impact penetration (R1).
RECORDED_LEAKS = {
    ("f1_projectile", "r5a_gravity_switch"): {"R1"},
    ("f1_projectile", "r5a_object_gravity"): {"R1"},
    ("f5_contact", "r4_teleport"): {"R1"},
}
# A re-simulation under another gravity changes Bullet's impact penetration and restitution,
# both speed dependent (R1, R3a). R4_jump is a third difference: any constant g leaves it at 0.
RESIMULATION_LEAKS = {"R1", "R3a"}
# Physical couplings: rotating g mid-flight is a jump of the acceleration, which leaves a
# one-row peak |delta g| dt^2 / (2 L) in R4_jump at the switch.
COUPLINGS = {("f1_projectile", "r5a_gravity_switch"): {"R4_jump"}}
# R5a needs two flights of R5A_MIN_FRAMES; a longer or shorter bounce flight of the f2 clip leaves
# one and R5a becomes undefined (NaN), which is not a value leak (ADR 0004 §9.2).
UNDEFINED = {
    ("f2_bounce", "r3_restitution_gain"): {"R5a"},
    ("f2_bounce", "r5b_gravity_scale"): {"R5a"},
}


def tolerance(value: float) -> float:
    return max(1e-3, 0.02 * abs(value))


def moved(base: dict[str, float], after: dict[str, float]) -> set[str]:
    """Entries whose value changed beyond the tolerance (both defined)."""
    return {
        k
        for k in base
        if math.isfinite(base[k])
        and math.isfinite(after[k])
        and abs(after[k] - base[k]) > tolerance(base[k])
    }


def undefined(base: dict[str, float], after: dict[str, float]) -> set[str]:
    """Entries that became measurable or stopped being measurable (NaN on one side)."""
    return {k for k in base if math.isfinite(base[k]) != math.isfinite(after[k])}


def report(base: dict[str, float], after: dict[str, float], keys: set[str]) -> dict[str, Any]:
    return {k: (base[k], after[k]) for k in keys}


def levels(plugin: Any, ds: DatasetSettings) -> tuple[float, ...]:
    return tuple(perturbation_params(type(plugin), ds).levels)


def perturbed(
    plugin: Any, clip: Any, family: str, level: int, target: float, seed: int
) -> tuple[ObjectStateTable, Any, PerturbationRecord]:
    key = SeedKey(seed, family, 0, level)
    result: tuple[ObjectStateTable, Any, PerturbationRecord] = perturb_trajectory(
        plugin, clip.scene, clip.states, clip.contacts, target, key
    )
    return result


def trajectory_names(plugins: dict[str, Any]) -> set[str]:
    return {
        name
        for name, plugin in plugins.items()
        if plugin.target is PerturbationTarget.TRAJECTORY and name != "none"
    }


@dataclass(frozen=True)
class Edit:
    target: float
    states: ObjectStateTable
    contacts: Any
    record: PerturbationRecord
    measures: dict[str, float]


class Sweep:
    """Every applicable plugin at every level on the clean clips, perturbed once per session."""

    def __init__(self, bank: Any, ds: DatasetSettings, plugins: dict[str, Any], truth: Any) -> None:
        self.bank, self.ds, self.plugins, self.truth = bank, ds, plugins, truth
        self._base: dict[tuple[str, str], dict[str, float]] = {}
        self._edits: dict[tuple[str, str], dict[tuple[str, int], Edit]] = {}

    def clip(self, source: str, family: str) -> Any:
        return getattr(self.bank, source)(family)

    def base(self, source: str, family: str) -> dict[str, float]:
        if (source, family) not in self._base:
            clip = self.clip(source, family)
            self._base[source, family] = measure_clip(
                self.truth, clip.scene, clip.states, clip.contacts, self.ds.contacts.tol_m
            )
        return self._base[source, family]

    def edits(self, source: str, family: str) -> dict[tuple[str, int], Edit]:
        if (source, family) not in self._edits:
            self._edits[source, family] = {
                (name, level): self._edit(source, family, name, level, target)
                for name in sorted(APPLICABLE[family])
                for level, target in enumerate(levels(self.plugins[name], self.ds), start=1)
            }
        return self._edits[source, family]

    def _edit(self, source: str, family: str, name: str, level: int, target: float) -> Edit:
        clip = self.clip(source, family)
        states, contacts, record = perturbed(
            self.plugins[name], clip, family, level, target, self.ds.seed
        )
        tol = self.ds.contacts.tol_m
        measures = measure_clip(self.truth, clip.scene, states, contacts, tol, record_cuts(record))
        return Edit(target, states, contacts, record, measures)


@pytest.fixture(scope="session")
def sweep(bank: Any, clean_ds: DatasetSettings, perturbations: Any, truth_measures: Any) -> Sweep:
    return Sweep(bank, clean_ds, instantiate(perturbations, clean_ds), truth_measures)


def test_the_families_cover_every_plugin(plugins: dict[str, Any]) -> None:
    covered = set().union(*APPLICABLE.values()) | {"r5b_gravity_scale"}
    assert covered == set(plugins) - {"none"} == set(OWN_KEY)


@pytest.mark.parametrize("source", ["recorded", "analytic"])
@pytest.mark.parametrize("family", sorted(APPLICABLE))
def test_each_level_is_reached_and_measured(family: str, source: str, sweep: Sweep) -> None:
    clip = sweep.clip(source, family)
    for name in sorted(trajectory_names(sweep.plugins) - APPLICABLE[family]):
        with pytest.raises(PerturbationNotApplicableError):
            perturbed(sweep.plugins[name], clip, family, 1, 0.1, sweep.ds.seed)
    for (name, level), edit in sweep.edits(source, family).items():
        assert edit.record.level == level
        if name == "r4_vanish":
            assert edit.record.magnitude == edit.target
        else:
            assert abs(edit.record.magnitude - edit.target) <= tolerance(edit.target), (name, level)
        plugin = sweep.plugins[name]
        assert plugin.measure(edit.states, edit.contacts, edit.record) == edit.record.magnitude
        assert edit.states.position.shape == clip.states.position.shape


@pytest.mark.parametrize("family", sorted(APPLICABLE))
def test_analytic_edits_move_only_their_own_measure(family: str, sweep: Sweep) -> None:
    base = sweep.base("analytic", family)
    for (name, level), edit in sweep.edits("analytic", family).items():
        own, after, magnitude = OWN_KEY[name], edit.measures, edit.record.magnitude
        allowed = {own} | COUPLINGS.get((family, name), set())
        assert moved(base, after) <= allowed, (name, level, report(base, after, moved(base, after)))
        assert undefined(base, after) <= UNDEFINED.get((family, name), set()), (name, level)
        if base[own] < magnitude - tolerance(magnitude):
            assert abs(after[own] - magnitude) <= tolerance(magnitude), (name, level)


@pytest.mark.parametrize("family", sorted(APPLICABLE))
def test_recorded_edits_move_only_documented_measures(family: str, sweep: Sweep) -> None:
    base = sweep.base("recorded", family)
    for (name, level), edit in sweep.edits("recorded", family).items():
        allowed = {OWN_KEY[name]} | RECORDED_LEAKS.get((family, name), set())
        allowed |= COUPLINGS.get((family, name), set())
        after = edit.measures
        assert moved(base, after) <= allowed, (name, level, report(base, after, moved(base, after)))
        assert not undefined(base, after), (name, level)


@pytest.mark.parametrize("family", GRAVITY_FAMILIES)
def test_gravity_scale_reaches_each_level_on_the_analytic_clip(
    family: str,
    bank: Any,
    ds: DatasetSettings,
    plugins: dict[str, Any],
    truth_measures: Any,
) -> None:
    plugin = plugins["r5b_gravity_scale"]
    assert plugin.target is PerturbationTarget.SCENE
    clean = bank.analytic(family)
    tol = ds.contacts.tol_m
    base = measure_clip(truth_measures, clean.scene, clean.states, clean.contacts, tol)
    for level, target in enumerate(levels(plugin, ds), start=1):
        scene, record = perturb_scene(
            plugin, clean.scene, target, SeedKey(ds.seed, family, 0, level)
        )
        assert scene.objects == clean.scene.objects
        clip = bank.of_scene(scene)
        magnitude = measured(plugin, clip.states, clip.contacts, record)
        assert abs(magnitude - target) <= tolerance(target), (level, magnitude)
        after = measure_clip(truth_measures, scene, clip.states, clip.contacts, tol)
        assert moved(base, after) <= {"R5b"}, report(base, after, moved(base, after))
        assert undefined(base, after) <= UNDEFINED.get((family, plugin.name), set())


@pytest.mark.parametrize("family", GRAVITY_FAMILIES)
def test_gravity_scale_is_measured_on_the_recorded_resimulation(
    family: str,
    bank: Any,
    ds: DatasetSettings,
    plugins: dict[str, Any],
    truth_measures: Any,
) -> None:
    plugin = plugins["r5b_gravity_scale"]
    clean = bank.recorded(family)
    clip, record = bank.recorded_gravity_scale(family)
    target = float(record.params["target_magnitude"])
    assert abs(measured(plugin, clip.states, clip.contacts, record) - target) <= tolerance(target)
    tol = ds.contacts.tol_m
    base = measure_clip(truth_measures, clean.scene, clean.states, clean.contacts, tol)
    after = measure_clip(truth_measures, clip.scene, clip.states, clip.contacts, tol)
    assert moved(base, after) <= {"R5b"} | RESIMULATION_LEAKS, report(
        base, after, moved(base, after)
    )
    assert not undefined(base, after)


def test_edits_are_seeded(bank: Any, ds: DatasetSettings, plugins: dict[str, Any]) -> None:
    clip = bank.recorded("f2_bounce")
    runs = [perturbed(plugins["r4_teleport"], clip, "f2_bounce", 3, 1.0, ds.seed) for _ in range(2)]
    assert runs[0][2] == runs[1][2]
    assert np.array_equal(runs[0][0].position, runs[1][0].position, equal_nan=True)


def test_not_applicable_names_the_plugin(
    bank: Any, ds: DatasetSettings, plugins: dict[str, Any]
) -> None:
    clip = bank.recorded("f3_collision")
    with pytest.raises(PerturbationNotApplicableError, match="r3_restitution_gain"):
        perturbed(plugins["r3_restitution_gain"], clip, "f3_collision", 4, 0.5, ds.seed)


def test_every_registered_plugin_measures_its_magnitude(
    perturbations: Any, plugins: dict[str, Any], bank: Any
) -> None:
    assert set(plugins) == set(perturbations)
    assert all(isinstance(plugin, MeasuredPerturbation) for plugin in plugins.values())
    clip, record = bank.recorded_gravity_scale("f2_bounce")
    with pytest.raises(TypeError, match="MeasuredPerturbation"):
        measured(object(), clip.states, clip.contacts, record)


def test_a_toml_override_changes_the_plugin_params(
    config_dir: Path, environ: dict[str, str], perturbations: Any
) -> None:
    override = (0.3, 0.6, 1.2, 2.4)
    with (config_dir / "dataset.toml").open("a", encoding="utf-8") as handle:
        handle.write(f"\n[dataset.perturbations.r4_teleport]\nlevels = {list(override)}\n")
    ds = load_dataset_settings(load_settings(config_dir, environ=environ))
    teleport = perturbations["r4_teleport"]
    assert tuple(perturbation_params(teleport, ds).levels) == override
    assert tuple(instantiate(perturbations, ds)["r4_teleport"].params.levels) == override
    assert perturbation_params(teleport, ds).contact_tol_m == ds.contacts.tol_m
    assert tuple(perturbation_params(perturbations["r4_resize"], ds).levels) != override


# --- r2_unsupported_rest target selection (ADR 0004 §4, test g) ---------------------------------


def box(template: ObjectSpec, object_id: str, size: float, xyz: tuple[float, float, float]) -> Any:
    return replace(
        template,
        object_id=object_id,
        label="cube",
        asset_id="cube",
        primitive=PrimitiveKind.BOX,
        extents=(size, size, size),
        position=xyz,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )


def stack_scene(base: SceneSpec, name: str, boxes: list[tuple[str, float, tuple]]) -> SceneSpec:
    template = base.objects[0]
    objects = tuple(box(template, oid, size, xyz) for oid, size, xyz in boxes)
    return replace(base, scene_id=f"{base.scene_id}-{name}", objects=objects)


def three_stack(base: SceneSpec) -> SceneSpec:
    return stack_scene(
        base,
        "stack3",
        [
            ("a_base", 0.12, (0.0, 0.0, 0.06)),
            ("b_middle", 0.08, (0.0, 0.0, 0.16)),
            ("c_top", 0.05, (0.0, 0.0, 0.225)),
            ("d_loose", 0.06, (0.2, -0.22, 0.03)),
        ],
    )


def gap_to(states: ObjectStateTable, top: int, other: int | None, frame: int) -> float:
    body = Plane.ground() if other is None else solid_at(states, other, frame)
    return float(signed_distance(solid_at(states, top, frame), body).distance)


def test_r2_shifts_the_top_of_a_three_stack_off_its_middle_support(
    bank: Any,
    ds: DatasetSettings,
    plugins: dict[str, Any],
    truth_measures: Any,
) -> None:
    plugin = plugins["r2_unsupported_rest"]
    clip = bank.of_scene(three_stack(bank.recorded("f4_stack").scene))
    ids = [str(i) for i in clip.states.object_ids]
    top = ids.index("c_top")
    targets = levels(plugin, ds)
    for seed in range(len(targets)):
        level = 1 + seed % len(targets)
        new_states, contacts, record = perturbed(
            plugin, clip, "f4_stack", level, targets[level - 1], seed
        )
        assert record.target_objects == ("c_top",)
        assert record.params["support"] == "b_middle"
        for index in (ids.index("a_base"), ids.index("b_middle"), ids.index("d_loose")):
            assert np.array_equal(new_states.position[:, index], clip.states.position[:, index])
            assert np.array_equal(new_states.quaternion[:, index], clip.states.quaternion[:, index])
        others = [None, ids.index("a_base"), ids.index("d_loose")]
        gaps = [gap_to(new_states, top, o, f) for o in others for f in (0, new_states.n_frames - 1)]
        assert min(gaps) >= plugin.params.min_clearance_m
        after = measure_clip(truth_measures, clip.scene, new_states, contacts, ds.contacts.tol_m)
        assert after["R2"] == record.magnitude


@pytest.mark.parametrize("source", ["recorded", "analytic"])
def test_r2_shifts_the_top_body_of_a_two_stack(source: str, sweep: Sweep) -> None:
    records = [
        edit.record
        for (name, _), edit in sweep.edits(source, "f4_stack").items()
        if name == "r2_unsupported_rest"
    ]
    assert len(records) == 4
    for record in records:
        assert record.target_objects == ("cylinder_0",)
        assert record.params["support"] == "cube_0"


@pytest.mark.parametrize(
    "boxes",
    [
        [
            ("a_base", 0.2, (0.0, 0.0, 0.1)),
            ("b_top", 0.05, (0.0, 0.0, 0.225)),
            ("c_side", 0.05, (0.05, 0.0, 0.225)),
        ],
        [("a_floor", 0.1, (0.0, 0.0, 0.05)), ("b_floor", 0.06, (0.2, -0.2, 0.03))],
    ],
    ids=["top-touches-a-neighbour", "floor-only"],
)
def test_r2_refuses_without_a_lone_top_body(
    boxes: list[tuple[str, float, tuple]], bank: Any, ds: DatasetSettings, plugins: dict[str, Any]
) -> None:
    clip = bank.of_scene(stack_scene(bank.recorded("f4_stack").scene, "refuse", boxes))
    with pytest.raises(PerturbationNotApplicableError, match="exactly one movable body"):
        perturbed(plugins["r2_unsupported_rest"], clip, "f4_stack", 1, 0.05, ds.seed)

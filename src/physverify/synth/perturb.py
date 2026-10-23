"""Applying perturbations: seeded attempts, the measured magnitude, and the ``perturb`` step of the
pipeline (variant states, provisional contacts, closest points and ground-truth contacts)."""

from __future__ import annotations

import dataclasses
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from physverify.schema.contacts import ContactTable, RawCollisions
from physverify.schema.scene import PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, ground_truth_source_key
from physverify.synth.gt_contacts import ClosestSamples, from_closest, from_raw_events
from physverify.synth.gt_velocity import position_consistent
from physverify.synth.perturbations._base import ScenePerturbation, TrajectoryPerturbation
from physverify.synth.placement import UnreachableTargetError
from physverify.synth.plan import DatasetPlan, VariantPlan
from physverify.synth.seeds import perturbation_seed
from physverify.synth.settings import DatasetSettings, PerturbationClass, perturbation_params
from physverify.synth.store import (
    CLOSEST,
    CONTACTS_WORLD,
    FAILURE_FILE,
    GT_META,
    KUBRIC_META,
    LAYERS,
    PROVISIONAL_CONTACTS,
    RAW_COLLISIONS,
    RECORD,
    SCENE_RECORD,
    SIM_STATES,
    SPEC,
    VARIANT_STATES,
    ContainerRunner,
    DatasetStore,
    Step,
    atomic_save,
    run_container,
)


class PerturbationNotApplicableError(Exception):
    """No valid edit exists for this scene; the variant is recorded as skipped with the reason."""


@runtime_checkable
class MeasuredPerturbation(Protocol):
    """Synthetic Kubric dataset extension of the frozen protocols: the magnitude measured on the
    perturbed truth."""

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float: ...


REFUSALS: tuple[type[Exception], ...] = (PerturbationNotApplicableError, UnreachableTargetError)


@dataclass(frozen=True)
class SeedKey:
    seed: int
    family: str
    index: int
    level: int

    def attempt_seed(self, name: str, attempt: int) -> int:
        return perturbation_seed(self.seed, self.family, self.index, name, self.level, attempt)


def instantiate(registry: Mapping[str, PerturbationClass], ds: DatasetSettings) -> dict[str, Any]:
    """One configured instance per registered perturbation."""
    return {name: cls(perturbation_params(cls, ds)) for name, cls in registry.items()}


def stamp(
    record: PerturbationRecord, *, level: int, seed: int, magnitude: float
) -> PerturbationRecord:
    return dataclasses.replace(record, level=level, seed=seed, magnitude=float(magnitude))


def measured(
    plugin: object, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
) -> float:
    """The plugin's magnitude measured on the perturbed truth; a plugin that cannot measure it is
    a registration error, never a silent fallback to the target."""
    if not isinstance(plugin, MeasuredPerturbation):
        raise TypeError(f"{record.name}: the plugin does not implement MeasuredPerturbation")
    value = plugin.measure(states, contacts, record)
    if not math.isfinite(value):
        raise PerturbationNotApplicableError(
            f"{record.name}: the magnitude cannot be measured on this clip"
        )
    return float(value)


def _attempts(plugin: object) -> int:
    return int(getattr(getattr(plugin, "params", None), "max_attempts", 1))


def _refused(name: str, reasons: list[str]) -> PerturbationNotApplicableError:
    return PerturbationNotApplicableError(f"{name}: " + "; ".join(dict.fromkeys(reasons)))


def perturb_scene(
    plugin: ScenePerturbation, scene: SceneSpec, target: float, key: SeedKey
) -> tuple[SceneSpec, PerturbationRecord]:
    """Scene edit with the provisional (target) magnitude; it is measured after simulation."""
    if not plugin.applicable(scene):
        raise PerturbationNotApplicableError(f"{plugin.name}: not applicable to {scene.family}")
    reasons: list[str] = []
    for attempt in range(_attempts(plugin)):
        seed = key.attempt_seed(plugin.name, attempt)
        try:
            edited, record = plugin.apply(scene, target, np.random.default_rng(seed))
        except REFUSALS as exc:
            reasons.append(str(exc))
            continue
        return edited, stamp(record, level=key.level, seed=seed, magnitude=record.magnitude)
    raise _refused(plugin.name, reasons)


def perturb_trajectory(
    plugin: TrajectoryPerturbation,
    scene: SceneSpec,
    states: ObjectStateTable,
    contacts: ContactTable,
    target: float,
    key: SeedKey,
) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
    """First successful seeded attempt, stamped with level, seed and the measured magnitude."""
    if not plugin.applicable(scene):
        raise PerturbationNotApplicableError(f"{plugin.name}: not applicable to {scene.family}")
    reasons: list[str] = []
    for attempt in range(_attempts(plugin)):
        seed = key.attempt_seed(plugin.name, attempt)
        try:
            new_states, new_contacts, record = plugin.apply(
                scene, states, contacts, target, np.random.default_rng(seed)
            )
            magnitude = measured(plugin, new_states, new_contacts, record)
        except REFUSALS as exc:
            reasons.append(str(exc))
            continue
        return (
            new_states,
            new_contacts,
            stamp(record, level=key.level, seed=seed, magnitude=magnitude),
        )
    raise _refused(plugin.name, reasons)


def relabel(states: ObjectStateTable, clip_id: str) -> ObjectStateTable:
    return dataclasses.replace(states, clip_id=clip_id, source_key=ground_truth_source_key(clip_id))


def same_poses(a: ObjectStateTable, b: ObjectStateTable) -> bool:
    """True when the edit left every position and orientation bit-identical (contacts unchanged)."""
    return bool(np.array_equal(a.position, b.position, equal_nan=True)) and bool(
        np.array_equal(a.quaternion, b.quaternion, equal_nan=True)
    )


def same_render_inputs(a: ObjectStateTable, b: ObjectStateTable) -> bool:
    """True when the render worker would keyframe the same poses, sizes and hidden frames."""
    same = all(
        np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True)
        for name in ("position", "quaternion", "size")
    )
    return same and bool(np.array_equal(a.visible, b.visible))


def discard_downstream(clip_dir: Path, states: ObjectStateTable) -> bool:
    """Before APPLY writes ``states``: the export and the contact tables of a previous APPLY are
    stale, and so are the render and the closest-point samples when the poses, sizes or hidden
    frames they were computed from changed (or cannot be compared). True when a render went."""
    previous = clip_dir / VARIANT_STATES
    keeps_render = previous.is_file() and same_render_inputs(
        ObjectStateTable.load_npz(previous), states
    )
    stale = [GT_META, PROVISIONAL_CONTACTS, CONTACTS_WORLD]
    if not keeps_render:
        stale += [CLOSEST, KUBRIC_META, LAYERS]
    rendered = (clip_dir / KUBRIC_META).is_file()
    for name in stale:
        (clip_dir / name).unlink(missing_ok=True)
    return rendered and not keeps_render


def load_simulation(
    store: DatasetStore, sim_id: str, clip_id: str, ds: DatasetSettings
) -> tuple[SceneSpec, ObjectStateTable, ContactTable]:
    """The simulated scene, its states and its event contacts, labelled as clip ``clip_id``.

    The contacts are computed on the simulator's own velocities on purpose (the approach speed
    that opens an impact must not move); the states follow ``ds.velocity_convention``."""
    sim_dir = store.scene_dir(sim_id)
    scene = SceneSpec.load_json(sim_dir / SPEC)
    simulated = relabel(ObjectStateTable.load_npz(sim_dir / SIM_STATES), clip_id)
    raw = RawCollisions.load_npz(sim_dir / RAW_COLLISIONS)
    clean = from_raw_events(raw, scene, simulated, ds.contacts)
    if ds.velocity_convention == "state":
        return scene, simulated, clean
    return scene, position_consistent(simulated, clean, scene), clean


@dataclass(frozen=True)
class ApplyReport:
    """Clips APPLY wrote or skipped, and those whose existing render it discarded."""

    applied: tuple[str, ...]
    discarded_renders: tuple[str, ...]


def apply_variants(
    plan: DatasetPlan, store: DatasetStore, plugins: Mapping[str, Any], ds: DatasetSettings
) -> ApplyReport:
    """Writes each pending variant's states, contacts and record; refusals become ``skipped``."""
    applied: list[str] = []
    discarded: list[str] = []
    for variant in plan.variants:
        clip_id = variant.clip_id
        if store.is_skipped(clip_id) or not _apply_pending(store, clip_id):
            continue
        if not store.is_done(Step.SIMULATE, variant.sim_id):
            continue
        try:
            if _apply_one(variant, store, plugins[variant.perturbation], ds):
                discarded.append(clip_id)
        except REFUSALS as exc:
            if discard_clip(store.clip_dir(clip_id)):
                discarded.append(clip_id)
            store.mark_skipped(clip_id, str(exc))
        applied.append(clip_id)
    return ApplyReport(tuple(applied), tuple(discarded))


def discard_clip(clip_dir: Path) -> bool:
    """A refused clip keeps nothing an earlier APPLY, render or export wrote (it may have been
    applied under other settings). True when a render went."""
    rendered = (clip_dir / KUBRIC_META).is_file()
    shutil.rmtree(clip_dir, ignore_errors=True)
    return rendered


def _apply_one(variant: VariantPlan, store: DatasetStore, plugin: Any, ds: DatasetSettings) -> bool:
    scene, states, clean = load_simulation(store, variant.sim_id, variant.clip_id, ds)
    if variant.scene_perturbation:
        provisional = PerturbationRecord.load_json(store.scene_dir(variant.sim_id) / SCENE_RECORD)
        magnitude = measured(plugin, states, clean, provisional)
        record = stamp(
            provisional, level=provisional.level, seed=provisional.seed, magnitude=magnitude
        )
        new_states, contacts = states, clean
    else:
        key = SeedKey(ds.seed, variant.family, variant.index, variant.level)
        new_states, contacts, record = perturb_trajectory(
            plugin, scene, states, clean, variant.target_magnitude, key
        )
    clip_dir = store.clip_dir(variant.clip_id)
    discarded = discard_downstream(clip_dir, new_states)
    atomic_save(new_states, clip_dir / VARIANT_STATES)
    if variant.scene_perturbation or same_poses(states, new_states):
        atomic_save(clean, clip_dir / CONTACTS_WORLD)
    else:
        atomic_save(contacts, clip_dir / PROVISIONAL_CONTACTS)
    atomic_save(record, clip_dir / RECORD)
    return discarded


def _apply_pending(store: DatasetStore, clip_id: str) -> bool:
    """APPLY never ran, or it ran and both of its contact outputs are gone: the event table it
    writes for an unedited trajectory and the provisional table of an edited one."""
    if not store.is_done(Step.APPLY, clip_id):
        return True
    provisional = (store.clip_dir(clip_id) / PROVISIONAL_CONTACTS).is_file()
    return not store.is_done(Step.PERTURB, clip_id) and not provisional


def _needs_contacts(plan: DatasetPlan, store: DatasetStore) -> list[VariantPlan]:
    """Edited trajectories (provisional table written by APPLY) still without contact truth."""
    return [
        v
        for v in plan.variants
        if store.is_done(Step.APPLY, v.clip_id)
        and not store.is_done(Step.PERTURB, v.clip_id)
        and not store.is_skipped(v.clip_id)
        and (store.clip_dir(v.clip_id) / PROVISIONAL_CONTACTS).is_file()
    ]


def closest_points(
    plan: DatasetPlan, store: DatasetStore, runner: ContainerRunner, ds: DatasetSettings
) -> int:
    """Runs the ``contacts`` task for edited trajectories that still lack ``closest.npz``."""
    pending = [
        v
        for v in _needs_contacts(plan, store)
        if not (store.clip_dir(v.clip_id) / CLOSEST).is_file()
    ]
    for v in pending:
        store.clear_failure(Step.PERTURB, v.clip_id)
    jobs = [
        {
            "clip_id": v.clip_id,
            "spec_path": runner.container_path(store.scene_dir(v.sim_id) / SPEC),
            "states_path": runner.container_path(store.clip_dir(v.clip_id) / VARIANT_STATES),
            "out_path": runner.container_path(store.clip_dir(v.clip_id) / CLOSEST),
            "failed_path": runner.container_path(store.clip_dir(v.clip_id) / FAILURE_FILE),
            "tol": ds.contacts.tol_m,
            "floor": ds.floor.model_dump(),
        }
        for v in pending
    ]
    if jobs:
        run_container(runner, "contacts", store.write_jobs("contacts", jobs))
    return len(jobs)


def ground_truth_contacts(plan: DatasetPlan, store: DatasetStore, ds: DatasetSettings) -> list[str]:
    done: list[str] = []
    for variant in _needs_contacts(plan, store):
        clip_dir = store.clip_dir(variant.clip_id)
        if not (clip_dir / CLOSEST).is_file():
            continue
        samples = ClosestSamples.load(clip_dir / CLOSEST)
        provisional = ContactTable.load_json(clip_dir / PROVISIONAL_CONTACTS)
        states = ObjectStateTable.load_npz(clip_dir / VARIANT_STATES)
        atomic_save(
            from_closest(samples, provisional, states, ds.contacts), clip_dir / CONTACTS_WORLD
        )
        done.append(variant.clip_id)
    return done

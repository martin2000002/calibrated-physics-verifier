"""Dataset settings, size priors, plugin registries and clean clips (the recorded Bullet truth of
one scene per family and its analytic twin) shared by the synth tests."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.kubric_worker.build_scene import native_entries
from physverify.kubric_worker.tasks.simulate import state_table
from physverify.physics.geometry import Plane, contact_manifold
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import (
    ContactKind,
    ContactRecord,
    ContactTable,
    ManifoldKind,
    RawCollisions,
)
from physverify.schema.scene import ObjectSpec, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.gt_contacts import (
    ClosestSamples,
    contact_table,
    from_closest,
    from_raw_events,
)
from physverify.synth.perturb import instantiate, relabel
from physverify.synth.perturbations._ballistics import Flight, GravitySchedule, integrate_sphere
from physverify.synth.perturbations._frame_windows import solid_at
from physverify.synth.perturbations._state_edit import ANALYTIC_SOURCE, bounce_record
from physverify.synth.settings import (
    ContactSettings,
    DatasetSettings,
    load_dataset_settings,
    load_size_priors,
)
from physverify.synth.store import DatasetStore

DATA = Path(__file__).parent / "data"
REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
Vec = np.ndarray


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ)


@pytest.fixture
def data_settings(config_dir: Path, environ: dict[str, str], tmp_path: Path) -> Settings:
    env = environ | {"PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}
    return load_settings(config_dir, environ=env)


@pytest.fixture
def ds(settings: Settings) -> DatasetSettings:
    return load_dataset_settings(settings)


@pytest.fixture
def priors(settings: Settings) -> dict[str, Any]:
    return load_size_priors(settings)


@pytest.fixture(scope="session")
def families() -> Any:
    return PLUGIN_KINDS["families"]()


@pytest.fixture(scope="session")
def perturbations() -> Any:
    return PLUGIN_KINDS["perturbations"]()


@pytest.fixture(scope="session")
def truth_measures() -> Any:
    return PLUGIN_KINDS["truth_measures"]()


@pytest.fixture
def plugins(perturbations: Any, ds: DatasetSettings) -> dict[str, Any]:
    return instantiate(perturbations, ds)


@pytest.fixture
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore(tmp_path / "synthetic" / "v1")


@pytest.fixture(scope="session")
def clean_ds(tmp_path_factory: pytest.TempPathFactory) -> DatasetSettings:
    """The committed dataset settings, loaded once per session (no local.toml)."""
    root = tmp_path_factory.mktemp("clean_ds")
    configs = root / "configs"
    configs.mkdir()
    for source in REPO_CONFIGS.glob("*.toml"):
        if not source.name.startswith("local"):
            shutil.copy(source, configs / source.name)
    lock = root / "heavy.lock"
    (configs / "local.toml").write_text(f'[runtime]\nheavy_lock = "{lock}"\n', encoding="utf-8")
    environ = {"HOME": str(root / "home"), "PATH": os.environ.get("PATH", "")}
    return load_dataset_settings(load_settings(configs, environ=environ))


@dataclass(frozen=True)
class Clip:
    scene: SceneSpec
    states: ObjectStateTable
    contacts: ContactTable


class ClipBank:
    """Clean clips, cached per family: ``recorded`` is the Bullet simulation of the committed
    scene, ``analytic`` its exact twin (no simulator penetration or restitution drift)."""

    def __init__(self, cfg: ContactSettings) -> None:
        self.cfg = cfg
        self._cache: dict[tuple[str, str], Clip] = {}

    def recorded(self, family: str) -> Clip:
        return self._cached("recorded", family, lambda: recorded_clip(DATA / family, self.cfg))

    def analytic(self, family: str) -> Clip:
        return self._cached("analytic", family, lambda: self.of_scene(self.recorded(family).scene))

    def recorded_gravity_scale(self, family: str) -> tuple[Clip, PerturbationRecord]:
        """The committed r5b level-3 re-simulation of the family's scene and its record."""
        directory = DATA / "r5b" / family
        record = PerturbationRecord.load_json(directory / "perturbation.json")
        return recorded_clip(directory, self.cfg), record

    def of_scene(self, scene: SceneSpec) -> Clip:
        return analytic_clip(scene, self.cfg)

    def _cached(self, source: str, family: str, build: Any) -> Clip:
        key = (source, family)
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]


@pytest.fixture(scope="session")
def bank(clean_ds: DatasetSettings) -> ClipBank:
    return ClipBank(clean_ds.contacts)


def recorded_clip(directory: Path, cfg: ContactSettings) -> Clip:
    scene = SceneSpec.load_json(directory / "spec.json")
    states = relabel(ObjectStateTable.load_npz(directory / "states.npz"), f"{scene.scene_id}-rec")
    raw = RawCollisions.load_npz(directory / "raw_collisions.npz")
    return Clip(scene, states, from_raw_events(raw, scene, states, cfg))


def analytic_clip(scene: SceneSpec, cfg: ContactSettings) -> Clip:
    """Exact clean truth of ``scene``: spheres fly and bounce analytically from the spec state
    (at native t = -settle / fps), an f3 pair collides once with the impulse law, every other body
    stays at its spec pose; resting contacts come from the primitive manifolds."""
    fps = float(scene.frame_rate)
    times = np.arange(scene.n_frames, dtype=np.float64) / fps
    flights, impacts = _flights(scene, times)
    tracks = [_track(obj, flights.get(obj.object_id), times) for obj in scene.objects]
    padded = [{k: _pad(v, scene.settle_frames) for k, v in t.items()} for t in tracks]
    states = relabel(state_table(scene, padded), f"{scene.scene_id}-analytic")
    radius = {o.object_id: o.extents[0] / 2.0 for o in scene.objects}
    for object_id, flight in flights.items():
        impacts.extend(
            bounce_record(object_id, b, radius[object_id], fps) for b in flight.bounces if b.t >= 0
        )
    provisional = contact_table(states.clip_id, impacts)
    samples = primitive_closest(states, cfg.tol_m)
    return Clip(scene, states, from_closest(samples, provisional, states, cfg))


def _pad(values: Vec, settle_frames: int) -> Vec:
    rows = native_entries(settle_frames, values.shape[0])
    out = np.repeat(values[:1], rows.stop, axis=0)
    out[rows] = values
    return out


def _track(obj: ObjectSpec, flight: Flight | None, times: Vec) -> dict[str, Vec]:
    n = times.size
    if flight is None:
        position = np.repeat(np.asarray([obj.position], dtype=np.float64), n, axis=0)
        velocity = np.zeros((n, 3))
    else:
        position, velocity = flight.positions, flight.velocities
    return {
        "position": position,
        "quaternion": np.repeat(np.asarray([obj.quaternion], dtype=np.float64), n, axis=0),
        "velocity": velocity,
        "angular_velocity": np.zeros((n, 3)),
    }


def _flights(scene: SceneSpec, times: Vec) -> tuple[dict[str, Flight], list[ContactRecord]]:
    t0 = -scene.settle_frames / float(scene.frame_rate)
    g = np.asarray(scene.gravity, dtype=np.float64)
    balls = [o for o in scene.objects if o.primitive is PrimitiveKind.SPHERE and not o.static]
    starts = {o.object_id: (t0, np.asarray(o.position), np.asarray(o.velocity)) for o in balls}
    flights = {o.object_id: _fly(o, starts[o.object_id], g, times) for o in balls}
    impacts: list[ContactRecord] = []
    if scene.family == "f3_collision":
        a, b = sorted(balls, key=lambda o: o.object_id)[:2]
        hit = sphere_collision(a, b, g, t0)
        if hit is not None and hit.t < times[-1]:
            impacts.append(hit.record(float(scene.frame_rate)))
            for obj, after in ((a, hit.after_a), (b, hit.after_b)):
                early = _fly(obj, starts[obj.object_id], g, times[times < hit.t])
                late = _fly(obj, (hit.t, *after), g, times[times >= hit.t])
                flights[obj.object_id] = Flight(
                    np.concatenate([early.positions, late.positions]),
                    np.concatenate([early.velocities, late.velocities]),
                    early.bounces + late.bounces,
                )
    return flights, impacts


def _fly(obj: ObjectSpec, start: tuple[float, Vec, Vec], g: Vec, times: Vec) -> Flight:
    if times.size == 0:
        return Flight(np.zeros((0, 3)), np.zeros((0, 3)))
    t, p, v = start
    schedule = GravitySchedule.constant(g)
    return integrate_sphere(p, v, t, obj.extents[0] / 2.0, schedule, obj.restitution, times)


@dataclass(frozen=True)
class SphereHit:
    t: float
    a: str
    b: str
    point: Vec
    normal: Vec
    after_a: tuple[Vec, Vec]
    after_b: tuple[Vec, Vec]

    def record(self, fps: float) -> ContactRecord:
        return ContactRecord(
            object_a=self.a,
            object_b=self.b,
            t_start=self.t * fps,
            t_end=self.t * fps,
            kind=ContactKind.IMPACT,
            normal=self.normal,
            points=self.point.reshape(1, 3),
            manifold=ManifoldKind.POINT,
            confidence=1.0,
            source=ANALYTIC_SOURCE,
        )


def sphere_collision(a: ObjectSpec, b: ObjectSpec, g: Vec, t0: float) -> SphereHit | None:
    """First touch of two spheres in the same gravity field (their separation d0 + dv t is
    linear) and the post-impact states from the impulse law with e = e_a e_b [sin verificar]."""
    ra, rb = a.extents[0] / 2.0, b.extents[0] / 2.0
    pa, pb = np.asarray(a.position, dtype=np.float64), np.asarray(b.position, dtype=np.float64)
    va, vb = np.asarray(a.velocity, dtype=np.float64), np.asarray(b.velocity, dtype=np.float64)
    d0, dv = pa - pb, va - vb
    qa, qb, qc = float(dv @ dv), 2.0 * float(d0 @ dv), float(d0 @ d0) - (ra + rb) ** 2
    disc = qb * qb - 4.0 * qa * qc
    if qa <= 0.0 or disc < 0.0:
        return None
    tau = (-qb - math.sqrt(disc)) / (2.0 * qa)
    if tau <= 0.0:
        return None
    pa_c, pb_c = pa + va * tau + 0.5 * g * tau**2, pb + vb * tau + 0.5 * g * tau**2
    va_c, vb_c = va + g * tau, vb + g * tau
    normal = (pa_c - pb_c) / (ra + rb)
    e = a.restitution * b.restitution
    impulse = -(1.0 + e) * float((va_c - vb_c) @ normal) / (1.0 / a.mass + 1.0 / b.mass)
    return SphereHit(
        t=t0 + tau,
        a=a.object_id,
        b=b.object_id,
        point=pb_c + rb * normal,
        normal=normal,
        after_a=(pa_c, va_c + impulse / a.mass * normal),
        after_b=(pb_c, vb_c - impulse / b.mass * normal),
    )


@pytest.fixture(scope="session")
def collide() -> Any:
    """``sphere_collision`` for tests that solve the first touch of two balls."""
    return sphere_collision


def primitive_closest(states: ObjectStateTable, tol: float) -> ClosestSamples:
    """What ``getClosestPoints`` would report within ``tol`` if Bullet sampled the whole
    primitive manifold: every manifold point of every touching pair, normals b to a."""
    rows: list[tuple[int, str, str, Vec, Vec, float]] = []
    ids = [str(i) for i in states.object_ids]
    for frame in range(states.n_frames):
        for i, a_id in enumerate(ids):
            body_a = solid_at(states, i, frame)
            others = [(ENV_ID, Plane.ground())] + [
                (ids[j], solid_at(states, j, frame)) for j in range(i + 1, len(ids))
            ]
            for b_id, body_b in others:
                manifold = contact_manifold(body_a, body_b, tol)
                if manifold is None:
                    continue
                for point, normal, distance in zip(
                    manifold.points, manifold.normals, manifold.distances, strict=True
                ):
                    rows.append((frame, a_id, b_id, point, normal, float(distance)))
    return _samples(rows)


def _samples(rows: list[tuple[int, str, str, Vec, Vec, float]]) -> ClosestSamples:
    points = np.array([r[3] for r in rows], dtype=np.float64).reshape(-1, 3)
    return ClosestSamples(
        frame=np.array([r[0] for r in rows], dtype=np.int64),
        a=np.array([r[1] for r in rows], dtype=str),
        b=np.array([r[2] for r in rows], dtype=str),
        point_a=points,
        point_b=points.copy(),
        normal_on_b=np.array([r[4] for r in rows], dtype=np.float64).reshape(-1, 3),
        distance=np.array([r[5] for r in rows], dtype=np.float64),
    )

"""ENV supports of R3b image_relspeed: the floor_plane rule changes the recorded f3_floor impact
only, and its exclusions (tilted floor, ENV impact, third body) keep the old guard."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest
from repo_configs import copy_repo_configs

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks.variants.env_supports._base import ENV_SUPPORTS, build_rules
from physverify.checks.variants.env_supports.floor_plane import FloorPlane
from physverify.checks.variants.r3b_image_relspeed import RelSpeedCheck, RelSpeedParams
from physverify.config.loader import load_settings
from physverify.core.errors import ConfigError
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, RawCollisions
from physverify.schema.io import dumps
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.scene import SceneSpec
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable
from physverify.synth.gt.camera import camera_contacts, camera_states
from physverify.synth.gt.contacts import from_raw_events
from physverify.synth.perturb import relabel
from physverify.synth.placement import spec_world_from_camera
from physverify.synth.settings import ContactSettings, load_dataset_settings

CLEAN = Path(__file__).parents[2] / "synth" / "data" / "clean"
FAMILIES = sorted(p.name for p in CLEAN.iterdir() if p.name != "f3_floor")
TRUTH = {Stage.CONTACTS.value: GROUND_TRUTH, Stage.LIFT.value: GROUND_TRUTH}
PERCEIVED: dict[str, str] = {}
PROVENANCES = pytest.mark.parametrize("provenance", [TRUTH, PERCEIVED], ids=["truth", "detected"])
FLOOR = RelSpeedParams(env_supports=("floor_plane",))
PAIR = ("ball_0", "ball_1")


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> ContactSettings:
    root = tmp_path_factory.mktemp("env_supports")
    configs = copy_repo_configs(root / "configs")
    (configs / "local.toml").write_text(
        f'[runtime]\nheavy_lock = "{root / "heavy.lock"}"\n', encoding="utf-8"
    )
    environ = {"HOME": str(root / "home"), "PATH": os.environ.get("PATH", "")}
    return load_dataset_settings(load_settings(configs, environ=environ)).contacts


def recorded(family: str, cfg: ContactSettings) -> sc.Case:
    """The committed Bullet clean clip of ``family`` in its spec camera's OpenCV frame."""
    return _recorded(family, cfg)


@cache
def _recorded(family: str, cfg: ContactSettings) -> sc.Case:
    directory = CLEAN / family
    scene = SceneSpec.load_json(directory / "spec.json")
    states = relabel(ObjectStateTable.load_npz(directory / "states.npz"), f"{scene.scene_id}-rec")
    contacts = from_raw_events(
        RawCollisions.load_npz(directory / "raw_collisions.npz"), scene, states, cfg
    )
    pose = spec_world_from_camera(scene.camera)
    width, height = scene.resolution
    fps = float(scene.frame_rate)
    clip = ClipManifest(
        dataset=sc.DATASET,
        clip_id=states.clip_id,
        split=Split.EVAL,
        video_path=f"{sc.DATASET}/{states.clip_id}",
        n_frames=scene.n_frames,
        width=width,
        height=height,
        container_fps=fps,
        seconds_per_frame=1.0 / fps,
        time_source=TimeSource.GROUND_TRUTH,
        gt_stages=STAGE_ORDER,
    )
    return sc.Case(
        clip=clip,
        states=camera_states(states, pose),
        contacts=camera_contacts(contacts, pose),
        scale=sc.oracle_scale(),
        view=sc.camera(),
    )


def evaluate(
    case: sc.Case, params: RelSpeedParams, provenance: dict[str, str]
) -> list[CheckResult]:
    return RelSpeedCheck(params).evaluate(case.inputs(InputLevel.PERCEIVED, provenance))


def outputs(case: sc.Case, params: RelSpeedParams, provenance: dict[str, str]) -> list[str]:
    return [dumps(r.to_dict()) for r in evaluate(case, params, provenance)]


def pair_result(case: sc.Case, params: RelSpeedParams, provenance: dict[str, str]) -> CheckResult:
    (result,) = [r for r in evaluate(case, params, provenance) if r.subject == PAIR]
    return result


@pytest.mark.parametrize("family", FAMILIES)
@PROVENANCES
def test_the_rule_leaves_every_other_family_unchanged(
    family: str, provenance: dict[str, str], cfg: ContactSettings
) -> None:
    case = recorded(family, cfg)
    assert outputs(case, FLOOR, provenance) == outputs(case, RelSpeedParams(), provenance)


@PROVENANCES
def test_the_rolling_impact_is_scored_with_its_floor_supports(
    provenance: dict[str, str], cfg: ContactSettings
) -> None:
    """The plain check abstains on the truth event and detects none among the RESTING rows."""
    case = recorded("f3_floor", cfg)
    before = [r for r in evaluate(case, RelSpeedParams(), provenance) if r.subject == PAIR]
    assert [r.abstain_reason for r in before] == (
        [AbstainReason.OUT_OF_SCOPE] if provenance else []
    )
    after = pair_result(case, FLOOR, provenance)
    assert after.abstain_reason is None and after.magnitude is not None
    assert math.isfinite(after.magnitude)
    supports = after.details["floor_supports"]
    assert supports["records"] and supports["bias_budget"] == FLOOR.support_bias_budget
    assert all(v <= FLOOR.support_tilt_deg for v in supports["angles"].values())


def with_floor(case: sc.Case, edit: Any) -> sc.Case:
    assert case.contacts is not None
    records = tuple(edit(r) if r.object_b == ENV_ID else r for r in case.contacts.records)
    return replace(case, contacts=replace(case.contacts, records=records))


def tilted(record: ContactRecord, case: sc.Case, deg: float) -> ContactRecord:
    """``record`` with its normal turned by ``deg`` towards the pair's mean motion."""
    v = np.nanmean(np.asarray(case.states.velocity[:, 0], dtype=np.float64), axis=0)
    n = np.asarray(record.normal, dtype=np.float64)
    along = v - (v @ n) * n
    along /= np.linalg.norm(along)
    a = math.radians(deg)
    return replace(record, normal=math.cos(a) * n + math.sin(a) * along)


@PROVENANCES
def test_a_floor_tilted_past_the_margin_keeps_the_guard(
    provenance: dict[str, str], cfg: ContactSettings
) -> None:
    """On truth the floor normal leaves the impact normal's plane; detected, the side velocities
    leave the floor's plane."""
    case = recorded("f3_floor", cfg)
    steep = with_floor(case, lambda r: tilted(r, case, 3.0))
    pair = [r for r in evaluate(steep, FLOOR, provenance) if r.subject == PAIR]
    assert all(r.abstain_reason is AbstainReason.OUT_OF_SCOPE for r in pair)
    assert all("floor_supports" not in r.details for r in pair)
    gentle = with_floor(case, lambda r: tilted(r, case, 1.0))
    assert pair_result(gentle, FLOOR, provenance).abstain_reason is None


def test_an_env_impact_keeps_the_guard(cfg: ContactSettings) -> None:
    case = recorded("f3_floor", cfg)
    bounced = with_floor(case, lambda r: replace(r, kind=ContactKind.IMPACT))
    assert pair_result(bounced, FLOOR, TRUTH).abstain_reason is AbstainReason.OUT_OF_SCOPE


def with_third_body(case: sc.Case, frames_after: float) -> sc.Case:
    """A third sphere far away that touches ball_0 ``frames_after`` the pair's impact."""
    assert case.contacts is not None
    states = case.states
    k = states.position.shape[1]

    def grown(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == k:
            return np.concatenate([value, value[:, 1:2]], axis=1)
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == k:
            return np.concatenate([value, value[1:2]])
        return value

    fields = {name: grown(getattr(states, name)) for name in states.__dataclass_fields__}
    fields["object_ids"] = np.array([*states.object_ids, "ball_9"])
    fields["position"][:, k] += np.array([0.0, 0.0, 5.0])
    (impact,) = [r for r in case.contacts.records if r.kind is ContactKind.IMPACT]
    t = impact.t_start + frames_after
    extra = replace(impact, object_b="ball_9", t_start=t, t_end=t)
    contacts = replace(case.contacts, records=(*case.contacts.records, extra))
    return replace(case, states=replace(states, **fields), contacts=contacts)


def test_a_third_body_between_the_sides_keeps_the_guard(cfg: ContactSettings) -> None:
    """A wide gap without margin leaves the side rows free around the third body's touch (a true
    record: detected bounds hold only ENV records and the other detected events)."""
    wide = FLOOR.model_copy(update={"gap_frames": 3.0, "bound_margin_rows": 0})
    case = recorded("f3_floor", cfg)
    assert pair_result(with_third_body(case, 30.0), wide, TRUTH).abstain_reason is None
    pair = [r for r in evaluate(with_third_body(case, 1.5), wide, TRUTH) if r.subject == PAIR]
    assert all(r.abstain_reason is AbstainReason.OUT_OF_SCOPE for r in pair)
    assert all("floor_supports" not in r.details for r in pair)


def test_rules_are_registered_injectable_and_unknown_names_fail() -> None:
    registry = PLUGIN_KINDS[ENV_SUPPORTS.name]()
    assert registry["floor_plane"] is FloorPlane
    with pytest.raises(ConfigError, match="unknown"):
        build_rules(("nope",), registry)
    with pytest.raises(ConfigError, match="unknown"):
        RelSpeedCheck(RelSpeedParams(env_supports=("nope",)))
    check = RelSpeedCheck(FLOOR, support_registry={"floor_plane": FloorPlane})
    assert [type(r) for r in check.env_supports] == [FloorPlane]
    assert RelSpeedCheck(RelSpeedParams()).env_supports == ()

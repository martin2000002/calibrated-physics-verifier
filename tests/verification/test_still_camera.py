"""The still-camera precondition gates the camera-frame checks of a clip
whose background moves, and changes nothing otherwise."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory
from synthetic_clip import SyntheticFixtureDataset
from wiring import resolve_plan

from physverify.app import calibration as app_calibration
from physverify.app import plugins
from physverify.app.context import AppContext, GlobalOptions, Invocation
from physverify.app.plugins import PLUGIN_KINDS
from physverify.app.plugins import build as plugin_registry
from physverify.checks._base import (
    Check,
    CheckInputs,
    assumes_still_camera,
    unscored,
    validate_check,
)
from physverify.checks._params import CommonParams
from physverify.config.models import Settings
from physverify.core.errors import InvalidPluginError
from physverify.extraction.cache import StageCache
from physverify.extraction.pipeline import PipelineDeps
from physverify.extraction.pipeline_resources import LazyLock, ProcRssProbe
from physverify.extraction.plan import ResolvedPlan
from physverify.finetune.scorers.graded import GRADING_REASONS, grades
from physverify.schema.camera_motion import CameraMotion
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.verification import check_routes
from physverify.verification.check_routes import perceived_routed_source, still_camera_settings
from physverify.verification.inputs import (
    ClipInputs,
    PerceivedSource,
    PipelineOutputLoader,
    refuse_heavy,
)
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import (
    INPUT_GAP_KEY,
    VerificationReport,
    VerificationRunner,
    configured_suite,
)
from physverify.verification.still_camera import StillCameraSettings, camera_gate
from physverify.verification.thresholding import ThresholdTable

LIMIT = 0.02
SETTINGS = StillCameraSettings(per_second=True, p90_limit=LIMIT)
ROUTED_R3A = (
    "verification.routes.area_depth.lift=backproject_area_depth",
    "verification.check_routes.R3a=area_depth",
)
GATED = {"R1", "R2", "R3a", "R3b", "R4_jump", "R4_size", "R5a", "R5b"}


def motion(per_pair: float, measured: bool = True, pairs: int = 10) -> CameraMotion:
    if not measured:
        return CameraMotion.unmeasured("affine_flow")
    return CameraMotion(
        "affine_flow", np.full(pairs, per_pair), np.full(pairs, 0.5), np.ones(pairs), True
    )


MOVING = motion(0.05)
"""0.1 diagonals per second against a limit of 0.02."""


def checks(settings: Settings) -> list[Check]:
    found = plugins.configured("checks", settings)
    return list(configured_suite(settings, found, PLUGIN_KINDS["policies"]()).checks)


def run(
    found: Sequence[Check], camera: CameraMotion | None, gate: StillCameraSettings | None
) -> VerificationReport:
    case = sc.parabola(clip_id="c")
    item = ClipInputs(case.clip, replace(case.inputs(), camera_motion=camera), None)
    runner = VerificationRunner(
        found, RawPolicy(), ThresholdTable.provisional({}), still_camera=gate
    )
    return runner.run(_One(item), [case.clip])


class _One:
    def __init__(self, item: ClipInputs) -> None:
        self._item = item

    def load(self, clips: Sequence[Any]) -> Any:
        yield self._item


def by_check(report: VerificationReport) -> dict[str, list[CheckResult]]:
    found: dict[str, list[CheckResult]] = {}
    for result in report.results:
        found.setdefault(result.check_id, []).append(result)
    return found


def test_the_nine_checks_but_r4_missing_assume_a_still_camera(
    make_settings: SettingsFactory,
) -> None:
    declared = {type(c).check_id: assumes_still_camera(c) for c in checks(make_settings())}
    assert {cid for cid, still in declared.items() if still} == GATED
    assert declared["R4_missing"] is False


@pytest.mark.parametrize(
    "camera",
    [None, motion(0.0, measured=False), motion(0.009), motion(float("nan"))],
    ids=["no_motion", "unmeasured", "under_the_limit", "no_background"],
)
def test_without_a_moving_camera_the_output_is_the_ungated_one(
    make_settings: SettingsFactory, camera: CameraMotion | None
) -> None:
    found = checks(make_settings())
    gated = run(found, camera, SETTINGS)
    assert gated.results == run(found, camera, None).results
    assert len(gated.table) == 9


def test_a_moving_camera_abstains_the_camera_frame_checks(make_settings: SettingsFactory) -> None:
    results = by_check(run(checks(make_settings()), MOVING, SETTINGS))
    for check_id in GATED:
        (gated,) = results[check_id]
        assert gated.verdict is Verdict.ABSTAIN
        assert gated.abstain_reason is AbstainReason.CAMERA_NOT_STILL
        assert gated.magnitude is None and math.isnan(gated.sigma)
        assert gated.details["camera_p90"] == pytest.approx(0.1)
        assert gated.details["p90_limit"] == LIMIT and gated.details["per_second"] is True
        assert INPUT_GAP_KEY not in gated.details
        assert not grades(gated)
    assert all(
        r.abstain_reason is not AbstainReason.CAMERA_NOT_STILL for r in results["R4_missing"]
    )
    assert results["R4_missing"] == by_check(run(checks(make_settings()), None, None))["R4_missing"]


def test_the_limit_is_strict_and_the_max_limit_is_optional(make_settings: SettingsFactory) -> None:
    (check,) = [c for c in checks(make_settings()) if type(c).check_id == "R1"]
    inputs = replace(sc.parabola().inputs(), camera_motion=motion(LIMIT / 2))
    assert camera_gate(check, inputs, SETTINGS) is None
    peaked = replace(
        inputs,
        camera_motion=CameraMotion(
            "affine_flow", np.array([0.0] * 9 + [0.05]), np.full(10, 0.5), np.ones(10), True
        ),
    )
    assert camera_gate(check, peaked, SETTINGS) is None
    capped = SETTINGS.model_copy(update={"max_limit": 0.05})
    gated = camera_gate(check, peaked, capped)
    assert gated is not None and gated.details["camera_peak"] == pytest.approx(0.1)


def test_the_gate_is_neither_a_missing_input_nor_a_grading_reason() -> None:
    assert AbstainReason.CAMERA_NOT_STILL not in GRADING_REASONS
    assert AbstainReason.CAMERA_NOT_STILL is not AbstainReason.MISSING_INPUT
    assert AbstainReason.CAMERA_NOT_STILL.value == "CAMERA_NOT_STILL"


def test_every_variant_declares_what_its_check_declares() -> None:
    defaults = plugins.build("checks")
    for key, variant in plugins.build("check_variants").items():
        assert assumes_still_camera(variant) is assumes_still_camera(defaults[variant.check_id]), (
            key
        )


def fake_check(**namespace: Any) -> type[Check]:
    def evaluate(self: Check, inputs: CheckInputs) -> Sequence[CheckResult]:
        return [unscored(type(self), inputs, ("a",), 0.1, 0.01, (0, 3), {})]

    def init(self: Check, params: CommonParams | None = None) -> None:
        return None

    base = {
        "check_id": "F1",
        "version": "1",
        "requires": frozenset(),
        "time_dependent": False,
        "magnitude_unit": "unit",
        "Params": CommonParams,
        "__init__": init,
        "evaluate": evaluate,
    }
    return type("FakeCheck", (), {**base, **namespace})


def test_an_undeclared_check_assumes_a_still_camera_and_is_still_a_check() -> None:
    undeclared = fake_check()
    validate_check(undeclared)
    assert assumes_still_camera(undeclared) and assumes_still_camera(undeclared())
    assert isinstance(undeclared(), Check)
    declared = fake_check(assumes_still_camera=False)
    validate_check(declared)
    assert not assumes_still_camera(declared) and isinstance(declared(), Check)


def test_a_non_bool_declaration_is_refused() -> None:
    with pytest.raises(InvalidPluginError, match="assumes_still_camera must be a bool"):
        validate_check(fake_check(assumes_still_camera="yes"))


def test_the_configured_section_reaches_the_suite_runner(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    gate = still_camera_settings(settings)
    assert gate is not None and gate.per_second
    assert still_camera_settings(make_settings()) == gate


def test_the_calibration_raw_runner_gates_a_moving_camera(
    config_dir: Path, environ: dict[str, str]
) -> None:
    context = AppContext(Invocation(("physverify",), environ), GlobalOptions(config_dir=config_dir))
    runner = app_calibration._raw_runner(context)
    case = sc.parabola(clip_id="c")
    item = ClipInputs(case.clip, replace(case.inputs(), camera_motion=MOVING), None)
    report = runner.run(_One(item), [case.clip])
    reasons = {r.check_id: r.abstain_reason for r in report.results}
    assert {cid for cid, r in reasons.items() if r is AbstainReason.CAMERA_NOT_STILL} == GATED


class _Recorder:
    """Stands for ``PipelineOutputLoader`` in ``check_routes``: records what each plan gets."""

    seen: ClassVar[list[tuple[ResolvedPlan, StillCameraSettings | None]]] = []

    def __init__(self, plan: ResolvedPlan, deps: object, still_camera: Any = None) -> None:
        self.seen.append((plan, still_camera))


def test_every_routed_plan_measures_on_the_main_plans_track(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check_routes, "PipelineOutputLoader", _Recorder)
    monkeypatch.setattr(_Recorder, "seen", [])
    settings = make_settings(*ROUTED_R3A, profile="precise")
    resolve = partial(resolve_plan, providers=PLUGIN_KINDS["providers"](), dataset="d")
    perceived_routed_source(
        settings,
        None,
        resolve,
        PLUGIN_KINDS["checks"](),
        None,
        InputLevel.PERCEIVED,  # type: ignore[arg-type]
    )
    plans = [plan for plan, _ in _Recorder.seen]
    assert len({plan.names[Stage.LIFT] for plan in plans}) >= 2
    assert {plan.step(Stage.TRACK) for plan in plans} == {plans[0].step(Stage.TRACK)}
    assert all(gate == still_camera_settings(settings) for _, gate in _Recorder.seen)


def gt_loader(settings: Settings, gate: StillCameraSettings | None) -> PipelineOutputLoader:
    deps = PipelineDeps(
        cache=StageCache(settings.paths.cache_dir),
        lock=LazyLock(partial(refuse_heavy, "synthetic_fixture")),
        paths=settings.paths,
        device="cpu",
        threads=1,
        error_profile=None,
        observer=None,
        rss=ProcRssProbe(),
        registry=plugin_registry,
    )
    plan = resolve_plan(
        settings, PLUGIN_KINDS["providers"](), dataset="synthetic_fixture", gt_stages=STAGE_ORDER
    )
    return PipelineOutputLoader(plan, deps, gate)


def camera_of(loader: PipelineOutputLoader) -> CameraMotion | None:
    dataset = SyntheticFixtureDataset()
    (item,) = PerceivedSource(dataset, loader, InputLevel.MIXED).load(dataset.clips())
    assert item.inputs is not None
    return item.inputs.camera_motion


def test_the_loader_measures_once_and_every_plan_on_that_track_reads_the_same_motion(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings()
    gate = still_camera_settings(settings)
    assert camera_of(gt_loader(settings, None)) is None
    first = camera_of(gt_loader(settings, gate))
    assert first is not None and first.measured
    assert camera_of(gt_loader(settings, gate)) == first
    cached = list(Path(settings.paths.cache_dir).rglob("camera_motion/*/output.json"))
    assert len(cached) == 1

"""AC11: building check inputs (time axis, scale binding, oracle, perceived, no heavy stage)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest
from synthetic_clip import SyntheticFixtureDataset

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import Check
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import MissingInputError
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.cache import StageCache
from physverify.extraction.matching import take_frames
from physverify.extraction.pipeline import LazyLock, PipelineDeps, ProcRssProbe
from physverify.extraction.plan import resolve_plan
from physverify.schema.checks import AbstainReason, InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import CoordFrame
from physverify.verification.inputs import (
    SCALE_SOURCE_KEY,
    ClipInputs,
    InputsError,
    OracleSource,
    PerceivedSource,
    PipelineOutputLoader,
    RawInputs,
    StageOutputs,
    build_inputs,
    oracle_scale,
    refuse_heavy,
)
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import RowStatus, VerificationRunner


def raw(case: sc.Case, scale: ScaleEstimate | None = None) -> RawInputs:
    clip_id = case.clip.clip_id
    return RawInputs(
        clip=case.clip,
        states=case.states,
        contacts=case.contacts,
        scale=oracle_scale(clip_id) if scale is None else scale,
        scene=case.scene,
        level=InputLevel.ORACLE,
        provenance={"kinematics": GROUND_TRUTH},
    )


def blank_check(index: int) -> Check:
    namespace = {
        "check_id": f"C{index}",
        "requires": frozenset(),
        "magnitude_unit": "unit",
        "Params": None,
        "evaluate": lambda self, inputs: [],
    }
    check: Check = type(f"Blank{index}", (), namespace)()
    return check


class ListSource:
    def __init__(self, items: Sequence[ClipInputs]) -> None:
        self._items = list(items)

    def load(self, clips: Sequence[ClipManifest]) -> Any:
        return iter(self._items)


def test_the_oracle_scale_is_exact_and_bound_to_the_table() -> None:
    inputs = build_inputs(raw(sc.parabola(clip_id="p")))
    assert inputs.scale is not None
    assert (inputs.scale.meters_per_unit, inputs.scale.rel_sigma) == (1.0, 0.0)
    assert inputs.scale.source is ScaleSource.GROUND_TRUTH


def test_a_scale_fitted_on_another_table_is_refused() -> None:
    wrong = ScaleEstimate(1.0, 0.0, ScaleSource.GROUND_TRUTH, "gt:another_clip")
    with pytest.raises(InputsError, match="^scale_binding"):
        build_inputs(raw(sc.parabola(clip_id="p"), wrong))


def test_without_scale_the_inputs_have_no_scale() -> None:
    none = ScaleEstimate(None, math.nan, ScaleSource.NONE, "gt:p")
    assert build_inputs(raw(sc.parabola(clip_id="p"), none)).scale is None


def strided_on_rows(case: sc.Case) -> sc.Case:
    """Frames 0, 2, 4... but ``t`` counted per row: the D14.2 mismatch."""
    states = sc.strided(case, 2, quantize=False).states
    return replace(case, states=replace(states, t=np.arange(len(states.t)) * sc.DT))


def non_uniform(case: sc.Case) -> sc.Case:
    kept = np.delete(np.unique(case.states.frame_index), 3)
    return replace(case, states=take_frames(case.states, kept))


@pytest.mark.parametrize("broken", [strided_on_rows, non_uniform], ids=["t_per_row", "gap"])
def test_a_bad_time_axis_abstains_on_every_check(broken: Any) -> None:
    case = broken(sc.parabola(clip_id="t"))
    with pytest.raises(InputsError, match="^time_axis"):
        build_inputs(raw(case))
    item = OracleSource(ScenarioDataset([case])).load([case.clip])
    report = VerificationRunner([blank_check(i) for i in range(9)], RawPolicy(), {}).run(
        ListSource(list(item)), [case.clip]
    )
    assert [row.status for row in report.table] == [RowStatus.ABSTAINED] * 9
    assert {r.abstain_reason for r in report.results} == {AbstainReason.MISSING_INPUT}
    assert all(str(r.details["error"]).startswith("time_axis") for r in report.results)


def test_a_strided_table_with_native_times_is_valid() -> None:
    inputs = build_inputs(raw(sc.strided(sc.parabola(clip_id="s"), 2, quantize=False)))
    assert inputs.states.frame_index[1] == 2


class CaseStore:
    def __init__(self, case: sc.Case) -> None:
        self._case = case

    def states(self, frame: CoordFrame) -> Any:
        return self._case.states

    def contacts(self, frame: CoordFrame) -> Any:
        return self._case.contacts

    def scene(self) -> Any:
        return self._case.scene

    def camera(self, sampling: Any) -> Any:
        return None


class ScenarioDataset:
    name = "scenario"

    def __init__(self, cases: Sequence[sc.Case], with_truth: bool = True) -> None:
        self._cases = {case.clip.clip_id: case for case in cases}
        self._with_truth = with_truth

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return [case.clip for case in self._cases.values()]

    def ground_truth(self, clip: ClipManifest) -> Any:
        return CaseStore(self._cases[clip.clip_id]) if self._with_truth else None


def test_the_oracle_labels_every_stage_as_ground_truth() -> None:
    case = sc.parabola(clip_id="o")
    (item,) = OracleSource(ScenarioDataset([case])).load([case.clip])
    assert item.inputs is not None
    assert item.inputs.input_level is InputLevel.ORACLE
    provenance = item.inputs.provenance
    assert {provenance[stage.value] for stage in STAGE_ORDER} == {GROUND_TRUTH}
    assert provenance[SCALE_SOURCE_KEY] == ScaleSource.GROUND_TRUTH.value


def test_a_dataset_without_ground_truth_has_no_oracle() -> None:
    case = sc.parabola(clip_id="o")
    (item,) = OracleSource(ScenarioDataset([case], with_truth=False)).load([case.clip])
    assert item.inputs is None and item.input_level is InputLevel.ORACLE
    assert item.error is not None and item.error.startswith("MissingInputError")


class FakeLoader:
    def __init__(self, found: Mapping[str, StageOutputs | str]) -> None:
        self._found = found

    def load(
        self, dataset: DatasetSource, clips: Sequence[ClipManifest]
    ) -> Mapping[str, StageOutputs | str]:
        return self._found


def perceived(case: sc.Case, outputs: Mapping[Stage, StageOutput]) -> ClipInputs:
    names = {stage: "fake" for stage in STAGE_ORDER}
    loader = FakeLoader({case.clip.clip_id: StageOutputs(outputs, names)})
    source = PerceivedSource(ScenarioDataset([case]), loader, InputLevel.MIXED)
    (item,) = source.load([case.clip])
    return item


def test_perceived_inputs_come_from_the_stage_outputs() -> None:
    case = sc.parabola(clip_id="m")
    scale = ScaleEstimate(2.0, 0.1, ScaleSource.SIZE_PRIOR, case.states.source_key)
    outputs = {Stage.KINEMATICS: case.states, Stage.SCALE: scale}
    item = perceived(case, outputs)
    assert item.inputs is not None
    assert item.inputs.input_level is InputLevel.MIXED
    assert item.inputs.contacts is None and item.inputs.scene is None
    assert item.inputs.scale is not None and item.inputs.scale.meters_per_unit == 2.0
    assert item.inputs.provenance[SCALE_SOURCE_KEY] == ScaleSource.SIZE_PRIOR.value
    assert item.inputs.provenance[Stage.DEPTH.value] == "fake"


def test_perceived_inputs_need_kinematics() -> None:
    item = perceived(sc.parabola(clip_id="m"), {})
    assert item.inputs is None and item.input_level is InputLevel.MIXED
    assert item.error == f"MissingInputError: no {Stage.KINEMATICS.value} output"


def test_a_loader_error_becomes_the_clip_error() -> None:
    case = sc.parabola(clip_id="m")
    source = PerceivedSource(
        ScenarioDataset([case]), FakeLoader({"m": "depth: boom"}), InputLevel.PERCEIVED
    )
    (item,) = source.load([case.clip])
    assert (item.inputs, item.error, item.input_level) == (
        None,
        "depth: boom",
        InputLevel.PERCEIVED,
    )


def test_clip_inputs_hold_inputs_or_an_error() -> None:
    case = sc.parabola(clip_id="c")
    with pytest.raises(ValueError):
        ClipInputs(case.clip, None, None, InputLevel.ORACLE)
    with pytest.raises(ValueError):
        ClipInputs(case.clip, None, "error")


def test_refuse_heavy_names_the_extract_command() -> None:
    with pytest.raises(MissingInputError, match="physverify extract run --dataset d"):
        refuse_heavy("d")


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ, profile=None, overrides=())


def loader_for(
    settings: Settings, gt_stages: tuple[Stage, ...], dataset: str
) -> tuple[PipelineOutputLoader, LazyLock]:
    lock = LazyLock(partial(refuse_heavy, dataset))
    deps = PipelineDeps(
        cache=StageCache(settings.paths.cache_dir),
        lock=lock,
        paths=settings.paths,
        device="cpu",
        threads=1,
        error_profile=None,
        observer=None,
        rss=ProcRssProbe(),
    )
    plan = resolve_plan(settings, PLUGIN_KINDS["providers"](), dataset=dataset, gt_stages=gt_stages)
    return PipelineOutputLoader(plan, deps), lock


def test_the_ground_truth_pipeline_feeds_the_checks_without_the_lock(settings: Settings) -> None:
    dataset = SyntheticFixtureDataset()
    loader, lock = loader_for(settings, STAGE_ORDER, dataset.name)
    (item,) = PerceivedSource(dataset, loader, InputLevel.MIXED).load(dataset.clips())
    assert lock.acquisitions == 0
    assert item.error is None and item.inputs is not None
    assert item.inputs.contacts is not None
    assert item.inputs.scale is not None
    assert item.inputs.provenance[Stage.LIFT.value] == GROUND_TRUTH
    assert item.inputs.provenance[SCALE_SOURCE_KEY] == ScaleSource.GROUND_TRUTH.value


class TwoClips:
    """The fixture clip twice, to exercise the per-clip fallback of the loader."""

    def __init__(self) -> None:
        self._inner = SyntheticFixtureDataset()
        self.name = self._inner.name
        first = self._inner.clips()[0]
        self._clips = [first, replace(first, clip_id=f"{first.clip_id}_copy")]

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return self._clips

    def video(self, clip: ClipManifest) -> Any:
        return self._inner.video(clip)

    def ground_truth(self, clip: ClipManifest) -> Any:
        return self._inner.ground_truth(clip)


def test_missing_weights_fail_each_clip_not_the_run(settings: Settings) -> None:
    dataset = TwoClips()
    loader, lock = loader_for(settings, (), dataset.name)
    found = loader.load(dataset, dataset.clips())  # type: ignore[arg-type]
    assert set(found) == {clip.clip_id for clip in dataset.clips()}
    assert all(isinstance(value, str) and "MissingInputError" in value for value in found.values())
    assert lock.acquisitions == 0

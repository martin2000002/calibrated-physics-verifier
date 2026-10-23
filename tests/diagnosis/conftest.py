"""Diagnosis test fixtures (ADR 0007 §7): no network, calib/eval copies of the analytic clip,
DETECT/TRACK/DEPTH fakes with known injected errors, a fake check suite and builders.

The light stages (CAMERA .. CONTACTS) are A5's real providers.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import math
import socket
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel
from synthetic_clip import DATASET, SyntheticFixtureDataset, build_clip, manifest
from wiring import checks

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import CheckInputs
from physverify.config.loader import load_settings
from physverify.config.models import RuntimeSettings, Settings
from physverify.datasets._base import ArrayGroundTruthStore, DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.measurement_cases import SampledMetric
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.extraction import pipeline as extraction_pipeline
from physverify.extraction._base import StageContext, StageProvider, WeightSpec, provider_key
from physverify.extraction.cache import StageCache
from physverify.extraction.pipeline import PipelineDeps
from physverify.extraction.pipeline_resources import LazyLock
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.runtime.runs import RunContext
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap, Detections, MaskTrack
from physverify.schema.stages import Stage
from physverify.verification.inputs import perceived_inputs

FAKE = "diag_fake"
PERCEIVED_ID_PREFIX = "p_"
CALIB_CLIP = "fixture_fall_calib"
EVAL_SHIFT_PX = 5
EVAL_C = 2.0
"""Eval-only errors of ``eval_error_providers`` (the plain fakes use 2 px and 1.25)."""
FAKE_PLAN = tuple(
    f"extraction.plan.{s.value}={FAKE}" for s in (Stage.DETECT, Stage.TRACK, Stage.DEPTH)
)
"""``--set`` overrides that put the fakes in the plan (every other stage keeps A5's provider)."""

FIXED_NOW = dt.datetime(2026, 9, 24, 12, 0, tzinfo=dt.UTC)
RUN_ID = "20260924T120000Z-diagtest"

FAKE_CHECK_ID = "fake_distance"
FAKE_REL_SIGMA = 0.05

SettingsFactory = Callable[..., Settings]
DepsFactory = Callable[..., PipelineDeps]
Evaluator = Callable[[CheckInputs], Sequence[CheckResult]]


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("diagnosis tests must not touch the network")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)


@pytest.fixture(autouse=True)
def no_collect_between_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline's per-stage ``gc.collect`` frees real frames; on these tiny clips it only
    costs ~0.1 s per stage, most of the level-1 tests' wall time."""
    monkeypatch.setattr(extraction_pipeline, "gc", SimpleNamespace(collect=lambda: 0))


# --- Calib and eval copies of the analytic clip ----------------------------------------------


def calib_manifest() -> ClipManifest:
    return dataclasses.replace(
        manifest(), clip_id=CALIB_CLIP, split=Split.CALIB, video_path=f"{DATASET}/{CALIB_CLIP}"
    )


class SplitFixtureDataset:
    """The analytic clip twice: ``fixture_fall_calib`` (calib) and ``fixture_fall`` (eval)."""

    name: ClassVar[str] = DATASET

    def __init__(self, settings: Any = None) -> None:
        self._base = SyntheticFixtureDataset(settings)
        self._clips = (calib_manifest(), manifest())

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return list(self._clips)

    def video(self, clip: ClipManifest) -> Any:
        return self._base.video(clip)

    def ground_truth(self, clip: ClipManifest) -> ArrayGroundTruthStore | None:
        return self._base.ground_truth(clip)

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


# --- DETECT / TRACK / DEPTH fakes: fixture truth plus a known error --------------------------


@functools.cache
def fixture_truth() -> ArrayGroundTruthStore:
    return build_clip(RuntimeSettings()).ground_truth


class FakeParams(ProviderParams):
    fail_clip: str = ""


class FakeDetectParams(FakeParams):
    id_prefix: str = PERCEIVED_ID_PREFIX
    shift_px: float = 0.0


class FakeTrackParams(FakeParams):
    shift_px: int = 2


class FakeDepthParams(FakeParams):
    """Depth = ``c`` * z (``c`` * (1 + ``object_eps``) on ``object_id``); disparity a / that + b."""

    kind: DepthKind = DepthKind.RELATIVE_DEPTH
    c: float = 1.25
    object_id: str = ""
    object_eps: float = 0.0
    a: float = 2.0
    b: float = 0.0


class _Fake:
    name: ClassVar[str] = FAKE
    version: ClassVar[str] = "1"
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    stage: ClassVar[Stage]
    Params: ClassVar[type[BaseModel]] = FakeParams

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, FakeParams)

    def _start(self, ctx: StageContext) -> None:
        if ctx.clip.clip_id == self.params.fail_clip:
            raise ValueError(f"fake {self.stage.value} fails on {ctx.clip.clip_id}")


def _on_eval[T](clip: ClipManifest, default: T, on_eval: T | None) -> T:
    return on_eval if on_eval is not None and clip.split is Split.EVAL else default


class FakeDetect(_Fake):
    """Truth boxes shifted ``shift_px`` along +x, ids renamed ``<prefix><truth id>``."""

    stage: ClassVar[Stage] = Stage.DETECT
    requires: ClassVar[tuple[Stage, ...]] = ()
    Params: ClassVar[type[BaseModel]] = FakeDetectParams

    def run(self, ctx: StageContext) -> Detections:
        self._start(ctx)
        params = expect_params(self.params, FakeDetectParams)
        truth = fixture_truth().detections(ctx.sampling)
        boxes = truth.boxes_xyxy.copy()
        boxes[:, [0, 2]] += params.shift_px
        return Detections(
            sampling=ctx.sampling,
            frame_index=truth.frame_index,
            boxes_xyxy=boxes,
            scores=truth.scores.copy(),
            labels=truth.labels.copy(),
            object_ids=np.array([f"{params.id_prefix}{i}" for i in truth.object_ids.tolist()]),
        )


def shift_columns(masks: NDArray[np.bool_], shift: int) -> NDArray[np.bool_]:
    """Masks moved ``shift`` pixels along +x (image columns), zero-filled."""
    out = np.zeros_like(masks)
    if shift > 0:
        out[..., shift:] = masks[..., :-shift]
    elif shift < 0:
        out[..., :shift] = masks[..., -shift:]
    else:
        out[...] = masks
    return out


class FakeTrack(_Fake):
    """Truth masks of the detected objects, shifted ``shift_px`` (``eval_shift_px`` on eval)."""

    stage: ClassVar[Stage] = Stage.TRACK
    eval_shift_px: ClassVar[int | None] = None
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DETECT,)
    Params: ClassVar[type[BaseModel]] = FakeTrackParams

    def run(self, ctx: StageContext) -> MaskTrack:
        self._start(ctx)
        params = expect_params(self.params, FakeTrackParams)
        detections = ctx.require(Stage.DETECT, Detections)
        truth = fixture_truth().masks(ctx.sampling)
        truth_ids = truth.object_ids.tolist()
        rows = [
            truth_ids.index(str(i).removeprefix(PERCEIVED_ID_PREFIX))
            for i in detections.object_ids.tolist()
        ]
        shift = _on_eval(ctx.clip, params.shift_px, self.eval_shift_px)
        return MaskTrack(
            sampling=ctx.sampling,
            object_ids=detections.object_ids.copy(),
            labels=detections.labels.copy(),
            masks=shift_columns(truth.masks[:, rows], shift),
            scores=truth.scores[:, rows],
        )


class FakeDepth(_Fake):
    """``c`` * z on calib clips, ``eval_c`` * z on eval clips when set."""

    stage: ClassVar[Stage] = Stage.DEPTH
    eval_c: ClassVar[float | None] = None
    requires: ClassVar[tuple[Stage, ...]] = ()
    Params: ClassVar[type[BaseModel]] = FakeDepthParams

    def run(self, ctx: StageContext) -> DepthMap:
        self._start(ctx)
        params = expect_params(self.params, FakeDepthParams)
        truth = fixture_truth()
        metric = truth.depth(ctx.sampling)
        z = metric.depth.astype(np.float64)
        valid = metric.valid & (z > 0)
        factor = np.full(z.shape, _on_eval(ctx.clip, params.c, self.eval_c))
        if params.object_id:
            masks = truth.masks(ctx.sampling)
            index = masks.object_ids.tolist().index(params.object_id)
            factor[masks.masks[:, index]] *= 1.0 + params.object_eps
        scaled = np.where(valid, factor * z, 1.0)
        if params.kind is DepthKind.AFFINE_DISPARITY:
            scaled = params.a / scaled + params.b
        return DepthMap(
            sampling=ctx.sampling,
            depth=np.where(valid, scaled, 0.0).astype(np.float32),
            kind=params.kind,
            valid=valid,
        )


FAKES: tuple[type[StageProvider], ...] = (FakeDetect, FakeTrack, FakeDepth)


# --- Fake check suite -------------------------------------------------------------------------


def mean_distance(inputs: CheckInputs) -> list[CheckResult]:
    """Per object, mean distance to the origin of its frame (meters when a scale is present)."""
    states = inputs.states
    factor = inputs.scale.meters_per_unit if inputs.scale is not None else 1.0
    unit = "m" if inputs.scale is not None else states.unit.value
    span = (int(states.frame_index[0]), int(states.frame_index[-1]))
    results = []
    for index, object_id in enumerate(states.object_ids.tolist()):
        norms = np.linalg.norm(states.position[:, index], axis=-1)
        finite = norms[np.isfinite(norms)]
        magnitude = float(finite.mean()) * factor if finite.size else None
        results.append(
            CheckResult(
                check_id=FAKE_CHECK_ID,
                clip_id=inputs.clip.clip_id,
                subject=(str(object_id),),
                frame_span=span,
                magnitude=magnitude,
                sigma=math.nan if magnitude is None else FAKE_REL_SIGMA * magnitude,
                unit=unit,
                verdict=Verdict.UNSCORED if magnitude is not None else Verdict.ABSTAIN,
                input_level=inputs.input_level,
                abstain_reason=None if magnitude is not None else AbstainReason.MISSING_INPUT,
                provenance=dict(inputs.provenance),
            )
        )
    return results


def threshold_decision(result: CheckResult, tau: float) -> CheckResult:
    """PASS when ``magnitude <= tau``; abstentions stay as they are."""
    if result.verdict is Verdict.ABSTAIN or result.magnitude is None:
        return result
    verdict = Verdict.PASS if result.magnitude <= tau else Verdict.FAIL
    return dataclasses.replace(result, verdict=verdict, threshold=tau)


@dataclasses.dataclass
class FakeCheckSuite:
    """``CheckSuite`` stand-in: real ``CheckInputs``, an injectable evaluator, optional taus.

    ``thresholds=None`` means no verdict policy (``decide`` returns None); a check without a tau
    stays UNSCORED. ``truth`` maps clip ids to scripted oracle inputs.
    """

    evaluator: Evaluator = mean_distance
    thresholds: Mapping[str, float] | None = None
    provisional: bool = True
    truth: Mapping[str, CheckInputs] = dataclasses.field(default_factory=dict)
    evaluated: list[CheckInputs] = dataclasses.field(default_factory=list)

    def inputs(
        self,
        clip: ClipManifest,
        outputs: Mapping[Stage, StageOutput],
        names: Mapping[Stage, str],
        level: InputLevel,
    ) -> CheckInputs:
        return perceived_inputs(clip, outputs, names, level)

    def truth_inputs(self, dataset: DatasetSource, clip: ClipManifest) -> CheckInputs | None:
        return self.truth.get(clip.clip_id)

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]:
        self.evaluated.append(inputs)
        return tuple(self.evaluator(inputs))

    def decide(self, results: Sequence[CheckResult]) -> Sequence[CheckResult] | None:
        if self.thresholds is None:
            return None
        taus = self.thresholds
        return tuple(
            threshold_decision(r, taus[r.check_id]) if r.check_id in taus else r for r in results
        )


# --- Fixtures ---------------------------------------------------------------------------------


class FakeRss:
    def reset(self) -> None:
        return None

    def peak_mb(self) -> float:
        return 1.0


@pytest.fixture(scope="session")
def real_providers() -> Mapping[str, type[StageProvider]]:
    return dict(PLUGIN_KINDS["providers"]())


@pytest.fixture
def providers(real_providers: Mapping[str, type[StageProvider]]) -> dict[str, type[StageProvider]]:
    """The real registry plus ``<stage>/diag_fake`` for DETECT, TRACK and DEPTH."""
    return {**real_providers, **{provider_key(fake): fake for fake in FAKES}}


class EvalShiftedTrack(FakeTrack):
    eval_shift_px: ClassVar[int | None] = EVAL_SHIFT_PX


class EvalScaledDepth(FakeDepth):
    eval_c: ClassVar[float | None] = EVAL_C


@pytest.fixture
def eval_error_providers(
    providers: Mapping[str, type[StageProvider]],
) -> dict[str, type[StageProvider]]:
    """``providers`` whose TRACK and DEPTH fakes err more on eval clips only.

    The eval error is a class attribute, not a parameter, so the params hashes, and with them
    the profile entry conditions, stay those of the plain fakes. Cached outputs share the same
    keys: run these against their own cache directory.
    """
    swapped = (EvalShiftedTrack, EvalScaledDepth)
    return {**providers, **{provider_key(fake): fake for fake in swapped}}


@pytest.fixture
def make_settings(config_dir: Path, environ: dict[str, str]) -> SettingsFactory:
    """Settings from the repo configs; the fakes are in the plan unless ``fakes=False``."""

    def build(*overrides: str, fakes: bool = True, profile: str | None = None) -> Settings:
        plan = FAKE_PLAN if fakes else ()
        return load_settings(
            config_dir, environ=environ, profile=profile, overrides=(*plan, *overrides)
        )

    return build


@pytest.fixture
def make_deps() -> DepsFactory:
    def build(settings: Settings, profile: ErrorProfile | None = None) -> PipelineDeps:
        return PipelineDeps(
            cache=StageCache(settings.paths.cache_dir),
            lock=LazyLock(nullcontext),
            paths=settings.paths,
            device="cpu",
            threads=1,
            error_profile=profile,
            observer=None,
            rss=FakeRss(),
        )

    return build


@pytest.fixture
def split_dataset() -> SplitFixtureDataset:
    return SplitFixtureDataset()


@pytest.fixture
def make_chain(
    providers: Mapping[str, type[StageProvider]],
    make_deps: DepsFactory,
    split_dataset: SplitFixtureDataset,
) -> Callable[..., ChainRunner]:
    def build(settings: Settings, profile: ErrorProfile | None = None) -> ChainRunner:
        return ChainRunner(
            settings, providers, checks(), make_deps(settings, profile), split_dataset
        )

    return build


@pytest.fixture
def fixed_clock() -> Callable[[], dt.datetime]:
    return lambda: FIXED_NOW


@pytest.fixture
def run_context(tmp_path: Path) -> RunContext:
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    return RunContext(
        run_id=RUN_ID,
        run_dir=run_dir,
        argv=("diagnose",),
        device="cpu",
        profile="fast",
        config_hash="cfg0123456789ab",
        commit="c0ffee0",
        created_utc=FIXED_NOW.isoformat(timespec="seconds"),
    )


@pytest.fixture
def fake_suite() -> Callable[..., FakeCheckSuite]:
    return FakeCheckSuite


@pytest.fixture
def make_diagnosis_context(
    providers: Mapping[str, type[StageProvider]],
    make_deps: DepsFactory,
    fixed_clock: Callable[[], dt.datetime],
    run_context: RunContext,
) -> Callable[..., DiagnosisContext]:
    """Context over the split fixture; ``metrics`` defaults to none, ``suite`` to no A7."""

    def build(
        settings: Settings,
        *,
        suite: CheckSuite | None = None,
        metrics: Mapping[str, SampledMetric] | None = None,
        profile: ErrorProfile | None = None,
        stage_providers: Mapping[str, type[StageProvider]] | None = None,
    ) -> DiagnosisContext:
        registry: dict[str, type[DatasetSource]] = {DATASET: SplitFixtureDataset}
        return DiagnosisContext(
            settings=settings,
            diag=diagnosis_settings(settings),
            resolve_dataset=lambda name: registry[name](settings),
            providers=providers if stage_providers is None else stage_providers,
            checks=checks(),
            metrics=dict(metrics or {}),
            deps=make_deps(settings, profile),
            suite=suite,
            clock=fixed_clock,
            run=run_context,
        )

    return build

"""Check inputs from ground truth (oracle) or from cached pipeline outputs (ADR 0006 §3.7, D8)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn, Protocol

from physverify.checks._base import CheckInputs
from physverify.core.errors import MissingInputError
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.pipeline import DOMAIN_ERRORS, ExtractionPipeline, PipelineDeps
from physverify.extraction.plan import ResolvedPlan
from physverify.physics import tracks
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.contacts import ContactTable
from physverify.schema.errors import ScaleBindingError
from physverify.schema.outputs import StageOutput
from physverify.schema.scale import ScaleEstimate, ScaleSource, effective_scale
from physverify.schema.scene import SceneSpec
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import CoordFrame, ObjectStateTable, ground_truth_source_key

SCALE_SOURCE_KEY = "scale_source"
CHECK_FRAME = CoordFrame.CAMERA_OPENCV
INPUT_STAGES = (Stage.KINEMATICS, Stage.CONTACTS, Stage.SCALE)


@dataclass(frozen=True)
class RawInputs:
    clip: ClipManifest
    states: ObjectStateTable
    contacts: ContactTable | None
    scale: ScaleEstimate | None
    scene: SceneSpec | None
    level: InputLevel
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class ClipInputs:
    """``inputs`` or ``error``; ``level`` labels the abstentions of a clip without inputs."""

    clip: ClipManifest
    inputs: CheckInputs | None
    error: str | None
    level: InputLevel | None = None

    def __post_init__(self) -> None:
        if (self.inputs is None) == (self.error is None):
            raise ValueError("ClipInputs needs exactly one of inputs and error")
        if self.inputs is None and self.level is None:
            raise ValueError("a ClipInputs without inputs needs its level")

    @property
    def input_level(self) -> InputLevel:
        if self.inputs is not None:
            return self.inputs.input_level
        assert self.level is not None
        return self.level


@dataclass(frozen=True)
class StageOutputs:
    """Cached outputs of one clip plus the provider name of every stage of the plan."""

    outputs: Mapping[Stage, StageOutput]
    names: Mapping[Stage, str]


class InputSource(Protocol):
    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]: ...


class StageOutputLoader(Protocol):
    def load(
        self, dataset: DatasetSource, clips: Sequence[ClipManifest]
    ) -> Mapping[str, StageOutputs | str]: ...


class InputsError(ValueError):
    pass


def build_inputs(raw: RawInputs) -> CheckInputs:
    """Validate the time axis (D14.2) and bind the scale to the table it was fitted on."""
    states = raw.states
    try:
        tracks.row_step(states.frame_index)
        tracks.check_time_axis(states.t, states.frame_index, raw.clip.seconds_per_frame)
    except ValueError as exc:
        raise InputsError(f"time_axis: {exc}") from exc
    try:
        scale = effective_scale(states, raw.scale)
    except ScaleBindingError as exc:
        raise InputsError(f"scale_binding: {exc}") from exc
    return CheckInputs(
        clip=raw.clip,
        states=states,
        contacts=raw.contacts,
        scale=scale,
        input_level=raw.level,
        provenance=dict(raw.provenance),
        scene=raw.scene,
    )


def oracle_scale(clip_id: str) -> ScaleEstimate:
    return ScaleEstimate(1.0, 0.0, ScaleSource.GROUND_TRUTH, ground_truth_source_key(clip_id))


def provenance(names: Mapping[Stage, str], scale: ScaleEstimate | None) -> dict[str, str]:
    source = ScaleSource.NONE if scale is None else scale.source
    return {stage.value: name for stage, name in names.items()} | {SCALE_SOURCE_KEY: source.value}


def perceived_inputs(
    clip: ClipManifest,
    outputs: Mapping[Stage, StageOutput],
    names: Mapping[Stage, str],
    level: InputLevel,
) -> CheckInputs:
    """States from KINEMATICS, contacts from CONTACTS, scale from SCALE; raises
    ``MissingInputError`` without states and ``InputsError`` on an invalid table."""
    states = _output(outputs, Stage.KINEMATICS, ObjectStateTable)
    if states is None:
        raise MissingInputError(f"no {Stage.KINEMATICS.value} output")
    scale = _output(outputs, Stage.SCALE, ScaleEstimate)
    raw = RawInputs(
        clip=clip,
        states=states,
        contacts=_output(outputs, Stage.CONTACTS, ContactTable),
        scale=scale,
        scene=None,
        level=level,
        provenance=provenance(names, scale),
    )
    return build_inputs(raw)


def refuse_heavy(dataset: str) -> NoReturn:
    """Lock factory of ``verify``: a heavy stage missing from the cache fails its clip."""
    raise MissingInputError(
        f"a heavy stage is not cached; run `physverify extract run --dataset {dataset}` first"
    )


def error_message(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, InputsError) else f"{type(exc).__name__}: {exc}"


def _prepared(raw: RawInputs) -> ClipInputs:
    try:
        return ClipInputs(raw.clip, build_inputs(raw), None)
    except DOMAIN_ERRORS as exc:
        return ClipInputs(raw.clip, None, error_message(exc), raw.level)


class OracleSource:
    """Native ground truth of the dataset, camera frame, exact scale."""

    def __init__(self, dataset: DatasetSource) -> None:
        self._dataset = dataset

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield self._one(clip)

    def _one(self, clip: ClipManifest) -> ClipInputs:
        store = self._dataset.ground_truth(clip)
        if store is None:
            return ClipInputs(
                clip, None, "MissingInputError: dataset has no ground truth", InputLevel.ORACLE
            )
        scale = oracle_scale(clip.clip_id)
        names = dict.fromkeys(STAGE_ORDER, GROUND_TRUTH)
        try:
            raw = RawInputs(
                clip=clip,
                states=store.states(CHECK_FRAME),
                contacts=store.contacts(CHECK_FRAME),
                scale=scale,
                scene=store.scene(),
                level=InputLevel.ORACLE,
                provenance=provenance(names, scale),
            )
        except DOMAIN_ERRORS as exc:
            return ClipInputs(clip, None, error_message(exc), InputLevel.ORACLE)
        return _prepared(raw)


class PipelineOutputLoader:
    """Runs the stage plan (light stages only: the lock refuses) and loads the cached outputs."""

    def __init__(self, plan: ResolvedPlan, deps: PipelineDeps) -> None:
        self._plan = plan
        self._deps = deps

    def load(
        self, dataset: DatasetSource, clips: Sequence[ClipManifest]
    ) -> Mapping[str, StageOutputs | str]:
        try:
            return self._load(dataset, clips)
        except DOMAIN_ERRORS as exc:
            if len(clips) <= 1:
                return {clip.clip_id: error_message(exc) for clip in clips}
        return {clip.clip_id: self.load(dataset, [clip])[clip.clip_id] for clip in clips}

    def _load(
        self, dataset: DatasetSource, clips: Sequence[ClipManifest]
    ) -> dict[str, StageOutputs | str]:
        report = ExtractionPipeline(self._plan, self._deps).run(dataset, clips)
        failed = {f.clip_id: f"{f.stage.value}: {f.error}" for f in report.failures}
        loaded: dict[str, StageOutputs | str] = {}
        for clip in clips:
            error = failed.get(clip.clip_id)
            loaded[clip.clip_id] = (
                error if error is not None else self._outputs(dataset.name, clip, report.keys)
            )
        return loaded

    def _outputs(
        self, dataset: str, clip: ClipManifest, keys: Mapping[tuple[str, Stage], str]
    ) -> StageOutputs:
        outputs: dict[Stage, StageOutput] = {}
        for step in self._plan.stages:
            key = keys.get((clip.clip_id, step.stage))
            if step.stage not in INPUT_STAGES or key is None:
                continue
            output = self._deps.cache.load(dataset, clip.clip_id, step.stage, step.name, key)
            if output is not None:
                outputs[step.stage] = output
        return StageOutputs(outputs, dict(self._plan.names))


class PerceivedSource:
    """Pipeline outputs: states from KINEMATICS, contacts from CONTACTS, scale from SCALE."""

    def __init__(
        self, dataset: DatasetSource, loader: StageOutputLoader, level: InputLevel
    ) -> None:
        self._dataset = dataset
        self._loader = loader
        self._level = level

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        loaded = self._loader.load(self._dataset, clips)
        for clip in clips:
            found = loaded.get(clip.clip_id, "MissingInputError: no pipeline outputs")
            if isinstance(found, str):
                yield ClipInputs(clip, None, found, self._level)
            else:
                yield self._one(clip, found)

    def _one(self, clip: ClipManifest, found: StageOutputs) -> ClipInputs:
        try:
            inputs = perceived_inputs(clip, found.outputs, found.names, self._level)
        except DOMAIN_ERRORS as exc:
            return ClipInputs(clip, None, error_message(exc), self._level)
        return ClipInputs(clip, inputs, None)


def _output[O](outputs: Mapping[Stage, StageOutput], stage: Stage, kind: type[O]) -> O | None:
    value = outputs.get(stage)
    return value if isinstance(value, kind) else None

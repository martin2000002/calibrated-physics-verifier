"""Stage-by-stage extraction over many clips with a lazy heavy lock (ADR 0005, 3.4)."""

from __future__ import annotations

import gc
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

from physverify.config.models import PathsSettings
from physverify.core.errors import MissingInputError, ResourceBudgetError
from physverify.core.hashing import error_profile_hash, sampling_hash, upstream_keys
from physverify.core.log import get_logger
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import StageContext, StageEvent, StageObserver, StageProvider
from physverify.extraction.cache import (
    StageCache,
    plan_params_hashes,
    profile_keys,
    stage_key,
    step_params_hash,
)
from physverify.extraction.plan import ResolvedPlan, StagePlan
from physverify.extraction.weights import missing_weights
from physverify.schema.clip import ClipManifest, FrameSampling, resolve_sampling
from physverify.schema.error_profile import PARAMS_HASH_CONDITION, ErrorProfile
from physverify.schema.errors import SchemaError
from physverify.schema.outputs import StageOutput, check_stage_output
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import Stage, StageArtifactMeta
from physverify.schema.states import ObjectStateTable

PENDING_SOURCE_KEY: Final = "pending"
"""LIFT providers emit this ``source_key``; the pipeline replaces it with LIFT's cache key."""

DOMAIN_ERRORS: Final = (MissingInputError, ResourceBudgetError, SchemaError, ValueError)
LIGHT_DEVICE: Final = "cpu"
KIB_PER_MIB: Final = 1024.0

log = get_logger("extraction.pipeline")


class LazyLock:
    """Enters the heavy lock on the first ``ensure()``; a fully cached run never builds it."""

    def __init__(self, factory: Callable[[], AbstractContextManager[object]]) -> None:
        self._factory = factory
        self._held: AbstractContextManager[object] | None = None
        self._acquisitions = 0

    def ensure(self) -> None:
        if self._held is not None:
            return
        manager = self._factory()
        manager.__enter__()
        self._held = manager
        self._acquisitions += 1

    def release(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            held.__exit__(None, None, None)

    @property
    def acquisitions(self) -> int:
        return self._acquisitions

    @property
    def held(self) -> bool:
        return self._held is not None


class RssProbe(Protocol):
    def reset(self) -> None: ...

    def peak_mb(self) -> float: ...


class ProcRssProbe:
    """Per-stage peak RSS from ``VmHWM`` after resetting it through ``clear_refs``.

    ``exact`` turns False when the reset or the read fails; the fallback ``ru_maxrss`` is the
    process-lifetime peak, an upper bound of the stage peak."""

    def __init__(self, proc: Path = Path("/proc/self")) -> None:
        self._proc = proc
        self.exact = True

    def reset(self) -> None:
        try:
            (self._proc / "clear_refs").write_text("5", encoding="ascii")
        except OSError:
            self.exact = False

    def peak_mb(self) -> float:
        if self.exact:
            hwm_kb = self._vm_hwm_kb()
            if hwm_kb is not None:
                return hwm_kb / KIB_PER_MIB
            self.exact = False
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KIB_PER_MIB

    def _vm_hwm_kb(self) -> float | None:
        try:
            lines = (self._proc / "status").read_text(encoding="ascii").splitlines()
        except OSError:
            return None
        for line in lines:
            if line.startswith("VmHWM:"):
                return float(line.split()[1])
        return None


@dataclass(frozen=True)
class ClipFailure:
    clip_id: str
    stage: Stage
    error: str


@dataclass(frozen=True)
class ExtractionReport:
    events: tuple[StageEvent, ...]
    failures: tuple[ClipFailure, ...]
    keys: Mapping[tuple[str, Stage], str]


@dataclass(frozen=True)
class PipelineDeps:
    cache: StageCache
    lock: LazyLock
    paths: PathsSettings
    device: str
    threads: int
    error_profile: ErrorProfile | None
    observer: StageObserver | None
    rss: RssProbe
    clock: Callable[[], float] = time.perf_counter


@dataclass
class _Cell:
    clip: ClipManifest
    sampling: FrameSampling
    gt_version: str | None
    keys: dict[Stage, str]
    cached: dict[Stage, bool]


@dataclass
class _Progress:
    events: list[StageEvent] = field(default_factory=list)
    failures: list[ClipFailure] = field(default_factory=list)
    keys: dict[tuple[str, Stage], str] = field(default_factory=dict)

    def fail(self, clip: ClipManifest, stage: Stage, exc: BaseException) -> None:
        self.failures.append(ClipFailure(clip.clip_id, stage, f"{type(exc).__name__}: {exc}"))


class ExtractionPipeline:
    """Runs every stage in order over all live clips; one provider instance alive at a time."""

    def __init__(self, plan: ResolvedPlan, deps: PipelineDeps) -> None:
        self._plan = plan
        self._deps = deps
        self._names = dict(plan.names)
        self._params = plan_params_hashes(plan.stages)

    def run(self, dataset: DatasetSource, clips: Sequence[ClipManifest]) -> ExtractionReport:
        self._warn_rejected_entries()
        progress = _Progress()
        cells = [cell for clip in clips if (cell := self._prepare(dataset, clip, progress))]
        self._check_weights(cells)
        try:
            for index, step in enumerate(self._plan.stages):
                cells = self._run_stage(step, dataset, cells, progress)
                if not self._heavy_pending(self._plan.stages[index + 1 :], cells):
                    self._deps.lock.release()
        finally:
            self._deps.lock.release()
        return ExtractionReport(
            tuple(progress.events), tuple(progress.failures), dict(progress.keys)
        )

    def _warn_rejected_entries(self) -> None:
        """Loud, once per entry: a params mismatch turns that sigma unknown (NaN)."""
        profile = self._deps.error_profile
        if profile is None:
            return
        keys = {
            key for step in self._plan.stages for key in profile_keys(step.provider, self._names)
        }
        for entry in profile.rejected(keys, self._params):
            log.warning(
                "error profile entry %s/%s %s was measured with params_hash %s but the plan uses "
                "%s; it is ignored and its sigma is unknown (re-run `diagnose stages "
                "--write-profile` for this profile)",
                entry.stage.value,
                entry.provider,
                entry.quantity,
                entry.conditions.get(PARAMS_HASH_CONDITION),
                self._params[entry.stage],
            )

    def _prepare(
        self, dataset: DatasetSource, clip: ClipManifest, progress: _Progress
    ) -> _Cell | None:
        gt_stages = self._plan.ground_truth_stages()
        missing = [stage for stage in gt_stages if stage not in clip.gt_stages]
        if missing:
            error = MissingInputError(f"no ground truth for {[s.value for s in missing]}")
            progress.fail(clip, missing[0], error)
            return None
        try:
            cell = _Cell(clip, resolve_sampling(clip, self._plan.sampling), None, {}, {})
        except DOMAIN_ERRORS as exc:
            progress.fail(clip, self._plan.stages[0].stage, exc)
            return None
        if gt_stages:
            store = dataset.ground_truth(clip)
            if store is None:
                progress.fail(clip, gt_stages[0], MissingInputError("dataset has no ground truth"))
                return None
            cell.gt_version = store.version
        for step in self._plan.stages:
            key = self._key(step, cell)
            cell.keys[step.stage] = key
            cell.cached[step.stage] = self._deps.cache.contains(
                dataset.name, clip.clip_id, step.stage, step.name, key
            )
        return cell

    def _key(self, step: StagePlan, cell: _Cell) -> str:
        return stage_key(
            step,
            clip_id=cell.clip.clip_id,
            object_queries=cell.clip.object_queries,
            sampling=cell.sampling,
            upstream=cell.keys,
            plan=self._names,
            plan_params=self._params,
            gt_version=cell.gt_version if step.provider.is_ground_truth else None,
            error_profile=self._deps.error_profile,
            device=self._deps.device,
        )

    def _check_weights(self, cells: Sequence[_Cell]) -> None:
        """D9: fail before taking the lock when a heavy provider that must run lacks weights."""
        needed = [
            step.provider
            for step in self._plan.stages
            if step.provider.heavy and any(not cell.cached[step.stage] for cell in cells)
        ]
        missing = missing_weights(needed, self._deps.paths)
        if missing:
            refs = ", ".join(f"{spec.kind.value}:{spec.ref}@{spec.revision}" for spec in missing)
            raise MissingInputError(
                f"missing weights {refs}; run `physverify extract fetch-weights` first"
            )

    @staticmethod
    def _heavy_pending(steps: Sequence[StagePlan], cells: Sequence[_Cell]) -> bool:
        return any(
            step.provider.heavy and not cell.cached[step.stage] for step in steps for cell in cells
        )

    def _run_stage(
        self, step: StagePlan, dataset: DatasetSource, cells: list[_Cell], progress: _Progress
    ) -> list[_Cell]:
        instance: StageProvider | None = None
        alive: list[_Cell] = []
        for cell in cells:
            event = StageEvent(
                cell.clip.clip_id,
                step.stage,
                step.name,
                cell.cached[step.stage],
                self._device(step),
            )
            self._notify_start(event)
            try:
                if not event.cached:
                    instance = instance or self._instantiate(step)
                    event = self._compute(step, instance, dataset, cell, event)
            except DOMAIN_ERRORS as exc:
                progress.fail(cell.clip, step.stage, exc)
                continue
            progress.events.append(event)
            progress.keys[(cell.clip.clip_id, step.stage)] = cell.keys[step.stage]
            self._notify_end(event)
            alive.append(cell)
        del instance
        gc.collect()
        return alive

    def _instantiate(self, step: StagePlan) -> StageProvider:
        if step.provider.heavy:
            self._deps.lock.ensure()
        return step.provider(step.params)

    def _device(self, step: StagePlan) -> str:
        return self._deps.device if step.provider.heavy else LIGHT_DEVICE

    def _compute(
        self,
        step: StagePlan,
        instance: StageProvider,
        dataset: DatasetSource,
        cell: _Cell,
        event: StageEvent,
    ) -> StageEvent:
        upstream = {stage: self._load(dataset, cell, stage) for stage in step.provider.requires}
        ctx = StageContext(
            clip=cell.clip,
            video=dataset.video(cell.clip),
            sampling=cell.sampling,
            upstream=upstream,
            plan=self._names,
            plan_params_hash=self._params,
            gt=dataset.ground_truth(cell.clip) if step.provider.is_ground_truth else None,
            error_profile=self._deps.error_profile,
            device=self._deps.device,
            threads=self._deps.threads,
            paths=self._deps.paths,
        )
        self._deps.rss.reset()
        started = self._deps.clock()
        output = instance.run(ctx)
        wall_s = self._deps.clock() - started
        peak_mb = self._deps.rss.peak_mb()
        output = self._bind(step.stage, output, upstream, cell)
        meta = self._meta(step, cell, wall_s, peak_mb)
        self._deps.cache.store(
            dataset.name, cell.clip.clip_id, step.stage, output, meta, step.params
        )
        return replace(event, wall_s=wall_s, peak_rss_mb=peak_mb)

    def _load(self, dataset: DatasetSource, cell: _Cell, stage: Stage) -> StageOutput:
        name = self._names[stage]
        output = self._deps.cache.load(
            dataset.name, cell.clip.clip_id, stage, name, cell.keys[stage]
        )
        if output is None:
            raise MissingInputError(f"{cell.clip.clip_id}: cached {stage.value}/{name} is missing")
        return output

    @staticmethod
    def _bind(
        stage: Stage, output: StageOutput, upstream: Mapping[Stage, StageOutput], cell: _Cell
    ) -> StageOutput:
        """D7: LIFT carries its own key; SCALE, KINEMATICS and CONTACTS must point back to it."""
        check_stage_output(stage, output)
        lift_key = cell.keys[Stage.LIFT]
        if stage is Stage.LIFT and isinstance(output, ObjectStateTable):
            return replace(output, source_key=lift_key)
        if isinstance(output, ScaleEstimate) and output.lift_key != lift_key:
            raise SchemaError(f"scale.lift_key {output.lift_key!r} != lift key {lift_key!r}")
        if stage in (Stage.KINEMATICS, Stage.CONTACTS):
            _check_binding(stage, output, upstream, lift_key)
        return output

    def _meta(
        self, step: StagePlan, cell: _Cell, wall_s: float, peak_mb: float
    ) -> StageArtifactMeta:
        provider = step.provider
        return StageArtifactMeta(
            stage=step.stage,
            provider=provider.name,
            provider_version=provider.version,
            params_hash=step_params_hash(step),
            sampling_hash=sampling_hash(cell.sampling),
            upstream=upstream_keys(provider.requires, cell.keys),
            error_profile_hash=error_profile_hash(
                self._deps.error_profile, profile_keys(provider, self._names), self._params
            ),
            gt_version=cell.gt_version if provider.is_ground_truth else None,
            device=self._deps.device if provider.heavy else None,
            cache_key=cell.keys[step.stage],
            clip_id=cell.clip.clip_id,
            is_ground_truth=provider.is_ground_truth,
            wall_s=wall_s,
            peak_rss_mb=peak_mb,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def _notify_start(self, event: StageEvent) -> None:
        if self._deps.observer is not None:
            self._deps.observer.on_stage_start(event)

    def _notify_end(self, event: StageEvent) -> None:
        if self._deps.observer is not None:
            self._deps.observer.on_stage_end(event)


def _check_binding(
    stage: Stage, output: StageOutput, upstream: Mapping[Stage, StageOutput], lift_key: str
) -> None:
    source_key = getattr(output, "source_key", None)
    if source_key != lift_key:
        raise SchemaError(f"{stage.value}.source_key {source_key!r} != lift key {lift_key!r}")
    reference = upstream.get(Stage.KINEMATICS) or upstream.get(Stage.LIFT)
    if not isinstance(reference, ObjectStateTable):
        return
    frame, unit = getattr(output, "frame", None), getattr(output, "unit", None)
    if (frame, unit) != (reference.frame, reference.unit):
        raise SchemaError(
            f"{stage.value} is {frame}/{unit} but its states are {reference.frame}/{reference.unit}"
        )

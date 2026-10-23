"""Level 1: each stage measured alone, pred from V_k against truth from V_{k+1} (ADR 0007 D6).

The calib pass runs without an error profile and builds the new one; the eval pass reads the
merged profile in memory, written or not (D3). Eval clips never enter the profile."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner, VariantOutputs, warm_variants
from physverify.diagnosis.chain_variants import (
    Variant,
    ladder,
    oracle_variant,
    scale_variants,
    truth_covers,
)
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.error_profile_writer import (
    MetricSamples,
    ProfileBuild,
    ProfileMeta,
    ProviderRef,
    StageSamples,
    build_profile,
    merge_profiles,
)
from physverify.diagnosis.error_profile_writer import write_profile as save_profile
from physverify.diagnosis.measurement_cases import (
    MetricCase,
    MetricOutcome,
    SampledMetric,
    pool,
    summarize,
)
from physverify.diagnosis.report import (
    ProfileEntryRow,
    StageErrorReport,
    StageMetricRow,
    build_provenance,
    dataset_versions,
    z_coverage,
)
from physverify.diagnosis.settings import DiagnosisSettings, MetricSettings
from physverify.extraction.cache import step_params_hash
from physverify.extraction.plan import ResolvedPlan
from physverify.physics.uncertainty import entry_rms
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import STAGE_ORDER, Stage

Z_SUFFIX = "_z"


@dataclass(frozen=True)
class StagePair:
    stage: Stage
    pred: Variant
    truth: Variant


def stage_pairs(diag: DiagnosisSettings) -> tuple[StagePair, ...]:
    """(V_k, V_{k+1}) per stage, plus V5 with each SCALE provider against V6."""
    variants = ladder()
    pairs = [StagePair(stage, variants[k], variants[k + 1]) for k, stage in enumerate(STAGE_ORDER)]
    scale_truth = variants[STAGE_ORDER.index(Stage.SCALE) + 1]
    pairs += [StagePair(Stage.SCALE, v, scale_truth) for v in scale_variants(diag.scale.providers)]
    for pair in pairs:
        if not truth_covers(pair.truth, pair.stage):
            raise ValueError(f"{pair.truth.name} does not cover {pair.stage.value} (D6)")
    return tuple(pairs)


def pair_variants(pairs: Sequence[StagePair]) -> tuple[Variant, ...]:
    """Every variant the pairs read, plus the oracle, once each by name."""
    unique: dict[str, Variant] = {}
    for variant in (*(v for p in pairs for v in (p.pred, p.truth)), oracle_variant()):
        unique.setdefault(variant.name, variant)
    return tuple(unique.values())


def provider_ref(plan: ResolvedPlan, stage: Stage) -> ProviderRef:
    step = plan.step(stage)
    return ProviderRef(
        stage=stage,
        name=step.provider.name,
        version=step.provider.version,
        params_hash=step_params_hash(step),
    )


@dataclass(frozen=True)
class MetricJob:
    key: str
    metric: SampledMetric
    pair: StagePair
    provider: ProviderRef


def metric_jobs(
    runner: ChainRunner, metrics: Mapping[str, SampledMetric], pairs: Sequence[StagePair]
) -> tuple[MetricJob, ...]:
    """One job per (metric, pair). A pair whose pred resolves to a provider already measured
    against the same truth (V5 and V5/scale=<default>) is measured once."""
    seen: set[tuple[Stage, str, str]] = set()
    jobs: list[MetricJob] = []
    for pair in pairs:
        ref = provider_ref(runner.plan(pair.pred), pair.stage)
        identity = (pair.stage, ref.name, pair.truth.name)
        if identity in seen:
            continue
        seen.add(identity)
        jobs += [MetricJob(k, m, pair, ref) for k, m in metrics.items() if m.stage is pair.stage]
    return tuple(jobs)


def last_uses(jobs: Sequence[MetricJob]) -> dict[str, int]:
    last: dict[str, int] = {}
    for index, job in enumerate(jobs):
        last[job.pair.pred.name] = index
        last[job.pair.truth.name] = index
    return last


class ClipOutputs:
    """Variant outputs of one clip, loaded on first use and released after their last job."""

    def __init__(self, runner: ChainRunner, clip: ClipManifest, last: Mapping[str, int]) -> None:
        self._runner = runner
        self._clip = clip
        self._last = last
        self._loaded: dict[str, VariantOutputs | None] = {}

    def get(self, variant: Variant) -> VariantOutputs | None:
        if variant.name not in self._loaded:
            self._loaded[variant.name] = self._runner.outputs(self._clip, variant)
        return self._loaded[variant.name]

    def release(self, index: int) -> None:
        for name in [n for n in self._loaded if self._last.get(n, -1) == index]:
            del self._loaded[name]


def truth_context(
    truth: VariantOutputs, oracle_outputs: VariantOutputs | None
) -> Mapping[Stage, StageOutput]:
    """Each stage from the truth variant when it is ground truth there, else from the oracle."""
    context: dict[Stage, StageOutput] = {}
    for stage in STAGE_ORDER:
        if stage in truth.variant.gt_stages and stage in truth.outputs:
            context[stage] = truth.outputs[stage]
        elif oracle_outputs is not None and stage in oracle_outputs.outputs:
            context[stage] = oracle_outputs.outputs[stage]
    return MappingProxyType(context)


def measure_clip(
    runner: ChainRunner, clip: ClipManifest, jobs: Sequence[MetricJob], options: MetricSettings
) -> Iterator[tuple[int, MetricOutcome]]:
    outputs = ClipOutputs(runner, clip, last_uses(jobs))
    reference = runner.outputs(clip, oracle_variant())
    gt = runner.dataset.ground_truth(clip)
    for index, job in enumerate(jobs):
        stage = job.pair.stage
        pred, truth = outputs.get(job.pair.pred), outputs.get(job.pair.truth)
        if pred is not None and truth is not None and stage in pred.outputs:
            case = MetricCase(
                clip=clip,
                pred=pred.outputs[stage],
                truth=truth.outputs[stage],
                context=truth_context(truth, reference),
                gt=gt,
                options=options,
                plan=pred.plan,
            )
            yield index, job.metric.samples(case)
        outputs.release(index)


def measure(
    runner: ChainRunner,
    clips: Sequence[ClipManifest],
    metrics: Mapping[str, SampledMetric],
    pairs: Sequence[StagePair],
    options: MetricSettings,
) -> StageSamples:
    """Per-clip outcomes of every metric on every pair; clips are processed one at a time."""
    jobs = metric_jobs(runner, metrics, pairs)
    outcomes: list[dict[str, MetricOutcome]] = [{} for _ in jobs]
    for clip in clips:
        for index, outcome in measure_clip(runner, clip, jobs, options):
            outcomes[index][clip.clip_id] = outcome
    return tuple(
        MetricSamples(
            metric=job.key,
            provider=job.provider,
            pred_variant=job.pair.pred.name,
            truth_variant=job.pair.truth.name,
            outcomes=MappingProxyType(found),
        )
        for job, found in zip(jobs, outcomes, strict=True)
    )


def clips_of(clips: Sequence[ClipManifest], split: Split) -> tuple[ClipManifest, ...]:
    return tuple(clip for clip in clips if clip.split is split)


def profile_meta(ctx: DiagnosisContext, dataset: DatasetSource, version: str) -> ProfileMeta:
    return ProfileMeta(
        created_utc=ctx.run.created_utc,
        run_id=ctx.run.run_id,
        commit=ctx.run.commit,
        split=ctx.diag.profile_split.value,
        subset=ctx.diag.subset,
        dataset=dataset.name,
        dataset_version=version,
        profile=ctx.settings.profile,
    )


def metric_rows(
    samples: StageSamples, split: Split, z_levels: Sequence[float]
) -> tuple[StageMetricRow, ...]:
    rows = []
    for measured in samples:
        pooled = pool(measured.outcomes.values())
        coverage = {
            key: z_coverage(values, z_levels)
            for key, values in pooled.samples.items()
            if key.endswith(Z_SUFFIX)
        }
        rows.append(
            StageMetricRow(
                stage=measured.provider.stage,
                metric=measured.metric,
                provider=measured.provider.name,
                pred_variant=measured.pred_variant,
                truth_variant=measured.truth_variant,
                split=split,
                n_clips=len(measured.outcomes),
                values=summarize(pooled),
                coverage=coverage,
            )
        )
    return tuple(rows)


def profile_rows(build: ProfileBuild, written: bool) -> tuple[ProfileEntryRow, ...]:
    measured = [
        ProfileEntryRow(
            stage=e.stage,
            provider=e.provider,
            quantity=e.quantity,
            bias=e.bias,
            sigma=e.sigma,
            rms=entry_rms(e),
            n=e.n,
            written=written,
        )
        for e in build.profile.entries
    ]
    skipped = [
        ProfileEntryRow(
            stage=s.stage,
            provider=s.provider,
            quantity=s.quantity,
            bias=s.bias,
            sigma=s.sigma,
            rms=math.hypot(s.bias, s.sigma),
            n=s.n,
            written=False,
        )
        for s in build.skipped
    ]
    return (*measured, *skipped)


def run_stage_errors(
    ctx: DiagnosisContext,
    clips: Sequence[ClipManifest],
    *,
    write_profile: bool,
    replace_mismatched: bool = False,
) -> StageErrorReport:
    diag = ctx.diag
    dataset = ctx.dataset()
    pairs = stage_pairs(diag)
    variants = pair_variants(pairs)
    calib, report = clips_of(clips, diag.profile_split), clips_of(clips, diag.report_split)
    runner = ctx.chain(dataset).with_error_profile(None)
    failures = warm_variants(runner, calib, variants, skip_failed=False)
    calib_samples = measure(runner, calib, ctx.metrics, pairs, diag.metrics)
    version = dataset_versions(dataset, calib).get(dataset.name, "")
    build = build_profile(calib_samples, profile_meta(ctx, dataset, version), diag.min_samples)
    merged = merge_profiles(
        ctx.deps.error_profile, build.profile, replace_mismatched=replace_mismatched
    )
    path = ctx.settings.paths.error_profile
    if write_profile:
        save_profile(path, merged)
    eval_runner = runner.with_error_profile(merged)
    failures += warm_variants(eval_runner, report, variants, skip_failed=False)
    eval_samples = measure(eval_runner, report, ctx.metrics, pairs, diag.metrics)
    z_levels = diag.end_to_end.z_levels
    return StageErrorReport(
        provenance=build_provenance(
            ctx.run,
            dataset=dataset.name,
            dataset_versions=dataset_versions(dataset, (*calib, *report)),
            subset=diag.subset,
            error_profile=merged,
            clips=[clip.clip_id for clip in (*calib, *report)],
        ),
        profile_split=diag.profile_split,
        report_split=diag.report_split,
        min_samples=diag.min_samples,
        profile_written=write_profile,
        profile_path=str(path) if write_profile else None,
        rows=(
            *metric_rows(calib_samples, diag.profile_split, z_levels),
            *metric_rows(eval_samples, diag.report_split, z_levels),
        ),
        profile=profile_rows(build, write_profile),
        failures=failures,
    )

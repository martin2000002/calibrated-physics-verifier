"""Fine-tuning wiring shared by the finetune and h3 commands: CUDA, generator, scoring and
the pilot stop rule.

The stop-rule verdict is always recomputed from the current ``pairs.jsonl``, ``report.json``,
base scores and pilot review; a ``stop_rule.json`` left from other pairs never counts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

from physverify.app.context import AppContext
from physverify.app.services import dataset_for, pipeline_deps, plan_for
from physverify.config.models import Settings
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import UsageError
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.finetune.candidate_scoring import ScoringDeps
from physverify.finetune.candidates import CandidateGenerator
from physverify.finetune.layout import CampaignStore
from physverify.finetune.scorer_scope import family_targets, scorer_exclusion
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scorers._calibration_status import CalibrationStatusSource
from physverify.finetune.settings import CandidateSettings, FinetuneSettings, load_finetune
from physverify.finetune.stop_rule import (
    StopVerdict,
    base_abstention,
    evaluate_stop,
    pairs_on_disk,
    stored_report,
    stored_review,
    verdict_lines,
    write_verdict,
)
from physverify.generation.wan21 import load_wan_pipeline, snapshot_dir, snapshot_problem
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import current_commit
from physverify.schema.checks import InputLevel
from physverify.schema.io import json_ready
from physverify.timing.environment import TorchCudaProbe
from physverify.timing.settings import load_timing
from physverify.verification.check_set import CHECKS_SECTION, ChecksSettings, build_checks
from physverify.verification.inputs import PerceivedSource, PipelineOutputLoader
from physverify.verification.runner import (
    VERIFICATION_SECTION,
    VerificationRunner,
    VerificationSettings,
    build_policy,
    load_thresholds,
)

NESTED_LOCK: Final = "held by finetune"
STOP_RULE_DEVIATION = "--stop-rule-deviation"


def require_cuda(settings: Settings) -> None:
    device = settings.runtime.device
    if not device.startswith("cuda"):
        raise UsageError(f"this subcommand needs --device cuda (device is {device!r})")
    if not TorchCudaProbe().available():
        raise UsageError("this subcommand needs a CUDA device and none is available")


def candidate_generator(
    context: AppContext,
    store: CampaignStore,
    adapter: Path | None,
    *,
    candidates: CandidateSettings | None = None,
) -> CandidateGenerator:
    settings = context.settings
    gen = load_timing(settings).generation
    problem = snapshot_problem(gen, settings.paths)
    if problem is not None:
        raise UsageError(f"Wan2.1 weights: {problem}")
    if adapter is not None and not adapter.exists():
        raise UsageError(f"adapter {adapter} does not exist")
    device = settings.runtime.device
    snapshot = snapshot_dir(gen, settings.paths)
    return CandidateGenerator(
        store,
        gen,
        load_finetune(settings).candidates if candidates is None else candidates,
        pipe_factory=lambda: load_wan_pipeline(snapshot, gen, device),
        device=device,
        adapter=adapter,
        commit=current_commit(settings.repo_root),
    )


def scoring_deps(context: AppContext) -> ScoringDeps:
    """Perceived inputs from the extraction pipeline over ``local_video``, checks from A7."""
    settings = context.settings
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    runner = VerificationRunner(
        build_checks(context.registry("checks"), settings.section(CHECKS_SECTION, ChecksSettings)),
        build_policy(context.registry("policies"), verification),
        load_thresholds(settings.paths.thresholds, verification.thresholds),
    )
    dataset = dataset_for(context, LOCAL_VIDEO)
    plan = plan_for(context, dataset=LOCAL_VIDEO)
    loader = PipelineOutputLoader(plan, pipeline_deps(context, None, lambda: NullLock(NESTED_LOCK)))
    return ScoringDeps(
        clips=dataset.clips,
        source=PerceivedSource(dataset, loader, InputLevel.PERCEIVED),
        runner=runner,
    )


def write_calibration_status(store: CampaignStore, model_tag: str, scorer: AggregateScorer) -> None:
    """Per check, the calibration state the scores were made under (calibrated scorer only)."""
    if not isinstance(scorer, CalibrationStatusSource):
        return
    status = scorer.calibration_status()
    if status is not None:
        text = json.dumps(json_ready(dict(status)), indent=2, sort_keys=True)
        write_text_atomic(store.calibration_status_path(model_tag, scorer.name), text)


def current_verdict(
    context: AppContext, cfg: FinetuneSettings, store: CampaignStore, scorer: AggregateScorer
) -> StopVerdict:
    name = scorer.name
    return evaluate_stop(
        stored_report(store, name),
        pairs_on_disk(store, name),
        base_abstention(store, scorer),
        stored_review(store, name),
        cfg.stop_rule,
        family_targets(cfg, scorer_exclusion(context.settings, cfg, context.registry("checks"))),
        (scorer.version, scorer.thresholds()),
    )


def require_stop_rule(
    context: AppContext,
    cfg: FinetuneSettings,
    store: CampaignStore,
    scorer: AggregateScorer,
    deviation: str | None,
    *,
    err: Callable[[str], None],
) -> StopVerdict:
    """Exit 2 when the rule fires, is enforced and no deviation is declared; the verdict, with
    the deviation that let it through, is written to ``stop_rule.json`` (and printed when it
    fired) either way."""
    verdict = current_verdict(context, cfg, store, scorer)
    if not verdict.passed and deviation is not None:
        verdict = verdict.model_copy(update={"deviation": deviation})
    write_verdict(store, scorer.name, verdict)
    if not verdict.passed:
        for line in verdict_lines(verdict):
            err(line)
    if not verdict.passed and verdict.enforced and deviation is None:
        raise UsageError(
            f"pilot stop rule fired: {'; '.join(verdict.reasons)}; H3 is not evaluable with "
            f"this verifier ({STOP_RULE_DEVIATION} TEXT, or finetune.stop_rule.enforce = false, "
            "to go on)"
        )
    return verdict

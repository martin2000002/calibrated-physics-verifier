"""Finetune dependencies shared by the A11 and H3 subcommands (not a plugin module)."""

from __future__ import annotations

import json
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
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scorers._calibration_status import CalibrationStatusSource
from physverify.finetune.settings import CandidateSettings, generator_settings, load_finetune
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import current_commit
from physverify.schema.checks import InputLevel
from physverify.schema.io import json_ready
from physverify.timing.envinfo import TorchCudaProbe
from physverify.timing.generation import load_wan_pipeline, snapshot_dir, snapshot_problem
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
    gen = generator_settings(settings)
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

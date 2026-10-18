"""Finetune dependencies shared by the A11 and H3 subcommands (not a plugin module)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from physverify.cli.commands.extract import pipeline_deps, resolve_dataset
from physverify.cli.context import AppContext
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.extraction.plan import resolve_plan
from physverify.finetune.generate import CandidateGenerator
from physverify.finetune.scoring import ScoringDeps
from physverify.finetune.settings import CandidateSettings, generator_settings, load_finetune
from physverify.finetune.store import CampaignStore
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import current_commit
from physverify.schema.checks import InputLevel
from physverify.timing.envinfo import TorchCudaProbe
from physverify.timing.generation import load_wan_pipeline, snapshot_dir, snapshot_problem
from physverify.verification.inputs import PerceivedSource, PipelineOutputLoader
from physverify.verification.runner import (
    CHECKS_SECTION,
    VERIFICATION_SECTION,
    ChecksSettings,
    VerificationRunner,
    VerificationSettings,
    build_checks,
    build_policy,
    load_thresholds,
)

DATASET: Final = "local_video"
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
    dataset = resolve_dataset(context, DATASET)
    plan = resolve_plan(settings, context.registry("providers"), dataset=DATASET, gt_stages=())
    loader = PipelineOutputLoader(plan, pipeline_deps(context, None, lambda: NullLock(NESTED_LOCK)))
    return ScoringDeps(
        clips=dataset.clips,
        source=PerceivedSource(dataset, loader, InputLevel.PERCEIVED),
        runner=runner,
    )

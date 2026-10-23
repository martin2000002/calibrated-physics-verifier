"""Fine-tuning wiring shared by the finetune and h3 commands: CUDA, generator, scoring and
the pilot stop rule.

The stop-rule verdict is always recomputed from the current ``pairs.jsonl``, ``report.json``,
base scores and pilot review; a ``stop_rule.json`` left from other pairs never counts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from physverify.app.context import AppContext
from physverify.app.services import dataset_for, pipeline_deps, plan_for, scorer_for
from physverify.checks._base import CHECKS
from physverify.config.models import Settings
from physverify.config.plugin_context import registry_of
from physverify.core.atomic import write_text_atomic
from physverify.core.choices import choice
from physverify.core.digests import file_sha256
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.datasets.local_video import NAME as LOCAL_VIDEO
from physverify.finetune.candidate_scoring import ScoringDeps, score_candidates
from physverify.finetune.candidates import CandidateGenerator, generate_candidates
from physverify.finetune.layout import (
    PAIRS_FILE,
    REPORT_FILE,
    VAL_PAIRS_FILE,
    VAL_REPORT_FILE,
    CampaignStore,
)
from physverify.finetune.pairs_report import check_current, pairs_of, report_lines
from physverify.finetune.pilot import fit_and_val, pilot_prompts
from physverify.finetune.prompts import build_prompt_set, campaign_prompts, sizing_labels
from physverify.finetune.records import BASE_TAG as BASE_TAG
from physverify.finetune.records import (
    CandidateScore,
    PromptRecord,
    write_record,
    write_records,
)
from physverify.finetune.scorer_scope import family_targets, scorer_exclusion
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scorers._calibration_status import CalibrationStatusSource
from physverify.finetune.seeds import parse_shard
from physverify.finetune.settings import (
    SPLITS,
    CandidateSettings,
    FinetuneSettings,
    PromptSplit,
    load_finetune,
)
from physverify.finetune.status import status_lines
from physverify.finetune.stop_rule import STOP_RULE_DEVIATION as STOP_RULE_DEVIATION
from physverify.finetune.stop_rule import (
    StopVerdict,
    base_abstention,
    evaluate_stop,
    pairs_on_disk,
    review_pilot,
    stored_report,
    stored_review,
    verdict_lines,
    write_verdict,
)
from physverify.generation.wan21 import load_wan_pipeline, snapshot_dir, snapshot_problem
from physverify.generator_comparison.arms import EVAL_SPLIT
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import current_commit
from physverify.schema.checks import InputLevel
from physverify.schema.io import json_ready
from physverify.timing.environment import TorchCudaProbe
from physverify.timing.finetune_budget import BudgetRequest as BudgetRequest
from physverify.timing.finetune_budget import budget_report
from physverify.timing.settings import load_timing
from physverify.verification._base import POLICIES
from physverify.verification.inputs import PerceivedSource, PipelineOutputLoader
from physverify.verification.runner import configured_suite

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
    """Perceived inputs from the extraction pipeline over ``local_video``; checks from the suite."""
    suite = configured_suite(
        context.settings, registry_of(context, CHECKS), registry_of(context, POLICIES)
    )
    dataset = dataset_for(context, LOCAL_VIDEO)
    plan = plan_for(context, dataset=LOCAL_VIDEO)
    loader = PipelineOutputLoader(plan, pipeline_deps(context, None, lambda: NullLock(NESTED_LOCK)))
    return ScoringDeps(
        clips=dataset.clips,
        source=PerceivedSource(dataset, loader, InputLevel.PERCEIVED),
        runner=suite.runner(),
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
        family_targets(cfg, scorer_exclusion(context.settings, cfg, registry_of(context, CHECKS))),
        (scorer.version, scorer.thresholds()),
    )


def require_stop_rule(
    context: AppContext,
    cfg: FinetuneSettings,
    store: CampaignStore,
    scorer: AggregateScorer,
    deviation: str | None,
    *,
    err: Echo,
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
            f"pilot stop rule fired: {'; '.join(verdict.reasons)}; the generator comparison is not "
            f"evaluable with this verifier ({STOP_RULE_DEVIATION} TEXT, or "
            "finetune.stop_rule.enforce = false, to go on)"
        )
    return verdict


def active_scorer(context: AppContext, name: str | None) -> AggregateScorer:
    built = scorer_for(context, name or load_finetune(context.settings).scorer)
    reason = built.unavailable()
    if reason is not None:
        raise UsageError(f"scorer {built.name} is unavailable: {reason}")
    return built


def score_prompts(
    context: AppContext,
    store: CampaignStore,
    model_tag: str,
    prompts: Sequence[PromptRecord],
    scorer: AggregateScorer,
    shard: str | None = None,
    *,
    echo: Echo,
) -> list[CandidateScore]:
    scores = score_candidates(
        store, model_tag, prompts, scorer, scoring_deps(context), parse_shard(shard)
    )
    write_calibration_status(store, model_tag, scorer)
    wanted = {p.prompt_id for p in prompts}
    selected = [s for s in scores if s.prompt_id in wanted]
    abstained = sum(s.score is None for s in selected)
    echo(f"{model_tag}/{scorer.name}: {len(selected)} scored, {abstained} abstained")
    return selected


def build_prompts(context: AppContext, check: bool, *, echo: Echo, err: Echo) -> bool:
    """Writes prompts.jsonl, or with ``check`` only compares it; False when the check fails."""
    store = CampaignStore.from_settings(context.settings)
    cfg = load_finetune(context.settings)
    records = build_prompt_set(cfg.prompts, sizing_labels(context.settings))
    text = "".join(record.model_dump_json() + "\n" for record in records)
    path = store.prompts_path
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if check:
        if current != text:
            err(f"{path} differs from the configuration")
            return False
    elif current != text:
        if current is not None and store.model_tags():
            raise UsageError(
                f"{path} differs from the configuration and candidates exist: "
                "use a new finetune.campaign"
            )
        write_records(path, records)
    echo(f"prompts: {path} sha256 {file_sha256(path)}")
    for split in SPLITS:
        echo(f"  {split}: {sum(r.split == split for r in records)}")
    return True


def campaign_split(split: str, comparison_step: str) -> PromptSplit:
    """A train or dev split; the eval candidates and scores belong to the generator comparison,
    whose generation stamp and pre-registration a finetune run would break."""
    parsed = choice(split, SPLITS, "--split")
    if parsed == EVAL_SPLIT:
        raise UsageError(
            "the eval split belongs to the generator comparison: run "
            f"`physverify compare-generators {comparison_step} --arm base|ft`"
        )
    return parsed


def generate_split(
    context: AppContext,
    split: str,
    adapter: Path | None,
    shard: str | None,
    limit: int | None,
    pilot: int | None,
    *,
    echo: Echo,
) -> None:
    which = campaign_split(split, "generate")
    if pilot is not None and which != "train":
        raise UsageError("--pilot draws train prompts: use it with --split train")
    require_cuda(context.settings)
    store = CampaignStore.from_settings(context.settings)
    prompts = campaign_prompts(store, which)
    if pilot is not None:
        prompts = pilot_prompts(prompts, pilot)
    generate_candidates(
        candidate_generator(context, store, adapter), prompts, shard, limit, echo=echo
    )


def score_split(
    context: AppContext,
    split: str,
    model_tag: str,
    scorer: str | None,
    shard: str | None,
    *,
    echo: Echo,
) -> None:
    which = campaign_split(split, "score")
    require_cuda(context.settings)
    store = CampaignStore.from_settings(context.settings)
    prompts = campaign_prompts(store, which)
    score_prompts(
        context, store, model_tag, prompts, active_scorer(context, scorer), shard, echo=echo
    )


def campaign_status_lines(context: AppContext) -> list[str]:
    return status_lines(CampaignStore.from_settings(context.settings))


def budget_lines(context: AppContext, request: BudgetRequest) -> list[str]:
    return budget_report(context.settings, request)


def write_pairs(context: AppContext, scorer: str | None, *, echo: Echo) -> None:
    """Winner-loser pairs of the fit prompts and of the val prompts, then the stop-rule verdict."""
    cfg = load_finetune(context.settings)
    store = CampaignStore.from_settings(context.settings)
    active = active_scorer(context, scorer)
    train = campaign_prompts(store, "train")
    train_ids = {p.prompt_id for p in train}
    scores = [s for s in store.scores(BASE_TAG, active.name) if s.prompt_id in train_ids]
    if not scores:
        raise UsageError(
            f"no {active.name} scores of {BASE_TAG}: run 'finetune score --split train'"
        )
    check_current(store, BASE_TAG, scores, active, "train")
    fit, held = fit_and_val(train)
    directory = store.pairs_dir(active.name)
    outputs = (
        (fit, PAIRS_FILE, REPORT_FILE, "train"),
        (held, VAL_PAIRS_FILE, VAL_REPORT_FILE, "val"),
    )
    for prompts, pairs_file, report_file, label in outputs:
        records, report = pairs_of(prompts, scores, cfg.pairs, active)
        write_records(directory / pairs_file, records)
        if report is not None:
            write_record(directory / report_file, report)
            for line in report_lines(label, report):
                echo(line)
    verdict = current_verdict(context, cfg, store, active)
    write_verdict(store, active.name, verdict)
    for line in verdict_lines(verdict):
        echo(line)
    echo(f"pairs: {directory}")


def pilot_review_lines(
    context: AppContext, scorer: str | None, better: int | None, note: str | None
) -> list[str]:
    cfg = load_finetune(context.settings)
    store = CampaignStore.from_settings(context.settings)
    return review_pilot(store, scorer or cfg.scorer, cfg.stop_rule, better, note)

"""Demo wiring: the verifier rule a demo report uses, its dependencies, and the run, scenes and
report services behind ``physverify demo``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.finetune import candidate_generator, require_cuda, scoring_deps
from physverify.app.services import scorer_for
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.demo.adhoc import (
    adhoc_prompt,
    adhoc_store,
    demo_prompts,
    parse_query,
    require_videos,
    tuned_tag,
)
from physverify.demo.pipeline import (
    DemoDeps,
    ReportDeps,
    rebuild_report,
    report_dir,
    run_demo,
    run_scenes,
)
from physverify.demo.report import RuleInfo, VideoReport, read_report, rule_info, score_text
from physverify.demo.settings import default_out, load_demo
from physverify.finetune.candidates import adapter_tag
from physverify.finetune.layout import CampaignStore
from physverify.finetune.prompts import check_vocabulary, sizing_labels
from physverify.finetune.records import BASE_TAG
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.settings import load_finetune
from physverify.generator_comparison.layout import ComparisonLayout
from physverify.generator_comparison.prereg import load_prereg
from physverify.generator_comparison.scoring import frozen_stamp_problem
from physverify.runtime.runs import current_commit

Rule = tuple[AggregateScorer, RuleInfo]


@dataclass(frozen=True)
class AdhocRequest:
    """``demo run``'s options: the prompt, what to look for, and where the video comes from."""

    prompt: str
    queries: Sequence[str] | None
    family: str
    target_checks: Sequence[str] | None
    adapter: Path | None
    video: Path | None
    out: Path | None


def target_checks(context: AppContext, wanted: Sequence[str] | None) -> list[str]:
    known = sorted(context.registry("checks"))
    unknown = sorted(set(wanted or ()) - set(known))
    if unknown:
        raise UsageError(f"unknown checks {unknown}; known: {known}")
    return list(wanted) if wanted else known


def demo_rule(context: AppContext) -> Rule:
    """``finetune.scorer``, or the provisional rule (labelled as such) when it is unavailable."""
    settings = context.settings
    name = load_finetune(settings).scorer
    scorer = scorer_for(context, name)
    reason = scorer.unavailable()
    if reason is None:
        return scorer, rule_info(scorer)
    if name == ProvisionalScorer.name or not load_demo(settings).allow_provisional:
        raise UsageError(f"scorer {name} is unavailable: {reason}")
    fallback = scorer_for(context, ProvisionalScorer.name)
    return fallback, rule_info(fallback, note=f"The {name} rule is unavailable: {reason}.")


def frozen_scorer(context: AppContext, prereg: Path) -> Rule:
    """``finetune.scorer`` as the generator-comparison freeze stamped it; never the provisional
    fallback."""
    name = load_finetune(context.settings).scorer
    scorer = scorer_for(context, name)
    reason = scorer.unavailable()
    if reason is not None:
        raise UsageError(f"scorer {name} is unavailable: {reason}")
    problem = frozen_stamp_problem(scorer, load_prereg(prereg))
    if problem is not None:
        raise UsageError(f"{prereg}: {problem}")
    return scorer, rule_info(scorer)


def report_deps(settings: Settings, provenance: dict[str, str], active: Rule) -> ReportDeps:
    scorer, rule = active
    return ReportDeps(
        scorer=scorer,
        rule=rule,
        style=load_demo(settings).overlay,
        max_frame_bytes=settings.runtime.max_frame_bytes,
        provenance=provenance,
    )


def demo_deps(
    context: AppContext, store: CampaignStore, settings: Settings, active: Rule
) -> DemoDeps:
    commit = current_commit(settings.repo_root)
    provenance = {
        "campaign": store.campaign,
        "commit": commit,
        "device": settings.runtime.device,
        "profile": settings.profile,
    }
    return DemoDeps(
        store=store,
        scoring=scoring_deps(context),
        report=report_deps(settings, provenance, active),
        commit=commit,
    )


def _echo(report: VideoReport, out: Path, *, echo: Echo) -> None:
    rule = report.rule
    echo(
        f"{report.candidate_id} ({report.model_tag}): {score_text(report.clip)}, coverage "
        f"{report.clip.coverage:.2f}, rule {rule.scorer} ({rule.status})"
    )
    echo(f"  {report_dir(out, report)}")


def run_adhoc(context: AppContext, request: AdhocRequest, *, echo: Echo) -> None:
    settings = context.settings
    video, adapter = request.video, request.adapter
    if video is not None and adapter is not None:
        raise UsageError("--adapter generates a video; it cannot be combined with --video")
    if video is None:
        require_cuda(settings)
    if not request.queries:
        raise UsageError("give at least one --query (the objects the verifier must find)")
    record = adhoc_prompt(
        request.prompt,
        [parse_query(q) for q in request.queries],
        request.family,
        target_checks(context, request.target_checks),
    )
    unscaled = load_finetune(settings).prompts.unscaled_queries
    check_vocabulary([record], sizing_labels(settings), unscaled)
    store = adhoc_store(settings)
    generator = None
    if video is None:
        cand = load_finetune(settings).candidates.model_copy(update={"store_latents": False})
        generator = candidate_generator(context, store, adapter, candidates=cand)
    target = request.out or default_out(settings)
    deps = demo_deps(context, store, settings, demo_rule(context))
    report = run_demo(record, deps, out=target, video=video, generator=generator)
    _echo(report, target, echo=echo)


def scene_reports(
    context: AppContext,
    *,
    from_select: bool,
    adapter: Path | None,
    out: Path | None,
    echo: Echo,
) -> None:
    settings = context.settings
    if from_select == (adapter is not None):
        raise UsageError("pass exactly one of --from-select or --adapter")
    demo = load_demo(settings)
    store = CampaignStore.from_settings(settings)
    prereg = ComparisonLayout.of(store.root).prereg
    if not prereg.is_file():
        raise UsageError(
            f"{prereg} does not exist: run 'compare-generators freeze' before scoring the scenes"
        )
    if adapter is not None and not adapter.exists():
        raise UsageError(f"adapter {adapter} does not exist")
    tuned = adapter_tag(adapter) if adapter is not None else tuned_tag(store, demo.select_run)
    prompts = demo_prompts(store)
    require_videos(store, prompts, (BASE_TAG, tuned))
    deps = demo_deps(context, store, settings, frozen_scorer(context, prereg))
    target = out or default_out(settings)
    for base, ft in run_scenes(prompts, deps, tuned, target):
        _echo(base, target, echo=echo)
        _echo(ft, target, echo=echo)


def rebuild(context: AppContext, results: Path, out: Path | None, *, echo: Echo) -> None:
    settings = context.settings
    if not results.exists():
        raise UsageError(f"{results} does not exist")
    old = read_report(results)
    provenance = {"rebuilt_commit": current_commit(settings.repo_root)}
    target = out or default_out(settings)
    deps = report_deps(settings, provenance, demo_rule(context))
    _echo(rebuild_report(old, deps, target), target, echo=echo)

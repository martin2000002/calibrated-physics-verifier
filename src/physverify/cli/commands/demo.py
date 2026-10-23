"""``physverify demo``: one video through the whole verifier, with its report and overlay.

Ad-hoc prompts and given videos live in their own campaign (``demo.adhoc_campaign``), never in
the H3 campaign. ``scenes`` reuses the H3 campaign's k = 0 videos of the fixed demo prompts and
scores them only once the H3 preregistration exists, with the rule it froze.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final

import typer

from physverify.cli.commands._finetune_deps import candidate_generator, require_cuda, scoring_deps
from physverify.cli.context import AppContext, app_context, heavy_command
from physverify.config.models import Settings
from physverify.core.errors import UsageError
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
from physverify.evaluation.layout import H3Layout
from physverify.evaluation.prereg import load_prereg
from physverify.evaluation.scoring import frozen_stamp_problem
from physverify.finetune.generate import adapter_tag
from physverify.finetune.prompts import check_vocabulary, fixed_record, sizing_labels
from physverify.finetune.records import BASE_TAG, DEMO_TAG, PromptRecord, read_record
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.scoring import build_scorer
from physverify.finetune.selection import Selection
from physverify.finetune.settings import FixedPrompt, QueryTemplate, load_finetune
from physverify.finetune.store import FINETUNE_DIR, LOCAL_VIDEO, SELECTED_FILE, CampaignStore
from physverify.runtime.runs import current_commit
from physverify.schema.io import stable_hash

NAME = "demo"
ADHOC_PREFIX: Final = "adhoc-"
ADHOC_TAG: Final = "adhoc"
ADHOC_FAMILY: Final = "adhoc"
ID_HASH_CHARS: Final = 8
COUNT_SEPARATOR: Final = ":"

app = typer.Typer(no_args_is_help=True, help="End-to-end demo: verifier report and overlay video.")

OutOption = Annotated[
    Path | None, typer.Option(help="Output directory (default: demo.out_dir under data_dir).")
]


def adhoc_store(settings: Settings) -> CampaignStore:
    campaign = load_demo(settings).adhoc_campaign
    return CampaignStore(
        root=settings.paths.data_dir / FINETUNE_DIR / campaign,
        raw_root=settings.paths.raw_dir / LOCAL_VIDEO,
        campaign=campaign,
    )


def parse_query(text: str) -> QueryTemplate:
    """``phrase`` or ``phrase:N`` (N objects expected)."""
    phrase, separator, count = text.rpartition(COUNT_SEPARATOR)
    if not separator or not count.isdigit():
        return QueryTemplate(phrase=text.strip())
    return QueryTemplate(phrase=phrase.strip(), expected_count=int(count))


def adhoc_prompt(
    text: str, queries: Sequence[QueryTemplate], family: str, target_checks: Sequence[str]
) -> PromptRecord:
    fields = {
        "text": text,
        "queries": [query.model_dump() for query in queries],
        "family": family,
        "target_checks": list(target_checks),
    }
    fixed = FixedPrompt(
        id=f"{ADHOC_PREFIX}{stable_hash(fields)[:ID_HASH_CHARS]}",
        family=family,
        text=text,
        queries=list(queries),
        target_checks=list(target_checks),
    )
    return fixed_record(fixed).model_copy(update={"tags": [ADHOC_TAG]})


def target_checks(context: AppContext, wanted: Sequence[str] | None) -> list[str]:
    known = sorted(context.registry("checks"))
    unknown = sorted(set(wanted or ()) - set(known))
    if unknown:
        raise UsageError(f"unknown checks {unknown}; known: {known}")
    return list(wanted) if wanted else known


Rule = tuple[AggregateScorer, RuleInfo]


def active_scorer(settings: Settings) -> Rule:
    """``finetune.scorer``, or the provisional rule (labelled as such) when it is unavailable."""
    name = load_finetune(settings).scorer
    scorer = build_scorer(name, settings)
    reason = scorer.unavailable()
    if reason is None:
        return scorer, rule_info(scorer)
    if name == ProvisionalScorer.name or not load_demo(settings).allow_provisional:
        raise UsageError(f"scorer {name} is unavailable: {reason}")
    fallback = build_scorer(ProvisionalScorer.name, settings)
    return fallback, rule_info(fallback, note=f"The {name} rule is unavailable: {reason}.")


def frozen_scorer(settings: Settings, prereg: Path) -> Rule:
    """``finetune.scorer`` exactly as the H3 freeze stamped it; never the provisional fallback."""
    name = load_finetune(settings).scorer
    scorer = build_scorer(name, settings)
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


def _echo(report: VideoReport, out: Path) -> None:
    rule = report.rule
    typer.echo(
        f"{report.candidate_id} ({report.model_tag}): {score_text(report.clip)}, coverage "
        f"{report.clip.coverage:.2f}, rule {rule.scorer} ({rule.status})"
    )
    typer.echo(f"  {report_dir(out, report)}")


@app.command("run", help="Verify one ad-hoc prompt: generate it (CUDA) or score --video.")
@heavy_command
def run(
    prompt: Annotated[str, typer.Option(help="Prompt text.")],
    query: Annotated[
        list[str] | None, typer.Option(help="Object to detect, 'phrase' or 'phrase:N'.")
    ] = None,
    family: Annotated[str, typer.Option(help="Scene family label.")] = ADHOC_FAMILY,
    target_check: Annotated[
        list[str] | None, typer.Option(help="Checks the prompt targets (default: all).")
    ] = None,
    adapter: Annotated[Path | None, typer.Option(help="LoRA adapter directory.")] = None,
    video: Annotated[Path | None, typer.Option(help="Score this mp4 instead.")] = None,
    out: OutOption = None,
) -> None:
    context = app_context()
    settings = context.settings
    if video is not None and adapter is not None:
        raise UsageError("--adapter generates a video; it cannot be combined with --video")
    if video is None:
        require_cuda(settings)
    if not query:
        raise UsageError("give at least one --query (the objects the verifier must find)")
    record = adhoc_prompt(
        prompt, [parse_query(q) for q in query], family, target_checks(context, target_check)
    )
    unscaled = load_finetune(settings).prompts.unscaled_queries
    check_vocabulary([record], sizing_labels(settings), unscaled)
    store = adhoc_store(settings)
    generator = None
    if video is None:
        cand = load_finetune(settings).candidates.model_copy(update={"store_latents": False})
        generator = candidate_generator(context, store, adapter, candidates=cand)
    target = out or default_out(settings)
    deps = demo_deps(context, store, settings, active_scorer(settings))
    report = run_demo(record, deps, out=target, video=video, generator=generator)
    _echo(report, target)


def tuned_tag(store: CampaignStore, run_id: str) -> str:
    path = store.select_dir(run_id) / SELECTED_FILE
    if not path.is_file():
        raise UsageError(f"{path} does not exist: run 'finetune select' or pass --adapter")
    selection = read_record(path, Selection)
    if selection.selected is None:
        raise UsageError(f"{path} selected no checkpoint; pass --adapter")
    return selection.selected.model_tag


def demo_prompts(store: CampaignStore) -> list[PromptRecord]:
    if not store.prompts_path.is_file():
        raise UsageError(f"{store.prompts_path} does not exist: run 'finetune prompts build'")
    prompts = [p for p in store.prompts() if DEMO_TAG in p.tags]
    if not prompts:
        raise UsageError(f"{store.prompts_path} has no prompts tagged {DEMO_TAG!r}")
    return prompts


def require_videos(
    store: CampaignStore, prompts: Sequence[PromptRecord], tags: Sequence[str]
) -> None:
    have = {
        (r.model_tag, r.prompt_id)
        for tag in tags
        for r in store.candidates(tag)
        if r.k == 0 and store.is_complete(r)
    }
    missing = [f"{p.prompt_id}/{t}" for p in prompts for t in tags if (t, p.prompt_id) not in have]
    if missing:
        raise UsageError(f"k = 0 videos missing: {missing}; run 'h3 generate' first")


@app.command(
    "scenes", help="Base vs fine-tuned reports of the fixed demo scenes (after h3 freeze)."
)
@heavy_command
def scenes(
    from_select: Annotated[
        bool, typer.Option(help="Fine-tuned tag from 'finetune select' (demo.select_run).")
    ] = False,
    adapter: Annotated[Path | None, typer.Option(help="LoRA adapter directory.")] = None,
    out: OutOption = None,
) -> None:
    context = app_context()
    settings = context.settings
    if from_select == (adapter is not None):
        raise UsageError("pass exactly one of --from-select or --adapter")
    demo = load_demo(settings)
    store = CampaignStore.from_settings(settings)
    prereg = H3Layout.of(store.root).prereg
    if not prereg.is_file():
        raise UsageError(f"{prereg} does not exist: run 'h3 freeze' before scoring the scenes")
    if adapter is not None and not adapter.exists():
        raise UsageError(f"adapter {adapter} does not exist")
    tuned = adapter_tag(adapter) if adapter is not None else tuned_tag(store, demo.select_run)
    prompts = demo_prompts(store)
    require_videos(store, prompts, (BASE_TAG, tuned))
    deps = demo_deps(context, store, settings, frozen_scorer(settings, prereg))
    target = out or default_out(settings)
    for base, ft in run_scenes(prompts, deps, tuned, target):
        _echo(base, target)
        _echo(ft, target)


@app.command("report", help="Rebuild a report and overlay from its stored results (CPU).")
def report(
    results: Annotated[Path, typer.Option(help="A report directory or its report.json.")],
    out: OutOption = None,
) -> None:
    settings = app_context().settings
    if not results.exists():
        raise UsageError(f"{results} does not exist")
    old = read_report(results)
    provenance = {"rebuilt_commit": current_commit(settings.repo_root)}
    target = out or default_out(settings)
    deps = report_deps(settings, provenance, active_scorer(settings))
    _echo(rebuild_report(old, deps, target), target)

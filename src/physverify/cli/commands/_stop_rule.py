"""The pilot's stop rule on disk, shared by ``finetune`` and ``h3`` (not a plugin module).

The verdict is always recomputed from the current ``pairs.jsonl``, ``report.json``, base scores
and pilot review; a ``stop_rule.json`` left from other pairs never counts.
"""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.finetune.pilot import fit_and_val
from physverify.finetune.records import (
    BASE_TAG,
    PairRecord,
    PairsReport,
    read_record,
    read_records,
    write_record,
)
from physverify.finetune.scope import family_targets, scorer_exclusion
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import same_stamp
from physverify.finetune.settings import FinetuneSettings
from physverify.finetune.stop_rule import (
    PilotReview,
    StopVerdict,
    candidate_abstention,
    evaluate_stop,
)
from physverify.finetune.store import (
    PAIRS_FILE,
    PILOT_REVIEW_FILE,
    REPORT_FILE,
    STOP_RULE_FILE,
    CampaignStore,
)

STOP_RULE_DEVIATION = "--stop-rule-deviation"
StopRuleDeviationOption = Annotated[
    str | None,
    typer.Option(
        STOP_RULE_DEVIATION,
        help="Go on although the pilot stop rule fired (recorded; H3 then has no verdict).",
    ),
]


def pairs_on_disk(store: CampaignStore, scorer: str) -> list[PairRecord]:
    path = store.pairs_dir(scorer) / PAIRS_FILE
    return read_records(path, PairRecord) if path.is_file() else []


def stored_review(store: CampaignStore, scorer: str) -> PilotReview | None:
    path = store.pairs_dir(scorer) / PILOT_REVIEW_FILE
    return read_record(path, PilotReview) if path.is_file() else None


def _report(store: CampaignStore, scorer: str) -> PairsReport | None:
    path = store.pairs_dir(scorer) / REPORT_FILE
    return read_record(path, PairsReport) if path.is_file() else None


def _abstention(store: CampaignStore, scorer: AggregateScorer) -> float | None:
    """Over the active scorer's base scores of the fit prompts (the pilot's)."""
    train = (
        [p for p in store.prompts() if p.split == "train"] if store.prompts_path.is_file() else []
    )
    fit_ids = {p.prompt_id for p in fit_and_val(train)[0]}
    scores = store.scores(BASE_TAG, scorer.name)
    return candidate_abstention(
        [s for s in scores if s.prompt_id in fit_ids and same_stamp(s, scorer)]
    )


def current_verdict(
    settings: Settings, cfg: FinetuneSettings, store: CampaignStore, scorer: AggregateScorer
) -> StopVerdict:
    name = scorer.name
    return evaluate_stop(
        _report(store, name),
        pairs_on_disk(store, name),
        _abstention(store, scorer),
        stored_review(store, name),
        cfg.stop_rule,
        family_targets(cfg, scorer_exclusion(settings, cfg)),
        (scorer.version, scorer.thresholds()),
    )


def write_verdict(store: CampaignStore, scorer: str, verdict: StopVerdict) -> None:
    write_record(store.pairs_dir(scorer) / STOP_RULE_FILE, verdict)


def verdict_lines(verdict: StopVerdict) -> list[str]:
    state = "passed" if verdict.passed else "FIRED"
    mode = "enforced" if verdict.enforced else "reported only"
    lines = [f"pilot stop rule: {state} ({mode}); measured {verdict.measured}"]
    lines += [f"  reason: {reason}" for reason in verdict.reasons]
    lines += [f"  family {name}: targets {kept}" for name, kept in verdict.families.items()]
    if verdict.deviation is not None:
        lines.append(f"  declared deviation: {verdict.deviation}")
    return lines


def require_stop_rule(
    settings: Settings,
    cfg: FinetuneSettings,
    store: CampaignStore,
    scorer: AggregateScorer,
    deviation: str | None,
) -> StopVerdict:
    """Exit 2 when the rule fires, is enforced and no deviation is declared; the verdict, with
    the deviation that let it through, is written to ``stop_rule.json`` (and printed when it
    fired) either way."""
    verdict = current_verdict(settings, cfg, store, scorer)
    if not verdict.passed and deviation is not None:
        verdict = verdict.model_copy(update={"deviation": deviation})
    write_verdict(store, scorer.name, verdict)
    if not verdict.passed:
        for line in verdict_lines(verdict):
            typer.echo(line, err=True)
    if not verdict.passed and verdict.enforced and deviation is None:
        raise UsageError(
            f"pilot stop rule fired: {'; '.join(verdict.reasons)}; H3 is not evaluable with "
            f"this verifier ({STOP_RULE_DEVIATION} TEXT, or finetune.stop_rule.enforce = false, "
            "to go on)"
        )
    return verdict

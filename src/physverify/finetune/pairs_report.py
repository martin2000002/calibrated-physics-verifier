"""The preference pairs of a campaign on disk: current scores, the pair sets, their yields and
report lines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Final

from physverify.core.errors import UsageError
from physverify.finetune.candidate_scoring import rescorer, stale_scores
from physverify.finetune.layout import PAIRS_FILE, REPORT_FILE, CampaignStore
from physverify.finetune.pairs import build_pairs
from physverify.finetune.records import (
    CandidateScore,
    PairRecord,
    PairsReport,
    PromptRecord,
    read_record,
    read_records,
)
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.settings import PairSettings, PromptSplit

SHOWN_STALE: Final = 5


def check_current(
    store: CampaignStore,
    model_tag: str,
    scores: Sequence[CandidateScore],
    scorer: AggregateScorer,
    split: PromptSplit,
) -> None:
    """Refuses scores of another scorer stamp or of another candidate config_hash."""
    stale = stale_scores(scores, scorer, store.candidates(model_tag))
    if stale:
        raise UsageError(
            f"{len(stale)} {model_tag} scores are stale (another {scorer.name} version or "
            f"thresholds, or another candidate configuration; e.g. {stale[:SHOWN_STALE]}): run "
            f"'finetune score --split {split} --model-tag {model_tag}' to score them again"
        )


def pairs_of(
    prompts: Sequence[PromptRecord],
    scores: Sequence[CandidateScore],
    rules: PairSettings,
    scorer: AggregateScorer,
) -> tuple[list[PairRecord], PairsReport | None]:
    wanted = {p.prompt_id for p in prompts}
    selected = [s for s in scores if s.prompt_id in wanted]
    if not selected:
        return [], None
    return build_pairs(prompts, selected, rules, rescorer(scorer))


def yields(report: PairsReport) -> str:
    """Both yields; a report older than the scored-prompt count has only the per-prompt one."""
    per_prompt = f"{report.yield_per_prompt:.3f} per prompt ({report.prompts})"
    if report.yield_per_scored_prompt is None:
        return f"{per_prompt}, no per-scored-prompt yield (older report)"
    return (
        f"{report.yield_per_scored_prompt:.3f} per scored prompt ({report.scored_prompts} "
        f"scored), {per_prompt}"
    )


def report_lines(label: str, report: PairsReport) -> list[str]:
    return [
        (
            f"{label}: {report.pairs} pairs; {yields(report)}; "
            f"{report.eligible}/{report.candidates} candidates eligible"
        ),
        f"  dropped: {json.dumps(report.dropped, sort_keys=True)}",
        f"  rejected: {json.dumps(report.rejected, sort_keys=True)}",
    ]


def load_pairs(store: CampaignStore, scorer: str, name: str = PAIRS_FILE) -> list[PairRecord]:
    path = store.pairs_dir(scorer) / name
    if not path.is_file():
        raise UsageError(f"{path} does not exist: run 'finetune pairs'")
    return read_records(path, PairRecord)


def require_calibrated(pairs: Sequence[PairRecord], allow_provisional: bool) -> None:
    if not pairs:
        raise UsageError("there are no training pairs")
    if not allow_provisional and any(not p.calibrated for p in pairs):
        raise UsageError(
            "the pairs come from an uncalibrated scorer; pass --allow-provisional to train on them"
        )


def pair_yield_of(store: CampaignStore, scorer: str, given: float | None) -> tuple[float, bool]:
    path = store.pairs_dir(scorer) / REPORT_FILE
    if given is not None:
        return given, False
    if path.is_file():
        return read_record(path, PairsReport).measured_yield, True
    raise UsageError(f"no measured pair yield ({path} missing): pass --pair-yield")

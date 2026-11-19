"""The three arms of the generator comparison per analysed prompt: base (K = 8), fine-tuned (K = 4,
same seeds), and best-of-8 picked from the base candidates. Pure: records in, arms out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from physverify.finetune.records import UNSEEN_TAG, CandidateScore, PromptRecord
from physverify.finetune.seeds import candidate_id

EVAL_SPLIT = "eval"


@dataclass(frozen=True)
class PromptArms:
    prompt: PromptRecord
    base: Mapping[int, CandidateScore]
    ft: Mapping[int, CandidateScore]

    @property
    def prompt_id(self) -> str:
        return self.prompt.prompt_id

    @property
    def unseen(self) -> bool:
        return UNSEEN_TAG in self.prompt.tags


def eval_prompts(prompts: Sequence[PromptRecord]) -> list[PromptRecord]:
    return [p for p in prompts if p.split == EVAL_SPLIT]


def analysis_prompts(
    prompts: Sequence[PromptRecord], exclude_tags: Sequence[str]
) -> list[PromptRecord]:
    """Eval prompts without an excluded tag (the demo scenes are shown, never analysed)."""
    excluded = set(exclude_tags)
    return [p for p in eval_prompts(prompts) if not excluded.intersection(p.tags)]


def _by_k(
    prompt_id: str, k_max: int, scores: Mapping[str, CandidateScore]
) -> dict[int, CandidateScore]:
    found = {k: scores.get(candidate_id(prompt_id, k)) for k in range(k_max)}
    return {k: s for k, s in found.items() if s is not None}


def build_arms(
    prompts: Sequence[PromptRecord],
    base: Sequence[CandidateScore],
    ft: Sequence[CandidateScore],
    k_base: int,
    k_ft: int,
) -> list[PromptArms]:
    base_by_id = {s.candidate_id: s for s in base}
    ft_by_id = {s.candidate_id: s for s in ft}
    return [
        PromptArms(
            prompt=p,
            base=_by_k(p.prompt_id, k_base, base_by_id),
            ft=_by_k(p.prompt_id, k_ft, ft_by_id),
        )
        for p in prompts
    ]

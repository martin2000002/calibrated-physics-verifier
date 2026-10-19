"""The fit/val partition of the train prompts and the pilot drawn from them (ADR 0012, 4.3).

The pilot measures the pair yield that sizes the run, so it takes fit prompts only (val prompts
land in another report) spread across the families: the prompt set is ordered by family, and its
first prompts alone would pilot a single family.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest

from physverify.finetune.records import VAL_TAG, PromptRecord


def is_val(prompt: PromptRecord) -> bool:
    return VAL_TAG in prompt.tags


def fit_and_val(train: Sequence[PromptRecord]) -> tuple[list[PromptRecord], list[PromptRecord]]:
    """The train prompts DPO fits on, and the held-out val prompts that choose beta."""
    return [p for p in train if not is_val(p)], [p for p in train if is_val(p)]


def pilot_prompts(train: Sequence[PromptRecord], count: int) -> list[PromptRecord]:
    """``count`` fit prompts taken round-robin over the families, in prompt-set order."""
    fit, _ = fit_and_val(train)
    families: dict[str, list[PromptRecord]] = {}
    for prompt in fit:
        families.setdefault(prompt.family, []).append(prompt)
    interleaved = [
        prompt for row in zip_longest(*families.values()) for prompt in row if prompt is not None
    ]
    return interleaved[:count]

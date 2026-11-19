from __future__ import annotations

from physverify.finetune.pilot import fit_and_val, pilot_prompts
from physverify.finetune.records import VAL_TAG, PromptRecord


def _prompt(family: str, i: int, val: bool = False) -> PromptRecord:
    return PromptRecord(
        prompt_id=f"{family}-{i:04d}",
        text=f"{family} {i}",
        queries=["ball"],
        family=family,
        template_id=f"{family}_t",
        target_checks=["R5a"],
        split="train",
        tags=[VAL_TAG] if val else [],
    )


def _train() -> list[PromptRecord]:
    """Ordered by family like the prompt set, with a val prompt at the head of each family."""
    prompts = []
    for family, n in (("fall", 6), ("slide", 4), ("stack", 2)):
        prompts.append(_prompt(family, 99, val=True))
        prompts += [_prompt(family, i) for i in range(n)]
    return prompts


def test_the_pilot_is_spread_across_families_and_holds_no_val_prompt() -> None:
    pilot = pilot_prompts(_train(), 7)
    assert [p.prompt_id for p in pilot] == [
        "fall-0000",
        "slide-0000",
        "stack-0000",
        "fall-0001",
        "slide-0001",
        "stack-0001",
        "fall-0002",
    ]
    assert not any(VAL_TAG in p.tags for p in pilot)


def test_a_pilot_larger_than_the_fit_set_takes_every_fit_prompt() -> None:
    fit, val = fit_and_val(_train())
    assert len(fit) == 12 and len(val) == 3
    assert sorted(p.prompt_id for p in pilot_prompts(_train(), 50)) == sorted(
        p.prompt_id for p in fit
    )

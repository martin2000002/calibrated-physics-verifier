"""The domain guard of ``prompts build``: every expansion of every template, every fixed prompt,
and the ``extra_train`` expansion (seen templates only), must lie in the set's domain. It looks at
the whole universe, not at the split, and names a template or fixed prompt id, never a text."""

from __future__ import annotations

from collections import Counter

from physverify.core.errors import ConfigError
from physverify.finetune.domains._base import DomainVerdict, PromptDomain, UnknownWordError
from physverify.finetune.prompts import expand_template, seen_items
from physverify.finetune.settings import PromptSetSettings


def _verdict(domain: PromptDomain, text: str, where: str) -> DomainVerdict:
    try:
        return domain.classify(text)
    except UnknownWordError as exc:
        raise ConfigError(f"{where}: domain {domain.name} knows no {exc.words}") from exc


def _require(domain: PromptDomain, texts: list[str], where: str) -> None:
    verdicts = [_verdict(domain, text, where) for text in texts]
    outside = [v for v in verdicts if not v.in_domain]
    if outside:
        reasons = Counter(reason for v in outside for reason in v.reasons)
        raise ConfigError(
            f"{where}: {len(outside)} of {len(verdicts)} texts are outside domain "
            f"{domain.name} (reasons {dict(sorted(reasons.items()))})"
        )


def check_universe(prompts: PromptSetSettings, domain: PromptDomain) -> None:
    for family in prompts.families:
        for template in family.templates:
            texts = [item.text for item in expand_template(template, prompts.slots)]
            _require(domain, texts, f"template {template.id}")
    for fixed in prompts.fixed_eval:
        _require(domain, [fixed.text], f"fixed prompt {fixed.id}")
    extra = prompts.extra_train
    if extra is not None:
        for family in extra.families:
            texts = [item.text for item in seen_items(family, extra.slots)]
            _require(domain, texts, f"extra {family.name}")

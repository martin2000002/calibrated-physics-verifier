"""The prompt set: template expansion, a deterministic stratified split, vocabulary checks (4.1).

Every query noun must have a size prior: it is checked against the labels the sizing model
can scale (``physverify.sizing.model.sized_labels``), except the declared ``unscaled_queries``,
whose checks that need a scale then abstain (no scale is invented).
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import string
import zlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from physverify.config.models import Settings
from physverify.core.errors import ConfigError, UsageError
from physverify.datasets.local_video import QueryEntry, QueryList
from physverify.finetune.layout import CampaignStore
from physverify.finetune.records import (
    DEMO_TAG,
    TEMPLATE_TAG_PREFIX,
    UNSEEN_TAG,
    VAL_TAG,
    PromptRecord,
)
from physverify.finetune.settings import (
    SPLITS,
    FamilySpec,
    FixedPrompt,
    PromptSetSettings,
    PromptSplit,
    QueryTemplate,
    TemplateSpec,
)
from physverify.schema.io import stable_hash

PROMPT_HASH_CHARS: Final = 10
FIXED_TEMPLATE: Final = "fixed"
SIZING_MODULE: Final = "physverify.sizing.model"
SIZING_FUNCTION: Final = "sized_labels"

LabelSource = Callable[[], Collection[str]]


@dataclass(frozen=True)
class Expanded:
    """One prompt of a family before it is assigned to a split."""

    template: TemplateSpec
    text: str
    queries: QueryList


def template_slots(text: str) -> list[str]:
    names = [field for _, field, _, _ in string.Formatter().parse(text) if field]
    return list(dict.fromkeys(names))


def _slots_of(template: TemplateSpec) -> list[str]:
    phrases = " ".join(query.phrase for query in template.queries)
    return sorted(set(template_slots(template.text)) | set(template_slots(phrases)))


def _query(query: QueryTemplate, values: Mapping[str, str]) -> str | QueryEntry:
    phrase = query.phrase.format(**values)
    if query.expected_count is None:
        return phrase
    return QueryEntry(phrase=phrase, expected_count=query.expected_count)


def expand_template(template: TemplateSpec, slots: Mapping[str, Sequence[str]]) -> list[Expanded]:
    """Cartesian product of the slots the template names, slot names sorted, values in order."""
    names = _slots_of(template)
    missing = sorted(set(names) - set(slots))
    if missing:
        raise ConfigError(f"template {template.id!r} names unknown slots {missing}")
    expanded = []
    for combo in itertools.product(*(slots[name] for name in names)):
        values = dict(zip(names, combo, strict=True))
        queries: QueryList = [_query(query, values) for query in template.queries]
        expanded.append(Expanded(template, template.text.format(**values), queries))
    return expanded


def expand(family: FamilySpec, slots: Mapping[str, Sequence[str]]) -> list[Expanded]:
    """Every prompt of a family, in canonical order, without repeated texts."""
    seen: set[str] = set()
    unique: list[Expanded] = []
    for template in family.templates:
        for item in expand_template(template, slots):
            if item.text not in seen:
                seen.add(item.text)
                unique.append(item)
    return unique


def prompt_id(family: str, text: str) -> str:
    return f"{family}-{stable_hash(text)[:PROMPT_HASH_CHARS]}"


def largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    """Integer quotas proportional to ``weights`` summing to ``total`` (ties to the first)."""
    scale = total / sum(weights)
    exact = [weight * scale for weight in weights]
    quotas = [int(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - quotas[i]), i))
    for index in order[: total - sum(quotas)]:
        quotas[index] += 1
    return quotas


def family_rng(seed: int, family: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(family.encode("utf-8"))])


def _permuted(items: Sequence[Expanded], rng: np.random.Generator) -> list[Expanded]:
    return [items[int(i)] for i in rng.permutation(len(items))]


def _record(
    family: FamilySpec, item: Expanded, split: PromptSplit, extra: Sequence[str] = ()
) -> PromptRecord:
    tags = [f"{TEMPLATE_TAG_PREFIX}{item.template.id}", *extra]
    if item.template.holdout:
        tags.append(UNSEEN_TAG)
    return PromptRecord(
        prompt_id=prompt_id(family.name, item.text),
        text=item.text,
        queries=item.queries,
        family=family.name,
        template_id=item.template.id,
        target_checks=list(family.target_checks),
        split=split,
        tags=tags,
    )


def _take(items: list[Expanded], n: int, family: str, what: str) -> list[Expanded]:
    if n > len(items):
        raise ConfigError(
            f"family {family}: {what} needs {n} prompts but only {len(items)} remain; "
            "add slot values or templates, or lower the split sizes"
        )
    taken = items[:n]
    del items[:n]
    return taken


def split_family(
    family: FamilySpec, cfg: PromptSetSettings, quotas: Mapping[PromptSplit, int]
) -> list[PromptRecord]:
    """Holdout templates feed only ``eval``; the seen ones fill ``eval``, ``dev``, ``train`` in
    that order from one seeded permutation. The first ``val_fraction`` of train is ``val``."""
    items = expand(family, cfg.slots)
    rng = family_rng(cfg.seed, family.name)
    seen = _permuted([i for i in items if not i.template.holdout], rng)
    unseen = _permuted([i for i in items if i.template.holdout], rng)
    n_unseen = min(len(unseen), round(quotas["eval"] * cfg.eval_unseen_fraction))
    eval_items = _take(unseen, n_unseen, family.name, "eval (unseen templates)")
    eval_items += _take(seen, quotas["eval"] - n_unseen, family.name, "eval")
    dev_items = _take(seen, quotas["dev"], family.name, "dev")
    train_items = _take(seen, quotas["train"], family.name, "train")
    n_val = round(len(train_items) * cfg.val_fraction)
    return [
        *(_record(family, item, "eval") for item in eval_items),
        *(_record(family, item, "dev") for item in dev_items),
        *(
            _record(family, item, "train", (VAL_TAG,) if index < n_val else ())
            for index, item in enumerate(train_items)
        ),
    ]


def fixed_record(fixed: FixedPrompt) -> PromptRecord:
    return PromptRecord(
        prompt_id=fixed.id,
        text=fixed.text,
        queries=[_query(query, {}) for query in fixed.queries],
        family=fixed.family,
        template_id=FIXED_TEMPLATE,
        target_checks=list(fixed.target_checks),
        split="eval",
        tags=[DEMO_TAG],
    )


def phrase_of(entry: str | QueryEntry) -> str:
    return entry if isinstance(entry, str) else entry.phrase


def normalized_label(phrase: str) -> str:
    return "_".join(phrase.strip().lower().split())


def is_covered(phrase: str, labels: Collection[str], unscaled: Collection[str]) -> bool:
    normalized = normalized_label(phrase)
    known = {normalized_label(label) for label in labels}
    allowed = {normalized_label(label) for label in unscaled}
    return normalized in known or normalized in allowed


def check_vocabulary(
    records: Sequence[PromptRecord], labels: Collection[str], unscaled: Collection[str]
) -> None:
    uncovered = sorted(
        {
            phrase_of(entry)
            for record in records
            for entry in record.queries
            if not is_covered(phrase_of(entry), labels, unscaled)
        }
    )
    if uncovered:
        raise ConfigError(
            f"query nouns without a size prior: {uncovered}; use sized labels or declare them "
            "in finetune.prompts.unscaled_queries"
        )


def build_prompt_set(cfg: PromptSetSettings, labels: Collection[str]) -> list[PromptRecord]:
    """The whole set, sorted by split (train, dev, eval), family order, then permutation."""
    weights = [family.weight for family in cfg.families]
    sizes = {"train": cfg.n_train, "dev": cfg.n_dev, "eval": cfg.n_eval}
    per_split = {split: largest_remainder(sizes[split], weights) for split in SPLITS}
    records: list[PromptRecord] = []
    for index, family in enumerate(cfg.families):
        quotas: dict[PromptSplit, int] = {split: per_split[split][index] for split in SPLITS}
        records.extend(split_family(family, cfg, quotas))
    records.extend(fixed_record(fixed) for fixed in cfg.fixed_eval)
    ids = [record.prompt_id for record in records]
    if len(set(ids)) != len(ids):
        raise ConfigError("two prompts share a prompt_id; fixed_eval ids must be new")
    check_vocabulary(records, labels, cfg.unscaled_queries)
    rank = {split: n for n, split in enumerate(SPLITS)}
    return sorted(records, key=lambda record: rank[record.split])


def sizing_labels(settings: Settings) -> Collection[str]:
    """Labels with a size prior, from the sizing model (imported lazily: another cycle's code)."""
    try:
        module = importlib.import_module(SIZING_MODULE)
    except ImportError as exc:
        raise UsageError(
            f"{SIZING_MODULE} is not available ({exc}); the prompt vocabulary is checked "
            "against its sized labels"
        ) from exc
    function = getattr(module, SIZING_FUNCTION)
    labels: Collection[str] = (
        function(settings) if inspect.signature(function).parameters else function()
    )
    return frozenset(labels)


def campaign_prompts(store: CampaignStore, split: PromptSplit | None = None) -> list[PromptRecord]:
    if not store.prompts_path.is_file():
        raise UsageError(f"{store.prompts_path} does not exist: run 'finetune prompts build'")
    prompts = store.prompts()
    return [p for p in prompts if split is None or p.split == split]

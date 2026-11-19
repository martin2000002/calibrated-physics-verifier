"""Agreement numbers of the final table from ``items.csv``: the decided fraction, the
verifier's accuracy against the VLM's on the decided items and the balanced accuracy of VLM OR
verifier against the VLM alone, with paired caption-bootstrap intervals."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from physverify.human_agreement.clip_decision import ClipDecision
from physverify.human_agreement.stats import caption_bootstrap
from physverify.reporting.final_rows import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    Criteria,
    Row,
    ci_text,
    fmt,
    missing,
)
from physverify.stats.intervals import clopper_pearson

DECIDED: frozenset[str] = frozenset({ClipDecision.PLAUSIBLE.value, ClipDecision.IMPLAUSIBLE.value})
ALL_VERSIONS = "all"


@dataclass(frozen=True)
class Item:
    caption: str
    version: str
    human: bool
    vlm: bool
    decided: bool
    verifier: bool
    """Flagged implausible by the verifier (an undecided item is not flagged)."""
    scoped: bool | None = None
    """At least one active check decided; None when ``items.csv`` has no ``decided_checks``."""


@dataclass(frozen=True)
class Bootstrap:
    n: int
    seed: int
    ci: float


@dataclass(frozen=True)
class Estimate:
    value: float | None
    low: float | None
    high: float | None
    n: int


def read_items(path: Path) -> tuple[Item, ...]:
    """Items with a VLM label; the comparison pairs only those."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(
        Item(
            caption=row["caption"],
            version=row["version"],
            human=row["human_implausible"] == "1",
            vlm=row["vlm_implausible"] == "1",
            decided=row["verifier_outcome"] in DECIDED,
            verifier=row["verifier_outcome"] == ClipDecision.IMPLAUSIBLE.value,
            scoped=None if row.get("decided_checks") is None else row["decided_checks"] != "",
        )
        for row in rows
        if row["vlm_implausible"] != ""
    )


def partition(items: Sequence[Item], version: str) -> tuple[Item, ...]:
    return (
        tuple(items) if version == ALL_VERSIONS else tuple(i for i in items if i.version == version)
    )


def _ratio(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def accuracy(items: Sequence[Item], flag: Callable[[Item], bool]) -> float:
    return _ratio(sum(flag(i) == i.human for i in items), len(items))


def balanced_accuracy(items: Sequence[Item], flag: Callable[[Item], bool]) -> float:
    positives = [i for i in items if i.human]
    negatives = [i for i in items if not i.human]
    tpr = _ratio(sum(flag(i) for i in positives), len(positives))
    tnr = _ratio(sum(not flag(i) for i in negatives), len(negatives))
    return (tpr + tnr) / 2


def _vlm(item: Item) -> bool:
    return item.vlm


def _verifier(item: Item) -> bool:
    return item.verifier


def _either(item: Item) -> bool:
    return item.vlm or item.verifier


def decided_accuracy_difference(items: Sequence[Item]) -> Mapping[str, float]:
    decided = [i for i in items if i.decided]
    return {"value": accuracy(decided, _verifier) - accuracy(decided, _vlm)}


def balanced_accuracy_difference(items: Sequence[Item]) -> Mapping[str, float]:
    """The verifier alone against the VLM alone (Part 3 of the comparison)."""
    return {"value": balanced_accuracy(items, _verifier) - balanced_accuracy(items, _vlm)}


def balanced_accuracy_gain(items: Sequence[Item]) -> Mapping[str, float]:
    return {"value": balanced_accuracy(items, _either) - balanced_accuracy(items, _vlm)}


def decided_fraction(items: Sequence[Item], ci: float) -> Estimate:
    k, n = sum(i.decided for i in items), len(items)
    bounds = clopper_pearson(k, n, ci)
    return Estimate(_ratio(k, n) if n else None, bounds.low, bounds.high, n)


def paired(
    items: Sequence[Item],
    statistic: Callable[[Sequence[Item]], Mapping[str, float]],
    boot: Bootstrap,
) -> Estimate:
    """The statistic on every item and its percentile interval over whole captions."""
    if not items:
        return Estimate(None, None, None, 0)
    point = statistic(items)["value"]
    result = caption_bootstrap(items, lambda i: i.caption, statistic, boot.n, boot.seed, boot.ci)
    found = result.intervals.get("value")
    low, high = (None, None) if found is None else (found.low, found.high)
    return Estimate(point if point == point else None, low, high, len(items))


PARTITIONS = (
    (ALL_VERSIONS, "VideoPhy-2 test and VideoPhy v1 object-object"),
    ("v2", "VideoPhy-2 test"),
    ("v1", "VideoPhy v1 object-object"),
)


def agreement_rows(
    items: Sequence[Item] | None, boot: Bootstrap, c: Criteria, src: str
) -> list[Row]:
    rows: list[Row] = []
    for version, label in PARTITIONS:
        part = None if items is None else partition(items, version)
        texts = (
            (f"R3.decided.{version}", f"{label}: decided fraction", f">= {c.generated_decided}"),
            (
                f"R3.accuracy.{version}",
                f"{label}: verifier minus VLM accuracy on decided",
                ">= 0 (point)",
            ),
            (
                f"R3.balanced.{version}",
                f"{label}: balanced accuracy, VLM+verifier minus VLM",
                "CI low > 0",
            ),
        )
        if not part:
            rows += [missing(i, t, th, "paired items") for i, t, th in texts]
            rows.append(scoped_row((), version, label, boot, src))
            continue
        share = decided_fraction(part, boot.ci)
        diff = paired(part, decided_accuracy_difference, boot)
        gain = paired(part, balanced_accuracy_gain, boot)
        rows += [
            _estimate_row(texts[0], share, (share.value or 0.0) >= c.generated_decided, src),
            _estimate_row(texts[1], diff, diff.value is not None and diff.value >= 0.0, src),
            _estimate_row(texts[2], gain, gain.low is not None and gain.low > 0.0, src),
            scoped_row(part, version, label, boot, src),
        ]
    return rows


def scoped_row(part: Sequence[Item], version: str, label: str, boot: Bootstrap, src: str) -> Row:
    """The balanced-accuracy gain on the items where at least one active check decides."""
    text = (
        f"R3.balanced_scoped.{version}",
        (
            f"{label}: balanced accuracy where an active check decides"
            " (conditional on the verifier), VLM+verifier minus VLM"
        ),
        "CI low > 0",
    )
    if not part:
        return missing(*text, "paired items")
    if any(i.scoped is None for i in part):
        return missing(*text, "decided_checks column")
    scoped = [i for i in part if i.scoped]
    gain = paired(scoped, balanced_accuracy_gain, boot)
    row = _estimate_row(text, gain, gain.low is not None and gain.low > 0.0, src)
    return replace(row, note=f"{row.note} of {len(part)}, coverage {fmt(len(scoped) / len(part))}")


def _estimate_row(text: tuple[str, str, str], found: Estimate, ok: bool, src: str) -> Row:
    status = NOT_EVALUABLE if found.value is None else (PASS if ok else FAIL)
    return Row(
        text[0],
        text[1],
        fmt(found.value),
        ci_text(found.low, found.high),
        text[2],
        status,
        src,
        f"n {found.n}",
    )

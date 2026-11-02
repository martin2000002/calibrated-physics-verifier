"""What the reference VLM and the verifier catch together and apart, on the paired items.

A system flags an item when it predicts implausible. The verifier's flags come from the
``as_plausible`` view (an undecided clip never flags: no evidence, no vote) and its decided
items from the ``decided_only`` view, so both reuse the undecided policy and its cut. The
OR-combination flags an item when either system does. Coverage is the fraction of human
implausible items flagged, precision the fraction of flagged items that are human implausible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from physverify.core.models import FrozenModel
from physverify.human_agreement.comparison_items import ComparisonItem
from physverify.human_agreement.stats import Labels, caption_bootstrap
from physverify.human_agreement.undecided_policy import (
    AS_PLAUSIBLE,
    DECIDED_ONLY,
    labels_of,
    policy_view,
)

SYSTEMS: Final = ("vlm", "verifier", "either")
SHARES: Final = ("verifier_only", "vlm_only", "both", "neither")
DETECTION: Final = ("coverage", "precision", "flagged_rate")
DECIDED_SYSTEMS: Final = ("verifier", "vlm", "difference")


@dataclass(frozen=True)
class Flags:
    """Per paired item (1 = implausible): the human label, each system's flag, and whether the
    verifier decided the item."""

    captions: tuple[str, ...]
    human: Labels
    vlm: Labels
    verifier: Labels
    decided: NDArray[np.bool_]


def flags(items: Sequence[ComparisonItem]) -> Flags:
    view = policy_view(items, AS_PLAUSIBLE)
    decided = set(policy_view(items, DECIDED_ONLY).index)
    return Flags(
        captions=tuple(items[i].row.caption for i in view.index),
        human=labels_of(items, view.index, lambda it: it.human_implausible),
        vlm=labels_of(items, view.index, lambda it: it.vlm_implausible),
        verifier=view.verifier,
        decided=np.asarray([i in decided for i in view.index], dtype=np.bool_),
    )


def _ratio(hits: int, total: int) -> float:
    return hits / total if total else math.nan


def _detection(y: Labels, p: Labels) -> dict[str, float]:
    tp, flagged = int(np.sum((y == 1) & (p == 1))), int(np.sum(p == 1))
    values = (_ratio(tp, int(np.sum(y == 1))), _ratio(tp, flagged), _ratio(flagged, int(y.size)))
    return dict(zip(DETECTION, values, strict=True))


def _shares(y: Labels, vlm: Labels, verifier: Labels) -> dict[str, float]:
    """Each true violation falls in exactly one share, so the four sum to 1."""
    positives = y == 1
    a, b = verifier[positives] == 1, vlm[positives] == 1
    masks = {"verifier_only": a & ~b, "vlm_only": ~a & b, "both": a & b, "neither": ~a & ~b}
    total = int(np.sum(positives))
    return {f"share.{name}": _ratio(int(np.sum(masks[name])), total) for name in SHARES}


def _decided(f: Flags, pick: NDArray[np.int_]) -> dict[str, float]:
    """On the items the verifier decided: its precision and the VLM's, paired."""
    keep = pick[f.decided[pick]]
    y = f.human[keep]
    verifier = _detection(y, f.verifier[keep])["precision"]
    vlm = _detection(y, f.vlm[keep])["precision"]
    values = (verifier, vlm, verifier - vlm)
    precision = {f"decided.{k}.precision": v for k, v in zip(DECIDED_SYSTEMS, values, strict=True)}
    return {"decided.fraction": _ratio(int(keep.size), int(pick.size)), **precision}


def _systems(vlm: Labels, verifier: Labels) -> dict[str, Labels]:
    return dict(zip(SYSTEMS, (vlm, verifier, vlm | verifier), strict=True))


def statistic(f: Flags, sample: Sequence[int]) -> Mapping[str, float]:
    pick = np.asarray(sample, dtype=np.int_)
    y, vlm, verifier = f.human[pick], f.vlm[pick], f.verifier[pick]
    systems = _systems(vlm, verifier)
    out = {f"{s}.{k}": v for s, p in systems.items() for k, v in _detection(y, p).items()}
    return {**out, **_shares(y, vlm, verifier), **_decided(f, pick)}


def counts(f: Flags) -> dict[str, int]:
    flagged = {f"flagged.{s}": int(np.sum(p)) for s, p in _systems(f.vlm, f.verifier).items()}
    return {
        "n": int(f.human.size),
        "human_implausible": int(np.sum(f.human)),
        "decided": int(np.sum(f.decided)),
        **flagged,
    }


class Complementarity(FrozenModel):
    """Point values and caption-bootstrap percentile intervals of ``statistic`` on the paired
    items; ``non_finite`` counts the replicas where a ratio had no denominator."""

    flag_policy: str
    decided_policy: str
    counts: dict[str, int]
    point: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    replicas: int
    non_finite: dict[str, int]
    ci: float


def complementarity(
    items: Sequence[ComparisonItem], replicas: int, seed: int, ci: float
) -> Complementarity:
    f = flags(items)
    positions = list(range(f.human.size))
    boot = caption_bootstrap(
        positions, lambda k: f.captions[k], lambda s: statistic(f, s), replicas, seed, ci
    )
    return Complementarity(
        flag_policy=AS_PLAUSIBLE,
        decided_policy=DECIDED_ONLY,
        counts=counts(f),
        point=dict(statistic(f, positions)),
        intervals={k: (v.low, v.high) for k, v in boot.intervals.items()},
        replicas=boot.replicas,
        non_finite=dict(boot.non_finite),
        ci=ci,
    )

"""Paired out-of-fold AUROC deltas b - a of two arms on the same clips (ADR 0036 clause (i)).

Both arms are read on the same scene draws. Readings: "scored" keeps the clips scored in both
arms, "operational" every labelled clip of both (unscored = 0.0). Pure, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from physverify.calibration.stratified import (
    GLOBAL,
    READINGS,
    STRATIFIED,
    Families,
    Index,
    Reading,
    Resampling,
    Valued,
    above,
    columns,
    finite,
    indices,
    key,
    metrics,
)
from physverify.core.models import ArtifactModel
from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap
from physverify.stats.intervals import Ci


class PairedReading(ArtifactModel):
    n: int
    a: float | None
    b: float | None
    delta: float | None
    ci: Ci
    """Of ``delta``, scene bootstrap with the same draws for both arms."""


class PairedUnit(ArtifactModel):
    auroc_cv: dict[Reading, PairedReading]
    stratified: dict[Reading, PairedReading] | None
    improves: dict[Reading, bool]
    """Every delta interval (global, and stratified where it applies) entirely above 0."""
    breaks: dict[Reading, bool]
    """Some delta interval entirely below 0."""


def paired_deltas(
    a: Mapping[str, Valued],
    b: Mapping[str, Valued],
    families: Families | None,
    settings: Resampling,
) -> PairedUnit:
    """Over the clips of both arms (keyed by clip id); scenes are arm ``a``'s."""
    clips = sorted(set(a) & set(b))
    ca, cb = columns([a[k] for k in clips], families), columns([b[k] for k in clips], families)
    common = ca.scored & cb.scored
    ca, cb = replace(ca, scored=common), replace(cb, scored=common)
    used = metrics(families is not None)

    def statistic(index: Index) -> dict[str, float]:
        return {
            key(r, name): metric(cb, cb.rows(r, index)) - metric(ca, ca.rows(r, index))
            for r in READINGS
            for name, metric in used.items()
        }

    blocks = [ClusterBlock(scene, i) for i, scene in enumerate(ca.scene)]
    cis = scene_bootstrap(
        blocks,
        lambda sample: statistic(indices(sample)),
        settings.bootstrap,
        settings.seed,
        settings.ci,
    )
    every = np.arange(len(clips))

    def read(r: Reading, name: str) -> PairedReading:
        rows, metric = ca.rows(r, every), used[name]
        point_a, point_b = finite(metric(ca, rows)), finite(metric(cb, rows))
        delta = None if point_a is None or point_b is None else point_b - point_a
        ci = cis.get(key(r, name), Ci())
        return PairedReading(n=int(rows.size), a=point_a, b=point_b, delta=delta, ci=ci)

    by_metric = {name: {r: read(r, name) for r in READINGS} for name in used}
    return PairedUnit(
        auroc_cv=by_metric[GLOBAL],
        stratified=by_metric.get(STRATIFIED),
        improves={r: all(above(m[r].ci.low, 0.0) for m in by_metric.values()) for r in READINGS},
        breaks={r: any(_below(m[r].ci.high) for m in by_metric.values()) for r in READINGS},
    )


def _below(bound: float | None) -> bool:
    return bound is not None and bound < 0.0

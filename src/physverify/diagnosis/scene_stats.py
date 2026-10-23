"""Per-clip stage-error statistics, per-scene tables and scene-bootstrap deltas (ADR 0014).

A collected file keeps, per clip and metric, the sufficient statistics of every sample quantity
([n, sum, sum of squares] over finite samples) and every rate ([numerator, denominator]), so
pooled and per-scene values, and deltas between two configurations, are exact sums."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from physverify.calibration.stats import SceneBlock, scene_bootstrap
from physverify.calibration.table import scene_of
from physverify.core.atomic import write_text_atomic
from physverify.diagnosis.cases import MetricOutcome, SampledMetric
from physverify.diagnosis.chain import ChainRunner
from physverify.diagnosis.error_profile_writer import StageSamples
from physverify.diagnosis.settings import DiagnosisSettings
from physverify.diagnosis.stage_errors import clips_of, measure, stage_pairs
from physverify.schema.clip import ClipManifest

Stat = Literal["rms", "mean", "rate"]
Kind = Literal["samples", "ratios"]
Stats = list[float]
Row = Mapping[str, Any]
DELTA = "delta"


def sufficient(outcome: MetricOutcome) -> tuple[dict[str, Stats], dict[str, Stats]]:
    samples: dict[str, Stats] = {}
    for name, values in outcome.samples.items():
        finite = np.asarray(values, dtype=np.float64).ravel()
        finite = finite[np.isfinite(finite)]
        samples[name] = [float(finite.size), float(finite.sum()), float(np.square(finite).sum())]
    ratios = {name: [float(num), float(den)] for name, (num, den) in outcome.ratios.items()}
    return samples, ratios


def clip_rows(found: StageSamples, split: str, scenes: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in found:
        for clip_id, outcome in metric.outcomes.items():
            samples, ratios = sufficient(outcome)
            rows.append(
                {
                    "split": split,
                    "metric": metric.metric,
                    "provider": str(metric.provider),
                    "clip_id": clip_id,
                    "scene": scenes[clip_id],
                    "samples": samples,
                    "ratios": ratios,
                }
            )
    return rows


def collect_rows(
    runner: ChainRunner,
    clips: Sequence[ClipManifest],
    metrics: Mapping[str, SampledMetric],
    diag: DiagnosisSettings,
    stages: Sequence[str],
) -> list[dict[str, Any]]:
    """Per-clip rows of the ``stages`` metrics, measured as ``diagnose stages`` does: the
    profile split without an error profile, the report split with ``paths.error_profile``."""
    wanted = {key: metric for key, metric in metrics.items() if key.split("/")[0] in stages}
    pairs = tuple(pair for pair in stage_pairs(diag) if pair.stage.value in stages)
    scenes = {clip.clip_id: scene_of(clip) for clip in clips}
    calib = clips_of(clips, diag.profile_split)
    report = clips_of(clips, diag.report_split)
    unprofiled = runner.with_error_profile(None)
    return [
        *clip_rows(
            measure(unprofiled, calib, wanted, pairs, diag.metrics),
            diag.profile_split.value,
            scenes,
        ),
        *clip_rows(
            measure(runner, report, wanted, pairs, diag.metrics), diag.report_split.value, scenes
        ),
    ]


@dataclass(frozen=True)
class Collected:
    label: str
    profile: str | None
    overrides: tuple[str, ...]
    rows: tuple[Row, ...]


def save(path: Path, collected: Collected) -> None:
    payload = {
        "label": collected.label,
        "profile": collected.profile,
        "overrides": list(collected.overrides),
        "rows": list(collected.rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(payload, indent=1))


def load(path: Path) -> Collected:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Collected(
        label=str(data["label"]),
        profile=data.get("profile"),
        overrides=tuple(data.get("overrides", ())),
        rows=tuple(data["rows"]),
    )


STATS: tuple[Stat, ...] = ("rms", "mean", "rate")


def parse_stat(text: str) -> Stat:
    for stat in STATS:
        if text == stat:
            return stat
    raise ValueError(f"unknown stat {text!r} ({', '.join(STATS)})")


def kind_of(stat: Stat) -> Kind:
    return "ratios" if stat == "rate" else "samples"


def summed(parts: Sequence[Sequence[float]]) -> Stats:
    """Element-wise sum of sufficient statistics; empty parts are skipped."""
    kept = [p for p in parts if p]
    return [float(sum(p[i] for p in kept)) for i in range(len(kept[0]))] if kept else []


def value(stats: Sequence[float], stat: Stat) -> float:
    """``rate`` = num/den; ``mean`` and ``rms`` from [n, sum, sumsq]; NaN when empty."""
    if stat == "rate":
        return stats[0] / stats[1] if len(stats) == 2 and stats[1] > 0 else math.nan
    if len(stats) != 3 or stats[0] == 0:
        return math.nan
    n, total, squares = stats
    return total / n if stat == "mean" else math.sqrt(squares / n)


def select_rows(rows: Sequence[Row], split: str, metric: str) -> list[Row]:
    return [r for r in rows if r["split"] == split and r["metric"] == metric]


def pooled(rows: Sequence[Row], kind: Kind, quantity: str) -> Stats:
    return summed([row[kind][quantity] for row in rows if quantity in row[kind]])


def by_scene(rows: Sequence[Row], kind: Kind, quantity: str) -> dict[str, Stats]:
    scenes = sorted({r["scene"] for r in rows})
    return {s: pooled([r for r in rows if r["scene"] == s], kind, quantity) for s in scenes}


@dataclass(frozen=True)
class SceneDelta:
    """candidate - base of one statistic over the scenes both share, with a scene-bootstrap
    interval (None bounds when it could not be computed)."""

    scenes: int
    point: float
    low: float | None
    high: float | None


@dataclass(frozen=True)
class DeltaQuery:
    split: str
    metric: str
    quantity: str
    stat: Stat


def scene_delta(
    base: Sequence[Row], candidate: Sequence[Row], query: DeltaQuery, n: int, seed: int, ci: float
) -> SceneDelta:
    kind = kind_of(query.stat)
    left = by_scene(select_rows(base, query.split, query.metric), kind, query.quantity)
    right = by_scene(select_rows(candidate, query.split, query.metric), kind, query.quantity)
    blocks = [
        SceneBlock(clip_id=s, item=(left[s], right[s])) for s in sorted(set(left) & set(right))
    ]

    def statistic(items: Sequence[tuple[Stats, Stats]]) -> dict[str, float]:
        b = value(summed([i[0] for i in items]), query.stat)
        c = value(summed([i[1] for i in items]), query.stat)
        return {DELTA: c - b}

    point = statistic([b.item for b in blocks])[DELTA] if blocks else math.nan
    interval = scene_bootstrap(blocks, statistic, n, seed, ci).get(DELTA) if blocks else None
    return SceneDelta(
        scenes=len(blocks),
        point=point,
        low=interval.low if interval is not None else None,
        high=interval.high if interval is not None else None,
    )


TABLE_HEADER = (
    "| label | split | metric | quantity | n | mean | rms / rate | per scene |",
    "|---|---|---|---|---|---|---|---|",
)


def table_lines(collected: Collected) -> list[str]:
    lines: list[str] = []
    for split, metric in sorted({(r["split"], r["metric"]) for r in collected.rows}):
        rows = select_rows(collected.rows, split, metric)
        for kind in ("samples", "ratios"):
            for quantity in sorted({q for r in rows for q in r[kind]}):
                lines.append(table_line(collected.label, split, metric, kind, quantity, rows))
    return lines


def table_line(
    label: str, split: str, metric: str, kind: Kind, quantity: str, rows: Sequence[Row]
) -> str:
    stat: Stat = "rate" if kind == "ratios" else "rms"
    total = pooled(rows, kind, quantity)
    per_scene = ", ".join(
        f"{scene}: {value(stats, stat):.4f}"
        for scene, stats in by_scene(rows, kind, quantity).items()
    )
    n = f"{total[0]:.0f}" if kind == "samples" else f"{total[0]:.0f}/{total[1]:.0f}"
    mean = f"{value(total, 'mean'):.4f}" if kind == "samples" else ""
    return (
        f"| {label} | {split} | {metric} | {quantity} | {n} | {mean} | "
        f"{value(total, stat):.4f} | {per_scene} |"
    )

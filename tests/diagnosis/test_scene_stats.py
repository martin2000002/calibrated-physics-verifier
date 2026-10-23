from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from cli_runner import CliRunner

from physverify.diagnosis import scene_stats
from physverify.diagnosis.measurement_cases import MetricOutcome


def row(
    scene: str, rms_parts: list[float], ratio: list[float], split: str = "eval"
) -> dict[str, Any]:
    return {
        "split": split,
        "metric": "m/x",
        "provider": "p",
        "clip_id": f"{scene}__none",
        "scene": scene,
        "samples": {"q": rms_parts},
        "ratios": {"r": ratio},
    }


def test_sufficient_drops_non_finite_samples() -> None:
    outcome = MetricOutcome(
        samples=MappingProxyType({"q": np.array([1.0, np.nan, -2.0, np.inf])}),
        ratios=MappingProxyType({"r": (3.0, 4.0)}),
    )
    samples, ratios = scene_stats.sufficient(outcome)
    assert samples == {"q": [2.0, -1.0, 5.0]}
    assert ratios == {"r": [3.0, 4.0]}


def test_summed_adds_elementwise_and_skips_empty_parts() -> None:
    assert scene_stats.summed([[1.0, 2.0, 3.0], [], [4.0, 5.0, 6.0]]) == [5.0, 7.0, 9.0]
    assert scene_stats.summed([[], []]) == []


@pytest.mark.parametrize(
    ("stats", "stat", "expected"),
    [
        ([2.0, -1.0, 5.0], "mean", -0.5),
        ([2.0, -1.0, 5.0], "rms", math.sqrt(2.5)),
        ([3.0, 4.0], "rate", 0.75),
    ],
)
def test_value(stats: list[float], stat: scene_stats.Stat, expected: float) -> None:
    assert scene_stats.value(stats, stat) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("stats", "stat"), [([], "rms"), ([0.0, 0.0, 0.0], "mean"), ([1.0, 0.0], "rate"), ([], "rate")]
)
def test_value_is_nan_when_undefined(stats: list[float], stat: scene_stats.Stat) -> None:
    assert math.isnan(scene_stats.value(stats, stat))


def test_parse_stat_rejects_unknown_names() -> None:
    assert scene_stats.parse_stat("rate") == "rate"
    with pytest.raises(ValueError, match="unknown stat"):
        scene_stats.parse_stat("median")


def test_pooled_rms_is_the_rms_over_every_scene_sample() -> None:
    rows = [row("a", [1.0, 1.0, 1.0], [1.0, 2.0]), row("b", [1.0, 3.0, 9.0], [2.0, 2.0])]
    assert scene_stats.value(scene_stats.pooled(rows, "samples", "q"), "rms") == pytest.approx(
        math.sqrt(5.0)
    )
    assert scene_stats.value(scene_stats.pooled(rows, "ratios", "r"), "rate") == 0.75
    assert scene_stats.by_scene(rows, "ratios", "r") == {"a": [1.0, 2.0], "b": [2.0, 2.0]}


QUERY = scene_stats.DeltaQuery(split="eval", metric="m/x", quantity="r", stat="rate")


def test_scene_delta_point_and_interval() -> None:
    base = [row(s, [], [1.0, 2.0]) for s in ("a", "b", "c")]
    cand = [row("a", [], [2.0, 2.0]), row("b", [], [2.0, 2.0]), row("c", [], [1.0, 2.0])]
    result = scene_stats.scene_delta(base, cand, QUERY, n=500, seed=0, ci=0.95)
    assert result.scenes == 3
    assert result.point == pytest.approx(2.0 / 6.0)
    assert result.low is not None and result.high is not None
    assert 0.0 <= result.low <= result.point <= result.high <= 0.5


def test_scene_delta_uses_only_shared_scenes_of_the_split() -> None:
    base = [row("a", [], [1.0, 2.0]), row("z", [], [0.0, 2.0]), row("a", [], [0.0, 9.0], "calib")]
    cand = [row("a", [], [2.0, 2.0])]
    result = scene_stats.scene_delta(base, cand, QUERY, n=50, seed=0, ci=0.95)
    assert result.scenes == 1
    assert result.point == pytest.approx(0.5)


def test_scene_delta_without_shared_scenes_is_undefined() -> None:
    result = scene_stats.scene_delta([row("a", [], [1.0, 2.0])], [], QUERY, n=50, seed=0, ci=0.95)
    assert result.scenes == 0
    assert math.isnan(result.point)
    assert result.low is None and result.high is None


def test_save_load_round_trip_and_table(tmp_path: Path) -> None:
    collected = scene_stats.Collected(
        label="A-fast",
        profile="fast",
        overrides=("a=1",),
        rows=(row("a", [2.0, 2.0, 4.0], [1.0, 2.0]),),
    )
    path = tmp_path / "out" / "a.json"
    scene_stats.save(path, collected)
    assert scene_stats.load(path) == collected
    lines = scene_stats.table_lines(collected)
    assert lines == [
        "| A-fast | eval | m/x | q | 2 | 1.0000 | 1.4142 | a: 1.4142 |",
        "| A-fast | eval | m/x | r | 1/2 |  | 0.5000 | a: 0.5000 |",
    ]


def test_delta_command(cli: CliRunner, tmp_path: Path) -> None:
    def write(name: str, rows: list[dict[str, Any]]) -> Path:
        path = tmp_path / name
        scene_stats.save(path, scene_stats.Collected(name, None, (), tuple(rows)))
        return path

    base = write("base.json", [row("a", [], [1.0, 2.0])])
    cand = write("cand.json", [row("a", [], [2.0, 2.0])])
    args = ("scene-stats", "delta", str(base), str(cand), "--metric", "m/x", "--quantity", "r")
    result = cli(*args, "--stat", "rate", "--n", "20")
    assert result.code == 0, result.err
    assert "delta +0.5000" in result.out
    assert cli(*args, "--stat", "median").code == 2

"""Monotonicity: Kendall tau_b of level vs the change against the clean clip."""

from __future__ import annotations

import numpy as np
import pytest
from calibration_cases import row, settings

from physverify.calibration.monotonicity import deltas, monotonicity, scene_kendall
from physverify.calibration.table import CalibrationDataError, CalibrationRow, Role

LEVELS = (1, 2, 3, 4)


def scene_rows(scene: str, clean: float, values: list[float], check_id: str = "R1") -> list:  # type: ignore[type-arg]
    found: list[CalibrationRow] = [row(clean, scene=scene, check_id=check_id)]
    found += [
        row(v, role=Role.POSITIVE, scene=scene, level=lv, mu=0.1 * lv, check_id=check_id)
        for lv, v in zip(LEVELS, values, strict=True)
    ]
    return found


def test_increasing_changes_give_tau_one_and_pass_monotonicity() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for s in range(10):
        clean = float(rng.normal(0.0, 0.01))
        rows += scene_rows(f"s{s}", clean, [clean + 0.1 * lv for lv in LEVELS])
    pairs = deltas(rows, 0.0)
    assert len(pairs) == 50 and {p.delta for p in pairs if p.level == 0} == {0.0}
    found = monotonicity(pairs, settings())
    assert found.kendall_tau_b == pytest.approx(1.0)
    assert found.h1_pass and found.monotone_medians
    assert found.n_scenes == 10


def test_random_changes_do_not_pass_monotonicity() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for s in range(10):
        rows += scene_rows(f"s{s}", 0.0, list(rng.normal(0.0, 1.0, len(LEVELS))))
    found = monotonicity(deltas(rows, 0.0), settings())
    assert found.ci.low is not None and found.ci.high is not None
    assert found.ci.low <= 0.0 <= found.ci.high
    assert not found.h1_pass


def test_a_scene_without_a_scored_clean_clip_is_left_out() -> None:
    rows = scene_rows("s0", 0.0, [0.1, 0.2, 0.3, 0.4])
    rows[0] = row(None, scene="s0")
    assert deltas(rows, 0.0) == ()


@pytest.mark.parametrize(
    ("check_id", "clean", "values"),
    [
        ("R5b", 0.000, [0.050, 0.100, 0.200, 0.400]),
        ("R3a", -0.102, [0.047, 0.097, 0.247, 0.496]),
        ("R1", 0.012, [0.050, 0.100, 0.200, 0.400]),
    ],
)
def test_the_levels_sweep_values_give_tau_one_per_scene(
    check_id: str, clean: float, values: list[float]
) -> None:
    """The oracle `levels` magnitudes of an earlier oracle sweep (one scene per perturbation)."""
    (found,) = scene_kendall(scene_rows("scene", clean, values, check_id), 0.0)
    assert found.kendall_tau_b == pytest.approx(1.0)
    assert (found.check_id, found.n_levels) == (check_id, 4)


def test_two_clean_clips_of_one_scene_are_rejected() -> None:
    rows = [*scene_rows("s0", 0.0, [0.1, 0.2, 0.3, 0.4]), row(0.0, scene="s0", clip_id="dup")]
    with pytest.raises(CalibrationDataError):
        deltas(rows, 0.0)

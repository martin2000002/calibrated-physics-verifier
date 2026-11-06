"""Out-of-fold contact-row noise: folds by scene index, band quantile (or Student t) of the
other folds; the match sigma on raw row residuals, the rate noise (row and size sigmas and
their autocorrelations) on series centred within each clip and body; the full biased
autocorrelation keeps every row Toeplitz positive semidefinite."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from physverify.calibration.row_sigma import (
    ROW_SIGMA_FOLDS,
    ROW_SIGMA_VERSION,
    ClipResiduals,
    RowLaw,
    RowNoise,
    RowSigma,
    autocorrelation,
    fit_row_sigma,
    fold_of,
    law_sigma,
)
from physverify.calibration.table import CalibrationDataError
from physverify.checks.separate._pass_through_noise import lag_correlation
from physverify.checks.separate.r1_dyn_pass_through import ROW_COVERAGE, TAIL_DOF, band_z
from physverify.core.errors import ConfigError
from physverify.core.jsonio import read_model
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.stages import STAGE_ORDER

SIZE_SHARE = 0.01
# Positive semidefinite up to rounding, fixed before running.
PSD_FLOOR = -1e-12
PSD_SIZES = range(2, 201)
PSD_SERIES = 60
# Long enough for the fitted lags to approach the true ones, where a cut lag breaks the Toeplitz.
PSD_ROWS = (24, 200)
PSD_GAPS = 0.10
PSD_REPLICATES = 10
TRIANGLE_SUPPORT = 7
AR_PHI = 0.8

Process = Callable[[np.random.Generator, int], np.ndarray]


def fitted(
    residuals: dict[str, list[float]], scenes: dict[str, str], law: RowLaw = RowLaw.QUANTILE
) -> RowSigma:
    return fit_row_sigma(
        {
            clip: ClipResiduals((np.asarray(values),), (SIZE_SHARE * np.asarray(values),))
            for clip, values in residuals.items()
        },
        scenes,
        check_id="R1_dyn",
        level=InputLevel.PERCEIVED,
        gt_stages=(),
        law=law,
    )


def quantile_sigma(values: list[float]) -> float:
    return float(np.quantile(np.abs(values), ROW_COVERAGE)) / band_z()


def clip(split: Split, scene_id: str | None) -> ClipManifest:
    return ClipManifest(
        dataset="kubric_synthetic",
        clip_id="c",
        split=split,
        video_path="kubric_synthetic/c",
        n_frames=4,
        width=8,
        height=8,
        container_fps=24.0,
        seconds_per_frame=1.0 / 24.0,
        time_source=TimeSource.GROUND_TRUTH,
        gt_stages=STAGE_ORDER,
        scene_id=scene_id,
    )


@pytest.mark.parametrize(
    ("scene_id", "fold"),
    [("f5_contact-s000", 0), ("f5_contact-s007", 2), ("f1-s054", 4), ("s003", None), (None, None)],
)
def test_the_fold_is_the_scene_index_mod_five(scene_id: str | None, fold: int | None) -> None:
    assert ROW_SIGMA_FOLDS == 5
    assert fold_of(scene_id) == fold


def test_each_fold_is_fitted_on_the_other_folds_only() -> None:
    values = {j: [float(j), 3.0 * j, -2.0 * j + 1.0, 0.5 * j * j] for j in range(5)}
    residuals = {f"c{j}": values[j] for j in range(5)}
    scenes = {f"c{j}": f"f5_contact-s{j + 5:03d}" for j in range(5)}
    result = fitted(residuals, scenes)
    for j in range(5):
        others = [v for i in range(5) if i != j for v in values[i]]
        assert result.by_fold[j].row_sigma == pytest.approx(quantile_sigma(others), rel=1e-9)
    assert result.pooled.row_sigma == pytest.approx(
        quantile_sigma([v for i in range(5) for v in values[i]]), rel=1e-9
    )
    assert result.residuals_per_fold == (4, 4, 4, 4, 4)
    assert result.scenes_per_fold[2] == ("f5_contact-s007",)


def test_bodies_and_clips_of_one_fold_are_pooled() -> None:
    residuals = {"a": [1.0, 2.0], "b": [3.0], "c": [10.0, 20.0, 30.0]}
    scenes = {"a": "f-s001", "b": "f-s006", "c": "f-s002"}
    result = fitted(residuals, scenes)
    assert result.residuals_per_fold == (0, 3, 3, 0, 0)
    assert result.by_fold[1].row_sigma == pytest.approx(
        quantile_sigma([10.0, 20.0, 30.0]), rel=1e-9
    )
    assert result.by_fold[0].row_sigma == pytest.approx(
        quantile_sigma([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), rel=1e-9
    )


def test_a_fold_value_does_not_move_its_own_sigma() -> None:
    base = {f"c{j}": [float(j), 2.0 * j + 1.0, -float(j)] for j in range(5)}
    scenes = {f"c{j}": f"f-s{j:03d}" for j in range(5)}
    moved = {**base, "c3": [1e6, -1e6, 5e5]}
    assert fitted(moved, scenes).by_fold[3] == fitted(base, scenes).by_fold[3]
    assert fitted(moved, scenes).by_fold[0].row_sigma != fitted(base, scenes).by_fold[0].row_sigma


def test_clips_without_scene_or_residuals_are_listed_and_left_out() -> None:
    residuals = {"a": [1.0, 2.0, 4.0], "b": [], "c": [100.0], "d": [0.5, 1.5]}
    result = fitted(residuals, {"a": "f-s001", "b": "f-s002", "d": "f-s003"})
    assert result.clips_without_residuals == ("b", "c")
    assert result.residuals_per_fold == (0, 3, 0, 2, 0)


@pytest.mark.parametrize(
    "residuals", [{}, {"a": [1.0, 2.0, 4.0]}], ids=["no_residuals", "one_fold_only"]
)
def test_a_fold_without_residuals_outside_it_is_refused(residuals: dict[str, list[float]]) -> None:
    with pytest.raises(CalibrationDataError, match="no residuals outside"):
        fitted(residuals, {"a": "f-s001"})


def test_calib_clips_take_their_fold_sigma_and_others_the_pooled_one() -> None:
    result = fitted(
        {f"c{j}": [float(j), 2.0 * j, 7.0] for j in range(5)},
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
    )
    assert result.noise_of(clip(Split.CALIB, "f-s013")) == result.by_fold[3]
    assert result.noise_of(clip(Split.EVAL, "f-s013")) == result.pooled
    assert result.noise_of(clip(Split.EVAL, None)) == result.pooled
    assert isinstance(result.pooled, RowNoise) and result.pooled.row_tail_dof is None


def test_a_calib_clip_without_a_scene_index_is_refused_a_sigma() -> None:
    result = fitted(
        {f"c{j}": [float(j), 2.0 * j, 7.0] for j in range(5)},
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
    )
    with pytest.raises(CalibrationDataError, match="no scene index"):
        result.noise_of(clip(Split.CALIB, None))


def test_the_artifact_round_trips() -> None:
    result = fitted({"a": [1.0, 2.0, 4.0], "b": [0.5, 3.0]}, {"a": "f-s001", "b": "f-s002"})
    again = RowSigma.model_validate_json(result.model_dump_json(by_alias=True))
    assert again == result and again.level is InputLevel.PERCEIVED


def test_the_rate_noise_is_fitted_on_series_centred_within_each_clip() -> None:
    values = {j: [float(j), 3.0 * j, -2.0 * j + 1.0] for j in range(5)}
    result = fitted(
        {f"c{j}": values[j] for j in range(5)}, {f"c{j}": f"f-s{j:03d}" for j in range(5)}
    )
    for j, fold in enumerate(result.by_fold):
        others = [v - np.mean(values[i]) for i in range(5) if i != j for v in values[i]]
        assert fold.rate_row_sigma == pytest.approx(quantile_sigma(others), rel=1e-9)
        assert fold.size_sigma == pytest.approx(SIZE_SHARE * fold.rate_row_sigma, rel=1e-9)
    assert result.size_residuals_per_fold == result.residuals_per_fold


def test_a_constant_offset_moves_the_match_sigma_but_not_the_rate_noise() -> None:
    base = {f"c{j}": [float(j), 2.0 * j + 1.0, -float(j), 0.5] for j in range(5)}
    scenes = {f"c{j}": f"f-s{j:03d}" for j in range(5)}
    shifted = {clip: [v + 40.0 for v in values] for clip, values in base.items()}
    moved, kept = fitted(shifted, scenes).pooled, fitted(base, scenes).pooled
    assert moved.row_sigma > kept.row_sigma
    assert moved.rate_row_sigma == pytest.approx(kept.rate_row_sigma, rel=1e-9)
    assert moved.row_autocorrelation == pytest.approx(kept.row_autocorrelation, rel=1e-9)


def test_the_student_t_law_is_its_band_half_width_over_the_normal_z() -> None:
    values = np.array([-3.0, -1.0, -0.5, 0.0, 0.2, 0.7, 1.5, 4.0, -0.1, 0.4])
    _, _, scale = stats.t.fit(values, f0=TAIL_DOF, floc=0.0)
    half = float(stats.t.ppf(0.5 + ROW_COVERAGE / 2.0, TAIL_DOF)) * scale
    assert law_sigma(values, RowLaw.STUDENT_T) == pytest.approx(half / band_z(), rel=1e-9)
    result = fitted(
        {f"c{j}": list(values + j) for j in range(5)},
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
        RowLaw.STUDENT_T,
    )
    assert result.pooled.row_tail_dof == TAIL_DOF
    assert result.noise_of(clip(Split.EVAL, None)).row_tail_dof == TAIL_DOF


def test_the_autocorrelation_is_the_biased_estimator_over_every_lag() -> None:
    series = np.array([1.0, 1.0, -1.0, -1.0] * 2)
    full = np.correlate(series, series, "full")[series.size :] / float(series @ series)
    assert autocorrelation([series]) == pytest.approx(tuple(full), rel=1e-12)
    assert autocorrelation([np.array([1.0, -1.0] * 4)])[0] == pytest.approx(-7.0 / 8.0)
    assert autocorrelation([np.zeros(6)]) == ()
    assert autocorrelation([np.array([1.0, np.nan])]) is None
    assert len(autocorrelation([np.arange(40.0), np.arange(9.0)]) or ()) == 39


def test_unread_rows_count_as_zero_in_the_autocorrelation() -> None:
    gappy = np.array([2.0, np.nan, -1.0, 0.5, -1.5])
    filled = np.nan_to_num(gappy - np.nanmean(gappy))
    full = np.correlate(filled, filled, "full")[filled.size :] / float(filled @ filled)
    assert autocorrelation([gappy]) == pytest.approx(tuple(full), rel=1e-12)


def white(rng: np.random.Generator, rows: int) -> np.ndarray:
    out: np.ndarray = rng.standard_normal(rows)
    return out


def triangle(rng: np.random.Generator, rows: int) -> np.ndarray:
    """Moving sum over ``TRIANGLE_SUPPORT`` rows: r_k = 1 - k / support up to the support."""
    out: np.ndarray = np.convolve(
        rng.standard_normal(rows + TRIANGLE_SUPPORT - 1), np.ones(TRIANGLE_SUPPORT), "valid"
    )
    return out


def ar1(rng: np.random.Generator, rows: int) -> np.ndarray:
    out = np.empty(rows)
    out[0] = rng.standard_normal() / np.sqrt(1.0 - AR_PHI**2)
    for t in range(1, rows):
        out[t] = AR_PHI * out[t - 1] + rng.standard_normal()
    return out


def gappy_series(rng: np.random.Generator, process: Process) -> list[np.ndarray]:
    """``PSD_SERIES`` series of ``PSD_ROWS`` rows with a share ``PSD_GAPS`` of unread rows."""
    out = []
    for _ in range(PSD_SERIES):
        series = process(rng, int(rng.integers(PSD_ROWS[0], PSD_ROWS[1] + 1)))
        series[rng.random(series.size) < PSD_GAPS] = np.nan
        out.append(series)
    return out


@pytest.mark.parametrize("process", [white, ar1, triangle], ids=["white", "ar1", "triangle"])
def test_the_fitted_lags_give_a_positive_semidefinite_toeplitz(process: Process) -> None:
    rng = np.random.default_rng(7)
    for _ in range(PSD_REPLICATES):
        lags = autocorrelation(gappy_series(rng, process))
        assert lags is not None
        lowest = min(
            float(np.linalg.eigvalsh(lag_correlation(np.arange(size), lags))[0])
            for size in PSD_SIZES
        )
        assert lowest >= PSD_FLOOR


def test_a_row_sigma_of_another_major_version_is_refused(tmp_path: Path) -> None:
    result = fitted(
        {f"c{j}": [float(j), 2.0 * j + 1.0, -float(j)] for j in range(5)},
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
    )
    assert result.version == ROW_SIGMA_VERSION == "4.0.0"
    path = tmp_path / "row_sigma.json"
    path.write_text(result.to_json(), encoding="utf-8")
    assert read_model(RowSigma, path) == result
    data = json.loads(result.to_json())
    for version in ("3.0.0", "5.0.0", "four"):
        with pytest.raises(ValidationError, match="version"):
            RowSigma.model_validate({**data, "version": version})
    path.write_text(json.dumps({**data, "version": "3.0.0"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="version"):
        read_model(RowSigma, path)


def test_the_size_autocorrelation_is_measured_apart_from_the_rows() -> None:
    rows = np.array([1.0, -1.0] * 6)
    sizes = np.array([1.0, 1.0, -1.0, -1.0] * 3)
    pieces = {f"c{j}": ClipResiduals((rows + j,), (0.01 * sizes,)) for j in range(5)}
    result = fit_row_sigma(
        pieces,
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
        check_id="R1_dyn",
        level=InputLevel.PERCEIVED,
        gt_stages=(),
    )
    assert result.pooled.row_autocorrelation == autocorrelation([rows] * 5)
    assert result.pooled.size_autocorrelation == pytest.approx(autocorrelation([sizes] * 5))
    assert result.pooled.row_autocorrelation[0] < 0.0 < result.pooled.size_autocorrelation[0]


def test_unread_rows_split_the_series_without_entering_the_sigma() -> None:
    series = np.array([1.0, np.nan, 2.0, -1.0])
    assert np.array_equal(ClipResiduals((series,)).flat_rows, [1.0, 2.0, -1.0])
    assert ClipResiduals((np.full(3, np.nan),)).empty

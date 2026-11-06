"""Out-of-fold contact-row noise: folds by scene index, band quantile (or Student t) of the
other folds; the match sigma on raw row residuals, the rate noise (row and size sigmas and
their autocorrelations) on series centred within each clip and body."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from physverify.calibration.row_sigma import (
    AUTOCORRELATION_MAX_LAG,
    ROW_SIGMA_FOLDS,
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
from physverify.checks.separate.r1_dyn_pass_through import ROW_COVERAGE, TAIL_DOF, band_z
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.stages import STAGE_ORDER

SIZE_SHARE = 0.01


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


def test_the_autocorrelation_keeps_lags_until_the_first_non_positive_one() -> None:
    assert autocorrelation([np.array([1.0, 1.0, -1.0, -1.0] * 2)]) == pytest.approx((0.125,))
    assert autocorrelation([np.array([1.0, -1.0] * 4)]) == ()
    assert autocorrelation([np.zeros(6)]) == ()
    assert autocorrelation([np.array([1.0, np.nan])]) is None
    ramp = np.arange(40.0)
    assert len(autocorrelation([ramp])) == AUTOCORRELATION_MAX_LAG


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
    assert result.pooled.row_autocorrelation == ()
    assert result.pooled.size_autocorrelation == autocorrelation([sizes] * 5)
    assert len(result.pooled.size_autocorrelation) == 1


def test_unread_rows_split_the_series_without_entering_the_sigma() -> None:
    series = np.array([1.0, np.nan, 2.0, -1.0])
    assert np.array_equal(ClipResiduals((series,)).flat_rows, [1.0, 2.0, -1.0])
    assert ClipResiduals((np.full(3, np.nan),)).empty

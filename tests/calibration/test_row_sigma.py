"""Out-of-fold contact-row noise: folds by scene index, band quantile (or Student t) of the
other folds for the row and size sigmas, autocorrelation factor of the row series."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from physverify.calibration.row_sigma import (
    ROW_SIGMA_FOLDS,
    ClipResiduals,
    RowLaw,
    RowNoise,
    RowSigma,
    autocorrelation_factor,
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
            clip: ClipResiduals((np.asarray(values),), SIZE_SHARE * np.asarray(values))
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
        assert result.by_fold[j] == pytest.approx(quantile_sigma(others), rel=1e-9)
    assert result.pooled == pytest.approx(
        quantile_sigma([v for i in range(5) for v in values[i]]), rel=1e-9
    )
    assert result.residuals_per_fold == (4, 4, 4, 4, 4)
    assert result.scenes_per_fold[2] == ("f5_contact-s007",)


def test_bodies_and_clips_of_one_fold_are_pooled() -> None:
    residuals = {"a": [1.0, 2.0], "b": [3.0], "c": [10.0, 20.0, 30.0]}
    scenes = {"a": "f-s001", "b": "f-s006", "c": "f-s002"}
    result = fitted(residuals, scenes)
    assert result.residuals_per_fold == (0, 3, 3, 0, 0)
    assert result.by_fold[1] == pytest.approx(quantile_sigma([10.0, 20.0, 30.0]), rel=1e-9)
    assert result.by_fold[0] == pytest.approx(
        quantile_sigma([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), rel=1e-9
    )


def test_a_fold_value_does_not_move_its_own_sigma() -> None:
    base = {f"c{j}": [float(j), 2.0 * j + 1.0, -float(j)] for j in range(5)}
    scenes = {f"c{j}": f"f-s{j:03d}" for j in range(5)}
    moved = {**base, "c3": [1e6, -1e6, 5e5]}
    assert fitted(moved, scenes).by_fold[3] == fitted(base, scenes).by_fold[3]
    assert fitted(moved, scenes).by_fold[0] != fitted(base, scenes).by_fold[0]


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
    assert result.noise_of(clip(Split.CALIB, "f-s013")) == RowNoise(
        result.by_fold[3], result.size_by_fold[3], result.autocorrelation_by_fold[3], None
    )
    pooled = RowNoise(result.pooled, result.size_pooled, result.autocorrelation_pooled, None)
    assert result.noise_of(clip(Split.EVAL, "f-s013")) == pooled
    assert result.noise_of(clip(Split.EVAL, None)) == pooled


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
    assert again.by_fold[0] == result.by_fold[0] and again.level is InputLevel.PERCEIVED


def test_the_size_sigma_follows_the_same_law_on_the_size_residuals() -> None:
    values = {j: [float(j), 3.0 * j, -2.0 * j + 1.0] for j in range(5)}
    result = fitted(
        {f"c{j}": values[j] for j in range(5)}, {f"c{j}": f"f-s{j:03d}" for j in range(5)}
    )
    assert result.size_by_fold == pytest.approx([SIZE_SHARE * s for s in result.by_fold], rel=1e-9)
    assert result.size_residuals_per_fold == result.residuals_per_fold


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
    assert result.tail_dof == TAIL_DOF
    assert result.noise_of(clip(Split.EVAL, None)).row_tail_dof == TAIL_DOF


def test_the_autocorrelation_sums_lags_until_the_first_non_positive_one() -> None:
    assert autocorrelation_factor([np.array([1.0, 1.0, -1.0, -1.0] * 2)]) == pytest.approx(1.25)
    assert autocorrelation_factor([np.array([1.0, -1.0] * 4)]) == 1.0
    assert autocorrelation_factor([np.zeros(6)]) == 1.0
    assert np.isnan(autocorrelation_factor([np.array([1.0, np.nan])]))


def test_unread_rows_split_the_series_without_entering_the_sigma() -> None:
    series = np.array([1.0, np.nan, 2.0, -1.0])
    assert np.array_equal(ClipResiduals((series,)).flat_rows, [1.0, 2.0, -1.0])
    assert ClipResiduals((np.full(3, np.nan),)).empty

"""Out-of-fold contact-row sigma: folds by scene index, MAD scale of the other folds."""

from __future__ import annotations

import numpy as np
import pytest

from physverify.calibration.row_sigma import ROW_SIGMA_FOLDS, RowSigma, fit_row_sigma, fold_of
from physverify.calibration.table import CalibrationDataError
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.stages import STAGE_ORDER

MAD_TO_SIGMA = 1.4826


def fitted(residuals: dict[str, list[float]], scenes: dict[str, str]) -> RowSigma:
    return fit_row_sigma(
        {clip: np.asarray(values) for clip, values in residuals.items()},
        scenes,
        check_id="R1_dyn",
        level=InputLevel.PERCEIVED,
        gt_stages=(),
    )


def mad_sigma(values: list[float]) -> float:
    array = np.asarray(values)
    return MAD_TO_SIGMA * float(np.median(np.abs(array - np.median(array))))


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
        assert result.by_fold[j] == pytest.approx(mad_sigma(others), rel=1e-4)
    assert result.pooled == pytest.approx(
        mad_sigma([v for i in range(5) for v in values[i]]), rel=1e-4
    )
    assert result.residuals_per_fold == (4, 4, 4, 4, 4)
    assert result.scenes_per_fold[2] == ("f5_contact-s007",)


def test_bodies_and_clips_of_one_fold_are_pooled() -> None:
    residuals = {"a": [1.0, 2.0], "b": [3.0], "c": [10.0, 20.0, 30.0]}
    scenes = {"a": "f-s001", "b": "f-s006", "c": "f-s002"}
    result = fitted(residuals, scenes)
    assert result.residuals_per_fold == (0, 3, 3, 0, 0)
    assert result.by_fold[1] == pytest.approx(mad_sigma([10.0, 20.0, 30.0]), rel=1e-4)
    assert result.by_fold[0] == pytest.approx(
        mad_sigma([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), rel=1e-4
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
    assert result.sigma_of(clip(Split.CALIB, "f-s013")) == result.by_fold[3]
    assert result.sigma_of(clip(Split.EVAL, "f-s013")) == result.pooled
    assert result.sigma_of(clip(Split.EVAL, None)) == result.pooled


def test_a_calib_clip_without_a_scene_index_is_refused_a_sigma() -> None:
    result = fitted(
        {f"c{j}": [float(j), 2.0 * j, 7.0] for j in range(5)},
        {f"c{j}": f"f-s{j:03d}" for j in range(5)},
    )
    with pytest.raises(CalibrationDataError, match="no scene index"):
        result.sigma_of(clip(Split.CALIB, None))


def test_the_artifact_round_trips() -> None:
    result = fitted({"a": [1.0, 2.0, 4.0], "b": [0.5, 3.0]}, {"a": "f-s001", "b": "f-s002"})
    again = RowSigma.model_validate_json(result.model_dump_json(by_alias=True))
    assert again.by_fold[0] == result.by_fold[0] and again.level is InputLevel.PERCEIVED

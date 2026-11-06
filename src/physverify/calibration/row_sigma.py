"""Out-of-fold noise of the R1_dyn contact row and image size: each body's lowest-point row and
image size read from perceived inputs against the same read from ground truth, on clean
calibration clips.

Scenes go to fold = scene index mod ``ROW_SIGMA_FOLDS``; every quantity of fold j is fitted on
the residuals of every other fold, pooled over bodies and rows; ``pooled`` uses every fold, for
clips outside calib. A sigma is the half-width of the two-sided ``ROW_COVERAGE`` band of the
law over the normal z of that band: the band is the |r| quantile (``quantile``) or the Student
t with ``TAIL_DOF`` dof, location 0 and maximum-likelihood scale (``student_t``).

The row match keeps each clip's offset (it moves the band), so its sigma is fitted on the raw
row residuals. A constant offset does not move a rate, so the rate noise (row sigma, relative
size sigma and their autocorrelations) is fitted on each body's series centred on its own mean
within the clip. The autocorrelations r_1, r_2, ... of the row and of the size series are
measured apart, from lag 1 until the first r_k <= 0 or lag ``AUTOCORRELATION_MAX_LAG``. Pure,
no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from physverify.calibration.table import CalibrationDataError
from physverify.checks._base import CheckInputs
from physverify.checks._image import FloatArray, camera_k
from physverify.checks.separate.r1_dyn_pass_through import (
    ROW_COVERAGE,
    TAIL_DOF,
    band_z,
    image_track,
)
from physverify.core.models import SCHEMA_KEY, ArtifactModel
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.matching import IdMap
from physverify.synth.scene_scope import scene_index

ROW_SIGMA_FOLDS: Final = 5
ROW_SIGMA_SCHEMA: Final = "physverify.row_sigma"
ROW_SIGMA_VERSION: Final = "3.0.0"
ROW_SIGMA_FILE: Final = "row_sigma.json"
AUTOCORRELATION_MAX_LAG: Final = 5


class RowLaw(StrEnum):
    QUANTILE = "quantile"
    STUDENT_T = "student_t"


DEFAULT_ROW_LAW: Final = RowLaw.QUANTILE


class RowNoise(BaseModel):
    """The R1_dyn parameters one clip runs with (field names are the check's ``Params``), one
    fit of a fold or the pooled one: ``row_sigma`` of the row match (raw residuals, pixels);
    ``rate_row_sigma`` (pixels), ``size_sigma`` (relative) and the lags r_1, r_2, ... of the
    rates' noise (centred series); ``row_tail_dof`` of the band law."""

    model_config = ConfigDict(frozen=True)

    row_sigma: float
    rate_row_sigma: float
    size_sigma: float
    row_autocorrelation: tuple[float, ...]
    size_autocorrelation: tuple[float, ...]
    row_tail_dof: float | None

    def params(self) -> dict[str, object]:
        return self.model_dump()


@dataclass(frozen=True)
class ClipResiduals:
    """Per matched body and on every perceived row (NaN where either side has none), perceived
    minus true contact row and the relative image-size residual (perceived - true) / true."""

    rows: tuple[FloatArray, ...] = ()
    sizes: tuple[FloatArray, ...] = ()

    @property
    def flat_rows(self) -> FloatArray:
        return _flat(self.rows)

    @property
    def flat_sizes(self) -> FloatArray:
        return _flat(self.sizes)

    @property
    def empty(self) -> bool:
        return not self.flat_rows.size


def _flat(series: tuple[FloatArray, ...]) -> FloatArray:
    values = np.concatenate(series) if series else np.empty(0)
    return values[np.isfinite(values)]


def centred(series: Sequence[FloatArray]) -> list[FloatArray]:
    """Each series minus its own mean (NaN rows skipped); series with fewer than two finite
    values carry no within-clip spread and are left out."""
    return [s - np.nanmean(s) for s in series if np.isfinite(s).sum() >= 2]


class RowSigma(ArtifactModel):
    """``by_fold[j]`` is the noise of clips of fold j, fitted without fold j."""

    schema_: str = Field(default=ROW_SIGMA_SCHEMA, alias=SCHEMA_KEY)
    version: str = ROW_SIGMA_VERSION
    check_id: str
    level: InputLevel
    gt_stages: tuple[str, ...]
    law: RowLaw
    coverage: float
    by_fold: tuple[RowNoise, ...]
    pooled: RowNoise
    residuals_per_fold: tuple[int, ...]
    size_residuals_per_fold: tuple[int, ...]
    scenes_per_fold: tuple[tuple[str, ...], ...]
    clips_without_residuals: tuple[str, ...]

    def noise_of(self, clip: ClipManifest) -> RowNoise:
        """A calib clip takes the noise of its scene's fold, any other clip the pooled noise.
        Refused for a calib clip without a scene index: it has no fold to stay out of."""
        if clip.split is not Split.CALIB:
            return self.pooled
        fold = fold_of(clip.scene_id, len(self.by_fold))
        if fold is None:
            raise CalibrationDataError(
                f"calib clip {clip.clip_id} has no scene index, so no out-of-fold row sigma"
            )
        return self.by_fold[fold]


def fold_of(scene_id: str | None, folds: int = ROW_SIGMA_FOLDS) -> int | None:
    index = None if scene_id is None else scene_index(scene_id)
    return None if index is None else index % folds


def body_readings(inputs: CheckInputs) -> dict[str, dict[int, tuple[float, float]]]:
    """Object id -> native frame -> (image row of its lowest point, image size), finite only."""
    k = camera_k(inputs)
    if k is None:
        return {}
    states = inputs.states
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for index, object_id in enumerate(states.object_ids):
        track = image_track(k, states, index)
        out[str(object_id)] = {
            int(frame): (float(row), float(size))
            for frame, row, size in zip(
                states.frame_index, track.contact[:, 1], track.size, strict=True
            )
            if np.isfinite(row) and np.isfinite(size)
        }
    return out


def contact_rows(inputs: CheckInputs) -> dict[str, dict[int, float]]:
    """Object id -> native frame -> image row of its lowest point (finite rows only)."""
    return {
        body: {frame: row for frame, (row, _) in rows.items()}
        for body, rows in body_readings(inputs).items()
    }


def clip_residuals(perceived: CheckInputs, truth: CheckInputs, ids: IdMap | None) -> ClipResiduals:
    """Residuals of every body matched by ``ids`` (None: the ids already agree), the row series
    on the perceived rows."""
    seen, true = body_readings(perceived), body_readings(truth)
    to_truth = {pred: pred for pred in seen} if ids is None else ids.pred_to_gt()
    frames = [int(f) for f in perceived.states.frame_index]
    rows: list[FloatArray] = []
    sizes: list[FloatArray] = []
    for pred, gt in to_truth.items():
        if pred not in seen or gt not in true:
            continue
        mine, theirs = seen[pred], true[gt]
        shared = [f for f in frames if f in mine and f in theirs]
        rows.append(
            np.asarray(
                [mine[f][0] - theirs[f][0] if f in shared else np.nan for f in frames],
                dtype=np.float64,
            )
        )
        sizes.append(
            np.asarray(
                [
                    (mine[f][1] - theirs[f][1]) / theirs[f][1] if f in shared else np.nan
                    for f in frames
                ],
                dtype=np.float64,
            )
        )
    return ClipResiduals(tuple(rows), tuple(sizes))


def row_residuals(perceived: CheckInputs, truth: CheckInputs, ids: IdMap | None) -> FloatArray:
    """Perceived minus true contact row on every native frame both read, bodies matched by
    ``ids`` (None: the ids already agree)."""
    return clip_residuals(perceived, truth, ids).flat_rows


def law_sigma(values: FloatArray, law: RowLaw) -> float:
    """Half-width of the ``ROW_COVERAGE`` band of ``law`` over the normal z of that band; NaN
    without finite values."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    if law is RowLaw.QUANTILE:
        half = float(np.quantile(np.abs(finite), ROW_COVERAGE))
    else:
        _, _, scale = stats.t.fit(finite, f0=TAIL_DOF, floc=0.0)
        half = float(stats.t.ppf(0.5 + ROW_COVERAGE / 2.0, TAIL_DOF)) * float(scale)
    return half / band_z()


def autocorrelation(series: Sequence[FloatArray]) -> tuple[float, ...] | None:
    """r_1, r_2, ... of the pooled centred series (NaN rows skipped), from lag 1 until the first
    r_k <= 0 or ``AUTOCORRELATION_MAX_LAG``; () for residuals that are all 0, None without a
    series of two finite values."""
    within = centred(series)
    if not within:
        return None
    total = float(sum(np.nansum(c**2) for c in within))
    lags: list[float] = []
    for lag in range(1, AUTOCORRELATION_MAX_LAG + 1):
        if total == 0.0:
            break
        r = float(sum(np.nansum(c[:-lag] * c[lag:]) for c in within if c.size > lag)) / total
        if not r > 0.0:
            break
        lags.append(r)
    return tuple(lags)


def _flat_centred(series: Sequence[FloatArray]) -> FloatArray:
    return _flat(tuple(centred(series)))


def _fit_group(group: Sequence[ClipResiduals], law: RowLaw) -> RowNoise | None:
    """The noise of one group of clips; None when some quantity has no residual."""
    rows = [s for c in group for s in c.rows]
    sizes = [s for c in group for s in c.sizes]
    row_lags, size_lags = autocorrelation(rows), autocorrelation(sizes)
    sigmas = (
        law_sigma(_flat(tuple(rows)), law),
        law_sigma(_flat_centred(rows), law),
        law_sigma(_flat_centred(sizes), law),
    )
    if row_lags is None or size_lags is None or not all(np.isfinite(sigmas)):
        return None
    return RowNoise(
        row_tail_dof=TAIL_DOF if law is RowLaw.STUDENT_T else None,
        row_sigma=sigmas[0],
        rate_row_sigma=sigmas[1],
        size_sigma=sigmas[2],
        row_autocorrelation=row_lags,
        size_autocorrelation=size_lags,
    )


def fit_row_sigma(
    residuals: Mapping[str, ClipResiduals],
    scenes: Mapping[str, str],
    *,
    check_id: str,
    level: InputLevel,
    gt_stages: Sequence[str],
    law: RowLaw = DEFAULT_ROW_LAW,
    folds: int = ROW_SIGMA_FOLDS,
) -> RowSigma:
    """``residuals`` and ``scenes`` keyed by clip id; a clip without a scene index or without
    row residuals is listed and left out. Refused when some fold has nothing outside it."""
    by_fold: list[list[ClipResiduals]] = [[] for _ in range(folds)]
    fold_scenes: list[set[str]] = [set() for _ in range(folds)]
    empty: list[str] = []
    for clip_id in sorted(residuals):
        values, scene_id = residuals[clip_id], scenes.get(clip_id)
        fold = fold_of(scene_id, folds)
        if fold is None or scene_id is None or values.empty:
            empty.append(clip_id)
            continue
        by_fold[fold].append(values)
        fold_scenes[fold].add(scene_id)
    groups = [[c for i, part in enumerate(by_fold) if i != j for c in part] for j in range(folds)]
    fits = [_fit_group(g, law) for g in [*groups, [c for part in by_fold for c in part]]]
    if any(f is None for f in fits):
        counts = [sum(c.flat_rows.size for c in part) for part in by_fold]
        raise CalibrationDataError(f"some fold has no residuals outside it (per fold: {counts})")
    done = [f for f in fits if f is not None]
    return RowSigma(
        check_id=check_id,
        level=level,
        gt_stages=tuple(gt_stages),
        law=law,
        coverage=ROW_COVERAGE,
        by_fold=tuple(done[:folds]),
        pooled=done[folds],
        residuals_per_fold=tuple(sum(c.flat_rows.size for c in part) for part in by_fold),
        size_residuals_per_fold=tuple(sum(c.flat_sizes.size for c in part) for part in by_fold),
        scenes_per_fold=tuple(tuple(sorted(s)) for s in fold_scenes),
        clips_without_residuals=tuple(empty),
    )

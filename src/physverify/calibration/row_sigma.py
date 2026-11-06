"""Out-of-fold sigma of the R1_dyn contact row: the image row of each body's lowest point read
from perceived inputs against the same row read from ground truth, on clean calibration clips.

Scenes go to fold = scene index mod ``ROW_SIGMA_FOLDS``. The sigma of fold j is the robust scale
(MAD / 0.6745) of the residuals of every other fold, pooled over bodies and rows; ``pooled`` uses
every fold, for clips outside calib. Pure, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from pydantic import Field

from physverify.calibration.table import CalibrationDataError
from physverify.checks._base import CheckInputs
from physverify.checks._image import FloatArray, camera_k
from physverify.checks.separate.r1_dyn_pass_through import image_track
from physverify.core.models import SCHEMA_KEY, ArtifactModel
from physverify.physics.uncertainty import robust_scale
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.matching import IdMap
from physverify.synth.scene_scope import scene_index

ROW_SIGMA_FOLDS: Final = 5
ROW_SIGMA_SCHEMA: Final = "physverify.row_sigma"
ROW_SIGMA_VERSION: Final = "1.0.0"
ROW_SIGMA_FILE: Final = "row_sigma.json"


class RowSigma(ArtifactModel):
    """``by_fold[j]`` is the sigma (pixels) for clips of fold j, fitted without fold j."""

    schema_: str = Field(default=ROW_SIGMA_SCHEMA, alias=SCHEMA_KEY)
    version: str = ROW_SIGMA_VERSION
    check_id: str
    level: InputLevel
    gt_stages: tuple[str, ...]
    by_fold: tuple[float, ...]
    pooled: float
    residuals_per_fold: tuple[int, ...]
    scenes_per_fold: tuple[tuple[str, ...], ...]
    clips_without_residuals: tuple[str, ...]

    def sigma_of(self, clip: ClipManifest) -> float:
        """A calib clip takes the sigma of its scene's fold, any other clip the pooled sigma.
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


def contact_rows(inputs: CheckInputs) -> dict[str, dict[int, float]]:
    """Object id -> native frame -> image row of its lowest point (finite rows only)."""
    k = camera_k(inputs)
    if k is None:
        return {}
    states = inputs.states
    rows: dict[str, dict[int, float]] = {}
    for index, object_id in enumerate(states.object_ids):
        contact = image_track(k, states, index).contact[:, 1]
        rows[str(object_id)] = {
            int(frame): float(row)
            for frame, row in zip(states.frame_index, contact, strict=True)
            if np.isfinite(row)
        }
    return rows


def row_residuals(perceived: CheckInputs, truth: CheckInputs, ids: IdMap | None) -> FloatArray:
    """Perceived minus true contact row on every native frame both read, bodies matched by
    ``ids`` (None: the ids already agree)."""
    seen, true = contact_rows(perceived), contact_rows(truth)
    to_truth = {pred: pred for pred in seen} if ids is None else ids.pred_to_gt()
    residuals = [
        row - true[gt][frame]
        for pred, gt in to_truth.items()
        if pred in seen and gt in true
        for frame, row in seen[pred].items()
        if frame in true[gt]
    ]
    return np.asarray(residuals, dtype=np.float64)


def fit_row_sigma(
    residuals: Mapping[str, FloatArray],
    scenes: Mapping[str, str],
    *,
    check_id: str,
    level: InputLevel,
    gt_stages: Sequence[str],
    folds: int = ROW_SIGMA_FOLDS,
) -> RowSigma:
    """``residuals`` and ``scenes`` keyed by clip id; a clip without a scene index or without
    residuals is listed and left out. Refused when some fold has no residuals outside it."""
    by_fold: list[list[FloatArray]] = [[] for _ in range(folds)]
    fold_scenes: list[set[str]] = [set() for _ in range(folds)]
    empty: list[str] = []
    for clip_id in sorted(residuals):
        values, scene_id = residuals[clip_id], scenes.get(clip_id)
        fold = fold_of(scene_id, folds)
        if fold is None or scene_id is None or not values.size:
            empty.append(clip_id)
            continue
        by_fold[fold].append(values)
        fold_scenes[fold].add(scene_id)
    pooled = [np.concatenate(part) if part else np.empty(0) for part in by_fold]
    sigmas = tuple(
        robust_scale(np.concatenate([p for i, p in enumerate(pooled) if i != j]))
        for j in range(folds)
    )
    if not all(np.isfinite(sigmas)):
        sizes = [int(p.size) for p in pooled]
        raise CalibrationDataError(f"some fold has no residuals outside it (per fold: {sizes})")
    return RowSigma(
        check_id=check_id,
        level=level,
        gt_stages=tuple(gt_stages),
        by_fold=sigmas,
        pooled=robust_scale(np.concatenate(pooled)),
        residuals_per_fold=tuple(int(p.size) for p in pooled),
        scenes_per_fold=tuple(tuple(sorted(s)) for s in fold_scenes),
        clips_without_residuals=tuple(empty),
    )

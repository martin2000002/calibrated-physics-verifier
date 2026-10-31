"""LIFT: orientation error of the estimated pose, up to the true solid's symmetry."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
)
from physverify.diagnosis.metrics._state_pairing import finite_rows, paired_states
from physverify.diagnosis.settings import MetricSettings
from physverify.physics.geometry import matrix_from_quaternion
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

EQUAL_EXTENT_REL = 0.02
"""Two true extents within 2 % of each other make the box symmetric under their swap."""
SIGNED_PERMUTATIONS = tuple(
    np.diag(signs)[list(order)]
    for order in itertools.permutations(range(3))
    for signs in itertools.product((1.0, -1.0), repeat=3)
)


def box_symmetries(extents: FloatArray) -> list[FloatArray]:
    """Proper rotations S of the local axes that map a box of these extents onto itself."""
    out = []
    for s in SIGNED_PERMUTATIONS:
        if np.linalg.det(s) < 0:
            continue
        swapped = np.abs(s) @ extents
        if np.all(np.abs(swapped - extents) <= EQUAL_EXTENT_REL * extents.max()):
            out.append(s)
    return out


def rotation_angle(a: FloatArray, b: FloatArray) -> float:
    """Radians: angle of the relative rotation a^T b."""
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.acos(min(max(cos, -1.0), 1.0))


def symmetric_angle(
    pred: FloatArray, truth: FloatArray, kind: PrimitiveKind, extents: FloatArray
) -> float:
    """Radians: smallest angle between the predicted and any equivalent true local frame; for a
    cylinder the angle between the axes (local z) up to sign."""
    if kind is PrimitiveKind.CYLINDER:
        cos = abs(float(pred[:, 2] @ truth[:, 2]))
        return math.acos(min(cos, 1.0))
    return min(rotation_angle(pred, truth @ s) for s in box_symmetries(extents))


@dataclass(frozen=True)
class LiftOrientationMetric:
    """``orientation_rad`` per visible non-sphere object row where both quaternions are known
    (a sphere has no orientation to get wrong); ``orientation_z`` divides it by the declared
    ``orientation_sigma``. Needs no context."""

    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "orientation"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, ObjectStateTable) or not isinstance(truth, ObjectStateTable):
            raise TypeError("lift metric compares two ObjectStateTable")
        pairs = paired_states(pred, truth)
        q_p, q_g = pairs.pred(pred.quaternion), pairs.truth(truth.quaternion)
        sizes, sigma = pairs.truth(truth.size), pairs.pred(pred.orientation_sigma)
        kinds = [PrimitiveKind(str(k)) for k in truth.primitive[pairs.truth_cols]]
        ok = pairs.visible(pred, truth) & finite_rows(q_p) & finite_rows(q_g) & finite_rows(sizes)
        angles, scores = [], []
        for t, n in zip(*np.nonzero(ok), strict=True):
            if kinds[n] is PrimitiveKind.SPHERE:
                continue
            angle = symmetric_angle(
                matrix_from_quaternion(q_p[t, n]),
                matrix_from_quaternion(q_g[t, n]),
                kinds[n],
                sizes[t, n],
            )
            angles.append(angle)
            if math.isfinite(sigma[t, n]) and sigma[t, n] > 0:
                scores.append(angle / sigma[t, n])
        samples = {"orientation_rad": np.array(angles), "orientation_z": np.array(scores)}
        return MetricOutcome(samples=MappingProxyType(samples))


PLUGINS = (LiftOrientationMetric,)

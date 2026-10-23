"""LIFT: position and size errors after the conditional scale fit, in object lengths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import MetricCase, MetricOutcome, bare_compute
from physverify.diagnosis.metrics._state_pairing import finite_rows, paired_states, sigma_scores
from physverify.diagnosis.normalizers import characteristic_lengths
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import conditional_scale_fit
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class LiftPositionMetric:
    """``centroid_rel``, ``size_rel``, ``position_err_m`` and ``position_z``. Needs no context."""

    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "position"
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
        p, g = pairs.pred(pred.position), pairs.truth(truth.position)
        ok = pairs.visible(pred, truth) & finite_rows(p) & finite_rows(g)
        present = truth.visible & finite_rows(truth.position)
        coverage = (float(np.count_nonzero(ok)), float(np.count_nonzero(present)))
        fit = conditional_scale_fit(pred, truth)
        if fit is None:
            empty = np.zeros(0)
            keys = ("centroid_rel", "size_rel", "position_err_m", "position_z")
            return MetricOutcome(
                samples=MappingProxyType(dict.fromkeys(keys, empty)),
                ratios=MappingProxyType({"coverage": coverage}),
            )
        s = fit.meters_per_unit
        e_p, e_g = pairs.pred(pred.size), pairs.truth(truth.size)
        length = characteristic_lengths(e_g)
        with_length = ok & np.isfinite(length) & (length > 0)
        sized = ok & finite_rows(e_p) & finite_rows(e_g) & (e_g > 0).all(axis=-1)
        residual = s * p - g
        samples = {
            "centroid_rel": (residual[with_length] / length[with_length][:, None]).ravel(),
            "size_rel": (s * e_p[sized] / e_g[sized] - 1.0).ravel(),
            "position_err_m": np.linalg.norm(residual[ok], axis=-1),
            "position_z": sigma_scores(p - g / s, pairs.pred(pred.position_sigma), ok),
        }
        return MetricOutcome(
            samples=MappingProxyType(samples), ratios=MappingProxyType({"coverage": coverage})
        )


PLUGINS = (LiftPositionMetric,)

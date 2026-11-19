"""PASS/FAIL when the k-sigma interval of a magnitude clears its threshold (``ThresholdRule``)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.verification.thresholding import THRESHOLD_POLICY, ThresholdParams, ThresholdRule


class ThresholdPolicy(ThresholdRule):
    name: ClassVar[str] = THRESHOLD_POLICY
    Params: ClassVar[type[BaseModel]] = ThresholdParams


PLUGINS = (ThresholdPolicy,)

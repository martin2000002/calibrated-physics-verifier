"""PASS/FAIL when the k-sigma interval of a magnitude clears its threshold (``ThresholdRule``)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.verification.thresholding import ThresholdParams, ThresholdRule


class ThresholdPolicy(ThresholdRule):
    name: ClassVar[str] = "threshold"
    Params: ClassVar[type[BaseModel]] = ThresholdParams


PLUGINS = (ThresholdPolicy,)

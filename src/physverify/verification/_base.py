"""Verdict policy contract: turns an UNSCORED result into PASS, FAIL or ABSTAIN."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from physverify.schema.checks import CheckResult


@runtime_checkable
class VerdictPolicy(Protocol):
    """Registered by ``name``; ``thresholds`` is the parsed calibration file (H1)."""

    name: ClassVar[str]

    def decide(self, result: CheckResult, thresholds: Mapping[str, Any]) -> CheckResult: ...

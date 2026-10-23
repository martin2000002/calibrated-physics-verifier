"""Verdict policy contract: turns an UNSCORED result into PASS, FAIL or ABSTAIN."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from physverify.schema.checks import CheckResult
from physverify.verification.thresholding import ThresholdTable


@runtime_checkable
class VerdictPolicy(Protocol):
    """Registered by ``name``; constructed as ``cls(params)`` with an instance of its ``Params``
    (from ``[verification.policies.<name>]``)."""

    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def decide(self, result: CheckResult, thresholds: ThresholdTable) -> CheckResult: ...

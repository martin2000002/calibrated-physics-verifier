"""Seam with cycle A7: the only diagnosis module that imports ``verification/``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from physverify.checks._base import Check, CheckInputs
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.verification._base import VerdictPolicy
from physverify.verification.check_set import CHECKS_SECTION, ChecksSettings, build_checks
from physverify.verification.inputs import ClipInputs, OracleSource, perceived_inputs
from physverify.verification.runner import (
    CALIBRATED,
    VERIFICATION_SECTION,
    VerificationRunner,
    VerificationSettings,
    build_policy,
    load_thresholds,
)


class CheckSuite(Protocol):
    @property
    def provisional(self) -> bool:
        """True while the thresholds behind ``decide`` are provisional (until A8)."""
        ...

    def inputs(
        self,
        clip: ClipManifest,
        outputs: Mapping[Stage, StageOutput],
        names: Mapping[Stage, str],
        level: InputLevel,
    ) -> CheckInputs: ...

    def truth_inputs(self, dataset: DatasetSource, clip: ClipManifest) -> CheckInputs | None: ...

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]: ...

    def decide(self, results: Sequence[CheckResult]) -> Sequence[CheckResult] | None: ...


class _Undecided:
    """Pass-through policy: A7's runner applies ``requires``; ``decide`` comes later."""

    name: ClassVar[str] = "undecided"

    def decide(self, result: CheckResult, thresholds: Mapping[str, Any]) -> CheckResult:
        return result


@dataclass(frozen=True)
class _Prepared:
    item: ClipInputs

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        yield self.item


class A7CheckSuite:
    """Thin adapter over ADR 0006 §3.7-3.8 (perceived_inputs, OracleSource, runner)."""

    def __init__(
        self,
        checks: Sequence[Check],
        policy: VerdictPolicy | None,
        thresholds: Mapping[str, Any],
    ) -> None:
        self._checks = tuple(checks)
        self._policy = policy
        self._thresholds = thresholds

    @classmethod
    def build(
        cls,
        settings: Settings,
        checks: Mapping[str, type[Check]],
        policies: Mapping[str, type[VerdictPolicy]],
    ) -> A7CheckSuite:
        verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
        built = build_checks(checks, settings.section(CHECKS_SECTION, ChecksSettings))
        policy = build_policy(policies, verification)
        thresholds = load_thresholds(settings.paths.thresholds, verification.thresholds)
        return cls(built, policy, thresholds)

    @property
    def provisional(self) -> bool:
        return self._thresholds.get("status") != CALIBRATED

    def inputs(
        self,
        clip: ClipManifest,
        outputs: Mapping[Stage, StageOutput],
        names: Mapping[Stage, str],
        level: InputLevel,
    ) -> CheckInputs:
        return perceived_inputs(clip, outputs, names, level)

    def truth_inputs(self, dataset: DatasetSource, clip: ClipManifest) -> CheckInputs | None:
        item = next(OracleSource(dataset).load([clip]), None)
        return None if item is None else item.inputs

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]:
        prepared = ClipInputs(clip=inputs.clip, inputs=inputs, error=None)
        runner = VerificationRunner(self._checks, _Undecided(), self._thresholds)
        return tuple(runner.run(_Prepared(prepared), [inputs.clip]).results)

    def decide(self, results: Sequence[CheckResult]) -> Sequence[CheckResult] | None:
        if self._policy is None:
            return None
        return tuple(self._policy.decide(result, self._thresholds) for result in results)

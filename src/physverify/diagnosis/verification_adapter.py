"""Seam with the check suite: the only diagnosis module that imports ``verification/``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from pydantic import BaseModel

from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.core.errors import MissingInputError
from physverify.core.models import FrozenModel
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner, VariantOutputs
from physverify.diagnosis.chain_variants import Variant
from physverify.extraction.pipeline import DOMAIN_ERRORS
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.verification._base import VerdictPolicy
from physverify.verification.check_routes import guarded_plans, routed_inputs
from physverify.verification.inputs import (
    ClipInputs,
    InputSource,
    OracleSource,
    error_message,
    perceived_inputs,
)
from physverify.verification.runner import VerificationRunner, configured_suite
from physverify.verification.thresholding import ThresholdTable


class CheckSuite(Protocol):
    @property
    def provisional(self) -> bool:
        """True while the thresholds behind ``decide`` are provisional (until the calibration chain
        sets them)."""
        ...

    def truth_inputs(self, dataset: DatasetSource, clip: ClipManifest) -> ClipInputs | None: ...

    def evaluate(self, item: ClipInputs) -> Sequence[CheckResult]: ...

    def decide(self, results: Sequence[CheckResult]) -> Sequence[CheckResult] | None: ...


class _Undecided:
    """Pass-through policy: the check suite's runner applies ``requires``; ``decide`` comes
    later."""

    name: ClassVar[str] = "undecided"
    Params: ClassVar[type[BaseModel]] = FrozenModel

    def decide(self, result: CheckResult, thresholds: ThresholdTable) -> CheckResult:
        return result


@dataclass(frozen=True)
class _Prepared:
    item: ClipInputs

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        yield self.item


@dataclass(frozen=True)
class _PlanSource:
    """Inputs of one plan's loaded stage outputs (None if a stage failed, ``failure`` saying so).
    The main plan's failure raises, so the clip fails as a whole; a route's failure leaves only
    the checks routed to it without inputs, as in ``verify``."""

    outputs: Mapping[Stage, StageOutput] | None
    names: Mapping[Stage, str]
    level: InputLevel
    route: bool
    failure: str = ""

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield self._one(clip)

    def _one(self, clip: ClipManifest) -> ClipInputs:
        try:
            if self.outputs is None:
                raise MissingInputError(self.failure)
            return ClipInputs(
                clip, perceived_inputs(clip, self.outputs, self.names, self.level), None
            )
        except DOMAIN_ERRORS as exc:
            if not self.route:
                raise
            return ClipInputs(clip, None, error_message(exc), self.level)


def require_routable(chain: ChainRunner, variants: Sequence[Variant]) -> None:
    """Resolve and guard every variant's plans before any clip, so a refused routing stops the run
    as it stops ``verify`` instead of failing each clip."""
    for variant in variants:
        guarded_plans(chain.settings_of(variant), chain.resolver(variant), chain.checks)


def variant_inputs(
    chain: ChainRunner, clip: ClipManifest, outputs: VariantOutputs, level: InputLevel
) -> ClipInputs:
    """``outputs``' inputs, each routed check reading its route's plan under the same variant, as
    ``verify`` reads them (``routed_inputs``)."""

    def source_of(plan: ResolvedPlan) -> InputSource:
        if plan == outputs.plan:
            return _PlanSource(outputs.outputs, plan.names, level, route=False)
        failure = (
            f"{clip.clip_id}: a route plan of variant {outputs.variant.name} has a failed stage"
        )
        return _PlanSource(chain.plan_outputs(clip, plan), plan.names, level, True, failure)

    variant = outputs.variant
    settings, resolve = chain.settings_of(variant), chain.resolver(variant)
    source = routed_inputs(settings, resolve, chain.checks, source_of)
    return next(iter(source.load([clip])))


class VerificationCheckSuite:
    """Thin adapter over the verification runner and the oracle inputs (``OracleSource``)."""

    def __init__(
        self,
        checks: Sequence[Check],
        policy: VerdictPolicy | None,
        thresholds: ThresholdTable,
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
    ) -> VerificationCheckSuite:
        suite = configured_suite(settings, checks, policies)
        return cls(suite.checks, suite.policy, suite.thresholds)

    @property
    def provisional(self) -> bool:
        return not self._thresholds.calibrated

    def truth_inputs(self, dataset: DatasetSource, clip: ClipManifest) -> ClipInputs | None:
        item = next(OracleSource(dataset).load([clip]), None)
        return None if item is None or item.inputs is None else item

    def evaluate(self, item: ClipInputs) -> Sequence[CheckResult]:
        runner = VerificationRunner(self._checks, _Undecided(), self._thresholds)
        return tuple(runner.run(_Prepared(item), [item.clip]).results)

    def decide(self, results: Sequence[CheckResult]) -> Sequence[CheckResult] | None:
        if self._policy is None:
            return None
        return tuple(self._policy.decide(result, self._thresholds) for result in results)

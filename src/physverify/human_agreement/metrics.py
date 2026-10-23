"""The content of ``metrics.json``: one comparison's numbers, which its summary reads back."""

from __future__ import annotations

import json
from typing import Any, Final

from physverify.core.models import FrozenModel
from physverify.human_agreement.comparison_items import JoinCounts
from physverify.human_agreement.coverage import InputGaps
from physverify.human_agreement.prereg import FreezeAudit
from physverify.schema.io import json_ready

METRICS_FILE: Final = "metrics.json"
SUMMARY_FILE: Final = "summary.md"

ViewBlock = dict[str, Any]
"""One policy view of one group, as ``policy_metrics.evaluate_view`` builds it."""


class PrimaryVerdict(FrozenModel):
    """``supported`` is None unless the verdict is ``applicable`` (calibrated rule, no gate)."""

    applicable: bool
    supported: bool | None
    status: str
    reason: str
    gate: str | None
    mcnemar_p: float
    accuracy_difference: float
    kappa_difference_ci: tuple[float, float]
    mde_accuracy: float
    mde_accuracy_clustered: float
    post_hoc: bool


class CalibrationState(FrozenModel):
    """``gate`` says why the calibrated stratum was not usable; ``stratum`` is its state."""

    gate: str | None
    stratum: dict[str, Any] | None


class RunInfo(FrozenModel):
    path: str
    profile: str
    device: str


class Coverage(FrozenModel):
    """Compared clips, the ones the verifier decided, and their fraction (NaN without clips)."""

    n: float
    decided: float
    coverage: float


class Sensitivity(FrozenModel):
    """The primary policy with the time-dependent checks moved, per group, without bootstrap."""

    time_dependent: list[str]
    time_dependent_in_primary: bool
    excluded: list[str]
    policy: str
    groups: dict[str, ViewBlock]


class ComparisonMetrics(FrozenModel):
    rule: str
    calibration: CalibrationState
    primary_excluded: list[str]
    preregistered: bool
    preregistration: FreezeAudit | None
    run: RunInfo
    primary_policy: str
    counts: JoinCounts
    abstention: dict[str, dict[str, int]]
    input_gaps: InputGaps | None
    coverage: dict[str, Coverage]
    h2_group: str
    descriptive_groups: list[str]
    groups: dict[str, dict[str, ViewBlock]]
    per_check: list[dict[str, object]]
    per_generator: list[dict[str, object]]
    secondary_cuts: dict[str, dict[str, dict[str, float]]]
    sensitivity_time_dependent: Sensitivity
    risk_coverage_vlm: dict[str, list[dict[str, float]]]
    h2_verdict: PrimaryVerdict

    @property
    def primary(self) -> ViewBlock:
        """The primary policy's view of the aggregate group: the one the verdict tests."""
        return self.groups[self.h2_group][self.primary_policy]

    def to_json(self) -> str:
        return json.dumps(json_ready(self.model_dump()), indent=2, sort_keys=True)

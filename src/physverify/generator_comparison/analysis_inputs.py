"""Inputs of the pre-registered generator comparison and the arm summaries it compares."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from physverify.core.models import FrozenModel
from physverify.finetune.pairs import Rescore
from physverify.finetune.settings import PairSettings, SelectSettings
from physverify.generator_comparison.arms import PromptArms
from physverify.generator_comparison.records import JudgeKey
from physverify.generator_comparison.settings import H3Settings
from physverify.generator_comparison.stats import Diff
from physverify.generator_comparison.verdict import ExternalStatus

JudgeValues = Mapping[JudgeKey, float | None]


class ArmSummary(FrozenModel):
    n: int
    n_scored: int
    abstention: float | None
    mean_score: float | None
    median_motion: float | None
    objects_ok_rate: float | None
    mean_pc: float | None
    mean_sa: float | None


@dataclass(frozen=True)
class AnalysisInputs:
    arms: Sequence[PromptArms]
    has_ft: bool
    k_ft: int
    rescore: Rescore
    rules: PairSettings
    select: SelectSettings
    cfg: H3Settings
    pc: JudgeValues
    sa: JudgeValues
    external_status: ExternalStatus


Values = dict[str, float]


def as_diffs(values: Values, unit: Mapping[str, str] | None = None) -> list[Diff]:
    return [Diff(unit[pid] if unit else pid, v) for pid, v in sorted(values.items())]

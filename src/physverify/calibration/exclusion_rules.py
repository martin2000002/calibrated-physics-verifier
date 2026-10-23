"""The preregistered exclusion rules 1.1, 1.2 (as amended) and 1.5.

Pure: the CLI loads the artifact, the evaluation and the two R2 runs, this module decides. Every
threshold comes from ``[exclusion_rules]`` (``configs/verifier/exclusion_rules.toml``): the model
fields have no defaults, so no number of a rule lives in code. It imports no consumer
(human_agreement, generator_comparison, finetune)."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.io import stable_hash

SECTION = "exclusion_rules"
DELTA_TPR = "delta_tpr"
KEPT = "kept"
ARMS = ("base", "tiny")
R5A_RULE_ID = "1.1"
R2_RULE_ID = "1.2"
REAL_FPR_RULE_ID = "1.5"
RULE_IDS = (R5A_RULE_ID, R2_RULE_ID, REAL_FPR_RULE_ID)
"""The preregistered rule ids, in the order ``exclusion_decision.RULES`` applies them."""


class R5aRule(FrozenModel):
    """1.1: kept only when the consumed stratum is usable, its weight > ``min_weight`` (status
    ok and Youden J low > 0) and the tau_b CI low > ``min_tau_b_low``."""

    check_ids: tuple[str, ...]
    min_weight: float
    min_tau_b_low: float


class R2Rule(FrozenModel):
    """1.2, as amended: both detector arms judged at the artifact's tau of the consumed stratum
    on the calib clips of ``family``, every one scored in both arms (the evaluation's
    denominators). ``rates = "total"``: a sigma abstention is no FAIL."""

    check_ids: tuple[str, ...]
    dataset: str
    subset: str
    family: str
    perturbation: str
    split: Split
    base_detector: str
    tiny_detector: str
    rates: Literal["total"]
    max_fpr_increase: float
    tpr_ci: float = Field(gt=0, lt=1)
    bootstrap_n: int = Field(ge=1)
    bootstrap_seed: int


class RealRule(FrozenModel):
    """1.5: excluded when the worst exact Clopper-Pearson low of the real FPR exceeds the max."""

    datasets: tuple[str, ...]
    max_fpr_lower: float
    ci: float
    sided: Literal["two_sided"]
    allowed_sets: tuple[dict[str, int], ...]

    @model_validator(mode="after")
    def _sets_name_the_datasets(self) -> RealRule:
        for allowed in self.allowed_sets:
            if set(allowed) != set(self.datasets):
                raise ValueError(f"allowed set {allowed} must name exactly {list(self.datasets)}")
        return self


class ExclusionRules(FrozenModel):
    version: str
    consumed_level: InputLevel
    consumed_profile: str
    rules: tuple[str, ...]
    r5a: R5aRule
    r2: R2Rule
    real: RealRule

    @model_validator(mode="after")
    def _known_rules(self) -> ExclusionRules:
        unknown = sorted(set(self.rules) - set(RULE_IDS))
        if unknown:
            raise ValueError(f"unknown rules {unknown}; known: {sorted(RULE_IDS)}")
        return self

    def checked(self, known: Collection[str]) -> ExclusionRules:
        unknown = sorted(set(self.r5a.check_ids + self.r2.check_ids) - set(known))
        if unknown:
            raise ConfigError(f"[{SECTION}] names unknown checks {unknown}; known: {sorted(known)}")
        return self

    def sha256(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def exclusion_rules(raw: Mapping[str, object], known: Collection[str]) -> ExclusionRules:
    """``[exclusion_rules]`` validated, its check ids against the registry ``known``."""
    try:
        rules = ExclusionRules.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"[{SECTION}] is invalid:\n{exc}") from exc
    return rules.checked(known)

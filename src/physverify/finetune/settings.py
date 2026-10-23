"""Settings of the preference fine-tuning cycle: the ``[finetune]`` section (ADR 0012).

Generator parameters are not here: they are read from ``[timing.generation]`` (D3), so the cost
measured by the timing harness is exactly the cost of these videos.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from physverify.config.models import Settings
from physverify.core.models import FrozenModel
from physverify.generation.settings import SLUG_PATTERN

SECTION = "finetune"
PromptSplit = Literal["train", "dev", "eval"]
SPLITS: tuple[PromptSplit, ...] = ("train", "dev", "eval")
TEMPLATE_ID_PATTERN = r"^[a-z0-9_]+$"


class QueryTemplate(FrozenModel):
    """A detection query whose phrase may name template slots, e.g. ``"{primitive}"``."""

    phrase: str = Field(min_length=1)
    expected_count: int | None = Field(None, ge=1)


class TemplateSpec(FrozenModel):
    """``holdout`` templates only ever produce ``eval`` prompts (tagged ``unseen_template``)."""

    id: str = Field(pattern=TEMPLATE_ID_PATTERN)
    text: str = Field(min_length=1)
    queries: list[QueryTemplate] = Field(min_length=1)
    holdout: bool = False


class FamilySpec(FrozenModel):
    name: str = Field(pattern=SLUG_PATTERN)
    target_checks: list[str] = Field(min_length=1)
    templates: list[TemplateSpec] = Field(min_length=2)
    weight: float = Field(1.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> FamilySpec:
        ids = [template.id for template in self.templates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"family {self.name}: template ids must be unique")
        holdout = sum(template.holdout for template in self.templates)
        if not 1 <= holdout < len(self.templates):
            raise ValueError(
                f"family {self.name}: needs at least one holdout and one seen template"
            )
        return self


class FixedPrompt(FrozenModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    family: str = Field(pattern=SLUG_PATTERN)
    text: str = Field(min_length=1)
    queries: list[QueryTemplate] = Field(min_length=1)
    target_checks: list[str] = Field(min_length=1)


class PromptSetSettings(FrozenModel):
    seed: int
    n_train: int = Field(ge=0)
    n_dev: int = Field(ge=0)
    n_eval: int = Field(ge=0)
    eval_unseen_fraction: float = Field(0.5, ge=0, le=1)
    val_fraction: float = Field(0.1, ge=0, lt=1)
    families: list[FamilySpec] = Field(min_length=1)
    slots: dict[str, list[str]] = Field(min_length=1)
    unscaled_queries: list[str] = Field(default_factory=list)
    fixed_eval: list[FixedPrompt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> PromptSetSettings:
        names = [family.name for family in self.families]
        if len(set(names)) != len(names):
            raise ValueError("family names must be unique")
        if any(not values or len(set(values)) != len(values) for values in self.slots.values()):
            raise ValueError("every slot needs distinct values")
        return self


class CandidateSettings(FrozenModel):
    """Candidates per prompt: ``k_train`` for pairs, ``k_eval`` (base, best-of-K baseline) and
    ``k_eval_finetuned`` with seeds k = 0..K-1 paired with the base, ``k_dev`` per checkpoint."""

    k_train: int = Field(4, ge=2)
    k_eval: int = Field(8, ge=1)
    k_eval_finetuned: int = Field(4, ge=1)
    k_dev: int = Field(2, ge=1)
    base_seed: int = Field(ge=0)
    store_latents: bool = True


class PairSettings(FrozenModel):
    min_margin: float = Field(ge=0)
    sigma_k: float = Field(1.0, ge=0)
    max_pairs_per_prompt: int = Field(1, ge=1)
    min_motion: float = Field(ge=0)
    min_motion_ratio: float = Field(0.5, ge=0)
    min_coverage: float = Field(0.0, ge=0, le=1)
    require_object_count: bool = True
    weight_by_margin: bool = False
    margin_ref: float = Field(0.5, gt=0)


class LoraSettings(FrozenModel):
    rank: int = Field(32, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.0, ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)


class TrainSettings(FrozenModel):
    beta: float = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_accum: int = Field(ge=1)
    epochs: float = Field(gt=0)
    max_steps: int | None = Field(None, ge=1)
    # The schedule is a fraction of the run's optimizer steps, resolved from the pair count.
    warmup_fraction: float = Field(0.1, ge=0, lt=1)
    checkpoint_fraction: float = Field(0.2, gt=0, le=1)
    keep_last: int | None = Field(None, ge=1)
    grad_clip: float = Field(gt=0)
    seed: int = 0
    timestep_sampling: Literal["uniform_shifted"] = "uniform_shifted"
    gradient_checkpointing: bool = True
    beta_grid: list[float] = Field(default_factory=lambda: [100.0, 500.0, 2500.0, 5000.0])
    beta_grid_fraction: float = Field(0.25, gt=0, le=1)
    beta_kl_factor: float = Field(2.0, ge=1)


class SelectSettings(FrozenModel):
    checkpoints: int = Field(3, ge=1)
    max_abstention_rise: float = Field(0.10, ge=0)
    max_motion_drop: float = Field(0.20, ge=0, le=1)
    ci_z: float = Field(1.96, gt=0)


class StopRuleSettings(FrozenModel):
    """The pilot's stop rule (ADR 0012): too few pairs, too many abstentions or too few better
    winners on inspection mean H3 is not evaluable with this verifier. ``enforce = false`` only
    reports it."""

    enforce: bool = True
    min_pairs_per_prompt: float = Field(0.3, ge=0)
    max_candidate_abstention: float = Field(0.5, ge=0, le=1)
    inspected: int = Field(10, ge=1)
    min_better_fraction: float = Field(0.6, ge=0, le=1)
    seed: int = 0


class FinetuneSettings(FrozenModel):
    campaign: str = Field(pattern=SLUG_PATTERN)
    scorer: str = "calibrated"
    exclude_time_dependent: bool = True
    """Leave ``[check_scope].time_dependent_checks`` out of every scorer (item 1.17)."""
    on_family_without_targets: Literal["error", "keep"] = "error"
    """A family whose ``target_checks`` are all excluded: refuse, or keep it scored by others."""
    prompts: PromptSetSettings
    candidates: CandidateSettings
    pairs: PairSettings
    lora: LoraSettings
    train: TrainSettings
    select: SelectSettings = Field(default_factory=SelectSettings)
    stop_rule: StopRuleSettings = Field(default_factory=StopRuleSettings)


def load_finetune(settings: Settings) -> FinetuneSettings:
    return settings.section(SECTION, FinetuneSettings)

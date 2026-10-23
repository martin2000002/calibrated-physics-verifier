"""``[h3]``: the pre-registered parameters of the generator comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from physverify.config.models import Settings
from physverify.core.models import FrozenModel

SECTION: Final = "h3"
AUTOEVAL_KEY: Final = "autoeval"


class GateSettings(FrozenModel):
    """Regression gate of the external judge on published VideoPhy-2 test videos."""

    n: int = Field(50, ge=1)
    subset: str = "v2_test"
    part: str = "v2_test"
    columns: dict[str, str] = Field(default_factory=lambda: {"pc": "pc_pred", "sa": "sa_pred"})
    min_exact: float = Field(0.90, ge=0, le=1)
    max_mae: float = Field(0.25, ge=0)
    max_parse_fail: int = Field(2, ge=0)


class UsabilitySettings(FrozenModel):
    """The external judge is unusable on our domain when it cannot tell the dev videos apart."""

    max_parse_fail_share: float = Field(0.10, ge=0, le=1)
    max_modal_share: float = Field(0.90, ge=0, le=1)


class AutoevalSettings(FrozenModel):
    """Pinned VideoPhy-2 AutoEval. Relative paths are under ``root`` (the H200 work directory,
    ``PHYSVERIFY__H3__AUTOEVAL__ROOT``); machine-local, so the section hash leaves it out."""

    root: str = ""
    venv: str = "venvs/videophy2"
    repo_dir: str = "autoeval/repo"
    checkpoint_dir: str = "autoeval/ckpt"
    freeze_file: str = "autoeval/autoeval_freeze.txt"
    hf_repo: str = "videophysics/videophy_2_auto"
    hf_revision: str = Field(min_length=12)
    code_commit: str = Field(min_length=12)
    num_frames: int = Field(32, ge=1)
    batch_size: int = Field(1, ge=1)
    timeout_s: float = Field(7200.0, gt=0)

    def path(self, value: str) -> Path | None:
        """``value`` as an absolute path, or None when it is relative and no root is set."""
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        return Path(self.root).expanduser() / candidate if self.root else None


class ComparisonSettings(FrozenModel):
    exclude_tags: list[str] = Field(default_factory=lambda: ["demo"])
    selection_run: str = "main"
    alpha: float = Field(0.05, gt=0, lt=1)
    ci: float = Field(0.95, gt=0, lt=1)
    power: float = Field(0.80, gt=0, lt=1)
    bootstrap_n: int = Field(10_000, ge=1)
    bootstrap_seed: int = 20261018
    wilcoxon_method: Literal["asymptotic"] = "asymptotic"
    zero_method: Literal["pratt"] = "pratt"
    correction: bool = False
    noninferiority_margin_pc: float = Field(0.13, gt=0)
    sa_guard_margin: float = Field(0.05, gt=0)
    margin_min_generator_n: int = Field(300, ge=1)
    generator_column: str = "model_name"
    max_objects_ok_drop: float = Field(0.10, ge=0, le=1)
    min_ft_pairs_per_prompt: int = Field(2, ge=1)
    cross_fit_checks_a: list[str] = Field(default_factory=list)
    eval_store_latents: bool = False
    external_judge: str = "videophy2_autoeval"
    gate: GateSettings = Field(default_factory=GateSettings)
    usability: UsabilitySettings = Field(default_factory=UsabilitySettings)
    autoeval: AutoevalSettings


def load_comparison(settings: Settings) -> ComparisonSettings:
    return settings.section(SECTION, ComparisonSettings)

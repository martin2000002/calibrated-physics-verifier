"""``[vlm_comparison]``: the sets, cuts, bootstraps and judges of the verifier-vs-VLM comparison."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel

SECTION: Final = "vlm_comparison"
Part = Literal["kubric", "real"]


class BootstrapSettings(FrozenModel):
    n: int = Field(ge=1)
    seed: int
    ci: float = Field(gt=0, lt=1)


class JudgeSpec(FrozenModel):
    """A judge plugin of ``generator_comparison.judges`` at its published cut: a clip scored
    below ``cut`` is flagged. ``prompt_file`` (under the judge's code checkout) must hash to
    ``prompt_sha256``."""

    plugin: str
    task: str
    cut: float
    prompt_file: str
    prompt_sha256: str = Field(min_length=64, max_length=64)
    clips_per_call: int = Field(100, ge=1)


class SetSpec(FrozenModel):
    part: Part
    caption: str = Field(min_length=1)


class VlmComparisonSettings(FrozenModel):
    clean_fpr: float = Field(gt=0, lt=1)
    max_uninterpretable: float = Field(ge=0, le=1)
    real_fpr: float = Field(gt=0, lt=1)
    real_fpr_upper: float = Field(gt=0, lt=1)
    agreement_partition: str
    pairs: BootstrapSettings
    agreement: BootstrapSettings
    judges: dict[str, JudgeSpec]
    sets: dict[str, SetSpec]

    def judge(self, key: str) -> JudgeSpec:
        found = self.judges.get(key)
        if found is None:
            raise ConfigError(f"[{SECTION}.judges] has no {key!r}; known: {sorted(self.judges)}")
        return found

    def set_of(self, dataset: str, part: Part | None = None) -> SetSpec:
        found = self.sets.get(dataset)
        if found is None or (part is not None and found.part != part):
            known = sorted(k for k, s in self.sets.items() if part is None or s.part == part)
            raise ConfigError(
                f"[{SECTION}.sets] has no {part or ''} set {dataset!r}; known: {known}"
            )
        return found


def load_vlm_comparison(settings: Settings) -> VlmComparisonSettings:
    return settings.section(SECTION, VlmComparisonSettings)

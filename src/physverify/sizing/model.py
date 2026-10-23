"""The fitted size model (``[size_model]``) and the one place labels resolve to a size.

Per class c, log L ~ N(mu_c, (f * sigma_c)^2) with L = (e_x e_y e_z)^(1/3) in meters and f the
conformal factor measured by grouped cross-validation. Every size is an estimate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from physverify.config.models import SIZE_PRIORS_SECTION, Settings, SizePrior, SizePriorTable
from physverify.core.models import FrozenModel
from physverify.schema.io import stable_hash
from physverify.sizing.priors import SizePriorParams, lookup_label
from physverify.sizing.settings import SECTION as SIZING_SECTION
from physverify.sizing.settings import SizeLookup

SECTION = "size_model"
HASH_CHARS = 8


class ClassSize(FrozenModel):
    mu_log: float
    sigma_log: float = Field(ge=0)
    n_objects: int = Field(ge=0)
    n_videos: int = Field(ge=0)


class SizeModelTable(FrozenModel):
    """Shape of ``configs/verifier/size_model.toml``; empty (no classes) until ``size fit`` runs."""

    source: str = "empty"
    estimated: Literal[True] = True
    sigma_factor: float = Field(default=1.0, gt=0)
    coverage_target: float = Field(default=0.683, gt=0, lt=1)
    classes: dict[str, ClassSize] = Field(default_factory=dict)


def class_prior(size: ClassSize, sigma_factor: float) -> SizePrior:
    """exp(mu) is the median length; its relative spread is approximated by sigma in log space."""
    return SizePrior(mean_m=math.exp(size.mu_log), rel_spread=sigma_factor * size.sigma_log)


def size_table(model: SizeModelTable, lookup: SizeLookup) -> dict[str, tuple[str, SizePrior]]:
    """``{phrase: (canonical key, size)}``: fixed sizes, model classes, then their synonyms."""
    canonical: dict[str, SizePrior] = {
        **{name: class_prior(size, model.sigma_factor) for name, size in model.classes.items()},
        **{
            name: SizePrior(mean_m=f.mean_m, rel_spread=f.rel_spread)
            for name, f in lookup.fixed.items()
        },
    }
    table = {name: (name, prior) for name, prior in canonical.items()}
    for phrase, target in lookup.labels.items():
        if target in canonical and phrase not in table:
            table[phrase] = (target, canonical[target])
    return table


def resolve(label: str, model: SizeModelTable, lookup: SizeLookup) -> tuple[str, SizePrior] | None:
    """Canonical key and size of a label (whole phrase first, then each word)."""
    return lookup_label(label, size_table(model, lookup))


def predict(label: str, model: SizeModelTable, lookup: SizeLookup) -> SizePrior | None:
    resolved = resolve(label, model, lookup)
    return None if resolved is None else resolved[1]


def model_hash(model: SizeModelTable, lookup: SizeLookup) -> str:
    return stable_hash(
        {"model": model.model_dump(mode="json"), "lookup": lookup.model_dump(mode="json")}
    )[:HASH_CHARS]


def prior_aliases() -> dict[str, str]:
    """Words that reach a frozen prior by alias ("box" -> "cube"); one table for both scale
    providers and for the label vocabulary."""
    return dict(SizePriorParams().label_to_prior)


def labels_with_size(
    priors: SizePriorTable,
    model: SizeModelTable,
    lookup: SizeLookup,
    aliases: Mapping[str, str],
) -> frozenset[str]:
    """Frozen prior keys, the aliases that reach one, and every phrase the size model resolves."""
    reachable = frozenset(a for a, target in aliases.items() if target in priors.root)
    return frozenset(priors.root) | reachable | frozenset(size_table(model, lookup))


def labels_in(sections: Mapping[str, object]) -> frozenset[str]:
    """Labels that get a size under these config sections (A10 and A11 validate prompts
    against it)."""
    return labels_with_size(
        SizePriorTable.model_validate(sections.get(SIZE_PRIORS_SECTION, {})),
        SizeModelTable.model_validate(sections.get(SECTION, {})),
        SizeLookup.model_validate(sections.get(SIZING_SECTION, {})),
        prior_aliases(),
    )


def sized_labels(settings: Settings) -> frozenset[str]:
    """``labels_in`` of the loaded settings; callers (CLI commands) inject them."""
    return labels_in(settings.sections)

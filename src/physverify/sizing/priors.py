"""Frozen size priors: their settings section, label lookup and provider parameters."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ConfigDict

from physverify.config.models import SizePriorTable
from physverify.core.models import FrozenModel


def label_candidates(label: str) -> tuple[str, ...]:
    """The whole label first, then its words from the last one: English puts the head noun of a
    query phrase last ("glass globe" is a globe, "red ball" a ball)."""
    text = label.strip().lower()
    return (text, *reversed(text.replace("_", " ").split()))


def lookup_label[V](label: str, table: Mapping[str, V]) -> V | None:
    """Exact match of the label, else of its last word that is a key of ``table``."""
    for candidate in label_candidates(label):
        if candidate in table:
            return table[candidate]
    return None


class SizePriorParams(FrozenModel):
    # The title keeps validation errors naming the provider's ``Params``.
    model_config = ConfigDict(title="Params")

    label_to_prior: dict[str, str] = {"box": "cube", "sphere": "ball"}
    priors: SizePriorTable = SizePriorTable({})

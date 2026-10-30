"""Frozen size priors: their settings section, label lookup and provider parameters."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ConfigDict

from physverify.config.models import SizePriorTable
from physverify.core.models import FrozenModel


PREPOSITIONS = frozenset(
    {
        "about", "above", "across", "against", "along", "at", "behind", "below", "beside",
        "between", "by", "for", "from", "in", "inside", "into", "near", "off", "on",
        "onto", "over", "through", "to", "under", "with", "without",
    }
)  # fmt: skip
"""Words that end the noun phrase of a query: what follows them modifies the head noun. "of" is
not one: in a partitive the bodies are named after it ("tower of blocks", "piece of slate")."""


def label_candidates(label: str) -> tuple[str, ...]:
    """The whole label first, then the words before its first preposition from the last one:
    English puts the head noun of a noun phrase last ("glass globe" is a globe, "red ball" a ball)
    and a prepositional phrase after it ("ball on book" is a ball)."""
    text = label.strip().lower()
    words = text.replace("_", " ").split()
    cut = next((i for i, word in enumerate(words) if word in PREPOSITIONS), len(words))
    return (text, *reversed(words[:cut]))


def lookup_label[V](label: str, table: Mapping[str, V]) -> V | None:
    """Exact match of the label, else of its last word before any preposition that is a key of
    ``table``."""
    for candidate in label_candidates(label):
        if candidate in table:
            return table[candidate]
    return None


class SizePriorParams(FrozenModel):
    # The title keeps validation errors naming the provider's ``Params``.
    model_config = ConfigDict(title="Params")

    label_to_prior: dict[str, str] = {"box": "cube", "sphere": "ball"}
    priors: SizePriorTable = SizePriorTable({})

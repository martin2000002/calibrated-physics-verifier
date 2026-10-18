"""Detection queries derived from a caption or prompt, shared by the real-video datasets."""

from __future__ import annotations

import string
from collections.abc import Sequence

from physverify.schema.clip import ObjectQuery

ARTICLES = ("a", "an", "the")


def derive_queries(prompt: str, fall_verbs: Sequence[str]) -> tuple[ObjectQuery, ...]:
    """The words before the first fall verb: 'A white bottle falls.' -> ('white bottle', 1)."""
    words = [w for w in (w.strip(string.punctuation) for w in prompt.lower().split()) if w]
    if words and words[0] in ARTICLES:
        words = words[1:]
    text = f" {' '.join(words)} "
    cuts = [i for verb in fall_verbs if (i := text.find(f" {verb.lower()} ")) >= 0]
    phrase = text[: min(cuts)] if cuts else text
    phrase = phrase.strip()
    return (ObjectQuery(phrase, 1),) if phrase else ()

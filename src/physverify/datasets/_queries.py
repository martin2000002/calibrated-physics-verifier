"""Detection queries: derived from a caption or prompt (the real-video datasets), or a clip's
queries with their counts withheld (the blind Kubric readings)."""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import replace

from physverify.schema.clip import ClipManifest, ObjectQuery

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


def blind(clip: ClipManifest) -> ClipManifest:
    """``clip`` with its object phrases kept and their expected counts withheld."""
    queries = tuple(ObjectQuery(q.phrase, expected_count=None) for q in clip.object_queries)
    return replace(clip, object_queries=queries)

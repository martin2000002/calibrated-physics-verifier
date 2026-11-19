"""Rigid bodies only, a rigid event, and at least two bodies.

The frozen rubric, its lexicons injected. Bodies are the rigid and articulated objects counted
with the quantity before them (numeral, singular determiner, anaphora = 0, bare = 1; colours and
materials skipped), plus furniture a rest verb puts something on before the clause ends. Multiword
names are joined before tokenizing, as the rubric does.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from pydantic import Field

from physverify.config.models import validated
from physverify.core.models import FrozenModel
from physverify.finetune.domains._base import DomainVerdict, UnknownWordError

MIN_BODIES: Final = 2
FLUID_OR_SOFT: Final = "fluid_or_soft"
FRACTURE: Final = "fracture_or_cut"
AGENT_MAIN_ACTOR: Final = "agent_main_actor"
NO_RIGID_EVENT: Final = "no_rigid_event"
ARTICULATED: Final = "articulated_or_soft_object"
SINGLE_BODY: Final = "single_body"


class Lexicon(FrozenModel):
    """``rigid`` adds to ``spheres`` and ``event_verbs`` to ``fall_verbs``, as in the rubric."""

    token_pattern: str = Field(min_length=1)
    multiword: list[tuple[str, str]]
    spheres: frozenset[str]
    rigid: frozenset[str]
    articulated_or_soft: frozenset[str]
    soft: frozenset[str]
    fluid: frozenset[str]
    fracture: frozenset[str]
    agents: frozenset[str]
    furniture: frozenset[str]
    ground: frozenset[str]
    colors: frozenset[str]
    materials: frozenset[str]
    numerals: dict[str, int]
    singular_det: frozenset[str]
    anaphora: frozenset[str]
    fall_verbs: frozenset[str]
    event_verbs: frozenset[str]
    rest_verbs: frozenset[str]
    function: frozenset[str]
    implied_agent: list[str]
    clause_break: frozenset[str]

    @property
    def all_rigid(self) -> frozenset[str]:
        return self.spheres | self.rigid

    @property
    def all_events(self) -> frozenset[str]:
        return self.fall_verbs | self.event_verbs

    def known(self) -> frozenset[str]:
        return frozenset().union(
            self.all_rigid,
            self.articulated_or_soft,
            self.soft,
            self.fluid,
            self.fracture,
            self.agents,
            self.furniture,
            self.ground,
            self.colors,
            self.materials,
            self.numerals,
            self.all_events,
            self.rest_verbs,
            self.function,
            self.clause_break,
        )


class RigidMultibody:
    name: ClassVar[str] = "rigid_multibody"
    version: ClassVar[str] = "1"

    def __init__(self, lexicon: Mapping[str, Any], section: str) -> None:
        self._lex = validated(Lexicon, lexicon, section)
        self._known = self._lex.known()
        self._pattern = re.compile(self._lex.token_pattern)

    def tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        for phrase, joined in self._lex.multiword:
            lowered = lowered.replace(phrase, joined)
        return self._pattern.findall(lowered)

    def classify(self, text: str) -> DomainVerdict:
        lex, toks = self._lex, self.tokens(text)
        unknown = sorted({tok for tok in toks if tok not in self._known})
        if unknown:
            raise UnknownWordError(unknown)
        words = set(toks)
        objects = [
            (i, tok)
            for i, tok in enumerate(toks)
            if tok in lex.all_rigid or tok in lex.articulated_or_soft or tok in lex.agents
        ]
        counted = sum(self._quantity(toks, i) for i, tok in objects if tok not in lex.agents)
        bodies = counted + self._supported_furniture(toks)
        reasons = self._exclusions(words, objects)
        articulated = bool(words & lex.articulated_or_soft)
        in_domain = not reasons and not articulated and bodies >= MIN_BODIES
        if articulated:
            reasons.append(ARTICULATED)
        if bodies < MIN_BODIES:
            reasons.append(SINGLE_BODY)
        implied = any(phrase in text.lower() for phrase in lex.implied_agent)
        return DomainVerdict(in_domain, bodies, implied, tuple(reasons))

    def _exclusions(self, words: set[str], objects: Sequence[tuple[int, str]]) -> list[str]:
        lex, reasons = self._lex, []
        if words & lex.soft or words & lex.fluid:
            reasons.append(FLUID_OR_SOFT)
        if words & lex.fracture:
            reasons.append(FRACTURE)
        if objects and objects[0][1] in lex.agents:
            reasons.append(AGENT_MAIN_ACTOR)
        if not words & lex.all_events:
            reasons.append(NO_RIGID_EVENT)
        return reasons

    def _quantity(self, toks: Sequence[str], index: int) -> int:
        lex, j = self._lex, index - 1
        while j >= 0 and (toks[j] in lex.colors or toks[j] in lex.materials):
            j -= 1
        if j < 0:
            return 1
        if toks[j] in lex.numerals:
            return lex.numerals[toks[j]]
        if toks[j] in lex.singular_det:
            return 1
        return 0 if toks[j] in lex.anaphora else 1

    def _supported_furniture(self, toks: Sequence[str]) -> int:
        lex, n = self._lex, 0
        for i, tok in enumerate(toks):
            if tok not in lex.rest_verbs:
                continue
            for nxt in toks[i + 1 :]:
                if nxt in lex.clause_break:
                    break
                if nxt in lex.furniture:
                    n += 1
                    break
        return n


PLUGINS = (RigidMultibody,)

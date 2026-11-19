"""Contract of a prompt domain rule: a text -> whether it lies in the domain the laws can judge.

A rule receives its lexicons from ``[prompt_domains.<name>]``; a word in none of them raises
``UnknownWordError`` instead of being guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import Field

from physverify.config.models import Settings, validated
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel

SECTION = "prompt_domains"


class UnknownWordError(ConfigError):
    """A word the rule's lexicons do not classify."""

    def __init__(self, words: list[str]) -> None:
        super().__init__(f"words in no lexicon: {words}")
        self.words = words


@dataclass(frozen=True)
class DomainVerdict:
    in_domain: bool
    n_bodies: int
    implied_agent: bool
    reasons: tuple[str, ...]


@runtime_checkable
class PromptDomain(Protocol):
    """Constructed as ``cls(lexicon, section)``; ``section`` names the table in errors."""

    name: ClassVar[str]
    version: ClassVar[str]

    def __init__(self, lexicon: Mapping[str, Any], section: str) -> None: ...

    def classify(self, text: str) -> DomainVerdict: ...


class DomainSettings(FrozenModel):
    rule: str = Field(min_length=1)
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lexicon: dict[str, Any]


PROMPT_DOMAINS = PluginKind[type[PromptDomain]]("prompt_domains")


def load_domain(
    settings: Settings, rules: Mapping[str, type[PromptDomain]], name: str
) -> PromptDomain:
    """The rule ``[prompt_domains.<name>]`` names, built with its lexicons."""
    tables = settings.sections.get(SECTION, {})
    if name not in tables:
        raise ConfigError(f"unknown prompt domain {name!r}; known: {sorted(tables)}")
    section = f"{SECTION}.{name}"
    spec = validated(DomainSettings, tables[name], section)
    if spec.rule not in rules:
        raise ConfigError(f"[{section}] names unknown rule {spec.rule!r}; known: {sorted(rules)}")
    return rules[spec.rule](spec.lexicon, f"{section}.lexicon")

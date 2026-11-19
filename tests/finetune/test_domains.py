"""The rigid multibody domain: lexicons only in configuration, the frozen rubric reproduced on the
train/dev texts of wan21_v1, and the guard that stops ``prompts build``."""

from __future__ import annotations

import ast
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from settings_factory import SettingsFactory

from physverify.app.plugins import build
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.finetune import domain_guard, domains
from physverify.finetune.domain_guard import check_universe
from physverify.finetune.domains._base import (
    PROMPT_DOMAINS,
    PromptDomain,
    UnknownWordError,
    load_domain,
)
from physverify.finetune.settings import PromptSetSettings

FIXTURE = Path(__file__).with_name("fixtures") / "domain_wan21_v1_traindev.jsonl"
"""232 train/dev prompts of wan21_v1 with the labels of the frozen rubric."""
DOMAIN_CODE = (*sorted(Path(domains.__file__).parent.glob("*.py")), Path(domain_guard.__file__))
DOMAIN = "rigid_multibody"


@pytest.fixture
def domain(settings: Settings) -> PromptDomain:
    return load_domain(settings, build(PROMPT_DOMAINS.name), DOMAIN)


def _words(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _words(item)
    elif isinstance(value, list):
        for item in value:
            yield from _words(item)


def _string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_no_lexicon_word_is_written_in_the_domain_code(settings: Settings) -> None:
    lexicon = settings.sections["prompt_domains"][DOMAIN]["lexicon"]
    words = {word for key, value in lexicon.items() for word in _words(value)}
    assert len(words) > 150 and len(DOMAIN_CODE) == 4
    for path in DOMAIN_CODE:
        assert not _string_constants(path) & words, path.name


def test_a_word_in_no_lexicon_is_an_error(domain: PromptDomain) -> None:
    with pytest.raises(UnknownWordError) as raised:
        domain.classify("A red ball falls onto a trampoline, static camera")
    assert raised.value.words == ["trampoline"]


def test_the_rule_reproduces_the_frozen_rubric(domain: PromptDomain) -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 232
    for row in rows:
        verdict = domain.classify(row["text"])
        got = (verdict.in_domain, verdict.n_bodies, verdict.implied_agent, list(verdict.reasons))
        assert got == (row["in_domain"], row["n_bodies"], row["implied_agent"], row["reasons"]), row
    counts = Counter((row["split"], row["in_domain"]) for row in rows)
    assert (counts["train", True], counts["train", False]) == (108, 92)
    assert (counts["dev", True], counts["dev", False]) == (16, 16)


def _one_set(text: str) -> PromptSetSettings:
    template = {"text": text, "queries": [{"phrase": "{ball}"}]}
    family = {
        "name": "fall",
        "target_checks": ["R3a"],
        "templates": [
            {**template, "id": "solo_ball"},
            {**template, "id": "solo_hold", "holdout": True},
        ],
    }
    return PromptSetSettings.model_validate(
        {
            "seed": 0,
            "n_train": 2,
            "n_dev": 0,
            "n_eval": 0,
            "families": [family],
            "slots": {"ball": ["tennis ball", "basketball"]},
            "domain": DOMAIN,
        }
    )


def test_the_guard_names_the_template_and_never_its_text(domain: PromptDomain) -> None:
    text = "On a {table}, a {ball} falls from about one meter and bounces, static camera"
    prompts = _one_set(text.replace("{table}", "wooden table"))
    with pytest.raises(ConfigError) as raised:
        check_universe(prompts, domain)
    message = str(raised.value)
    assert "template solo_ball:" in message and "single_body" in message
    assert "2 of 2" in message
    assert "falls" not in message and "basketball" not in message


def test_the_guard_reports_an_unknown_word_with_the_template(domain: PromptDomain) -> None:
    prompts = _one_set("On a trampoline, a {ball} and a cube collide, static camera")
    with pytest.raises(ConfigError, match=r"template solo_ball: .*\['trampoline'\]"):
        check_universe(prompts, domain)


def test_an_unknown_domain_or_rule_is_a_config_error(make_settings: SettingsFactory) -> None:
    rules = build(PROMPT_DOMAINS.name)
    with pytest.raises(ConfigError, match="unknown prompt domain"):
        load_domain(make_settings(), rules, "nowhere")
    with pytest.raises(ConfigError, match="unknown rule"):
        load_domain(make_settings(f"prompt_domains.{DOMAIN}.rule=other"), rules, DOMAIN)

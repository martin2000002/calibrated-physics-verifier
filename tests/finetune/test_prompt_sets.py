"""The preference prompt sets: one template library rendered per surface, the in-domain
universe, the pilot disjoint from the final set, wan21_v1 unchanged, and texts never printed."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cli_runner import CliRunner
from settings_factory import SettingsFactory

from physverify.app.plugins import build
from physverify.config.loader import read_config_files
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.finetune.domain_guard import check_universe
from physverify.finetune.domains._base import PROMPT_DOMAINS, PromptDomain, load_domain
from physverify.finetune.prompts import build_prompt_set, expand_template
from physverify.finetune.settings import PromptSetSettings, load_finetune, load_prompt_set
from physverify.schema.io import stable_hash

FINAL = "rigid_tabletop"
PILOTS = ("tt_pilot_table", "tt_pilot_floor")
V1 = "wan21_v1"
DOMAIN = "rigid_multibody"
SURFACE_PREFIXES = ("On a {table}, ", "On a {floor}, ")
V1_HASH = "039c2e823b5ffa0dc8e58e66028647571a55e1b6256530e91a1043b1e99724c4"
"""``finetune.prompts`` at 43b9b03, before the move to ``prompt_sets``."""
LIBRARY_HASH = "5f9d6dfd78a8bba7623e2633e1c703c034c17af8c76aa6a6fc3d93ed487dc9be"
"""The template table the pilot ran on plus three colors only the final set selects and seven
only its extra train prompts select; the final build may only select from it."""
PILOT_UNIVERSES = {
    "tt_pilot_table": "98b34364428e01db5d6f8d2224983108df904ab131c4934e8789ca7255a917e6",
    "tt_pilot_floor": "19912e36d30ac19f6cfccb9c53aed758ebeed7920ee3022c0d9fea326c8da17d",
}
"""Sorted texts each pilot could draw when its prompts were sealed (5a58ac9; unchanged since)."""
FINAL_COLORS = ("red", "blue", "green", "white", "black", "pink", "purple", "cyan")
"""Never yellow, the pilots' color; none starts with a vowel, since templates write "a {color}"."""
EXTRA_COLORS = ("orange", "brown", "gray", "magenta", "teal", "navy", "maroon")
"""In no other color list, so every extra train text is new by construction."""
EXTRA_TAG = "extra:amendment4"
EXTRA_TEMPLATES = {"tt_collide_blocks_floor": False, "tt_collide_balls_floor": True}
FINAL_TEMPLATES = {
    "tt_bounce_near_floor": False,
    "tt_bounce_rests_floor": False,
    "tt_bounce_onto_floor": True,
    "tt_collide_blocks_floor": False,
    "tt_collide_balls_floor": True,
}
"""Template id -> holdout: the floor-only domain the design pilot left."""
WHOLE_LIBRARY = (
    f"prompt_sets.{FINAL}.library.families=['tt_bounce', 'tt_collide', 'tt_roll']",
    f"prompt_sets.{FINAL}.library.drop_shapes=[]",
    f"prompt_sets.{FINAL}.library.surfaces=['table', 'floor']",
    f"prompt_sets.{FINAL}.library.scene_surface=table",
)
TARGET_CHECKS = {
    "tt_bounce": ["R3a", "R4_size", "R4_missing"],
    "tt_collide": ["R4_size", "R4_missing", "R3b"],
}
ONE_BODY_SET = """
[prompt_sets.one_body]
seed = 0
n_train = 2
n_dev = 0
n_eval = 0
val_fraction = 0.0
domain = "rigid_multibody"
slots = { ball = ["tennis ball", "basketball"] }
[[prompt_sets.one_body.families]]
name = "fall"
target_checks = ["R3a"]
[[prompt_sets.one_body.families.templates]]
id = "solo_drop"
text = "On a wooden table, a {ball} falls from about one meter and bounces, static camera"
queries = [{ phrase = "{ball}" }]
[[prompt_sets.one_body.families.templates]]
id = "solo_roll"
holdout = true
text = "On a wooden table, a {ball} rolls, static camera"
queries = [{ phrase = "{ball}" }]
"""


@pytest.fixture
def sets(settings: Settings) -> dict[str, PromptSetSettings]:
    return {name: load_prompt_set(settings, name) for name in (FINAL, *PILOTS, V1)}


@pytest.fixture
def env(data_dir: Path) -> dict[str, str]:
    return {"PHYSVERIFY__PATHS__DATA_DIR": str(data_dir)}


@pytest.fixture
def labels(monkeypatch: pytest.MonkeyPatch, fake_labels: frozenset[str]) -> None:
    monkeypatch.setattr("physverify.app.finetune.sizing_labels", lambda settings: fake_labels)


def _universe(prompts: PromptSetSettings) -> Iterator[str]:
    for family in prompts.families:
        for template in family.templates:
            yield from (item.text for item in expand_template(template, prompts.slots))


def _texts(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from [item] if key == "text" else _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


def test_every_expansion_and_scene_has_two_rigid_bodies(
    settings: Settings, sets: dict[str, PromptSetSettings]
) -> None:
    domain: PromptDomain = load_domain(settings, build(PROMPT_DOMAINS.name), DOMAIN)
    for name in (FINAL, *PILOTS):
        texts = [*_universe(sets[name]), *(fixed.text for fixed in sets[name].fixed_eval)]
        verdicts = [domain.classify(text) for text in texts]
        assert all(v.in_domain and v.n_bodies >= 2 for v in verdicts), name
    scenes = [fixed.text for fixed in sets[FINAL].fixed_eval]
    assert len(scenes) == 2 and not set(scenes) & set(_universe(sets[FINAL]))


def test_templates_name_their_surface_first_and_v1_never_does(
    sets: dict[str, PromptSetSettings],
) -> None:
    for name in (FINAL, *PILOTS):
        texts = [t.text for family in sets[name].families for t in family.templates]
        assert texts and all(text.startswith(SURFACE_PREFIXES) for text in texts), name
    v1 = [t.text for family in sets[V1].families for t in family.templates]
    assert not any(text.startswith(SURFACE_PREFIXES) for text in v1)


def test_each_surface_renders_every_shape_once(make_settings: SettingsFactory) -> None:
    whole = load_prompt_set(make_settings(*WHOLE_LIBRARY), FINAL)
    ids = [t.id for family in whole.families for t in family.templates]
    assert len(ids) == 18 and len(set(ids)) == 18
    assert {i.rsplit("_", 1)[1] for i in ids} == {"table", "floor"}
    assert whole.fixed_eval[1].text.endswith("onto the table, static camera")


def test_the_final_set_is_the_floor_domain_the_pilot_left(
    sets: dict[str, PromptSetSettings],
) -> None:
    final = {t.id: t.holdout for family in sets[FINAL].families for t in family.templates}
    assert final == FINAL_TEMPLATES
    for pilot, surface in zip(PILOTS, ("table", "floor"), strict=True):
        assert all(t.id.endswith(f"_{surface}") for f in sets[pilot].families for t in f.templates)
    assert sets[FINAL].fixed_eval[1].text.endswith("onto the floor, static camera")


def test_the_pilot_texts_are_the_ones_it_ran_on(sets: dict[str, PromptSetSettings]) -> None:
    for pilot, expected in PILOT_UNIVERSES.items():
        assert stable_hash(sorted(_universe(sets[pilot]))) == expected, pilot


def test_v1_is_the_table_it_was() -> None:
    raw = read_config_files(Path(__file__).resolve().parents[2] / "configs")[0]
    assert stable_hash(raw["prompt_sets"][V1]) == V1_HASH


def test_the_template_table_is_the_one_the_pilot_ran_on() -> None:
    raw = read_config_files(Path(__file__).resolve().parents[2] / "configs")[0]
    assert stable_hash(raw["prompt_library"]) == LIBRARY_HASH


def test_the_fixed_preference_values(
    settings: Settings, sets: dict[str, PromptSetSettings]
) -> None:
    cfg = load_finetune(settings)
    final = cfg.prompts
    assert cfg.prompt_set == FINAL and cfg.scorer == "graded" and final.domain == DOMAIN
    assert (final.seed, final.n_train, final.n_dev, final.n_eval) == (20261116, 200, 32, 80)
    assert (final.eval_unseen_fraction, final.val_fraction) == (0.5, 0.1)
    k = cfg.candidates
    assert (k.k_train, k.k_eval, k.k_eval_finetuned, k.k_dev) == (4, 8, 4, 2)
    assert (cfg.pairs.min_margin, cfg.pairs.sigma_k) == (0.10, 1.0)
    rule = cfg.stop_rule
    assert (rule.min_pairs_per_prompt, rule.max_candidate_abstention) == (0.3, 0.5)
    assert (rule.inspected, rule.min_better_fraction, rule.min_pairs) == (10, 0.6, 54)
    extra = final.extra_train
    assert extra is not None and (extra.seed, extra.n, extra.tag) == (20261117, 120, EXTRA_TAG)
    assert {t.id: t.holdout for f in extra.families for t in f.templates} == EXTRA_TEMPLATES
    assert extra.slots["color"] == [*EXTRA_COLORS] and extra.slots["floor"] == final.slots["floor"]
    assert {f.name: f.target_checks for f in final.families} == TARGET_CHECKS
    assert final.slots["floor"] == ["wooden floor", "concrete floor", "tiled floor"]
    assert final.slots["color"] == [*FINAL_COLORS]
    for pilot in PILOTS:
        assert (sets[pilot].seed, sets[pilot].n_train) == (20261115, 36)
        assert sets[pilot].slots["color"] == ["yellow"]
        assert sets[pilot].n_dev == sets[pilot].n_eval == 0 == sets[pilot].val_fraction


def test_the_extra_train_prompts_are_seen_blocks_in_one_new_color(
    settings: Settings, sets: dict[str, PromptSetSettings], fake_labels: frozenset[str]
) -> None:
    final = sets[FINAL]
    check_universe(final, load_domain(settings, build(PROMPT_DOMAINS.name), DOMAIN))
    records = build_prompt_set(final, fake_labels)
    base, extras = records[:-120], records[-120:]
    assert not any(EXTRA_TAG in r.tags for r in base)
    assert {r.split for r in extras} == {"train"}
    assert all(r.tags == ["template:tt_collide_blocks_floor", EXTRA_TAG] for r in extras)
    colors = [set(r.text.replace(",", " ").split()) & set(EXTRA_COLORS) for r in extras]
    assert all(len(found) == 1 for found in colors)
    others = {*_universe(final), *(f.text for f in final.fixed_eval)}
    others |= {text for pilot in PILOTS for text in _universe(sets[pilot])}
    texts = {r.text for r in extras}
    assert len(texts) == 120 and not texts & others
    assert not {r.prompt_id for r in extras} & {r.prompt_id for r in base}


def test_the_pilots_share_no_text_with_the_final_set(sets: dict[str, PromptSetSettings]) -> None:
    final = {*_universe(sets[FINAL]), *(fixed.text for fixed in sets[FINAL].fixed_eval)}
    for pilot in PILOTS:
        assert not set(_universe(sets[pilot])) & final, pilot
    assert not set(_universe(sets[PILOTS[0]])) & set(_universe(sets[PILOTS[1]]))


def test_no_template_text_is_written_twice_in_configs() -> None:
    raw = read_config_files(Path(__file__).resolve().parents[2] / "configs")[0]
    texts = list(_texts(raw))
    assert len(texts) > 30 and len(set(texts)) == len(texts)


def test_a_dropped_shape_must_be_a_seen_one(make_settings: SettingsFactory) -> None:
    holdout = make_settings(f"prompt_sets.{FINAL}.library.drop_shapes=['tt_bounce_onto']")
    with pytest.raises(ConfigError, match="seen shapes to drop"):
        load_prompt_set(holdout, FINAL)
    dropped = load_prompt_set(
        make_settings(f"prompt_sets.{FINAL}.library.drop_shapes=['tt_bounce_near']"), FINAL
    )
    ids = {t.id for family in dropped.families for t in family.templates}
    assert len(ids) == 5 and "tt_collide_mixed_floor" in ids
    assert not {"tt_bounce_near_table", "tt_bounce_near_floor"} & ids


def test_finetune_names_its_prompts_only_through_a_set(make_settings: SettingsFactory) -> None:
    with pytest.raises(ConfigError, match="unknown prompt set"):
        load_finetune(make_settings("finetune.prompt_set=nope"))


@pytest.mark.parametrize("name", PILOTS)
def test_prompts_build_prints_counts_and_hashes_never_texts(
    cli: CliRunner, env: dict[str, str], labels: None, name: str
) -> None:
    built = cli(
        "--profile", "precise", "--set", f"finetune.prompt_set={name}",
        "--set", f"finetune.campaign=wan21_{name}", "finetune", "prompts", "build", env=env,
    )  # fmt: skip
    assert built.code == 0, built.err
    path = Path(re.findall(r"^prompts: (\S+) sha256 [0-9a-f]{64}$", built.out, re.MULTILINE)[0])
    texts = [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(texts) == 36 and "train: 36 (" in built.out and "eval lines sha256" in built.out
    assert not any(text in built.out + built.err for text in texts)


def test_a_one_body_template_stops_the_build_before_writing(
    cli: CliRunner, config_dir: Path, env: dict[str, str], data_dir: Path, labels: None
) -> None:
    with (config_dir / "studies" / "prompt_sets.toml").open("a", encoding="utf-8") as handle:
        handle.write(ONE_BODY_SET)
    result = cli(
        "--set", "finetune.prompt_set=one_body", "finetune", "prompts", "build", env=env
    )  # fmt: skip
    assert result.code == 2
    assert "template solo_drop:" in result.err and "single_body" in result.err
    assert "falls" not in result.err + result.out
    assert not list(data_dir.rglob("prompts.jsonl"))

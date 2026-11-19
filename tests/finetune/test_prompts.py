from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pytest

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.finetune.prompts import (
    build_prompt_set,
    check_extra,
    check_vocabulary,
    is_covered,
    largest_remainder,
    only_appends,
)
from physverify.finetune.records import DEMO_TAG, UNSEEN_TAG, VAL_TAG, PromptRecord
from physverify.finetune.settings import PromptSetSettings, load_finetune, load_prompt_set

V1 = "wan21_v1"


def _cfg(settings: Settings) -> PromptSetSettings:
    return load_finetune(settings).prompts


def _text(records: list[PromptRecord]) -> str:
    return "".join(r.model_dump_json() + "\n" for r in records)


def _n_extra(cfg: PromptSetSettings) -> int:
    return 0 if cfg.extra_train is None else cfg.extra_train.n * len(cfg.extra_train.families)


def test_the_set_is_byte_identical_across_builds(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    assert _text(build_prompt_set(cfg, fake_labels)) == _text(build_prompt_set(cfg, fake_labels))


def test_another_seed_changes_the_set(settings: Settings, fake_labels: frozenset[str]) -> None:
    cfg = _cfg(settings)
    other = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert _text(build_prompt_set(cfg, fake_labels)) != _text(build_prompt_set(other, fake_labels))


def test_splits_are_disjoint_and_have_the_configured_sizes(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    records = build_prompt_set(cfg, fake_labels)
    counts = Counter(r.split for r in records)
    assert counts == {
        "train": cfg.n_train + _n_extra(cfg),
        "dev": cfg.n_dev,
        "eval": cfg.n_eval + len(cfg.fixed_eval),
    }
    texts = [r.text for r in records]
    assert len(set(texts)) == len(texts)
    assert len({r.prompt_id for r in records}) == len(records)


def test_family_quotas_differ_by_at_most_one(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    records = build_prompt_set(cfg, fake_labels)[: -_n_extra(cfg) or None]
    for split in ("train", "dev"):
        per_family = Counter(r.family for r in records if r.split == split)
        assert max(per_family.values()) - min(per_family.values()) <= 1


def test_largest_remainder_sums_to_the_total() -> None:
    assert largest_remainder(10, [1, 1, 1]) == [4, 3, 3]
    assert sum(largest_remainder(200, [1.0] * 7)) == 200


def test_holdout_templates_only_reach_eval_tagged_unseen(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    holdout = {t.id for f in cfg.families for t in f.templates if t.holdout}
    records = build_prompt_set(cfg, fake_labels)
    for record in records:
        if record.template_id in holdout:
            assert record.split == "eval"
            assert UNSEEN_TAG in record.tags
        else:
            assert UNSEEN_TAG not in record.tags
    unseen = sum(UNSEEN_TAG in r.tags for r in records)
    assert unseen == pytest.approx(cfg.n_eval * cfg.eval_unseen_fraction, abs=len(cfg.families))


def test_about_a_tenth_of_train_is_tagged_val(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    records = build_prompt_set(cfg, fake_labels)
    val = [r for r in records if VAL_TAG in r.tags]
    assert all(r.split == "train" for r in val)
    assert len(val) == pytest.approx(cfg.n_train * cfg.val_fraction, abs=len(cfg.families))


def test_demo_prompts_are_in_eval(settings: Settings, fake_labels: frozenset[str]) -> None:
    cfg = _cfg(settings)
    demos = [r for r in build_prompt_set(cfg, fake_labels) if DEMO_TAG in r.tags]
    assert {r.prompt_id for r in demos} == {f.id for f in cfg.fixed_eval}
    assert all(r.split == "eval" for r in demos)


def test_a_noun_without_a_size_prior_is_a_config_error(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    with pytest.raises(ConfigError, match="cup"):
        build_prompt_set(load_prompt_set(settings, V1), fake_labels - {"cup"})


def test_unscaled_queries_are_the_declared_exceptions(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = load_prompt_set(settings, V1)
    records = build_prompt_set(cfg, fake_labels)
    assert {"domino", "table"} == set(cfg.unscaled_queries)
    with pytest.raises(ConfigError, match="domino"):
        check_vocabulary(records, fake_labels, ["table"])


def test_labels_match_with_spaces_or_underscores() -> None:
    assert is_covered("Cereal box", {"cereal_box"}, ())
    assert is_covered("tennis ball", {"tennis ball"}, ())
    assert not is_covered("chair", {"cup"}, ())


def test_the_fast_profile_is_a_two_prompt_smoke_set(
    make_settings: Callable[..., Settings], fake_labels: frozenset[str]
) -> None:
    cfg = load_finetune(make_settings(profile="fast"))
    records = build_prompt_set(cfg.prompts, fake_labels)
    assert Counter(r.split for r in records)["train"] == 2
    assert cfg.candidates.k_train == 2


def test_the_extra_train_prompts_follow_the_unchanged_set(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    assert cfg.extra_train is not None and _n_extra(cfg) == 120
    records = build_prompt_set(cfg, fake_labels)
    without = build_prompt_set(cfg.model_copy(update={"extra_train": None}), fake_labels)
    assert records[:-120] == without
    extras = records[-120:]
    assert {r.split for r in extras} == {"train"} and not any(VAL_TAG in r.tags for r in extras)
    assert all(cfg.extra_train.tag in r.tags and UNSEEN_TAG not in r.tags for r in extras)


def test_only_extra_train_lines_may_be_appended(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
    records = build_prompt_set(cfg, fake_labels)
    base, full, last = _text(records[:-120]), _text(records), records[-1]
    assert only_appends(base, full, cfg) and only_appends(_text(records[:-1]), full, cfg)
    assert not only_appends(_text(records[:-121]), full, cfg)
    assert not only_appends(full, base, cfg) and not only_appends(base.rstrip("\n"), full, cfg)
    for changed in (
        last.model_copy(update={"split": "dev"}),
        last.model_copy(update={"tags": last.tags[:1]}),
        last.model_copy(update={"tags": [*last.tags, VAL_TAG]}),
    ):
        assert not only_appends(base, _text([*records[:-1], changed]), cfg)
    assert not only_appends(base, full, cfg.model_copy(update={"extra_train": None}))


def test_extra_train_prompts_are_new_seen_prompts(
    make_settings: Callable[..., Settings], fake_labels: frozenset[str]
) -> None:
    extra = "prompt_sets.rigid_tabletop.extra_train"
    old_colors = make_settings(
        f"{extra}.slots.color=['red', 'blue']", f"{extra}.n=10", profile="precise"
    )
    with pytest.raises(ConfigError, match="10 extra_train prompts lie in the base universe"):
        build_prompt_set(load_prompt_set(old_colors, "rigid_tabletop"), fake_labels)
    too_many = make_settings(f"{extra}.n=127", profile="precise")
    with pytest.raises(ConfigError, match="needs 127 prompts but only 126 remain"):
        build_prompt_set(load_prompt_set(too_many, "rigid_tabletop"), fake_labels)
    cfg = load_prompt_set(make_settings(profile="precise"), "rigid_tabletop")
    records = build_prompt_set(cfg, fake_labels)
    balls = [r for r in records if r.template_id == "tt_collide_balls_floor"]
    assert balls and UNSEEN_TAG in balls[0].tags
    with pytest.raises(ConfigError, match="holdout template"):
        check_extra(cfg, balls[:1])

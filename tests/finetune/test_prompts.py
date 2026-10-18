from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pytest

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.finetune.prompts import (
    build_prompt_set,
    check_vocabulary,
    is_covered,
    largest_remainder,
)
from physverify.finetune.records import DEMO_TAG, UNSEEN_TAG, VAL_TAG, PromptRecord
from physverify.finetune.settings import PromptSetSettings, load_finetune


def _cfg(settings: Settings) -> PromptSetSettings:
    return load_finetune(settings).prompts


def _text(records: list[PromptRecord]) -> str:
    return "".join(r.model_dump_json() + "\n" for r in records)


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
        "train": cfg.n_train,
        "dev": cfg.n_dev,
        "eval": cfg.n_eval + len(cfg.fixed_eval),
    }
    texts = [r.text for r in records]
    assert len(set(texts)) == len(texts)
    assert len({r.prompt_id for r in records}) == len(records)


def test_family_quotas_differ_by_at_most_one(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    records = build_prompt_set(_cfg(settings), fake_labels)
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
        build_prompt_set(_cfg(settings), fake_labels - {"cup"})


def test_unscaled_queries_are_the_declared_exceptions(
    settings: Settings, fake_labels: frozenset[str]
) -> None:
    cfg = _cfg(settings)
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

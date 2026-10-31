"""One function lists every label that gets a size."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import SizePrior, SizePriorTable
from physverify.sizing.model import (
    ClassSize,
    SizeModelTable,
    labels_with_size,
    model_hash,
    pooled_prior,
    predict,
    prior_aliases,
    resolve,
    sized_labels,
)
from physverify.sizing.priors import GENERIC_PRIOR, PRIMITIVE_WORDS
from physverify.sizing.settings import FixedSize, SizeLookup, sizing_settings

CUP = ClassSize(mu_log=math.log(0.09), sigma_log=0.15, n_objects=10, n_videos=9)
MODEL = SizeModelTable(sigma_factor=0.9, classes={"cup": CUP, "cereal_box": CUP})
LOOKUP = SizeLookup(
    labels={"mug": "cup", "cereal box": "cereal_box", "dodo": "absent"},
    fixed={"tennis ball": FixedSize(mean_m=0.067, rel_spread=0.05, source="ITF")},
)


def test_classes_synonyms_and_fixed_sizes_resolve() -> None:
    cup = predict("red mug", MODEL, LOOKUP)
    assert cup is not None
    assert cup.mean_m == pytest.approx(0.09)
    assert cup.rel_spread == pytest.approx(0.135)
    assert predict("cereal box", MODEL, LOOKUP) == predict("cereal_box", MODEL, LOOKUP)
    assert predict("tennis ball", MODEL, LOOKUP) == SizePrior(mean_m=0.067, rel_spread=0.05)
    assert predict("a tennis racket", MODEL, LOOKUP) is None
    assert predict("dodo", MODEL, LOOKUP) is None
    assert predict("ball", MODEL, LOOKUP) is None


def test_fixed_sizes_declare_at_least_five_percent() -> None:
    with pytest.raises(ValueError, match=r"greater than or equal to 0\.05"):
        FixedSize(mean_m=0.067, rel_spread=0.01, source="ITF")


def test_the_label_set_joins_priors_classes_and_synonyms() -> None:
    priors = SizePriorTable({"cube": SizePrior(mean_m=0.1, rel_spread=0.5)})
    labels = labels_with_size(priors, MODEL, LOOKUP, {"box": "cube", "orb": "absent"})
    assert {"cube", "box", "cup", "cereal_box", "mug", "cereal box", "tennis ball"} <= labels
    assert {"dodo", "orb"}.isdisjoint(labels)


REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_sized_labels_of_the_repo_configs_include_the_provider_aliases() -> None:
    settings = load_settings(REPO_CONFIGS, environ={})
    labels = sized_labels(settings)
    assert {"cube", "ball", "cylinder", "tennis ball", "basketball", "mug"} <= labels
    assert set(prior_aliases()) <= labels


@pytest.mark.parametrize("label", ["tennis racket", "football", "american football"])
def test_a_lone_sport_word_gets_no_regulation_ball_size(label: str) -> None:
    sizing = load_settings(REPO_CONFIGS, environ={}).sections
    lookup = SizeLookup.model_validate(sizing.get("sizing", {}))
    assert resolve(label, SizeModelTable(), lookup) is None
    assert resolve("tennis ball", SizeModelTable(), lookup) is not None


def test_the_hash_follows_the_model() -> None:
    other = MODEL.model_copy(update={"sigma_factor": 1.0})
    assert model_hash(MODEL, LOOKUP) != model_hash(other, LOOKUP)
    assert len(model_hash(MODEL, LOOKUP)) == 8


def test_the_pooled_prior_matches_the_log_moments_of_the_class_mixture() -> None:
    small = ClassSize(mu_log=-2.0, sigma_log=0.1, n_objects=10, n_videos=9)
    large = ClassSize(mu_log=0.0, sigma_log=0.3, n_objects=99, n_videos=90)
    prior = pooled_prior(SizeModelTable(classes={"a": small, "b": large}))
    assert prior.mean_m == pytest.approx(math.exp(-1.0))
    assert prior.rel_spread == pytest.approx(math.sqrt((0.1**2 + 0.3**2) / 2 + 1.0))


def test_the_generic_prior_is_the_pooled_objectron_train_model_rounded() -> None:
    settings = load_settings(REPO_CONFIGS, environ={})
    priors = settings.section("size_priors", SizePriorTable).root
    pooled = pooled_prior(settings.section("size_model", SizeModelTable))
    assert set(priors) == {GENERIC_PRIOR}
    assert priors[GENERIC_PRIOR].mean_m == pytest.approx(pooled.mean_m, abs=0.005)
    assert priors[GENERIC_PRIOR].rel_spread == pytest.approx(pooled.rel_spread, abs=0.005)
    assert sizing_settings(settings).baseline_prior == GENERIC_PRIOR
    assert all(prior_aliases()[word] == GENERIC_PRIOR for word in PRIMITIVE_WORDS)

"""The held-out monotonicity primary scenes per dataset; a fresh render was never seen."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from calibration_cases import clip

from physverify.calibration.session import primary_scope
from physverify.config.loader import load_settings
from physverify.core.errors import ConfigError
from physverify.schema.clip import Split
from physverify.synth.scene_scope import SceneScope, h1_scope

SEEN = frozenset(range(11, 40))
FRESH = frozenset(range(1, 21))
SMALL = frozenset(range(1, 11))


@pytest.fixture
def shipped(repo_root: Path) -> SceneScope:
    return h1_scope(load_settings(repo_root / "configs", environ={}))


def test_each_dataset_has_its_primary_scenes(shipped: SceneScope) -> None:
    assert shipped.indices_for("kubric_synthetic") == SEEN
    assert shipped.indices_for("kubric_fresh") == FRESH
    assert shipped.indices_for("kubric_fresh_ood") == SMALL
    assert shipped.indices_for("kubric_fresh_blind") == SMALL
    assert shipped.indices_for("anything_else") == shipped.indices == SEEN


def test_s000_is_primary_nowhere(shipped: SceneScope) -> None:
    names = ("kubric_synthetic", *shipped.by_dataset)
    assert all(0 not in shipped.indices_for(name) for name in names)


def test_a_bad_per_dataset_spec_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="h1_scope"):
        SceneScope(by_dataset={"kubric_fresh": ("s3-s1",)})


class _Source:
    def __init__(self, name: str) -> None:
        self.name = name

    def ground_truth(self, clip: Any) -> None:
        return None


@pytest.mark.parametrize(
    ("dataset", "kept"),
    [("kubric_synthetic", {"a-s011", "a-s039"}), ("kubric_fresh", {"a-s001", "a-s011"})],
)
def test_the_evaluated_dataset_picks_the_scope(
    repo_root: Path, dataset: str, kept: set[str]
) -> None:
    settings = load_settings(repo_root / "configs", environ={})
    session = cast(Any, SimpleNamespace(settings=settings))
    clips = [clip(f"c{i}", f"a-s{i:03d}", Split.EVAL) for i in (0, 1, 11, 39)]
    primary = primary_scope(session, cast(Any, _Source(dataset)), "eval", clips)
    assert primary is not None
    assert primary.scope.primary_scenes == h1_scope(settings).scenes_for(dataset)
    assert primary.dataset.name == dataset
    assert primary.dataset.n_scenes == len(kept)

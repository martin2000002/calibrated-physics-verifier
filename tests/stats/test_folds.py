"""Scene-grouped folds."""

from __future__ import annotations

from physverify.stats.folds import grouped_folds


def test_folds_never_split_a_scene() -> None:
    scenes = [f"s{i % 7}" for i in range(30)]
    folds = grouped_folds(scenes, 5, seed=3)
    assert len(folds) == 5
    assert set().union(*folds) == set(scenes)
    assert sum(len(f) for f in folds) == 7
    assert grouped_folds(["a", "b"], 5, seed=0) == grouped_folds(["b", "a"], 5, seed=0)
    assert len(grouped_folds(["a", "b"], 5, seed=0)) == 2

"""The scene bootstrap."""

from __future__ import annotations

from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap


def test_the_bootstrap_resamples_whole_scenes() -> None:
    blocks = [
        ClusterBlock("A", 1),
        ClusterBlock("A", 1),
        ClusterBlock("B", 0),
        ClusterBlock("C", 0),
    ]
    ci = scene_bootstrap(blocks, lambda s: {"odd": float(sum(s) % 2)}, 300, 0, 0.95)
    assert (ci["odd"].low, ci["odd"].high) == (0.0, 0.0)

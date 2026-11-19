"""``synth.plan.family_of``: the exact inverse of ``scene_id_of``."""

from __future__ import annotations

import pytest

from physverify.calibration.table import in_family
from physverify.synth.plan import family_of, scene_id_of

FAMILIES = ("f1_projectile", "f2_bounce", "a-s", "a-s1", "x")


@pytest.mark.parametrize(
    ("scene_id", "family"),
    [
        ("s000", None),
        ("f2-s007", "f2"),
        ("a-s-s001", "a-s"),
        ("f1_projectile-s1234", "f1_projectile"),
        ("f1-s01", None),
        ("f1-s0001", None),
        ("f1-s", None),
        ("-s001", None),
        ("f1-s00x", None),
        ("f1_projectile_s000", None),
    ],
)
def test_family_of_reads_only_scene_ids_of_the_convention(
    scene_id: str, family: str | None
) -> None:
    assert family_of(scene_id) == family


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("index", [0, 7, 999, 1000, 12345])
def test_family_of_inverts_scene_id_of(family: str, index: int) -> None:
    assert family_of(scene_id_of(family, index)) == family


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "scene_id", [*(scene_id_of(f, i) for f in FAMILIES for i in (0, 42, 1000)), "s000", "x-s01"]
)
def test_in_family_is_family_of(scene_id: str, family: str) -> None:
    assert in_family(scene_id, family) == (family_of(scene_id) == family)

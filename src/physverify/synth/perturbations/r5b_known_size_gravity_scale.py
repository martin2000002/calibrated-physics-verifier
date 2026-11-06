"""R5b on the known-size family: the gravity scale k = 1 + sigma m of r5b_gravity_scale on f6."""

from __future__ import annotations

from typing import ClassVar

from physverify.synth.perturbations._gravity_scale import GravityScaleBase


class KnownSizeGravityScale(GravityScaleBase):
    name: ClassVar[str] = "r5b_known_size_gravity_scale"
    families: ClassVar[tuple[str, ...]] = ("f6_known_size",)


PLUGINS = (KnownSizeGravityScale,)

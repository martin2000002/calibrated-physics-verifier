"""R5b on the known-size family: ``r5b_gravity_scale`` (k = 1 + sigma m) applied to f6 scenes."""

from __future__ import annotations

from typing import ClassVar

from physverify.synth.perturbations.r5b_gravity_scale import GravityScale


class KnownSizeGravityScale(GravityScale):
    name: ClassVar[str] = "r5b_known_size_gravity_scale"
    families: ClassVar[tuple[str, ...]] = ("f6_known_size",)


PLUGINS = (KnownSizeGravityScale,)

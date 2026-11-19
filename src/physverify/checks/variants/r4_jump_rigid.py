"""R4_jump rigid2: v2 without the stencils where the object's size steps, in the statistic and in
its noise estimate."""

from __future__ import annotations

from physverify.checks.variants._rigid_jump import RigidJumpCheck

PLUGINS = (RigidJumpCheck,)

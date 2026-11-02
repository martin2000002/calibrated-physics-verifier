"""R4_jump rigid2_dfloor: rigid2 with the depth axis of sigma-hat floored at the depth sigma the
lift declares."""

from __future__ import annotations

from physverify.checks.variants._declared_depth_jump import DeclaredDepthJumpCheck

PLUGINS = (DeclaredDepthJumpCheck,)

"""Back-projection of tracked masks through depth and intrinsics."""

from __future__ import annotations

from physverify.extraction.providers._backproject import BackprojectLift

PLUGINS = (BackprojectLift,)

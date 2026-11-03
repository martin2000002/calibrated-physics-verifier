"""R4_size v3_window_depth: v3 on extents rescaled to the object's running-median depth."""

from __future__ import annotations

from physverify.checks.variants._window_depth_size import WindowDepthSizeCheck

__all__ = ("WindowDepthSizeCheck",)

PLUGINS = (WindowDepthSizeCheck,)

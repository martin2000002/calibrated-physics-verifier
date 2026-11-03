"""R4_size v3: a persistent change of the size relative to the other bodies, against its running
median, with time windows in seconds and rows at the image border left out."""

from __future__ import annotations

from physverify.checks.variants._detrended_size import DetrendedSizeCheck, DetrendedSizeParams

__all__ = ("DetrendedSizeCheck", "DetrendedSizeParams")

PLUGINS = (DetrendedSizeCheck,)

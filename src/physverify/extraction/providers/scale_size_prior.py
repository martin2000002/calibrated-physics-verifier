"""Scale from the known typical size of labelled objects."""

from physverify.extraction.providers._size_prior import SizePriorScale

PLUGINS = (SizePriorScale,)
__all__ = ("PLUGINS", "SizePriorScale")

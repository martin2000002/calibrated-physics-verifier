"""R4_jump rigid2_dfloor_null: rigid2_dfloor that abstains when the track's own stencils away
from the peak reject the white-noise null the statistic is read under."""

from __future__ import annotations

from physverify.checks.variants._null_tail_jump import NullTailJumpCheck

PLUGINS = (NullTailJumpCheck,)

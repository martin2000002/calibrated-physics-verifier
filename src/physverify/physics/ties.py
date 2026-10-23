"""Tolerance under which two candidates for a largest value count as tied."""

from __future__ import annotations

TIE_RTOL = 1e-9
"""Relative slack under which two values count as the same maximum, so a tie resolves to the
earliest candidate (a teleport inflates two consecutive rows equally; two kink gains of one row)."""

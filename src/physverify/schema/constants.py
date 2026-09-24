"""Physical and structural constants shared by host and container."""

from __future__ import annotations

STANDARD_GRAVITY = 9.81
"""Reference gravity g0 in m/s^2 (not Kubric's default of 10)."""

ENV_ID = "__env__"
"""Object id of the static environment (ground and fixed scenery) in contact records."""

MAX_MANIFOLD_POINTS = 8
"""Upper bound on the vertices kept per contact manifold."""

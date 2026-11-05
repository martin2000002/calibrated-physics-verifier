"""Targets and clearance shared by the interpenetration perturbations (R1 and R1_dyn)."""

from __future__ import annotations

PEN_LEVELS: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)
MIN_CLEARANCE_M = 0.01
SOLVE_TOL = 1e-6

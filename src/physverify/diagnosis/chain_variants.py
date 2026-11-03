"""Chain variants: the prefix ladder V0..V8, scale variants and the 2^3 factorial cells.

V_k takes ground truth for the first k stages of ``STAGE_ORDER``: stage k is measured as
V_k (prediction) against V_{k+1} (truth), so both sides share the same true upstream.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from physverify.config.models import Settings
from physverify.extraction.routes import with_providers
from physverify.schema.stages import STAGE_ORDER, Stage

FACTORS: tuple[Stage, ...] = (Stage.TRACK, Stage.DEPTH, Stage.SCALE)

Cell = tuple[bool, bool, bool]
"""Whether each of ``FACTORS`` runs perceived (True) or ground truth (False)."""

CELLS: tuple[Cell, ...] = tuple(
    (track, depth, scale) for track, depth, scale in itertools.product((False, True), repeat=3)
)


@dataclass(frozen=True, eq=False)
class Variant:
    """One chain configuration. ``name`` is unique and is the key to use in mappings."""

    name: str
    gt_stages: frozenset[Stage]
    overrides: Mapping[Stage, str] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def gt_order(self) -> tuple[Stage, ...]:
        return tuple(stage for stage in STAGE_ORDER if stage in self.gt_stages)


def ladder() -> tuple[Variant, ...]:
    """V0 (fully perceived) .. V8 (fully ground truth, the oracle)."""
    return tuple(Variant(f"V{k}", frozenset(STAGE_ORDER[:k])) for k in range(len(STAGE_ORDER) + 1))


def perceived_variant() -> Variant:
    """V0: every stage perceived."""
    return ladder()[0]


def oracle_variant() -> Variant:
    """V8: every stage ground truth."""
    return ladder()[-1]


def truth_covers(truth: Variant, stage: Stage) -> bool:
    """Stage k is measured against a truth whose ground truth covers DETECT .. stage k."""
    return frozenset(STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]) <= truth.gt_stages


def scale_variants(providers: Sequence[str]) -> tuple[Variant, ...]:
    """V5 (true DETECT..LIFT) with each listed SCALE provider."""
    base = frozenset(STAGE_ORDER[: STAGE_ORDER.index(Stage.SCALE)])
    return tuple(
        Variant(f"V5/scale={name}", base, MappingProxyType({Stage.SCALE: name}))
        for name in providers
    )


def cell_name(cell: Cell) -> str:
    return "F_" + "".join("p" if perceived else "t" for perceived in cell)


def cell_ground_truth(cell: Cell, fixed: frozenset[Stage]) -> frozenset[Stage]:
    truth = {stage for stage, perceived in zip(FACTORS, cell, strict=True) if not perceived}
    return fixed | truth


def factorial_cells(fixed: frozenset[Stage]) -> tuple[Variant, ...]:
    """The 8 cells in ``CELLS`` order, F_ttt .. F_ppp."""
    return tuple(Variant(cell_name(cell), cell_ground_truth(cell, fixed)) for cell in CELLS)


def apply_overrides(settings: Settings, variant: Variant) -> Settings:
    return with_providers(settings, variant.overrides)

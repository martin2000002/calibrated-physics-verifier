"""Substitution arms: each of masks, depth and contacts comes from ground truth or perception.

A training clip draws its arm once, each stage independently with probability ``share`` of
ground truth, from a hash of the seed, the clip and the stage, so the draw is reproducible and
does not depend on the order clips are visited. Every arm is a chain variant (``Variant``), so
the same cache and plans serve training, the factorial and the substitution attribution.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping

from physverify.diagnosis.chain_variants import Variant
from physverify.schema.checks import InputLevel
from physverify.schema.stages import Stage

ARM_STAGES: Mapping[str, tuple[Stage, ...]] = {
    "masks": (Stage.DETECT, Stage.TRACK),
    "depth": (Stage.DEPTH,),
    "contacts": (Stage.CONTACTS,),
}
Arm = tuple[bool, ...]
"""Whether each of ``ARM_STAGES`` (in order) is ground truth."""

ARMS: tuple[Arm, ...] = tuple(itertools.product((False, True), repeat=len(ARM_STAGES)))
HASH_RANGE = 2**64


def arm_name(arm: Arm) -> str:
    return "A_" + "".join("t" if truth else "p" for truth in arm)


def arm_variant(arm: Arm) -> Variant:
    stages = (
        s for truth, group in zip(arm, ARM_STAGES.values(), strict=True) if truth for s in group
    )
    return Variant(arm_name(arm), frozenset(stages))


def arm_level(arm: Arm) -> InputLevel:
    if all(arm):
        return InputLevel.ORACLE
    return InputLevel.MIXED if any(arm) else InputLevel.PERCEIVED


def uniform(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / HASH_RANGE


def drawn_arm(clip_id: str, seed: int, share: float) -> Arm:
    return tuple(uniform(seed, clip_id, stage) < share for stage in ARM_STAGES)

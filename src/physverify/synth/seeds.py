"""Deterministic generators: each scene and each perturbation attempt derives its own seed from the
dataset seed and its identity, so adding scenes, subsets or plugins never shifts existing draws."""

from __future__ import annotations

import math

import numpy as np

from physverify.core.cache_keys import params_hash

_HEX_SCALE = 16**8


def _hash_int(payload: dict[str, object], digits: int) -> int:
    return int(params_hash(payload)[:digits], 16)


def scene_seed(seed: int, family: str, index: int) -> int:
    """Seed of scene ``index`` of ``family``, kept in ``SceneSpec.seed`` to redraw the scene."""
    return _hash_int({"seed": seed, "family": family, "index": index}, 15)


def scene_rng(seed: int, family: str, index: int) -> np.random.Generator:
    return np.random.default_rng(scene_seed(seed, family, index))


def perturbation_seed(
    seed: int, family: str, index: int, name: str, level: int, attempt: int
) -> int:
    payload = {
        "seed": seed,
        "family": family,
        "index": index,
        "perturbation": name,
        "level": level,
        "attempt": attempt,
    }
    return _hash_int(payload, 15)


def split_phase(seed: int, family: str) -> float:
    """Offset in [0, 1) of the low-discrepancy calib/eval sequence of one family."""
    return _hash_int({"seed": seed, "family": family}, 8) / _HEX_SCALE


def is_calib(index: int, fraction: float, phase: float) -> bool:
    """Scene ``index`` is calib when floor((i + 1) f + phase) - floor(i f + phase) == 1: any prefix
    of scenes is split within one of ``fraction``, so smaller subsets nest in larger ones."""
    return math.floor((index + 1) * fraction + phase) - math.floor(index * fraction + phase) == 1

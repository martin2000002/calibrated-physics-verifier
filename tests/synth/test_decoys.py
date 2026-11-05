"""Decoys on clean f5 and f4 clips: they draw the positive's seed stream, so when both accept the
same attempt they pick its body and direction; and they leave every truth measure where the clean
clip had it."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from physverify.synth.magnitudes import record_cuts
from physverify.synth.perturb import SeedKey, instantiate, perturb_trajectory
from physverify.synth.settings import DatasetSettings, perturbation_params
from physverify.synth.truth_measures._base import measure_truth

PAIRS = {
    "r1_penetration_decoy": ("r1_penetration", "f5_contact", "R1"),
    "r2_unsupported_rest_decoy": ("r2_unsupported_rest", "f4_stack", "R2"),
}
CASES = [(decoy, source) for decoy in sorted(PAIRS) for source in ("recorded", "analytic")]


def tolerance(value: float) -> float:
    return max(1e-3, 0.02 * abs(value))


def edited(plugins: dict[str, Any], name: str, clip: Any, family: str, level: int, ds: Any) -> Any:
    target = perturbation_params(type(plugins[name]), ds).levels[level - 1]
    key = SeedKey(ds.seed, family, 0, level)
    return perturb_trajectory(plugins[name], clip.scene, clip.states, clip.contacts, target, key)


@pytest.fixture(scope="module")
def plugins(perturbations: Any, clean_ds: DatasetSettings) -> dict[str, Any]:
    return instantiate(perturbations, clean_ds)


def test_decoys_are_labelled_clean(plugins: dict[str, Any]) -> None:
    for decoy, (positive, family, _) in PAIRS.items():
        assert plugins[decoy].target_checks == ()
        assert plugins[decoy].seed_stream == positive
        assert plugins[decoy].families == plugins[positive].families == (family,)
        assert plugins[decoy].magnitude_unit == plugins[positive].magnitude_unit


@pytest.mark.parametrize(("decoy", "source"), CASES)
def test_decoy_shares_the_positive_choices(
    decoy: str, source: str, bank: Any, clean_ds: DatasetSettings, plugins: dict[str, Any]
) -> None:
    positive, family, _ = PAIRS[decoy]
    clip = getattr(bank, source)(family)
    paired = 0
    for level in (1, 2, 3, 4):
        *_, fake = edited(plugins, decoy, clip, family, level, clean_ds)
        *_, real = edited(plugins, positive, clip, family, level, clean_ds)
        key = SeedKey(clean_ds.seed, family, 0, level)
        stream = {key.attempt_seed(positive, a) for a in range(plugins[decoy].params.max_attempts)}
        assert fake.seed in stream, level
        if fake.seed != real.seed:
            continue
        paired += 1
        assert fake.target_objects == real.target_objects, level
        fake_offset = np.asarray(fake.params["offset_m"])
        real_offset = np.asarray(real.params["offset_m"])
        if decoy == "r1_penetration_decoy":
            assert np.allclose(fake_offset, -real_offset), level
        else:
            assert fake.params["support"] == real.params["support"]
            cos = (
                fake_offset
                @ real_offset
                / np.linalg.norm(fake_offset)
                / np.linalg.norm(real_offset)
            )
            assert abs(abs(cos) - 1.0) < 1e-9, level
    assert paired >= 1


@pytest.mark.parametrize(("decoy", "source"), CASES)
def test_decoy_leaves_the_truth_clean(
    decoy: str,
    source: str,
    bank: Any,
    clean_ds: DatasetSettings,
    plugins: dict[str, Any],
    truth_measures: Any,
) -> None:
    _, family, own = PAIRS[decoy]
    clip = getattr(bank, source)(family)
    tol = clean_ds.contacts.tol_m
    base = measure_truth(truth_measures, clip.scene, clip.states, clip.contacts, tol)
    for level, target in enumerate(perturbation_params(type(plugins[decoy]), clean_ds).levels, 1):
        states, contacts, record = edited(plugins, decoy, clip, family, level, clean_ds)
        after = measure_truth(
            truth_measures, clip.scene, states, contacts, tol, record_cuts(record)
        )
        assert after[own] <= max(base[own], 0.0) + tolerance(base[own]), (level, after[own])
        rose = {
            k
            for k in base
            if math.isfinite(base[k]) and after[k] > base[k] + tolerance(base[k]) and k != own
        }
        assert not rose, (level, {k: (base[k], after[k]) for k in rose})
        if decoy == "r2_unsupported_rest_decoy":
            assert abs(record.magnitude + target) <= tolerance(target), level
        else:
            assert record.magnitude <= tolerance(target), level

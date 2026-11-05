from dataclasses import replace
from pathlib import Path

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check
from physverify.checks.r2_static_equilibrium import EquilibriumCheck
from physverify.checks.variants._learned import (
    HAS_IMAGE,
    LeakageError,
    LearnedArtifact,
    dense_layout,
    image_key,
    model_inputs,
    require_disjoint,
)
from physverify.checks.variants.r1_learned import (
    TRACK_COLUMNS as R1_TRACK,
)
from physverify.checks.variants.r1_learned import (
    LearnedInterpenetrationCheck,
    LearnedInterpenetrationParams,
)
from physverify.checks.variants.r2_learned import (
    TRACK_COLUMNS as R2_TRACK,
)
from physverify.checks.variants.r2_learned import (
    LearnedEquilibriumCheck,
    LearnedEquilibriumParams,
)
from physverify.core.errors import ConfigError
from physverify.core.feature_table import FeatureTable
from physverify.schema.checks import Verdict
from physverify.schema.constants import ENV_ID
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.stats.mlp import Standardiser, TwoHeadWeights

IMAGE = ("zone_px_share", "depth_gap_ab")
TOKENS = ("token_000", "token_001")
RMS = 0.04


def artifact(check_id: str, track: tuple[str, ...], seeds: frozenset[int]) -> LearnedArtifact:
    dense = dense_layout(track, IMAGE)
    rng = np.random.default_rng(0)
    weights = TwoHeadWeights(
        ((rng.normal(size=(len(dense), 4)), rng.normal(size=4)),),
        (rng.normal(size=(4, 1)), np.array([0.1])),
        (np.array([[0.5], [-0.25]]), np.array([0.0])),
    )
    ones = Standardiser(np.zeros(len(dense)), np.ones(len(dense)))
    token_std = Standardiser(np.zeros(2), np.ones(2))
    return LearnedArtifact(
        check_id, "u", dense, TOKENS, weights, ones, token_std, RMS, "enc", seeds
    )


def files(tmp_path: Path, art: LearnedArtifact, rows: dict[str, list[float]]) -> dict[str, str]:
    art_path, table_path = tmp_path / "artifact.npz", tmp_path / "features.npz"
    sha = art.save(art_path)
    FeatureTable.from_rows(rows, (*IMAGE, *TOKENS), {}).save(table_path)
    from physverify.core.digests import file_sha256

    return {
        "artifact_path": str(art_path),
        "artifact_sha256": sha,
        "features_path": str(table_path),
        "features_sha256": file_sha256(table_path),
    }


def with_seed(case: sc.Case, seed: int) -> sc.Case:
    record = PerturbationRecord(
        "r1_penetration", ("R1",), 0.1, "pen_over_size", MagnitudeKind.CHECK_UNIT, 1, (), None, seed
    )
    return replace(case, clip=replace(case.clip, perturbation=record))


def test_r1_rescores_object_pairs_with_an_estimated_magnitude(tmp_path: Path) -> None:
    case = sc.overlapping_spheres(0.05, (0.1, 0.15))
    inputs = case.inputs()
    key = image_key(inputs.clip.clip_id, inputs.provenance, "ball_b", "ball_a")
    art = artifact("R1", R1_TRACK, frozenset({7}))
    params = LearnedInterpenetrationParams(**files(tmp_path, art, {key: [0.2, -0.1, 1.0, 2.0]}))
    check = LearnedInterpenetrationCheck(params)
    assert isinstance(check, Check)
    results = check.evaluate(inputs)
    assert all(ENV_ID not in r.subject for r in results)
    (result,) = results
    subject = check.subjects(inputs)[0]
    dense, tokens = model_inputs(subject.track, (np.array([0.2, -0.1]), np.array([1.0, 2.0])), 2, 2)
    expected = art.estimate(dense[None, :], tokens[None, :])[0]
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(expected)
    assert result.sigma == RMS
    assert result.details["estimated"] is True
    assert result.details["base_magnitude"] == pytest.approx(0.25)
    assert result.unit == "pen_over_size"


def test_a_trained_clip_is_refused(tmp_path: Path) -> None:
    case = with_seed(sc.overlapping_spheres(), 7)
    params = LearnedInterpenetrationParams(
        **files(tmp_path, artifact("R1", R1_TRACK, frozenset({7})), {})
    )
    with pytest.raises(LeakageError):
        LearnedInterpenetrationCheck(params).evaluate(case.inputs())
    other = with_seed(sc.overlapping_spheres(), 8)
    assert LearnedInterpenetrationCheck(params).evaluate(other.inputs())
    with pytest.raises(LeakageError):
        require_disjoint([1, 2, 3], [3, 4])
    require_disjoint([1, 2], [3])


def test_artifact_and_features_are_named_by_hash_and_paths_are_not_hashed(tmp_path: Path) -> None:
    named = files(tmp_path, artifact("R1", R1_TRACK, frozenset()), {})
    params = LearnedInterpenetrationParams(**named)
    dumped = params.model_dump()
    assert dumped["artifact_sha256"] == named["artifact_sha256"]
    assert "artifact_path" not in dumped and "features_path" not in dumped
    wrong = LearnedInterpenetrationParams(**{**named, "artifact_sha256": "0" * 64})
    with pytest.raises(ConfigError):
        LearnedInterpenetrationCheck(wrong).evaluate(sc.overlapping_spheres().inputs())
    other_track = files(tmp_path, artifact("R1", R2_TRACK, frozenset()), {})
    with pytest.raises(ConfigError):
        LearnedInterpenetrationCheck(LearnedInterpenetrationParams(**other_track)).evaluate(
            sc.overlapping_spheres().inputs()
        )


def test_artifact_round_trip(tmp_path: Path) -> None:
    art = artifact("R2", R2_TRACK, frozenset({1, 5}))
    art.save(tmp_path / "a.npz")
    loaded = LearnedArtifact.load(tmp_path / "a.npz")
    assert loaded.meta == art.meta and loaded.training_seeds == art.training_seeds
    x = np.arange(len(art.dense_columns), dtype=float)[None, :]
    np.testing.assert_allclose(
        loaded.estimate(x, np.ones((1, 2))), art.estimate(x, np.ones((1, 2)))
    )
    assert HAS_IMAGE in art.dense_columns


def test_r2_is_in_overhang_units_and_scores_without_an_image_row(tmp_path: Path) -> None:
    params = LearnedEquilibriumParams(**files(tmp_path, artifact("R2", R2_TRACK, frozenset()), {}))
    check = LearnedEquilibriumCheck(params)
    assert check.magnitude_unit == "overhang_over_size"
    assert check.replaced_unit == EquilibriumCheck.magnitude_unit
    (result,) = check.evaluate(sc.overhang(0.25).inputs())
    assert result.verdict is Verdict.UNSCORED
    assert result.details["image_pairs"] == 0
    assert result.unit == "overhang_over_size"

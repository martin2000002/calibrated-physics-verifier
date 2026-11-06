from dataclasses import replace

import numpy as np
import pytest

from physverify.app.learned import fit
from physverify.checks.variants._learned import LeakageError
from physverify.extraction.contact_crops import CropSettings
from physverify.extraction.crop_encoder import CropEncoder
from physverify.extraction.pair_features import (
    ZONE_MAX,
    PairFrames,
    dense_columns,
    middle_frame,
    pair_row,
    token_columns,
)
from physverify.learned.arms import ARMS, arm_level, arm_name, arm_variant, drawn_arm
from physverify.learned.controls import (
    ClipScore,
    ControlSettings,
    clip_scores,
    decoy_control,
    hdri_control,
    invisible_control,
    oracle_gate,
    perceived_gate,
    single_frame_control,
)
from physverify.learned.samples import Role, Sample, SampleSet, clip_role, target
from physverify.learned.train import TrainSettings, is_validation, split, train
from physverify.learned.visibility import (
    PositiveEvidence,
    invisible,
    offset_invisible,
    ray_angle_deg,
    zone_invisible,
)
from physverify.schema.checks import InputLevel
from physverify.schema.scene import CameraSpec, MagnitudeKind, PerturbationRecord
from physverify.schema.stages import Stage

FAST = ControlSettings(n_boot=200)


def record(
    name: str, checks: tuple[str, ...], objects: tuple[str, ...] = ("a", "b")
) -> PerturbationRecord:
    return PerturbationRecord(
        name, checks, 0.2, "pen_over_size", MagnitudeKind.CHECK_UNIT, 1, objects, None, 7
    )


def test_arms_cover_every_truth_pattern_and_name_their_stages() -> None:
    assert len(set(ARMS)) == 8
    assert arm_level((True, True, True)) is InputLevel.ORACLE
    assert arm_level((False, False, False)) is InputLevel.PERCEIVED
    assert arm_level((True, False, False)) is InputLevel.MIXED
    variant = arm_variant((True, False, True))
    assert variant.name == arm_name((True, False, True)) == "A_tpt"
    assert variant.gt_stages == frozenset({Stage.DETECT, Stage.TRACK, Stage.CONTACTS})


def test_the_arm_draw_is_reproducible_and_near_its_share() -> None:
    assert drawn_arm("clip-1", 3, 0.5) == drawn_arm("clip-1", 3, 0.5)
    draws = np.array([drawn_arm(f"clip-{i}", 3, 0.5) for i in range(2000)])
    assert np.allclose(draws.mean(axis=0), 0.5, atol=0.05)
    assert not any(any(drawn_arm(f"c{i}", 3, 0.0)) for i in range(50))


def test_roles_and_targets() -> None:
    positive = record("r1_penetration", ("R1",))
    decoy = record("r1_penetration_decoy", ())
    assert clip_role(None, "R1") is Role.CLEAN
    assert clip_role(record("none", ()), "R1") is Role.CLEAN
    assert clip_role(decoy, "R1") is Role.DECOY
    assert clip_role(positive, "R1") is Role.POSITIVE
    assert clip_role(record("r2_offset", ("R2",)), "R1") is None
    assert target(positive, Role.POSITIVE, ("a", "b")) == pytest.approx(0.2)
    assert target(positive, Role.POSITIVE, ("a", "c")) == 0.0
    assert target(decoy, Role.DECOY, ("a", "b")) == 0.0


def synthetic(n_scenes: int, seed: int = 0) -> SampleSet:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_scenes):
        for role in Role:
            x = rng.normal(size=3)
            label = 0.5 * max(x[0], 0.0) if role is Role.POSITIVE else 0.0
            dense = np.array([x[0] if role is Role.POSITIVE else -abs(x[0]), x[1], x[2]])
            samples.append(
                Sample(f"s{i}-{role}", f"s{i}", ("a", "b"), "A_ppp", role, label, dense, x[:2])
            )
    return SampleSet(tuple(samples), ("d0", "d1", "d2"), ("token_000", "token_001"))


def test_the_split_keeps_a_scene_together() -> None:
    fit_set, held = split(synthetic(100), 0.2)
    assert {s.scene_id for s in fit_set.samples}.isdisjoint({s.scene_id for s in held.samples})
    assert all(is_validation(s.scene_id, 0.2) for s in held.samples)
    assert 0.1 < len(held.samples) / 300 < 0.3


def test_training_learns_and_stops_on_validation_loss_only() -> None:
    settings = TrainSettings(max_epochs=80, patience=10)
    trained = train(synthetic(150), settings, ("R1", "pen_over_size", "sha"), frozenset({1}))
    log = trained.log
    assert log.best_epoch == int(np.argmin(log.validation_loss))
    assert len(log.validation_loss) <= settings.max_epochs
    assert trained.artifact.residual_rms == pytest.approx(np.sqrt(min(log.validation_loss)))
    dense, tokens, y = trained.validation.arrays()
    assert np.mean((trained.artifact.estimate(dense, tokens) - y) ** 2) < 0.5 * np.var(y)
    assert trained.artifact.training_seeds == frozenset({1})
    again = train(synthetic(150), settings, ("R1", "pen_over_size", "sha"), frozenset({1}))
    assert again.log == log


def test_training_refuses_an_evaluated_set_sharing_a_seed() -> None:
    with pytest.raises(LeakageError):
        fit(synthetic(5), ("R1", "u", "sha"), frozenset({1, 2}), [{9}, {2}], TrainSettings())


def test_sample_set_round_trips_through_a_feature_table() -> None:
    samples = synthetic(4)
    back = SampleSet.from_table(samples.to_table())
    assert back.dense_columns == samples.dense_columns
    assert [s.key for s in back.samples] == [s.key for s in samples.samples]
    assert np.allclose(back.arrays()[0], samples.arrays()[0])


def scores(role: Role, values: list[float], prefix: str = "") -> list[ClipScore]:
    return [ClipScore(f"{prefix}{role}{i}", f"scene{i}", role, v) for i, v in enumerate(values)]


def test_a_clip_scores_its_largest_subject() -> None:
    keys = [("c1", "s1", Role.CLEAN), ("c1", "s1", Role.CLEAN), ("c2", "s2", Role.DECOY)]
    best = {c.clip_id: c.score for c in clip_scores(keys, [0.1, 0.4, 0.2])}
    assert best == {"c1": 0.4, "c2": 0.2}


def test_decoy_control_fails_when_decoys_fire_above_clean() -> None:
    clean = scores(Role.CLEAN, list(np.linspace(0.0, 1.0, 100)))
    quiet = decoy_control(clean + scores(Role.DECOY, [0.5] * 50), FAST)
    loud = decoy_control(clean + scores(Role.DECOY, [2.0] * 50), FAST)
    assert quiet.passed
    assert not loud.passed
    assert loud.value == pytest.approx(1.0 - 0.0, abs=0.05)


def test_invisible_control_passes_at_chance_and_fails_when_separable() -> None:
    rng = np.random.default_rng(0)
    clean = scores(Role.CLEAN, list(rng.normal(size=60)), "c")
    chance = scores(Role.POSITIVE, list(rng.normal(size=60)), "p")
    separable = scores(Role.POSITIVE, list(rng.normal(size=60) + 5.0), "p")
    assert invisible_control(chance, clean, FAST).passed
    assert not invisible_control(separable, clean, FAST).passed


def test_single_frame_control_fails_only_when_clip_features_add_signal() -> None:
    rng = np.random.default_rng(1)
    clip = scores(Role.POSITIVE, list(rng.normal(size=60) + 3.0)) + scores(
        Role.CLEAN, list(rng.normal(size=60))
    )
    blind = [replace(c, score=float(rng.normal())) for c in clip]
    assert not single_frame_control(clip, blind, FAST).passed
    assert single_frame_control(clip, clip, FAST).passed


def test_hdri_control_flags_a_large_drop() -> None:
    assert hdri_control(0.80, 0.75, FAST).passed
    assert not hdri_control(0.80, 0.60, FAST).passed


def test_gates_read_the_auroc_lower_bound() -> None:
    rng = np.random.default_rng(2)
    decoys = scores(Role.DECOY, list(rng.normal(size=60)), "d")
    strong = scores(Role.POSITIVE, list(rng.normal(size=60) + 4.0), "p")
    weak = scores(Role.POSITIVE, list(rng.normal(size=60) + 0.1), "p")
    assert oracle_gate(strong, decoys, FAST).passed
    assert perceived_gate(strong, decoys, FAST).passed
    assert not oracle_gate(weak, decoys, FAST).passed


def test_invisibility_markers() -> None:
    camera = CameraSpec((0.0, -5.0, 1.0), (0.0, 0.0, 1.0))
    along = replace(record("r2_offset", ("R2",)), params={"offset_m": [0.0, -0.1, 0.0]})
    across = replace(record("r2_offset", ("R2",)), params={"offset_m": [0.1, 0.0, 0.0]})
    assert ray_angle_deg([0.0, 0.1, 0.0], camera) == pytest.approx(0.0)
    assert offset_invisible(along, camera)
    assert not offset_invisible(across, camera)
    assert offset_invisible(along, camera, max_angle_deg=0.0) is False
    assert offset_invisible(record("r2_offset", ("R2",)), camera) is None
    assert zone_invisible(0.0)
    assert not zone_invisible(0.01)


def test_invisibility_per_check_is_undetermined_without_its_evidence() -> None:
    camera = CameraSpec((0.0, -5.0, 1.0), (0.0, 0.0, 1.0))
    along = replace(record("r2_offset", ("R2",)), params={"offset_m": [0.0, -0.1, 0.0]})
    assert invisible("R1", PositiveEvidence(along, None, 0.0)) is True
    assert invisible("R1", PositiveEvidence(along, None, 0.3)) is False
    assert invisible("R1", PositiveEvidence(along, camera, None)) is None
    assert invisible("R1", PositiveEvidence(along, camera, float("nan"))) is None
    assert invisible("R2", PositiveEvidence(along, camera, None)) is True
    assert invisible("R2", PositiveEvidence(along, None, 0.0)) is None
    no_offset = record("r2_offset", ("R2",))
    assert invisible("R2", PositiveEvidence(no_offset, camera, 0.0)) is None
    assert invisible("R5", PositiveEvidence(along, camera, 0.0)) is None


def frames(meet: bool) -> PairFrames:
    t, h, w = 5, 32, 32
    a = np.zeros((t, h, w), dtype=bool)
    b = np.zeros((t, h, w), dtype=bool)
    a[:, 8:16, 8:16] = True
    b[:, 8:16, (16 if meet else 26) : (24 if meet else 30)] = True
    depth = np.full((t, h, w), 2.0)
    return PairFrames(np.zeros((t, h, w, 3)), a, b, depth, np.ones((t, h, w), dtype=bool))


def test_pair_row_matches_its_columns_and_reads_the_chosen_frames() -> None:
    encoder = CropEncoder(lambda batch: batch.mean(axis=(2, 3)), "sha")
    settings = CropSettings(size_px=16)
    columns = (*dense_columns(), *token_columns(3))
    touching = pair_row(frames(True), encoder, settings)
    apart = pair_row(frames(False), encoder, settings, middle_frame(5))
    assert touching.shape == apart.shape == (len(columns),)
    assert touching[columns.index(ZONE_MAX)] > 0.0
    assert apart[columns.index(ZONE_MAX)] == 0.0

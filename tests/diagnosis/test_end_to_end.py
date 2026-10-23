"""Level 2: pairing, agreement measures, flips and the fixture run."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from synthetic_clip import OBJECT_IDS

from physverify.config.models import Settings
from physverify.core.errors import MissingInputError
from physverify.diagnosis.chain_variants import oracle_variant, perceived_variant
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.end_to_end import (
    NO_POLICY,
    PERCEIVED_VS_V8,
    V8_VS_ORACLE,
    Decision,
    agreement,
    clip_rows,
    flip_stats,
    run_end_to_end,
    variant_outputs,
    variant_results,
)
from physverify.diagnosis.pairing import Paired, ResultPair, pair_results, span_iou
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.matching import IdMap, MatchMethod

CHECK = "r_test"
CLIP = "clip"
MIN_IOU = 0.1
PERCEIVED_PREFIX = "p_"
"""Id prefix of the conftest's perceived DETECT fake."""


def result(
    subject: tuple[str, ...],
    magnitude: float | None,
    span: tuple[int, int] | None = (0, 20),
    *,
    sigma: float = math.nan,
    verdict: Verdict | None = None,
    reason: AbstainReason | None = None,
    check_id: str = CHECK,
) -> CheckResult:
    abstains = magnitude is None
    return CheckResult(
        check_id=check_id,
        clip_id=CLIP,
        subject=subject,
        frame_span=span,
        magnitude=magnitude,
        sigma=sigma,
        unit="m",
        verdict=verdict or (Verdict.ABSTAIN if abstains else Verdict.UNSCORED),
        input_level=InputLevel.ORACLE,
        abstain_reason=reason or (AbstainReason.MISSING_INPUT if abstains else None),
    )


def id_map(pairs: dict[str, str], unmatched: tuple[str, ...] = ()) -> IdMap:
    return IdMap(
        pairs=tuple((pred, gt, 1.0) for pred, gt in pairs.items()),
        unmatched_pred=unmatched,
        unmatched_gt=(),
        method=MatchMethod.MASK_IOU,
        frames_used=(0,),
        min_score=0.3,
    )


def test_span_iou_on_inclusive_native_frames() -> None:
    assert span_iou((0, 9), (5, 14), MIN_IOU) == pytest.approx(5 / 15)
    assert span_iou((0, 9), (10, 19), MIN_IOU) == 0.0
    assert span_iou(None, None, MIN_IOU) == 1.0
    assert span_iou(None, (0, 9), MIN_IOU) == MIN_IOU


def test_pairs_by_subject_map_and_span() -> None:
    first, second = result(("ball",), 1.0, (0, 20)), result(("box",), 2.0, (0, 20))
    late = result(("box",), 3.0, (40, 60))
    env, pair = result((), 0.5, None), result(("ball", "box"), 0.2, (5, 9))
    oracle = (late, first, second, env, pair)
    p_ball = result(("p_0",), 1.5, (2, 18))
    p_box_late = result(("p_1",), 2.5, (41, 59))
    ghost = result(("p_2",), 9.0, (0, 20))
    p_env = result((), 0.4, None)
    p_pair = result(("p_1", "p_0"), 0.3, (5, 9))
    far = result(("p_0",), 7.0, (100, 120))
    other = (ghost, p_box_late, p_ball, p_env, p_pair, far)
    subjects = id_map({"p_0": "ball", "p_1": "box"}, unmatched=("p_2",))

    paired = pair_results(oracle, other, subjects, MIN_IOU)

    by_oracle = {id(p.oracle): p.other for p in paired.pairs}
    assert by_oracle == {
        id(first): p_ball,
        id(late): p_box_late,
        id(env): p_env,
        id(pair): p_pair,
    }
    assert [p.subject for p in paired.pairs] == [(), ("ball",), ("ball", "box"), ("box",)]
    assert paired.missing == (second,)
    assert set(map(id, paired.spurious)) == {id(ghost), id(far)}
    assert paired.check_ids == (CHECK,)


def test_sampled_spans_pair_with_native_spans() -> None:
    """V8 runs on sampled frames (native indices, step 3) and OracleSource on every frame."""
    native = (result(("ball",), 1.0, (0, 59)), result(("ball",), 2.0, (60, 89)))
    sampled = (result(("ball",), 1.1, (0, 57)), result(("ball",), 2.1, (60, 87)))
    paired = pair_results(native, sampled, None, MIN_IOU)
    assert [(p.oracle.magnitude, p.other.magnitude) for p in paired.pairs] == [
        (1.0, 1.1),
        (2.0, 2.1),
    ]
    assert paired.missing == paired.spurious == ()


def test_pairs_below_the_span_floor_stay_apart() -> None:
    paired = pair_results(
        (result(("ball",), 1.0, (0, 99)),), (result(("ball",), 1.0, (99, 120)),), None, MIN_IOU
    )
    assert paired.pairs == ()
    assert len(paired.missing) == len(paired.spurious) == 1


def scripted() -> Paired:
    pairs = (
        (
            result(("a",), 1.0, verdict=Verdict.PASS),
            result(("a",), 1.5, sigma=0.25, verdict=Verdict.FAIL),
        ),
        (
            result(("b",), 2.0, verdict=Verdict.FAIL),
            result(("b",), 1.0, sigma=0.5, verdict=Verdict.FAIL),
        ),
        (result(("c",), 4.0, verdict=Verdict.PASS), result(("c",), 4.0, verdict=Verdict.PASS)),
        (result(("d",), 3.0, verdict=Verdict.PASS), result(("d",), None)),
    )
    missing = result(("e",), None, reason=AbstainReason.OCCLUDED)
    spurious = result(("f",), 7.0, verdict=Verdict.PASS)
    return Paired(
        pairs=tuple(ResultPair(CHECK, CLIP, o.subject, o, x) for o, x in pairs),
        missing=(missing,),
        spurious=(spurious,),
    )


def test_agreement_values_are_exact() -> None:
    (row,) = agreement(scripted(), (1.0, 2.0), Decision(available=True, provisional=True))
    assert (row.check_id, row.unit) == (CHECK, "m")
    assert (row.n_pairs, row.n_missing, row.n_spurious) == (4, 1, 1)
    deltas = (0.5, -1.0, 0.0)
    assert row.bias == pytest.approx(sum(deltas) / 3)
    assert row.mae == pytest.approx(0.5)
    assert row.rms == pytest.approx(math.sqrt((0.25 + 1.0) / 3))
    assert row.rel_error == pytest.approx((0.5 / 1.0 + 1.0 / 2.0 + 0.0) / 3)
    assert row.n_z == 2
    assert [(c.k, c.observed, c.n) for c in row.coverage] == [(1.0, 0.0, 2), (2.0, 1.0, 2)]
    assert row.abstain_reference == {
        AbstainReason.MISSING_INPUT: 0.0,
        AbstainReason.OCCLUDED: pytest.approx(0.2),
    }
    assert row.abstain_other == {
        AbstainReason.MISSING_INPUT: pytest.approx(0.2),
        AbstainReason.OCCLUDED: 0.0,
    }


def test_flip_rate_and_confusion() -> None:
    flips = flip_stats(scripted().pairs, Decision(available=True, provisional=True))
    assert flips.available and flips.provisional
    assert (flips.n_decided, flips.n_flips) == (3, 1)
    assert flips.rate == pytest.approx(1 / 3)
    expected = {ref: dict.fromkeys(Verdict, 0) for ref in Verdict}
    expected[Verdict.PASS][Verdict.FAIL] = 1
    expected[Verdict.FAIL][Verdict.FAIL] = 1
    expected[Verdict.PASS][Verdict.PASS] = 1
    expected[Verdict.PASS][Verdict.ABSTAIN] = 1
    assert flips.confusion == expected
    assert sum(sum(row.values()) for row in flips.confusion.values()) == 4


def test_flips_are_unavailable_without_a_policy() -> None:
    flips = flip_stats(scripted().pairs, Decision(available=False, reason=NO_POLICY))
    assert not flips.available
    assert flips.reason == NO_POLICY
    assert flips.rate is None and flips.confusion == {}


def test_clip_rows_list_pairs_missing_and_spurious() -> None:
    rows = clip_rows(PERCEIVED_VS_V8, scripted())
    assert [(r.subject, r.m_reference, r.m_other) for r in rows] == [
        (("a",), 1.0, 1.5),
        (("b",), 2.0, 1.0),
        (("c",), 4.0, 4.0),
        (("d",), 3.0, None),
        (("e",), None, None),
        (("f",), None, 7.0),
    ]
    assert rows[0].delta == pytest.approx(0.5)
    assert rows[3].delta is None
    assert (rows[0].verdict_reference, rows[0].verdict_other) == (Verdict.PASS, Verdict.FAIL)


ContextFactory = Callable[..., DiagnosisContext]
Suite = Any
"""The conftest's ``FakeCheckSuite`` (a fixture: conftest is not importable)."""


@pytest.fixture
def context(
    make_settings: Callable[..., Settings], make_diagnosis_context: ContextFactory
) -> Callable[[CheckSuite | None], DiagnosisContext]:
    settings = make_settings("diagnosis.dataset=synthetic_fixture")
    return lambda suite: make_diagnosis_context(settings, suite=suite)


def eval_clip(clips: Sequence[ClipManifest]) -> ClipManifest:
    return next(c for c in clips if c.split is Split.EVAL)


def magnitudes(
    ctx: DiagnosisContext, suite: CheckSuite, clip: ClipManifest, perceived: bool
) -> dict[tuple[str, ...], float]:
    """Magnitudes of one variant's results by subject, straight from the suite."""
    variant = perceived_variant() if perceived else oracle_variant()
    chain = ctx.chain(ctx.dataset())
    chain.warm([clip], [variant])
    level = InputLevel.PERCEIVED if perceived else InputLevel.ORACLE
    results = variant_results(suite, clip, variant_outputs(chain, clip, variant), level)
    return {r.subject: r.magnitude for r in results if r.magnitude is not None}


def script_truth_from_v8(ctx: DiagnosisContext, suite: Suite, clip: ClipManifest) -> None:
    """``OracleSource`` inputs equal to V8's, so the sampling comparison is exactly zero."""
    chain = ctx.chain(ctx.dataset())
    chain.warm([clip], [oracle_variant()])
    outputs = variant_outputs(chain, clip, oracle_variant())
    inputs = suite.inputs(clip, outputs.outputs, outputs.plan.names, InputLevel.ORACLE)
    suite.truth = {clip.clip_id: inputs}


def test_run_end_to_end_on_the_fixture(
    context: Callable[[CheckSuite | None], DiagnosisContext], fake_suite: Callable[..., Suite]
) -> None:
    suite = fake_suite()
    ctx = context(suite)
    clips = ctx.dataset().clips()
    clip = eval_clip(clips)
    script_truth_from_v8(ctx, suite, clip)
    oracle = magnitudes(ctx, suite, clip, perceived=False)
    perceived = magnitudes(ctx, suite, clip, perceived=True)
    (check_id,) = {r.check_id for r in suite.evaluator(suite.evaluated[0])}
    suite.thresholds = {check_id: 1.0}
    suite.evaluated.clear()

    report = run_end_to_end(ctx, clips)

    assert report.provenance.clips == (clip.clip_id,)
    assert report.provenance.provisional == ("flips",)
    assert report.failures == ()
    levels = [i.input_level for i in suite.evaluated]
    assert levels == [InputLevel.ORACLE, InputLevel.PERCEIVED, InputLevel.ORACLE]
    first, second = report.comparisons
    assert (first.name, first.reference, first.other) == (PERCEIVED_VS_V8, "V8", "V0")
    assert (second.name, second.reference, second.other) == (V8_VS_ORACLE, "oracle", "V8")
    (row,) = first.checks
    assert (row.check_id, row.n_pairs, row.n_missing, row.n_spurious) == (check_id, 2, 0, 0)
    deltas = [
        perceived[(f"{PERCEIVED_PREFIX}{object_id}",)] - oracle[(object_id,)]
        for object_id in OBJECT_IDS
    ]
    assert row.bias == pytest.approx(sum(deltas) / 2)
    assert row.rms == pytest.approx(math.sqrt(sum(d * d for d in deltas) / 2))
    assert row.flips.available and row.flips.provisional
    (sampling,) = second.checks
    assert (sampling.n_pairs, sampling.bias, sampling.rms) == (2, 0.0, 0.0)
    assert {(r.comparison, r.subject) for r in report.clips} == {
        (name, (object_id,)) for name in (PERCEIVED_VS_V8, V8_VS_ORACLE) for object_id in OBJECT_IDS
    }


def test_missing_oracle_source_and_policy_are_reported(
    context: Callable[[CheckSuite | None], DiagnosisContext], fake_suite: Callable[..., Suite]
) -> None:
    ctx = context(fake_suite())
    report = run_end_to_end(ctx, ctx.dataset().clips())
    (failure,) = report.failures
    assert (failure.variant, failure.error) == ("oracle", "no OracleSource inputs")
    first, second = report.comparisons
    assert second.checks == ()
    assert all(not row.flips.available and row.flips.reason == NO_POLICY for row in first.checks)
    assert report.provenance.provisional == ()


def test_end_to_end_needs_a_check_suite(
    context: Callable[[CheckSuite | None], DiagnosisContext],
) -> None:
    ctx = context(None)
    with pytest.raises(MissingInputError, match="check suite"):
        run_end_to_end(ctx, ctx.dataset().clips())


def test_end_to_end_needs_report_split_clips(
    context: Callable[[CheckSuite | None], DiagnosisContext], fake_suite: Callable[..., Suite]
) -> None:
    ctx = context(fake_suite())
    calib = [c for c in ctx.dataset().clips() if c.split is not Split.EVAL]
    with pytest.raises(MissingInputError, match="eval"):
        run_end_to_end(ctx, calib)


def test_abstaining_results_keep_their_reason_in_rates() -> None:
    oracle = (result(("a",), 1.0), result(("b",), None, reason=AbstainReason.TOO_FEW_FRAMES))
    other = (result(("a",), None, reason=AbstainReason.TOO_FEW_FRAMES), result(("b",), 1.0))
    (row,) = agreement(pair_results(oracle, other, None, MIN_IOU), (1.0,), Decision(False))
    assert row.n_pairs == 2 and row.n_z == 0 and math.isnan(row.bias)
    assert row.abstain_reference == {AbstainReason.TOO_FEW_FRAMES: 0.5}
    assert row.abstain_other == {AbstainReason.TOO_FEW_FRAMES: 0.5}
    assert not row.flips.available

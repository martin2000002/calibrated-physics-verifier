"""Report models: deterministic JSON, markdown tables, estimated/provisional labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from physverify.diagnosis.report import (
    MISSING,
    PROVISIONAL_NOTE,
    REAL_VIDEO_SCOPE,
    REPORT_DIR,
    AttributionReport,
    BudgetRow,
    CheckAgreementRow,
    CheckAttribution,
    ClipCheckRow,
    Comparison,
    Coverage,
    EndToEndReport,
    Failure,
    FallRow,
    FlipStats,
    MotionRow,
    ProfileEntryRow,
    Provenance,
    RealVideoReport,
    Report,
    StageErrorReport,
    StageMetricRow,
    VerdictDistribution,
    build_provenance,
    nominal_coverage,
    profile_hash,
    z_coverage,
)
from physverify.diagnosis.report_markdown import render_markdown, write_report
from physverify.runtime.runs import RunContext
from physverify.schema.checks import AbstainReason, Verdict
from physverify.schema.clip import Split
from physverify.schema.error_profile import ErrorEntry, ErrorProfile, no_profile_hash
from physverify.schema.stages import Stage
from physverify.stats.intervals import Interval

LEVELS = (1.0, 2.0)
CELLS = ("F_ttt", "F_ttp", "F_tpt", "F_tpp", "F_ptt", "F_ptp", "F_ppt", "F_ppp")


def provenance(
    run: RunContext, estimated: tuple[str, ...] = (), provisional: tuple[str, ...] = ()
) -> Provenance:
    return build_provenance(
        run,
        dataset="synthetic_fixture",
        dataset_versions={"synthetic_fixture": "analytic-1"},
        subset="local",
        error_profile=None,
        clips=("fixture_fall", "fixture_fall_calib"),
        estimated=estimated,
        provisional=provisional,
    )


def stage_report(run: RunContext) -> StageErrorReport:
    row = StageMetricRow(
        stage=Stage.TRACK,
        metric="track/mask",
        provider="diag_fake",
        pred_variant="V1",
        truth_variant="V2",
        split=Split.EVAL,
        n_clips=1,
        values={
            "centroid_px_rel.mean": 0.25,
            "centroid_px_rel.std": math.nan,
            "centroid_px_rel.rms": 0.25,
            "centroid_px_rel.n": 12.0,
            "id_switch_rate": 0.0,
            "id_switch_rate.num": 0.0,
            "id_switch_rate.den": 22.0,
        },
        coverage={"centroid_px_rel": z_coverage([0.5, -1.5, 2.5], LEVELS)},
    )
    entry = ProfileEntryRow(
        stage=Stage.TRACK,
        provider="diag_fake",
        quantity="centroid_px_rel",
        bias=0.25,
        sigma=math.nan,
        rms=0.25,
        n=12,
        written=False,
    )
    return StageErrorReport(
        provenance=provenance(run, estimated=("depth/absrel",)),
        profile_split=Split.CALIB,
        report_split=Split.EVAL,
        min_samples=30,
        profile_written=False,
        rows=(row,),
        profile=(entry,),
        failures=(Failure(clip_id="fixture_fall", error="boom", stage=Stage.DEPTH, variant="V0"),),
    )


def agreement(check_id: str, flips: FlipStats) -> CheckAgreementRow:
    return CheckAgreementRow(
        check_id=check_id,
        unit="m",
        n_pairs=4,
        n_missing=1,
        n_spurious=0,
        bias=0.01,
        mae=0.02,
        rms=0.03,
        rel_error=0.1,
        n_z=4,
        coverage=z_coverage([0.1, 0.9, 1.5, 3.0], LEVELS),
        abstain_reference={AbstainReason.OCCLUDED: 0.0},
        abstain_other={AbstainReason.OCCLUDED: 0.25},
        flips=flips,
    )


def end_to_end_report(run: RunContext) -> EndToEndReport:
    decided = FlipStats(
        available=True,
        provisional=True,
        rate=0.25,
        n_decided=4,
        n_flips=1,
        confusion={Verdict.PASS: {Verdict.PASS: 3, Verdict.FAIL: 1}},
    )
    missing = FlipStats(available=False, reason="no verdict policy")
    return EndToEndReport(
        provenance=provenance(run, provisional=("flips",)),
        report_split=Split.EVAL,
        comparisons=(
            Comparison(
                name="perceived_vs_v8",
                reference="V8",
                other="V0",
                checks=(agreement("r1", decided), agreement("r2", missing)),
            ),
            Comparison(
                name="v8_vs_oracle",
                reference="oracle",
                other="V8",
                checks=(agreement("r1", missing),),
            ),
        ),
        clips=(
            ClipCheckRow(
                comparison="perceived_vs_v8",
                clip_id="fixture_fall",
                check_id="r1",
                subject=("ball", "box"),
                m_reference=0.1,
                m_other=None,
                delta=None,
                verdict_reference=Verdict.PASS,
                verdict_other=None,
            ),
        ),
    )


def attribution_report(run: RunContext) -> AttributionReport:
    cells = {name: 0.01 * (i + 1) for i, name in enumerate(CELLS)}
    check = CheckAttribution(
        check_id="r5a",
        unit="m/s^2",
        n_blocks=3,
        n_clips=2,
        cells=cells,
        cells_ci={"F_ppp": Interval(low=0.05, high=0.11)},
        v0=0.1,
        effects={"TRACK": 0.04, "TRACK:DEPTH": 0.0},
        shapley={Stage.TRACK: 0.04, Stage.DEPTH: 0.02, Stage.SCALE: 0.01},
        budget=(
            BudgetRow(row="floor", value=0.01, percent=10.0),
            BudgetRow(row="track", value=0.04, percent=40.0, stage_metric="track/mask"),
        ),
        incomplete={"abstain:F_ppp": 2, "no_pair": 1},
        bootstrap_dropped=1,
    )
    return AttributionReport(
        provenance=provenance(run),
        report_split=Split.EVAL,
        fixed_ground_truth=(Stage.DETECT, Stage.CAMERA),
        lift_unit="relative",
        bootstrap=1000,
        seed=0,
        ci=0.95,
        checks=(check,),
    )


def real_report(run: RunContext) -> RealVideoReport:
    return RealVideoReport(
        provenance=provenance(run, estimated=("g",), provisional=("verdicts",)),
        motion=(
            MotionRow(
                dataset="physics_iq",
                clip_id="c1",
                scenario="balls-collide",
                dynamic=True,
                n_frames=48,
                motion_iou=0.5,
            ),
        ),
        falls=(
            FallRow(
                dataset="pisa_real",
                clip_id="p1",
                object_id="ball",
                n_frames=20,
                g_est=9.5,
                g_rel_error=-0.03,
                gravity_dir_estimated=True,
            ),
        ),
        verdicts=(
            VerdictDistribution(
                dataset="pisa_real",
                check_id="r5a",
                counts={Verdict.PASS: 2, Verdict.ABSTAIN: 1},
                fail_rate=0.0,
                abstentions={AbstainReason.SCALE_UNAVAILABLE: 1},
            ),
        ),
    )


BUILDERS = (stage_report, end_to_end_report, attribution_report, real_report)


def test_failure_json_is_sorted_with_null() -> None:
    text = Failure(clip_id="c", error="boom", stage=Stage.TRACK).to_json()
    assert text == (
        '{\n  "clip_id": "c",\n  "error": "boom",\n  "stage": "track",\n  "variant": null\n}\n'
    )


@pytest.mark.parametrize("build", BUILDERS)
def test_json_is_byte_identical_sorted_and_finite(run_context: RunContext, build: object) -> None:
    first: Report = build(run_context)  # type: ignore[operator]
    second: Report = build(run_context)  # type: ignore[operator]
    text = first.to_json()
    assert text == second.to_json()
    assert "NaN" not in text and "Infinity" not in text
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert render_markdown(first) == render_markdown(second)


def test_nan_becomes_null(run_context: RunContext) -> None:
    payload = json.loads(stage_report(run_context).to_json())
    assert payload["rows"][0]["values"]["centroid_px_rel.std"] is None
    assert payload["profile"][0]["sigma"] is None
    assert payload["provenance"]["run_id"] == run_context.run_id
    assert payload["provenance"]["created_utc"] == run_context.created_utc


def test_value_order_does_not_change_the_bytes(run_context: RunContext) -> None:
    report = stage_report(run_context)
    row = report.rows[0]
    reordered = row.model_copy(update={"values": dict(reversed(list(row.values.items())))})
    assert report.model_copy(update={"rows": (reordered,)}).to_json() == report.to_json()


def test_write_report_paths_and_contents(run_context: RunContext) -> None:
    report = end_to_end_report(run_context)
    json_path, md_path = write_report(run_context.run_dir, "end_to_end", report)
    assert json_path == run_context.run_dir / REPORT_DIR / "end_to_end.json"
    assert md_path == run_context.run_dir / REPORT_DIR / "end_to_end.md"
    assert json_path.read_text(encoding="utf-8") == report.to_json()
    assert md_path.read_text(encoding="utf-8") == render_markdown(report)
    before = json_path.read_bytes()
    write_report(run_context.run_dir, "end_to_end", end_to_end_report(run_context))
    assert json_path.read_bytes() == before


def test_provenance_sorts_and_hashes_the_profile(run_context: RunContext) -> None:
    prov = build_provenance(
        run_context,
        dataset="d",
        dataset_versions={},
        subset="local",
        error_profile=None,
        clips=("b", "a"),
        estimated=("x", "x", "a"),
    )
    assert prov.clips == ("a", "b")
    assert prov.estimated == ("a", "x")
    assert prov.error_profile_hash == no_profile_hash() == profile_hash(None)
    entry = ErrorEntry(
        stage=Stage.TRACK,
        provider="diag_fake",
        provider_version="1",
        quantity="centroid_px_rel",
        unit="",
        bias=0.0,
        sigma=0.1,
        n=40,
    )
    profile = ErrorProfile(
        created_utc=run_context.created_utc, run_id="r", commit="c", entries=(entry,)
    )
    assert profile_hash(profile) == profile.entries_hash([entry.key])
    assert profile_hash(profile) != no_profile_hash()


def test_nominal_and_observed_coverage() -> None:
    assert nominal_coverage(1.0) == pytest.approx(0.682689492, abs=1e-9)
    assert nominal_coverage(2.0) == pytest.approx(0.954499736, abs=1e-9)
    one, two = z_coverage([0.5, -1.5, 2.5, math.nan, math.inf], LEVELS)
    assert (one.k, one.n, two.k, two.n) == (1.0, 3, 2.0, 3)
    assert one.observed == pytest.approx(1 / 3)
    assert two.observed == pytest.approx(2 / 3)
    (empty,) = z_coverage([], (1.0,))
    assert empty.n == 0 and math.isnan(empty.observed)
    assert isinstance(empty, Coverage)


def test_stage_markdown_tables_and_labels(run_context: RunContext) -> None:
    text = render_markdown(stage_report(run_context))
    assert text.startswith("# Per-stage extractor errors (Level 1)\n")
    assert "| quantity | mean | std | rms | n |" in text
    assert f"| centroid_px_rel | 0.25 | {MISSING} | 0.25 | 12 |" in text
    assert "| id_switch_rate | 0 | 0 | 22 |" in text
    assert "| quantity | z coverage |" in text
    assert "\\|z\\|<=1: 0.3333 (nominal 0.6827)" in text
    assert "- [estimated] (not exact measurements): depth/absrel" in text
    assert "## Error profile entries" in text and "## Failures" in text


def test_end_to_end_flips_side_by_side(run_context: RunContext) -> None:
    text = render_markdown(end_to_end_report(run_context))
    assert f"- [provisional] {PROVISIONAL_NOTE}: flips" in text
    assert "| check | perceived_vs_v8 | v8_vs_oracle |" in text
    assert "| r1 | 0.25 (1/4) [provisional] | unavailable: no verdict policy |" in text
    assert f"| r2 | unavailable: no verdict policy | {MISSING} |" in text
    assert "## perceived_vs_v8: V0 against V8" in text
    assert "## v8_vs_oracle: V8 against oracle" in text
    assert (
        "| perceived_vs_v8 | fixture_fall | r1 | ball, box | 0.1 | n/a | n/a | PASS | n/a |" in text
    )


def test_attribution_lists_the_eight_cells_with_unit(run_context: RunContext) -> None:
    report = attribution_report(run_context)
    text = render_markdown(report)
    for name in CELLS:
        assert f"| {name} |" in text
    assert "unit `m/s^2`" in text
    assert "LIFT unit in every cell: `relative`" in text
    assert "Incomplete blocks: abstain:F_ppp 2, no_pair 1." in text
    assert "| F_ppp | 0.08 | 0.05 | 0.11 |" in text
    assert "| track | 0.04 | 40 | n/a | n/a | track/mask |" in text
    payload = json.loads(report.to_json())
    assert payload["lift_unit"] == "relative"
    assert sorted(payload["checks"][0]["cells"]) == sorted(CELLS)


def test_real_report_is_end_to_end_only(run_context: RunContext) -> None:
    report = real_report(run_context)
    payload = json.loads(report.to_json())
    assert payload["error_scope"] == "end_to_end_only"
    assert payload["thresholds_provisional"] is True
    text = render_markdown(report)
    assert REAL_VIDEO_SCOPE in text
    assert f"Thresholds are {PROVISIONAL_NOTE}." in text
    assert "g [estimated]" in text and "[estimated] camera +y" in text
    assert "SCALE_UNAVAILABLE 1" in text
    assert "IoU is agreement, not accuracy" in text


def test_real_report_rejects_another_scope(run_context: RunContext) -> None:
    with pytest.raises(ValueError):
        RealVideoReport(provenance=provenance(run_context), error_scope="per_stage")


def test_write_report_creates_the_directory(tmp_path: Path, run_context: RunContext) -> None:
    run_dir = tmp_path / "fresh"
    json_path, _ = write_report(run_dir, "real", real_report(run_context))
    assert json_path.parent == run_dir / REPORT_DIR and json_path.exists()

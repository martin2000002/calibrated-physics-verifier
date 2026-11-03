"""AC12: the runner with check doubles (9 rows per clip, aggregation, row_step, outputs)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import check_scenarios as sc
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.settings import calibration_settings
from physverify.checks._base import Check, CheckInputs, abstain, unscored
from physverify.checks._params import CommonParams
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.extraction.settings import SECTION as EXTRACTION
from physverify.extraction.settings import ExtractionSettings
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.io import read_jsonl, stable_hash
from physverify.schema.stages import Stage
from physverify.verification._base import POLICIES
from physverify.verification.check_set import CHECKS_SECTION, ChecksSettings, active_checks_hash
from physverify.verification.inputs import ClipInputs
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import (
    CHECKS_FILE,
    TABLE_FILE,
    RowStatus,
    TableRow,
    VerificationReport,
    VerificationRunner,
    VerificationSettings,
    aggregate,
    configured_suite,
    write_report,
)
from physverify.verification.sweep import CheckUnit
from physverify.verification.thresholding import THRESHOLD_POLICY, ThresholdTable

Behavior = Callable[[type[Check], CheckInputs], Sequence[CheckResult]]
REASONS = AbstainReason


def make_check(check_id: str, behavior: Behavior, requires: frozenset[str] = frozenset()) -> Any:
    def evaluate(self: Check, inputs: CheckInputs) -> Sequence[CheckResult]:
        return behavior(type(self), inputs)

    def init(self: Check, params: CommonParams | None = None) -> None:
        return None

    namespace = {
        "check_id": check_id,
        "requires": requires,
        "magnitude_unit": f"{check_id}_unit",
        "Params": CommonParams,
        "__init__": init,
        "evaluate": evaluate,
    }
    return type(f"Fake{check_id}", (), namespace)


def scored(*values: tuple[str, float, float]) -> Behavior:
    def run(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
        return [unscored(check, inputs, (s,), m, sd, (0, 3), {}) for s, m, sd in values]

    return run


def abstaining(*reasons: AbstainReason) -> Behavior:
    def run(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
        return [abstain(check, inputs, (f"o{i}",), r, None, {}) for i, r in enumerate(reasons)]

    return run


def no_event(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
    return []


def boom(check: type[Check], inputs: CheckInputs) -> Sequence[CheckResult]:
    raise RuntimeError("boom")


NINE = (
    make_check("F1", scored(("a", 0.2, 0.01), ("b", 0.4, 0.02))),
    make_check("F2", no_event),
    make_check("F3", abstaining(REASONS.OCCLUDED, REASONS.TOO_FEW_FRAMES, REASONS.OCCLUDED)),
    make_check("F4", boom),
    make_check("F5", scored(("a", 0.1, 0.0)), requires=frozenset({"contacts"})),
    make_check("F6", scored(("a", 0.1, 0.0)), requires=frozenset({"scene"})),
    make_check("F7", scored(("a", 5.0, 0.0))),
    make_check("F8", scored(("a", 0.5, math.nan))),
    make_check("F9", scored(("a", 0.1, 0.0), ("b", 0.9, 0.0))),
)
STALE_TABLE = "calibrated with other check versions, parameters or variants"
FORMER_FAST_CHECKS_HASH = "d01a5232ec8cb0ab038923811724d8e725347616e5d5a0aa123b90f1e3536448"
"""The fast ``checks_hash`` as f133fc8 computed it (parameters alone), over the current parameters:
891fd358... there, until R4_size gained ``axis_errors``."""
THRESHOLDS = ThresholdTable.provisional(dict.fromkeys(("F1", "F5", "F7", "F8"), 1.0))


class ListSource:
    def __init__(self, items: Sequence[ClipInputs]) -> None:
        self._items = {item.clip.clip_id: item for item in items}

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield self._items[clip.clip_id]


def prepared(case: sc.Case) -> ClipInputs:
    return ClipInputs(case.clip, case.inputs(), None)


def run(items: Sequence[ClipInputs], checks: Sequence[Any] = NINE) -> VerificationReport:
    runner = VerificationRunner([c() for c in checks], ThresholdPolicy(), THRESHOLDS)
    return runner.run(ListSource(items), [item.clip for item in items])


def rows_of(report: VerificationReport, clip_id: str) -> dict[str, TableRow]:
    return {row.check_id: row for row in report.table if row.clip_id == clip_id}


def test_nine_rows_per_clip_whatever_the_checks_do() -> None:
    a, b = sc.parabola(clip_id="a"), sc.overhang(0.0, clip_id="b")
    report = run([prepared(a), prepared(b)])
    assert len(report.table) == 9 * 2
    rows = rows_of(report, "a")
    assert rows["F1"].status is RowStatus.SCORED
    assert rows["F2"].status is RowStatus.NO_EVENT
    assert rows["F2"].verdict is None and rows["F2"].magnitude is None
    assert rows["F3"].status is RowStatus.ABSTAINED
    assert rows["F4"].status is RowStatus.ERROR
    assert rows["F5"].status is RowStatus.SCORED
    assert rows["F6"].status is RowStatus.ABSTAINED
    assert rows["F6"].abstain_reason is REASONS.MISSING_INPUT
    assert [c.clip_id for c in report.clips] == ["a", "b"]


def test_aggregation_takes_the_worst_subject_and_its_sigma() -> None:
    rows = rows_of(run([prepared(sc.parabola(clip_id="a"))]), "a")
    assert (rows["F1"].magnitude, rows["F1"].sigma) == (0.4, 0.02)
    assert (rows["F1"].n_subjects, rows["F1"].verdict) == (2, Verdict.PASS)
    assert (rows["F3"].abstain_reason, rows["F3"].n_abstained) == (REASONS.OCCLUDED, 3)
    assert rows["F7"].verdict is Verdict.FAIL
    assert rows["F8"].verdict is Verdict.ABSTAIN
    assert rows["F8"].abstain_reason is REASONS.UNKNOWN_UNCERTAINTY
    assert rows["F9"].verdict is Verdict.UNSCORED
    assert rows["F1"].unit == "F1_unit"


def test_any_failing_subject_fails_the_row() -> None:
    check = NINE[0]()
    inputs = sc.parabola().inputs()
    results = [
        replace(unscored(type(check), inputs, ("a",), 0.1, 0.0, None, {}), verdict=Verdict.PASS),
        replace(unscored(type(check), inputs, ("b",), 3.0, 0.0, None, {}), verdict=Verdict.FAIL),
    ]
    assert aggregate("clip", check, results).verdict is Verdict.FAIL
    assert aggregate("clip", check, results[:1]).verdict is Verdict.PASS
    assert aggregate("clip", check, []).status is RowStatus.NO_EVENT


def test_the_raw_policy_keeps_rows_unscored() -> None:
    items = [prepared(sc.parabola(clip_id="a"))]
    runner = VerificationRunner([c() for c in NINE], RawPolicy(), THRESHOLDS)
    report = runner.run(ListSource(items), [items[0].clip])
    assert rows_of(report, "a")["F7"].verdict is Verdict.UNSCORED
    assert all(r.threshold is None for r in report.results)


def test_row_step_is_the_native_stride_of_the_table() -> None:
    case = sc.strided(sc.parabola(clip_id="s"), 2, quantize=False)
    report = run([prepared(case)])
    assert {row.row_step for row in report.table} == {2}


def test_a_failing_check_gives_an_error_row_and_the_others_run() -> None:
    report = run([prepared(sc.parabola(clip_id="a"))])
    errors = [r for r in report.results if r.check_id == "F4"]
    assert len(errors) == 1
    assert errors[0].verdict is Verdict.ABSTAIN
    assert errors[0].abstain_reason is REASONS.MISSING_INPUT
    assert errors[0].details["error"] == "RuntimeError: boom"
    assert rows_of(report, "a")["F7"].status is RowStatus.SCORED


def test_missing_required_inputs_abstain_naming_the_field() -> None:
    report = run([prepared(sc.parabola(clip_id="a"))])
    (missing,) = [r for r in report.results if r.check_id == "F6"]
    assert missing.details["missing"] == ["scene"]


def test_a_clip_without_inputs_abstains_on_every_check() -> None:
    clip = sc.parabola(clip_id="bad").clip
    item = ClipInputs(clip, None, "time_axis: not uniform", InputLevel.PERCEIVED)
    report = run([item])
    assert [r.status for r in report.table] == [RowStatus.ABSTAINED] * 9
    assert all(r.abstain_reason is REASONS.MISSING_INPUT for r in report.results)
    assert all(r.details["error"].startswith("time_axis") for r in report.results)
    assert {r.input_level for r in report.results} == {InputLevel.PERCEIVED}


def test_results_carry_the_threshold_origin() -> None:
    report = run([prepared(sc.parabola(clip_id="a"))])
    assert {r.provenance["thresholds"] for r in report.results} == {"provisional"}


def test_outputs_round_trip(tmp_path: Path) -> None:
    report = run([prepared(sc.parabola(clip_id="a")), prepared(sc.overhang(0.0, clip_id="b"))])
    write_report(tmp_path, report)
    assert read_jsonl(CheckResult, tmp_path / CHECKS_FILE) == list(report.results)
    first = json.loads((tmp_path / CHECKS_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert CheckResult.from_dict(first["data"]) == report.results[0]
    with (tmp_path / TABLE_FILE).open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 18
    assert {row["status"] for row in table} <= {s.value for s in RowStatus}
    assert {row["row_step"] for row in table} == {"1"}


@pytest.mark.parametrize("status", list(RowStatus))
def test_row_status_values_are_the_declared_ones(status: RowStatus) -> None:
    assert status.value in {"scored", "abstained", "no_event", "error"}


def test_configured_suite_builds_the_registered_checks_and_the_chosen_policy(
    make_settings: SettingsFactory,
) -> None:
    checks, policies = PLUGIN_KINDS["checks"](), PLUGIN_KINDS["policies"]()
    suite = configured_suite(make_settings(), checks, policies, policy=RawPolicy.name)
    assert [type(check).check_id for check in suite.checks] == list(checks)
    assert isinstance(suite.policy, RawPolicy)
    assert suite.verification.policy == RawPolicy.name
    assert not suite.thresholds.calibrated
    assert isinstance(suite.runner(), VerificationRunner)
    with pytest.raises(ConfigError, match="unknown verdict policy 'nope'"):
        configured_suite(make_settings(), checks, policies, policy="nope")


def test_a_provisional_tau_in_another_unit_than_its_check_stops_the_suite(
    make_settings: SettingsFactory,
) -> None:
    policies = PLUGIN_KINDS[POLICIES.name]()
    whitened = make_settings("check_variants.R5a=whitened")
    with pytest.raises(ConfigError, match=r"R5a \(tau in accel_rel_diff, check reports accel_wald"):
        configured_suite(whitened, plugins.configured("checks", whitened), policies)
    declared = make_settings(
        "check_variants.R5a=whitened",
        "verification.thresholds.R5a=11.34",
        "verification.threshold_units.R5a=accel_wald_chi2_3",
    )
    suite = configured_suite(declared, plugins.configured("checks", declared), policies)
    assert suite.thresholds.tau_of("R5a") == 11.34
    same_unit = make_settings("check_variants.R4_size=v3", "check_variants.R4_jump=rigid2")
    assert not configured_suite(
        same_unit, plugins.configured("checks", same_unit), policies
    ).thresholds.calibrated
    with pytest.raises(ConfigError, match=r"R1 \(tau in seconds, check reports pen_over_size"):
        mislabelled = make_settings("verification.threshold_units.R1=seconds")
        configured_suite(mislabelled, plugins.configured("checks", mislabelled), policies)


def test_the_precise_profile_runs_the_whitened_fall_with_a_threshold_in_its_unit(
    make_settings: SettingsFactory,
) -> None:
    policies = PLUGIN_KINDS[POLICIES.name]()
    precise = make_settings(profile="precise")
    checks = plugins.configured("checks", precise)
    assert checks["R5a"] is plugins.build("check_variants")["R5a/whitened"]
    assert configured_suite(precise, checks, policies).thresholds.tau_of("R5a") == 11.34
    v1 = make_settings("check_variants.R5a=default", profile="precise")
    mismatch = r"R5a \(tau in accel_wald_chi2_3, check reports accel_rel"
    with pytest.raises(ConfigError, match=mismatch):
        configured_suite(v1, plugins.configured("checks", v1), policies)


ARMS = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "perception_arms.tsv"


def _arm_swaps(name: str) -> list[str]:
    rows = ARMS.read_text(encoding="utf-8").splitlines()
    arms = dict(row.split("\t") for row in rows if row and not row.startswith("#"))
    return arms[name].split()


def test_the_k_arm_runs_the_line_of_sight_falls_under_precise(
    make_settings: SettingsFactory,
) -> None:
    k = make_settings(*_arm_swaps("k"), profile="precise")
    checks = plugins.configured("checks", k)
    variants = plugins.build("check_variants")
    assert checks["R5a"] is variants["R5a/los_tilt"]
    assert checks["R5b"] is variants["R5b/los_ballistic"]
    suite = configured_suite(k, checks, PLUGIN_KINDS["policies"]())
    assert suite.thresholds.tau_of("R5a") == 13.28
    assert calibration_settings(k).tolerance_of(CheckUnit.of(checks["R5a"])) == 1e-5
    assert k.section(EXTRACTION, ExtractionSettings).plan[Stage.DEPTH] == "vda_large"


def test_the_k_variants_without_the_tau_and_units_of_the_arm_are_refused_under_precise(
    make_settings: SettingsFactory,
) -> None:
    swaps = [s for s in _arm_swaps("k") if s.startswith(("extraction.", "check_variants."))]
    bare = make_settings(*swaps, profile="precise")
    with pytest.raises(
        ConfigError, match=r"R5a \(tau in accel_wald_chi2_3, check reports accel_tilt"
    ):
        configured_suite(bare, plugins.configured("checks", bare), PLUGIN_KINDS["policies"]())


@pytest.mark.parametrize("recorded", ["default", "whitened", None])
def test_a_calibrated_table_judges_only_the_checks_it_was_fitted_with(
    make_settings: SettingsFactory, recorded: str | None
) -> None:
    policies = PLUGIN_KINDS[POLICIES.name]()
    default = make_settings()
    fitted = None if recorded is None else make_settings(f"check_variants.R5a={recorded}")
    write_table(
        default,
        None
        if fitted is None
        else active_checks_hash(fitted, plugins.configured("checks", fitted)),
    )
    checks = plugins.configured("checks", default)
    if recorded == "default":
        assert configured_suite(default, checks, policies).thresholds.calibrated
        return
    with pytest.raises(ConfigError, match=STALE_TABLE):
        configured_suite(default, checks, policies)


def test_a_table_calibrated_before_check_versions_is_refused(
    make_settings: SettingsFactory,
) -> None:
    """The hash a table recorded before versions entered ``checks_hash`` (the
    parameters alone, as on f133fc8) no longer matches, so verify refuses the table."""
    policies = PLUGIN_KINDS[POLICIES.name]()
    settings = make_settings()
    checks = plugins.configured("checks", settings)
    former = former_checks_hash(checks, settings.section(CHECKS_SECTION, ChecksSettings).root)
    assert former == FORMER_FAST_CHECKS_HASH
    write_table(settings, former)
    with pytest.raises(ConfigError, match=STALE_TABLE):
        configured_suite(settings, checks, policies)


def former_checks_hash(checks: Mapping[str, type[Check]], overrides: Mapping[str, Any]) -> str:
    """``checks_hash`` as f133fc8 computed it, before versions: the effective parameters alone."""
    return stable_hash(
        {
            check_id
            if getattr(check, "variant", None) is None
            else f"{check_id}/{check.variant}": (
                check.Params.model_validate(overrides.get(check_id, {})).model_dump(mode="json")
            )
            for check_id, check in sorted(checks.items())
        }
    )


def write_table(settings: Settings, checks_hash: str | None) -> None:
    """A calibrated threshold table recorded with ``checks_hash``, where verify reads it."""
    table = ThresholdTable(
        status="calibrated",
        source="calibration.json#oracle/fast@run",
        checks={},
        checks_hash=checks_hash,
    )
    path = settings.paths.thresholds
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table.model_dump(mode="json")), encoding="utf-8")


def test_the_default_policy_is_the_registered_threshold_policy() -> None:
    policies = PLUGIN_KINDS[POLICIES.name]()
    assert VerificationSettings().policy == THRESHOLD_POLICY
    assert policies[THRESHOLD_POLICY] is ThresholdPolicy

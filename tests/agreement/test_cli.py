"""``physverify agreement`` end to end on the miniature tables and fake verify runs (C2, C5, C7)."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

CALIBRATION_TESTS = str(Path(__file__).resolve().parent.parent / "calibration")
if CALIBRATION_TESTS not in sys.path:
    sys.path.insert(0, CALIBRATION_TESTS)

from exclusion_cases import write_record  # noqa: E402
from stratum_cases import write_artifact  # noqa: E402

from physverify.agreement.prereg import file_sha256, load_preregistration  # noqa: E402
from physverify.agreement.report import REPORT_DIR, _sign_flip_lines  # noqa: E402
from physverify.calibration.loader import default_artifact_path  # noqa: E402
from physverify.calibration.settings import active_checks_hash  # noqa: E402
from physverify.datasets._videophy_review import PENDING, load_review  # noqa: E402
from physverify.schema.checks import InputLevel  # noqa: E402

FAST = ("--set", "agreement.bootstrap_n=50")


def _dev(world) -> Path:
    clips = {"v1d/g1": [world.PASS], "v1d/g2": [world.FAIL], "v1d/g3": [world.CROSSING]}
    return world.run("dev", clips)


def _test(world, **kwargs) -> Path:
    clips = {
        "v1t/a1": [world.PASS],
        "v1t/a2": [world.FAIL],
        "v1t/b1": [world.FAIL],
        "v1t/b2": [world.CROSSING],
        "v2t/f1": [world.NO_EVENT],
        "v2t/f2": [world.FAIL],
    }
    return world.run("test", clips, **kwargs)


def _frozen(cli, world) -> Path:
    world.keep_all()
    assert cli(*FAST, "agreement", "freeze", "--verify-run", str(_dev(world))).code == 0
    return _test(world)


def test_subset_writes_the_pending_review_and_the_query_drafts(cli, world) -> None:
    result = cli(*FAST, "agreement", "subset", "--audit")
    assert result.code == 0, result.err
    review = load_review(world.dataset.subset_path)
    assert review is not None and review.status == PENDING
    assert f"[?] {world.BALL}" in result.out
    queries = json.loads(world.dataset.queries_path.read_text(encoding="utf-8"))
    assert queries[world.BALL] == ["ball"]


def test_compare_on_the_test_subset_is_refused_while_the_review_is_pending(cli, world) -> None:
    assert cli(*FAST, "agreement", "subset").code == 0
    result = cli(*FAST, "agreement", "compare", "--verify-run", str(_test(world)))
    assert result.code == 1 and "hand review" in result.err  # MissingInputError


def test_freeze_then_compare_writes_every_output_without_an_h2_verdict(cli, world) -> None:
    run = _frozen(cli, world)
    result = cli(*FAST, "agreement", "compare", "--verify-run", str(run))
    assert result.code == 0, result.err
    assert "no H2 verdict" in result.out
    out = run / REPORT_DIR
    names = ("items.csv", "per_check.csv", "per_generator.csv", "errors.csv", "summary.md")
    assert all((out / name).is_file() for name in names)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["h2_verdict"]["applicable"] is False
    assert "no H2 verdict" in (out / "summary.md").read_text(encoding="utf-8")


def test_a_changed_config_breaks_the_preregistration(cli, world) -> None:
    run = _frozen(cli, world)
    result = cli(
        *FAST, "--set", "agreement.alpha=0.1", "agreement", "compare", "--verify-run", str(run)
    )
    assert result.code == 2 and "agreement_config_sha256" in result.err


def test_the_committed_rule_is_calibrated_with_the_descriptive_fallback(repo_root: Path) -> None:
    with (repo_root / "configs" / "agreement.toml").open("rb") as handle:
        agreement = tomllib.load(handle)["agreement"]
    assert agreement["rule"] == "calibrated"
    assert agreement["on_unusable_stratum"] == "descriptive"


def test_the_calibrated_rule_without_its_artefact_is_a_config_error(cli, world) -> None:
    run = _frozen(cli, world)
    result = cli(*FAST, "agreement", "compare", "--verify-run", str(run), "--rule", "calibrated")
    assert result.code == 2


def test_a_run_with_foreign_clips_or_other_thresholds_is_refused(cli, world) -> None:
    run = _frozen(cli, world)
    foreign = cli(*FAST, "agreement", "compare", "--verify-run", str(run), "--subset", "h2_dev")
    assert foreign.code == 2 and "outside the subset" in foreign.err
    relabelled = world.run("relabelled", {"v1t/a1": [world.FAIL]}, label="calibrated:x")
    result = cli(*FAST, "agreement", "compare", "--verify-run", str(relabelled))
    assert result.code == 2 and "thresholds" in result.err


def test_a_run_that_is_not_perceived_is_refused(cli, world) -> None:
    run = _frozen(cli, world)
    checks = run / "checks.jsonl"
    lines = checks.read_text(encoding="utf-8").replace(
        f'"{InputLevel.PERCEIVED.value}"', f'"{InputLevel.ORACLE.value}"'
    )
    checks.write_text(lines, encoding="utf-8")
    result = cli(*FAST, "agreement", "compare", "--verify-run", str(run))
    assert result.code == 2 and "not a perceived run" in result.err


def test_rerunning_freeze_on_an_unchanged_state_keeps_the_preregistration(cli, world) -> None:
    world.keep_all()
    dev = str(_dev(world))
    first = cli(*FAST, "agreement", "freeze", "--verify-run", dev)
    assert first.code == 0, first.err
    path = Path(first.out.split("preregistration: ", 1)[1].splitlines()[0])
    before = path.read_bytes()
    again = cli(*FAST, "agreement", "freeze", "--verify-run", dev)
    assert again.code == 0, again.err
    assert f"preregistration: {path}" in again.out and "kept" in again.out
    assert path.read_bytes() == before


def test_freeze_refuses_a_different_frozen_state_without_force(cli, world) -> None:
    world.keep_all()
    dev = str(_dev(world))
    assert cli(*FAST, "agreement", "freeze", "--verify-run", dev).code == 0
    changed = ("--set", "agreement.alpha=0.1")
    refused = cli(*FAST, *changed, "agreement", "freeze", "--verify-run", dev)
    assert refused.code == 2 and "already pre-registered" in refused.err
    assert "agreement_config_sha256" in refused.err
    forced = cli(*FAST, *changed, "agreement", "freeze", "--verify-run", dev, "--force")
    assert forced.code == 0, forced.err
    assert "replacing pre-registration" in forced.err
    path = Path(forced.out.split("preregistration: ", 1)[1].splitlines()[0])
    assert load_preregistration(path).replaces_sha256 is not None


def test_a_refreeze_after_a_test_comparison_is_post_hoc_and_audited(cli, world) -> None:
    world.keep_all()
    dev, changed = str(_dev(world)), ("--set", "agreement.alpha=0.1")
    assert cli(*FAST, "agreement", "freeze", "--verify-run", dev).code == 0
    run = _test(world)
    assert cli(*FAST, "agreement", "compare", "--verify-run", str(run)).code == 0
    metrics = json.loads((run / REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    audit = metrics["preregistration"]
    assert audit["versions"] == 1 and audit["post_hoc"] is False
    assert metrics["h2_verdict"]["post_hoc"] is False
    refused = cli(*FAST, *changed, "agreement", "freeze", "--verify-run", dev, "--force")
    assert refused.code == 2 and "--declare-deviation" in refused.err
    late = ("--force", "--declare-deviation", "alpha corrected")
    forced = cli(*FAST, *changed, "agreement", "freeze", "--verify-run", dev, *late)
    assert forced.code == 0, forced.err
    assert "post hoc" in forced.err
    path = Path(forced.out.split("preregistration: ", 1)[1].splitlines()[0])
    record = load_preregistration(path)
    assert record.post_hoc and record.deviations[0].startswith("alpha corrected")
    assert cli(*FAST, *changed, "agreement", "compare", "--verify-run", str(run)).code == 0
    metrics = json.loads((run / REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["preregistration"]["versions"] == 2
    assert metrics["preregistration"]["post_hoc"] is True
    assert metrics["h2_verdict"]["post_hoc"] is True
    assert "post hoc" in (run / REPORT_DIR / "summary.md").read_text(encoding="utf-8").lower()


def test_subset_check_fails_while_pending_and_passes_once_recorded(cli, world) -> None:
    assert cli(*FAST, "agreement", "subset").code == 0
    pending = cli(*FAST, "agreement", "subset", "--check")
    assert pending.code == 1 and "hand review" in pending.err
    world.keep_all()
    assert cli(*FAST, "agreement", "subset", "--check").code == 0


def test_per_version_results_are_labelled_descriptive(cli, world) -> None:
    run = _frozen(cli, world)
    assert cli(*FAST, "agreement", "compare", "--verify-run", str(run)).code == 0
    text = (run / REPORT_DIR / "summary.md").read_text(encoding="utf-8")
    metrics = json.loads((run / REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["h2_group"] == "all"
    assert metrics["descriptive_groups"] and "all" not in metrics["descriptive_groups"]
    assert "## Group all (aggregate: decides H2)" in text
    for group in metrics["descriptive_groups"]:
        assert f"## Group {group} (descriptive only" in text


def test_the_summary_renders_every_c5_block(cli, world) -> None:
    run = _frozen(cli, world)
    assert cli(*FAST, "agreement", "compare", "--verify-run", str(run)).code == 0
    text = (run / REPORT_DIR / "summary.md").read_text(encoding="utf-8")
    metrics = json.loads((run / REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    for needle in (
        "| system | n | acc | bal acc | F1 | kappa | Pearson | Spearman | AUC |",
        "constants.implausible",
        "constants.plausible",
        "constants.dev_majority",
        "McNemar b (verifier right only)",
        "kappa [",
        "## Per check",
        "## Per generator",
        "## Sensitivity without the time-dependent checks",
        "Time-dependent: R4_jump, R5b",
        "Coverage (decided fraction): all ",
        "verifier predicts implausible on",
        "Sign flip between versions",
    ):
        assert needle in text, needle
    assert {"per_check", "per_generator", "coverage"} <= set(metrics)
    coverage = metrics["coverage"]["all"]
    assert coverage["coverage"] == coverage["decided"] / coverage["n"]


def test_a_sign_flip_between_versions_is_reported() -> None:
    def block(acc: float, kappa: float) -> dict[str, object]:
        return {
            "verifier": {"accuracy": acc, "kappa": kappa},
            "vlm": {"accuracy": 0.5, "kappa": 0.0},
        }

    groups = {"v1": {"p": block(0.6, 0.1)}, "v2": {"p": block(0.4, 0.2)}}
    (line,) = _sign_flip_lines(groups, "p")
    assert "accuracy (v1 0.100, v2 -0.100)" in line and "kappa" not in line
    assert "none" in _sign_flip_lines({"v1": groups["v1"], "v2": groups["v1"]}, "p")[0]


CALIBRATED = ("--set", 'agreement.rule="calibrated"')
STOP = ("--set", 'agreement.on_unusable_stratum="error"')
STRICT = ("--set", "calibration.require_calibrated_artifact=true")


def _artifact(world, *, record: bool = True, **kwargs) -> Path:
    """The artifact and, unless ``record`` is False, its ADR 0015 decision record."""
    active = active_checks_hash(world.settings)
    options = {"profile": "precise", "checks_hash": active, **kwargs}
    artifact = write_artifact(default_artifact_path(world.settings), **options)
    if record:
        path = world.settings.paths.calibration_dir / "exclusions.json"
        write_record(path, artifact_sha256=file_sha256(artifact) or "", checks_hash=active)
    return artifact


def _metrics(run: Path) -> dict:
    return json.loads((run / REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))


def _freeze(cli, world, *options: str, after: tuple[str, ...] = ()):
    world.keep_all()
    dev = world.root / "runs" / "dev"
    dev = dev if dev.is_dir() else _dev(world)
    return cli(*FAST, *options, "agreement", "freeze", "--verify-run", str(dev), *after)


def test_an_unusable_stratum_runs_descriptive_and_h2_is_not_evaluable(cli, world) -> None:
    artifact = _artifact(world, ready=False)
    frozen = _freeze(cli, world, *CALIBRATED)
    assert frozen.code == 0, frozen.err
    path = Path(frozen.out.split("preregistration: ", 1)[1].splitlines()[0])
    prereg = load_preregistration(path)
    assert "aggregate threshold" in str(prereg.calibration_gate)
    assert prereg.calibration_artifact_sha256 == file_sha256(artifact)
    run = _test(world)
    result = cli(*FAST, *CALIBRATED, "agreement", "compare", "--verify-run", str(run))
    assert result.code == 0, result.err
    verdict = _metrics(run)["h2_verdict"]
    assert verdict["status"] == "not_evaluable" and verdict["supported"] is None
    assert _metrics(run)["calibration"]["gate"] == verdict["gate"]
    text = (run / REPORT_DIR / "summary.md").read_text(encoding="utf-8")
    assert "H2 not evaluable with the calibrated rule" in text


def test_a_calibrated_freeze_needs_the_decision_record_of_its_artifact(cli, world) -> None:
    _artifact(world, record=False)
    missing = _freeze(cli, world, *CALIBRATED)
    assert missing.code == 2 and "exclusions decide" in missing.err
    _artifact(world)
    frozen = _freeze(cli, world, *CALIBRATED)
    assert frozen.code == 0, frozen.err
    prereg = load_preregistration(Path(frozen.out.split("preregistration: ", 1)[1].splitlines()[0]))
    record = (world.settings.paths.calibration_dir / "exclusions.json").read_text(encoding="utf-8")
    assert prereg.exclusion_record_sha256 == json.loads(record)["sha256"]
    assert prereg.secondary_tests == ["mcnemar_caption_permutation"]


def test_on_unusable_stratum_error_stops(cli, world) -> None:
    _artifact(world, ready=False)
    result = _freeze(cli, world, *CALIBRATED, *STOP)
    assert result.code == 2 and "aggregate threshold" in result.err


def test_a_stale_artifact_stops_in_both_modes(cli, world) -> None:
    _artifact(world, checks_hash="another")
    for mode in ((), STOP):
        result = _freeze(cli, world, *CALIBRATED, *mode)
        assert result.code == 2 and "other check parameters" in result.err


def test_the_fallback_still_checks_the_thresholds_label(cli, world) -> None:
    _artifact(world, ready=False)
    assert _freeze(cli, world, *CALIBRATED).code == 0
    relabelled = world.run("relabelled", {"v1t/a1": [world.FAIL]}, label="calibrated:x")
    result = cli(*FAST, *CALIBRATED, "agreement", "compare", "--verify-run", str(relabelled))
    assert result.code == 2 and "thresholds" in result.err


def test_require_calibrated_turns_an_insufficient_artifact_descriptive(cli, world) -> None:
    _artifact(world)  # status "insufficient", stratum ready
    assert _freeze(cli, world, *CALIBRATED, *STRICT).code == 0
    run = _test(world)
    result = cli(*FAST, *CALIBRATED, *STRICT, "agreement", "compare", "--verify-run", str(run))
    assert result.code == 0, result.err
    gate = _metrics(run)["h2_verdict"]["gate"]
    assert "not 'calibrated'" in gate and str(world.settings.paths.calibration_dir) not in gate


def test_a_usable_stratum_reports_its_state_per_check(cli, world) -> None:
    _artifact(world)
    frozen = _freeze(cli, world, *CALIBRATED)
    assert frozen.code == 0, frozen.err
    run = _test(world)
    result = cli(*FAST, *CALIBRATED, "agreement", "compare", "--verify-run", str(run))
    assert result.code == 0, result.err
    calibration = _metrics(run)["calibration"]
    assert calibration["gate"] is None
    assert calibration["stratum"]["checks"]["R1"]["status"] == "ok"
    assert _metrics(run)["h2_verdict"]["status"] == "evaluated"


def test_the_dev_design_effect_is_used_only_where_it_was_estimated(cli, world) -> None:
    frozen = _freeze(cli, world)
    assert frozen.code == 0, frozen.err
    path = Path(frozen.out.split("preregistration: ", 1)[1].splitlines()[0])
    dev = load_preregistration(path).dev_design_effect
    run = _test(world)
    assert cli(*FAST, "agreement", "compare", "--verify-run", str(run)).code == 0
    metrics = _metrics(run)
    primary = (metrics["h2_group"], metrics["primary_policy"])
    sources = {
        (group, policy): view.get("design_effect_source")
        for group, views in metrics["groups"].items()
        for policy, view in views.items()
    }
    assert sources.pop(primary) == ("this_data" if dev is None else "dev")
    assert sources and set(sources.values()) <= {"this_data", None}


def test_a_calibrated_refreeze_after_a_fallback_comparison_is_post_hoc(cli, world) -> None:
    _artifact(world, ready=False)
    assert _freeze(cli, world, *CALIBRATED).code == 0
    run = _test(world)
    assert cli(*FAST, *CALIBRATED, "agreement", "compare", "--verify-run", str(run)).code == 0
    _artifact(world)
    stale = cli(*FAST, *CALIBRATED, "agreement", "compare", "--verify-run", str(run))
    assert stale.code == 2 and "calibration_gate" in stale.err
    late = ("--force", "--declare-deviation", "stratum became usable")
    refrozen = _freeze(cli, world, *CALIBRATED, after=late)
    assert refrozen.code == 0, refrozen.err
    assert cli(*FAST, *CALIBRATED, "agreement", "compare", "--verify-run", str(run)).code == 0
    assert _metrics(run)["h2_verdict"]["post_hoc"] is True


def test_an_unknown_check_scope_id_is_a_config_error(cli, world) -> None:
    result = _freeze(cli, world, "--set", 'check_scope.exclude_checks=["R99"]')
    assert result.code == 2 and "R99" in result.err


def test_the_time_dependent_checks_can_join_the_primary_exclusion(cli, world) -> None:
    timed = ("--set", "agreement.exclude_time_dependent=true")
    assert _freeze(cli, world, *timed).code == 0
    run = _test(world)
    assert cli(*FAST, *timed, "agreement", "compare", "--verify-run", str(run)).code == 0
    metrics = _metrics(run)
    assert metrics["primary_excluded"] == ["R4_jump", "R5b"]
    assert metrics["sensitivity_time_dependent"]["time_dependent_in_primary"] is False
    text = (run / REPORT_DIR / "summary.md").read_text(encoding="utf-8")
    assert "## Sensitivity with the time-dependent checks" in text

"""``physverify agreement`` end to end on the miniature tables and fake verify runs (C2, C5, C7)."""

from __future__ import annotations

import json
from pathlib import Path

from physverify.agreement.prereg import load_preregistration
from physverify.agreement.report import REPORT_DIR, _sign_flip_lines
from physverify.datasets._videophy_review import PENDING, load_review
from physverify.schema.checks import InputLevel

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

"""``physverify results`` over a fake A8 v2 chain (plan WP-M, acceptance criteria 1-7)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from calibration_cases import PROFILE
from cli_runner import CliRunner
from reporting_cases import Chain, build_chain, evaluation, write, write_json, write_metrics

from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import load_calibration
from physverify.reporting import calibration_numbers
from physverify.reporting.calibration_numbers import DESCRIPTIVE
from physverify.reporting.numbers import NOT_PREREGISTERED
from physverify.reporting.render import value_text
from physverify.reporting.sources import Doc, Request, collect_inputs, resolve
from physverify.reporting.values import detection_limit
from physverify.schema.checks import InputLevel
from physverify.verification.scope import CheckScope

EXIT_USAGE = 2
SECTIONS = {
    "per_check",
    "aggregate",
    "monotonicity",
    "real_fpr",
    "abstention",
    "agreement",
    "h2_verdict",
    "exclusions",
    "oracle_reference",
}


@pytest.fixture
def chain(tmp_path: Path) -> Chain:
    return build_chain(tmp_path)


def _run(cli: CliRunner, chain: Chain, *extra: str, h2: bool = True) -> tuple[int, str]:
    args = ["--profile", "fast", "results", "--a8v2-dir", str(chain.dir), *extra]
    if h2:
        args += ["--h2-run", str(chain.h2_run)]
    result = cli(*args)
    return result.code, result.out + result.err


def _results(cli: CliRunner, chain: Chain, *extra: str, h2: bool = True) -> dict[str, Any]:
    code, output = _run(cli, chain, *extra, h2=h2)
    assert code == 0, output
    data: dict[str, Any] = json.loads((chain.dir / "results" / "results.json").read_text())
    return data


def _values(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{name}.{key}": value
        for name, section in data["sections"].items()
        for key, value in section["values"].items()
    }


def test_every_section_is_written_and_every_source_resolves(cli: CliRunner, chain: Chain) -> None:
    """Criterion 1: each value read again from its ``path#pointer`` is the same number."""
    data = _results(cli, chain)
    assert set(data["sections"]) == SECTIONS
    assert all(s["missing"] is None for s in data["sections"].values()), data["sections"]
    assert (chain.dir / "results" / "results.md").is_file()
    values = _values(data)
    assert len(values) > 50
    for key, value in values.items():
        found = resolve(value["source"])
        if isinstance(found, list):
            assert found == [value["ci_low"], value["ci_high"]], key
        else:
            assert found == value["value"], key


def test_the_values_are_the_typed_fields(cli: CliRunner, chain: Chain) -> None:
    """Criterion 2, spot checks against the loaded models and files; NaN is null and '-'."""
    values = _values(_results(cli, chain))
    stratum = load_calibration(chain.artifact).stratum(InputLevel.PERCEIVED, PROFILE).model
    r1 = stratum.checks["R1"]
    assert values["per_check.R1.tau"]["value"] == r1.tau
    assert values["per_check.R1.fpr_cv"]["k"] == r1.fpr.cv.k
    assert values["per_check.R1.fpr_cv"]["ci_high"] == r1.fpr.cv.ci.high
    assert values["aggregate.score_threshold"]["value"] == stratum.aggregate.score_threshold
    assert values["monotonicity.R1.calib_all.tau_b"]["value"] == r1.monotonicity.kendall_tau_b
    report = EvaluationReport.model_validate_json(chain.evaluation.read_text())
    perceived = next(s for s in report.strata if s.level is InputLevel.PERCEIVED)
    heldout = next(c for c in perceived.checks if c.check_id == "R1")
    assert values["per_check.R1.fpr_heldout"]["value"] == heldout.fpr_clean.value
    assert values["real_fpr.real_fixture.R1"]["n"] == perceived.real[0].fpr.n
    assert values["real_fpr.real_fixture.score"]["k"] == perceived.real_clip_flags[0].fpr.k
    fit = json.loads(chain.abstention.read_text())
    assert fit["k_star"] is not None
    assert values["abstention.k_star_calib"]["value"] == fit["k_star"]
    curve = json.loads((chain.curve_run / "abstention_report.json").read_text())
    row = next(r for r in curve["tables"]["calibrated"]["curve"] if r["k_sigma"] == fit["k_star"])
    assert values["abstention.eval_at_k_star.risk"]["ci_high"] == row["risk_ci"][1]
    found = json.loads((chain.h2_run / "agreement" / "metrics.json").read_text())
    primary = found["groups"][found["h2_group"]][found["primary_policy"]]
    intervals = primary["bootstrap"]["intervals"]
    assert values["agreement.verifier.accuracy"]["value"] == primary["verifier"]["accuracy"]
    assert values["agreement.verifier.accuracy"]["ci_low"] == intervals["verifier.accuracy"][0]
    assert values["agreement.difference.kappa"]["ci_high"] == intervals["difference.kappa"][1]
    assert values["agreement.mcnemar.p_value"]["value"] == primary["mcnemar"]["p_value"]
    assert values["h2_verdict.supported"]["value"] == found["h2_verdict"]["supported"]
    assert values["h2_verdict.gate"]["value"] is None
    assert values["exclusions.1.5.R5b.excluded"]["note"] == "undecided: no events"
    assert values["per_check.R5b.tau"]["value"] is None
    markdown = (chain.dir / "results" / "results.md").read_text()
    assert "| R5b.tau | - |" in markdown


def test_the_estimates_are_labelled(cli: CliRunner, chain: Chain) -> None:
    """Criterion 5: L_D, sigma, p_c, k* and risks are estimates; s* is in-sample."""
    values = _values(_results(cli, chain))
    for key in (
        "per_check.R1.detection_limit",
        "per_check.R1.sigma_factor",
        "aggregate.probability.R1.a",
        "abstention.k_star_calib",
        "abstention.k_star_eval.calibrated",
        "abstention.calib_at_k_star.risk",
        "abstention.real.physics_iq.k_star.risk",
    ):
        assert values[key]["estimated"] is True, key
    assert values["per_check.R1.tau"]["estimated"] is False
    assert values["aggregate.score_threshold"]["note"] == "in-sample, optimistic"


def _doc(data: dict[str, Any]) -> Doc:
    return Doc(path=Path("artifact.json"), data=data, sha256="x")


@pytest.mark.parametrize(("censored", "text"), [("below", "<= 0.05"), ("above", "> 0.05")])
def test_a_censored_detection_limit_is_a_bound(censored: str, text: str) -> None:
    ld = {"value": None, "ci": {"low": None, "high": None}, "censored": censored,
          "bound": 0.05, "unit": "m", "n": 9}  # fmt: skip
    value = detection_limit("ld", _doc({"ld": ld}), "/ld")
    assert value.value is None and value.bound == 0.05 and value.estimated
    assert value_text(value) == text
    assert value_text(dataclasses.replace(value, censored="none")) == "-"


def test_h1_pass_is_reported_as_fitted_and_the_medians_are_descriptive(
    cli: CliRunner, chain: Chain
) -> None:
    """Criterion 7: medians that are not monotone leave the reported h1_pass as it is."""
    inputs = collect_inputs(
        Request(chain.dir, check_scope=CheckScope(decision_record="exclusions.json"))
    )
    data = json.loads(json.dumps(inputs.artifact.data))
    index = next(i for i, s in enumerate(data["strata"]) if s["level"] == "perceived")
    tau = data["strata"][index]["checks"]["R1"]["monotonicity"]
    tau.update(monotone_medians=False, h1_pass=True)
    altered = dataclasses.replace(inputs, artifact=_doc(data))
    by_key = {v.key: v for v in calibration_numbers.monotonicity(altered).values}
    assert by_key["R1.calib_all.h1_pass"].value is True
    assert by_key["R1.calib_all.monotone_medians"].value is False
    assert by_key["R1.calib_all.monotone_medians"].note == DESCRIPTIVE
    medians = [v for k, v in by_key.items() if k.startswith("R1.calib_all.median.L")]
    assert medians and all(v.note == DESCRIPTIVE for v in medians)
    _results(cli, chain)
    markdown = (chain.dir / "results" / "results.md").read_text()
    assert "| R1.calib_all.tau_b |" in markdown and "| R1.calib_all.median.L1 |" in markdown


def test_the_primary_tau_comes_from_the_h1_view(cli: CliRunner, chain: Chain) -> None:
    values = _values(_results(cli, chain))
    primary = values["monotonicity.R1.calib_primary.tau_b"]
    assert "/h1_primary/strata/" in primary["source"]
    assert primary["note"] == "30 scenes"
    assert values["monotonicity.R1.calib_all.tau_b"]["note"] == "40 scenes"


def test_a_1_3_0_evaluation_leaves_the_primary_tau_missing(
    cli: CliRunner, chain: Chain, tmp_path: Path
) -> None:
    data = json.loads(chain.evaluation.read_text())
    for stratum in (*data["strata"], *data["h1_primary"]["strata"]):
        del stratum["monotonicity"]
    old = write_json(tmp_path / "old" / "evaluation.json", data)
    results = _results(cli, chain, "--h1-evaluation", str(old))
    section = results["sections"]["monotonicity"]
    assert "calib_primary missing" in section["caveat"]
    assert not any(".calib_primary." in key for key in section["values"])


def test_a_cuda_chain_is_a_rehearsal(cli: CliRunner, tmp_path: Path) -> None:
    """Criterion 4."""
    chain = build_chain(tmp_path, device="cuda")
    data = _results(cli, chain, h2=False)
    assert data["reportable"] is False
    assert data["banner"] == "REHEARSAL (device cuda): not reportable"
    assert all(v["reportable"] is False for v in _values(data).values())
    assert data["sections"]["agreement"]["missing"] == "no --h2-run given"
    markdown = (chain.dir / "results" / "results.md").read_text()
    assert "**REHEARSAL (device cuda): not reportable**" in markdown


def test_a_provisional_h2_rule_is_marked_not_preregistered(cli: CliRunner, chain: Chain) -> None:
    write_metrics(chain, provisional=True)
    data = _results(cli, chain)
    assert data["sections"]["agreement"]["caveat"] == NOT_PREREGISTERED
    assert data["sections"]["h2_verdict"]["caveat"] == NOT_PREREGISTERED
    assert f"Note: {NOT_PREREGISTERED}" in (chain.dir / "results" / "results.md").read_text()


def test_the_abstention_run_option_replaces_the_log(cli: CliRunner, chain: Chain) -> None:
    for log in (chain.dir / "logs").glob("7-*.log"):
        log.unlink()
    caveat = _results(cli, chain)["sections"]["abstention"]["caveat"]
    assert caveat.startswith("held-out curve missing: no `run:` line")
    values = _results(cli, chain, "--abstention-run", str(chain.curve_run))
    assert "k_star_eval.calibrated" in values["sections"]["abstention"]["values"]


def test_without_step_9_the_record_is_missing(cli: CliRunner, tmp_path: Path) -> None:
    chain = build_chain(tmp_path, codes={9: "running"})
    missing = _results(cli, chain)["sections"]["exclusions"]["missing"]
    assert missing == "step 9 (exclusions decide) has not completed"


def _tree(root: Path) -> dict[Path, str]:
    return {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "results" not in p.relative_to(root).parts
    }


def test_nothing_is_written_outside_out(cli: CliRunner, chain: Chain, tmp_path: Path) -> None:
    before = _tree(tmp_path / "data") | _tree(chain.dir)
    out = tmp_path / "elsewhere"
    code, output = _run(cli, chain, "--out", str(out))
    assert code == 0, output
    assert _tree(tmp_path / "data") | _tree(chain.dir) == before
    assert sorted(p.name for p in out.iterdir()) == ["results.json", "results.md"]


# --- refusals (criterion 3, criterion 6) -------------------------------------------------------


def _refused(cli: CliRunner, chain: Chain, match: str, *extra: str, h2: bool = True) -> None:
    code, output = _run(cli, chain, *extra, h2=h2)
    assert code == EXIT_USAGE, output
    assert match in output, output


def test_a_refitted_artifact_is_refused(cli: CliRunner, chain: Chain) -> None:
    write(chain.artifact, chain.artifact.read_text() + "\n")
    _refused(cli, chain, "refitted since?")


@pytest.mark.parametrize("step", [5, 6, 7])
def test_an_unfinished_fit_step_is_refused(cli: CliRunner, tmp_path: Path, step: int) -> None:
    chain = build_chain(tmp_path, codes={step: "running"})
    _refused(cli, chain, f"step {step} of")


def test_an_h2_run_of_another_artifact_is_refused(cli: CliRunner, chain: Chain) -> None:
    path = write_metrics(chain)
    data = json.loads(path.read_text())
    data["rule"] = data["rule"].replace("@run-a", "@run-b")
    write_json(path, data)
    _refused(cli, chain, "stratum of the artifact run-a")


def test_an_h2_run_of_another_stratum_is_refused(cli: CliRunner, chain: Chain) -> None:
    """The oracle stratum of the same artifact is not the one this report covers."""
    write_metrics(chain, level=InputLevel.ORACLE)
    rule = json.loads((chain.h2_run / "agreement" / "metrics.json").read_text())["rule"]
    assert rule.endswith("#oracle/fast@run-a")
    _refused(cli, chain, "is not the perceived/fast stratum")


def test_an_h2_run_with_a_cuda_chain_is_refused(cli: CliRunner, tmp_path: Path) -> None:
    chain = build_chain(tmp_path, device="cuda")
    _refused(cli, chain, "H2 runs only on the CPU calibration")


def test_an_h1_evaluation_of_another_artifact_is_refused(
    cli: CliRunner, chain: Chain, tmp_path: Path
) -> None:
    data = json.loads(chain.evaluation.read_text())
    data["artifact_sha256"] = "0" * 64
    other = write_json(tmp_path / "other" / "evaluation.json", data)
    _refused(cli, chain, "evaluated artifact sha256 000000000000", "--h1-evaluation", str(other))


def test_a_pooled_tau_that_differs_from_the_artifact_is_refused(
    cli: CliRunner, chain: Chain, tmp_path: Path
) -> None:
    """Criterion 6 on the collector's side: the equality control runs on the given data."""
    data = json.loads(chain.evaluation.read_text())
    data["strata"][0]["monotonicity"][0]["monotonicity"]["kendall_tau_b"] = 0.123
    changed = write_json(tmp_path / "changed" / "evaluation.json", data)
    _refused(cli, chain, "differs from the artifact's", "--h1-evaluation", str(changed))


def test_an_abstention_of_another_table_is_refused(cli: CliRunner, chain: Chain) -> None:
    data = json.loads(chain.abstention.read_text())
    data["thresholds"]["perceived"]["artifact"]["sha256"] = "f" * 64
    write_json(chain.abstention, data)
    _refused(cli, chain, "not the artifact run-a")


def test_a_record_of_another_evaluation_is_refused(cli: CliRunner, chain: Chain) -> None:
    report = evaluation().model_copy(update={"artifact_path": str(chain.artifact), "run_id": "b"})
    write(chain.evaluation, report.to_json())
    _refused(cli, chain, "does not match the evaluation sha256")

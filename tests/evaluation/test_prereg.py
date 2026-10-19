from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physverify.agreement.prereg import file_sha256
from physverify.core.errors import ConfigError, PhysverifyError, UsageError
from physverify.evaluation.layout import H3Layout
from physverify.evaluation.margins import MarginDerivation
from physverify.evaluation.prereg import (
    AUDIT_FIELDS,
    EVAL_DATA_DEVIATION,
    H3_FIELDS,
    CandidateSnapshot,
    FreezeRequest,
    H3Preregistration,
    Mde,
    assert_complete,
    chain,
    chain_shas,
    first_created,
    identity,
    load_prereg,
    plan_freeze,
    require_unchanged,
    stamp_problem,
    write_prereg,
)
from physverify.finetune.stop_rule import StopRuleState

DERIVATION = MarginDerivation(
    column="pc_pred",
    min_n=300,
    n_generators=2,
    lo_generator="A",
    lo=3.6,
    hi_generator="B",
    hi=3.86,
    spread=0.26,
    se=0.01,
    value=0.13,
)


def record(**update: Any) -> H3Preregistration:
    fields: dict[str, Any] = {
        "commit": "c0ffee",
        "section_hashes": {"h3": "h"},
        "prompts_sha256": "p",
        "eval_counts": {"n": 8},
        "selection_run_id": "main",
        "adapter": "/a",
        "adapter_tag": "lora-abc",
        "calibration_artifact_sha256": None,
        "rule": "provisional",
        "scorer_version": "2",
        "rule_label": "provisional",
        "rule_status": "provisional",
        "calibration_dataset": None,
        "contaminated": False,
        "judge_stamps": {"pc": "s-pc", "sa": "s-sa"},
        "gate_sha256": None,
        "margins": {"pc": 0.13, "sa": 0.05},
        "margin_derivations": {"pc": DERIVATION},
        "margin_source_sha256": "m",
        "bootstrap_seed": 1,
        "mde": Mde(
            n=8,
            alpha=0.05,
            power=0.8,
            verifier_sd=None,
            verifier_mde=None,
            pc_sd=None,
            pc_mde=None,
            pc_underpowered=None,
        ),
        "usability": None,
        "scipy_version": "1.14",
        "eval_candidates_at_freeze": CandidateSnapshot(n=0, sha256="e"),
        "stop_rule": StopRuleState(passed=True, enforced=True, reasons=[], pairs_sha256="p"),
        "created_utc": "2026-10-18T00:00:00+00:00",
    }
    return H3Preregistration(**{**fields, **update})


def request(
    force: bool = True, deviation: str | None = None, eval_data: bool = False, gate: bool = False
) -> FreezeRequest:
    return FreezeRequest(force, deviation, eval_data, gate)


def test_identity_has_every_field_but_the_audit_ones() -> None:
    assert set(H3_FIELDS) == set(H3Preregistration.model_fields) - AUDIT_FIELDS
    assert {"commit", "created_utc", "eval_candidates_at_freeze"} <= AUDIT_FIELDS
    assert set(identity(record())) == set(H3_FIELDS)


def test_the_first_freeze_must_be_blind() -> None:
    current = record()
    assert plan_freeze(None, None, current, request(force=False)) == current
    with pytest.raises(UsageError, match="not blind"):
        plan_freeze(None, None, current, request(eval_data=True))


def test_an_unchanged_state_writes_nothing() -> None:
    later = record(commit="other", created_utc="2026-10-19T00:00:00+00:00")
    assert plan_freeze(record(), "sha0", later, request(force=False)) is None


def test_a_code_commit_after_the_freeze_is_not_refused() -> None:
    require_unchanged(record(), identity(record(commit="fixed")))


def test_a_changed_state_needs_force() -> None:
    with pytest.raises(UsageError, match="--force"):
        plan_freeze(record(), "sha0", record(rule_label="new"), request(force=False))


def test_a_recalibration_refreezes_without_a_deviation() -> None:
    new = plan_freeze(record(), "sha0", record(rule_label="calibrated:x"), request())
    assert new is not None
    assert (new.rule_label, new.replaces_sha256, new.deviations) == ("calibrated:x", "sha0", [])


@pytest.mark.parametrize(
    ("version", "label", "ok"),
    [("2", "provisional", True), ("3", "provisional", False), ("2", "calibrated:x", False)],
)
def test_stamp_problem_compares_version_and_label(version: str, label: str, ok: bool) -> None:
    problem = stamp_problem(version, label, record())
    assert (problem is None) is ok
    if not ok:
        assert problem is not None and repr(version) in problem and "'2'" in problem


def test_a_new_scorer_version_refreezes_like_a_recalibration() -> None:
    assert "scorer_version" in H3_FIELDS
    new = plan_freeze(record(), "sha0", record(scorer_version="3"), request())
    assert new is not None and new.scorer_version == "3" and not new.post_hoc


@pytest.mark.parametrize(
    "update",
    [
        {"rule": "calibrated"},
        {"rule_status": "unavailable"},
        {"rule_label": "calibrated:x"},
        {"scorer_version": "3"},
        {"calibration_artifact_sha256": "new"},
    ],
)
def test_after_eval_data_a_rule_refreeze_without_a_declaration_fails(
    update: dict[str, Any],
) -> None:
    with pytest.raises(UsageError, match="--declare-deviation"):
        plan_freeze(record(), "sha0", record(**update), request(eval_data=True))


def test_a_declared_rule_refreeze_after_eval_data_is_post_hoc() -> None:
    changed = record(rule_label="calibrated:x")
    new = plan_freeze(record(), "sha0", changed, request(deviation="recalibrated", eval_data=True))
    assert new is not None and new.post_hoc
    assert new.deviations == [
        "recalibrated (changed: ['rule_label'])",
        f"{EVAL_DATA_DEVIATION} (changed: ['rule_label'])",
    ]
    later = plan_freeze(new, "sha1", record(rule_label="calibrated:y"), request())
    assert later is not None and later.post_hoc


def test_an_unknown_calibration_dataset_blocks_a_calibrated_freeze() -> None:
    unknown = record(rule_status="calibrated", contaminated=None)
    with pytest.raises(UsageError, match="unknown"):
        plan_freeze(None, None, unknown, request(force=False))
    with pytest.raises(UsageError, match="unknown"):
        plan_freeze(record(), "sha0", unknown, request())
    provisional = record(contaminated=None)
    assert plan_freeze(None, None, provisional, request(force=False)) == provisional


def test_other_fields_need_a_declared_deviation() -> None:
    changed = record(margins={"pc": 0.2, "sa": 0.05})
    with pytest.raises(UsageError, match="--declare-deviation"):
        plan_freeze(record(), "sha0", changed, request())
    new = plan_freeze(record(deviations=["old"]), "sha0", changed, request(deviation="typo"))
    assert new is not None
    assert new.deviations == ["old", "typo (changed: ['margins'])"]


def test_after_eval_data_only_mutable_fields_change_and_it_is_recorded() -> None:
    late = request(deviation="x", eval_data=True)
    new = plan_freeze(record(), "sha0", record(rule_label="c"), late)
    assert new is not None
    assert new.deviations[-1] == f"{EVAL_DATA_DEVIATION} (changed: ['rule_label'])"
    changed = record(margins={"pc": 0.2, "sa": 0.05})
    with pytest.raises(UsageError, match="eval data"):
        plan_freeze(record(), "sha0", changed, request(deviation="x", eval_data=True))


def test_judge_stamps_change_freely_only_before_the_gate() -> None:
    changed = record(judge_stamps={"pc": "new", "sa": "s-sa"})
    assert plan_freeze(record(), "sha0", changed, request()) is not None
    with pytest.raises(UsageError, match="--declare-deviation"):
        plan_freeze(record(), "sha0", changed, request(gate=True))


def test_an_incomplete_state_is_a_bug() -> None:
    state = identity(record())
    del state["margins"]
    with pytest.raises(PhysverifyError, match="margins"):
        assert_complete(state)
    with pytest.raises(PhysverifyError):
        require_unchanged(record(), state)


def test_a_changed_field_is_refused() -> None:
    with pytest.raises(ConfigError, match="margins"):
        require_unchanged(record(), identity(record(margins={"pc": 0.2, "sa": 0.05})))


def test_refreezes_keep_the_chain_to_the_first_freeze(tmp_path: Path) -> None:
    layout = H3Layout(tmp_path / "h3")
    first = record()
    write_prereg(layout, first)
    sha1 = file_sha256(layout.prereg)
    second = record(rule_label="c", created_utc="2026-10-20T00:00:00+00:00", replaces_sha256=sha1)
    write_prereg(layout, second)
    sha2 = file_sha256(layout.prereg)
    assert sha1 is not None and sha2 is not None
    loaded = load_prereg(layout.prereg)
    assert [r.rule_label for r in chain(layout, loaded)] == ["c", "provisional"]
    assert chain_shas(layout, loaded, sha2) == {sha1, sha2}
    assert first_created(layout, loaded) == first.created_utc


def test_a_missing_prereg_names_the_freeze(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="h3 freeze"):
        load_prereg(tmp_path / "none.json")


def test_a_new_stop_rule_state_refreezes_and_its_deviation_is_audit_only() -> None:
    fired = StopRuleState(passed=False, enforced=True, reasons=["r"], pairs_sha256="q")
    new = plan_freeze(record(), "sha0", record(stop_rule=fired), request())
    assert new is not None and new.deviations == []
    assert "stop_rule" in identity(new) and "stop_rule_deviation" not in identity(new)

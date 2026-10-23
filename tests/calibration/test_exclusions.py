"""The preregistered exclusion rules (1.1, 1.2 as amended, 1.5) and their record."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from exclusion_cases import (
    FAMILY,
    PRECISE,
    arm,
    arms,
    calibration,
    evaluation,
    inputs,
    r2_rows,
    r5a_check,
    real_fpr,
    real_set,
    record,
    rules,
    rules_raw,
    scene,
)
from stratum_cases import FIXTURE_CHECKS_HASH, check_calibration, write_artifact

from physverify.calibration import exclusion_detector_arms
from physverify.calibration.artifact import RealFpr
from physverify.calibration.artifact_io import load_calibration
from physverify.calibration.exclusion_decision import (
    RULES,
    decide,
    r5a_rule,
    real_fpr_rule,
    record_markdown,
)
from physverify.calibration.exclusion_detector_arms import r2_rule
from physverify.calibration.exclusion_inputs import R2Arms, consumed_stratum
from physverify.calibration.exclusion_rules import ExclusionRules, exclusion_rules
from physverify.core.errors import ConfigError
from physverify.stats.intervals import Ci
from physverify.verification.exclusion_record import (
    ExclusionInputError,
    load_record,
    parse_record,
)
from physverify.verification.runner import RowStatus


def _one(evidence: tuple) -> object:  # type: ignore[type-arg]
    assert len(evidence) == 1
    return evidence[0]


def test_every_threshold_comes_from_the_toml() -> None:
    """No rule number has a code default: the model without the TOML values does not load."""
    for name, field in ExclusionRules.model_fields.items():
        assert field.is_required(), name
    for sub in ("r5a", "r2", "real"):
        model = ExclusionRules.model_fields[sub].annotation
        assert all(f.is_required() for f in model.model_fields.values())  # type: ignore[union-attr]
    loaded = rules()
    assert loaded.r2.max_fpr_increase == 0.10 and loaded.real.max_fpr_lower == 0.10
    assert set(loaded.rules) == set(RULES)


@pytest.mark.parametrize(("key", "value"), [("bootstrap_n", 0), ("tpr_ci", 1.0), ("tpr_ci", 0.0)])
def test_an_r2_bootstrap_that_cannot_give_an_interval_is_refused(key: str, value: float) -> None:
    raw = rules_raw()
    raw["r2"] = {**raw["r2"], key: value}  # type: ignore[dict-item]
    with pytest.raises(ConfigError):
        exclusion_rules(raw, ["R2", "R5a"])


def test_unknown_rules_and_checks_are_configuration_errors() -> None:
    raw = rules_raw()
    with pytest.raises(ConfigError, match="unknown rules"):
        exclusion_rules({**raw, "rules": ["1.1", "9.9"]}, {"R2", "R5a"})
    with pytest.raises(ConfigError, match="unknown checks"):
        exclusion_rules(raw, {"R2"})
    bad_set = {**raw["real"], "allowed_sets": [{"physics_iq": 54}]}  # type: ignore[dict-item]
    with pytest.raises(ConfigError, match="must name exactly"):
        exclusion_rules({**raw, "real": bad_set}, {"R2", "R5a"})


# Rule 1.1.


def test_r5a_is_kept_when_usable_weighted_and_monotone() -> None:
    found = _one(r5a_rule(inputs(), rules()))
    assert (found.excluded, found.reason) == (False, "kept")
    assert found.inputs["informedness_low"] == 0.2 and found.inputs["kendall_tau_b_low"] == 0.1


@pytest.mark.parametrize(
    ("check", "reason"),
    [
        (r5a_check(tau_b_low=-0.162), "kendall_tau_b_low_not_positive"),
        (r5a_check(tau_b_low=None), "kendall_tau_b_low_not_positive"),
        (r5a_check(weight=0.0, j_low=0.0), "weight_not_positive"),
        (
            r5a_check().model_copy(update={"status": "insufficient"}),
            "weight_not_positive",
        ),
    ],
)
def test_r5a_is_excluded_when_one_condition_fails(check: object, reason: str) -> None:
    cal = calibration({"R2": check_calibration(tau=1.0), "R5a": check})  # type: ignore[dict-item]
    found = _one(r5a_rule(inputs(cal), rules()))
    assert (found.excluded, found.reason) == (True, reason)


def test_r5a_is_excluded_when_the_stratum_is_unusable_or_lacks_it() -> None:
    unusable = _one(r5a_rule(inputs(problem="aggregate insufficient"), rules()))
    assert (unusable.excluded, unusable.reason) == (True, "stratum_unusable")
    missing = _one(r5a_rule(inputs(calibration({"R2": check_calibration(tau=1.0)})), rules()))
    assert (missing.excluded, missing.reason) == (True, "not_in_stratum")


def test_the_consumed_stratum_goes_through_the_consumers_gate(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "c.json", ready=False, profile=PRECISE)
    cal = load_calibration(path)
    found = consumed_stratum(
        path, cal, rules(), checks_hash=FIXTURE_CHECKS_HASH, require_calibrated=False
    )
    assert found.problem is not None and "insufficient" in found.problem
    ready = write_artifact(tmp_path / "r.json", ready=True, profile=PRECISE)
    ok = consumed_stratum(
        ready,
        load_calibration(ready),
        rules(),
        checks_hash=FIXTURE_CHECKS_HASH,
        require_calibrated=False,
    )
    assert ok.problem is None
    strict = consumed_stratum(
        ready,
        load_calibration(ready),
        rules(),
        checks_hash=FIXTURE_CHECKS_HASH,
        require_calibrated=True,
    )
    assert strict.problem is not None
    other = write_artifact(tmp_path / "f.json", profile="fast")
    with pytest.raises(ExclusionInputError):
        consumed_stratum(
            other,
            load_calibration(other),
            rules(),
            checks_hash=FIXTURE_CHECKS_HASH,
            require_calibrated=False,
        )


# Rule 1.2.


def test_identical_arms_have_no_difference_and_keep_r2() -> None:
    found = _one(r2_rule(inputs(), rules()))
    assert found.decided and not found.excluded
    assert found.inputs["delta_fpr"] == 0.0
    assert found.inputs["base.tpr"] == found.inputs["tiny.tpr"] == 0.5
    assert found.inputs["tau"] == 1.0 and found.inputs["bias"] == 0.0


def test_a_higher_base_fpr_at_the_same_tau_excludes_r2() -> None:
    positives = [2.0] * 20
    base = r2_rows([2.0] * 3 + [0.1] * 17, positives)
    tiny = r2_rows([0.1] * 20, positives)
    found = _one(r2_rule(inputs(base=base, tiny=tiny), rules()))
    assert found.inputs["delta_fpr"] == pytest.approx(0.15)
    assert (found.excluded, found.reason) == (True, "fpr_increase")


def test_an_fpr_increase_at_the_threshold_keeps_r2() -> None:
    positives = [2.0] * 20
    base = r2_rows([2.0] * 2 + [0.1] * 18, positives)
    found = _one(r2_rule(inputs(base=base, tiny=r2_rows([0.1] * 20, positives)), rules()))
    assert found.inputs["delta_fpr"] == pytest.approx(0.10) and not found.excluded


def test_a_tpr_drop_in_every_scene_excludes_r2() -> None:
    clean = [0.1] * 20
    found = _one(
        r2_rule(inputs(base=r2_rows(clean, [0.5] * 20), tiny=r2_rows(clean, [2.0] * 20)), rules())
    )
    assert found.inputs["delta_tpr_high"] is not None and found.inputs["delta_tpr_high"] < 0
    assert (found.excluded, found.reason) == (True, "tpr_decrease")


def test_a_tpr_drop_in_one_scene_does_not_exclude() -> None:
    clean = [0.1] * 20
    base = r2_rows(clean, [0.5] + [2.0] * 19)
    found = _one(r2_rule(inputs(base=base, tiny=r2_rows(clean, [2.0] * 20)), rules()))
    assert found.inputs["delta_tpr_high"] == 0.0 and not found.excluded


def test_without_a_usable_tau_the_rule_does_not_decide() -> None:
    cal = calibration({"R2": check_calibration(tau=None, status="insufficient")})
    found = _one(r2_rule(inputs(cal), rules()))
    assert (found.decided, found.excluded, found.reason) == (False, False, "no_usable_tau")
    unusable = _one(r2_rule(inputs(problem="no aggregate"), rules()))
    assert not unusable.decided and not unusable.excluded


def test_without_positives_the_tpr_test_does_not_decide() -> None:
    clean = r2_rows([0.1] * 20, [])
    found = _one(r2_rule(inputs(base=clean, tiny=clean), rules()))
    assert (found.decided, found.excluded, found.reason) == (False, False, "no_positive")
    assert found.inputs["delta_tpr_high"] is None


def test_an_fpr_increase_excludes_even_without_positives() -> None:
    base, tiny = r2_rows([2.0] * 3 + [0.1] * 17, []), r2_rows([0.1] * 20, [])
    found = _one(r2_rule(inputs(base=base, tiny=tiny), rules()))
    assert (found.decided, found.excluded, found.reason) == (True, True, "fpr_increase")


def test_a_non_finite_tpr_interval_does_not_decide(monkeypatch: pytest.MonkeyPatch) -> None:
    nan = float("nan")
    monkeypatch.setattr(
        exclusion_detector_arms, "scene_bootstrap", lambda *a: {"delta_tpr": Ci(low=nan, high=nan)}
    )
    found = _one(r2_rule(inputs(), rules()))
    assert (found.decided, found.excluded, found.reason) == (False, False, "tpr_ci_undefined")


@pytest.mark.parametrize("status", [RowStatus.ERROR, RowStatus.ABSTAINED, RowStatus.NO_EVENT])
def test_an_unscored_row_in_either_arm_is_refused(status: RowStatus) -> None:
    """An arm that fails to extract the clips it would FAIL cannot lower its FPR."""
    same = r2_rows([0.1] * 20, [2.0] * 20)
    holed = (dataclasses.replace(same[0], status=status, magnitude=None), *same[1:])
    for base, tiny in ((same, holed), (holed, same)):
        with pytest.raises(ExclusionInputError, match="1 unscored"):
            r2_rule(inputs(base=base, tiny=tiny), rules())


@pytest.mark.parametrize(("base_fails", "tiny_fails"), [(8, 6), (16, 14)])
def test_an_exact_gap_of_the_threshold_keeps_r2(base_fails: int, tiny_fails: int) -> None:
    """2/20 gaps are 0.1 exactly: float rounding (0.4 - 0.3 > 0.1) must not exclude."""
    positives = [2.0] * 20
    base = r2_rows([2.0] * base_fails + [0.1] * (20 - base_fails), positives)
    tiny = r2_rows([2.0] * tiny_fails + [0.1] * (20 - tiny_fails), positives)
    found = _one(r2_rule(inputs(base=base, tiny=tiny), rules()))
    assert found.inputs["delta_fpr"] == pytest.approx(0.10)
    assert (found.decided, found.excluded, found.reason) == (True, False, "kept")


def test_only_calib_family_clips_are_judged() -> None:
    same = r2_rows([0.1] * 20, [2.0] * 20)
    other = tuple(
        dataclasses.replace(r, scene_id=r.scene_id.replace(FAMILY, "f1_drop"), clip_id=f"x{i}")
        for i, r in enumerate(same)
    )
    found = _one(r2_rule(inputs(base=(*same, *other), tiny=(*same, *other)), rules()))
    assert found.inputs["base.n_clean"] == 20 and found.inputs["n_scenes"] == 20


def test_swapped_or_mismatched_arms_are_refused() -> None:
    same = r2_rows([0.1] * 20, [2.0] * 20)
    cfg = rules().r2
    base = inputs()
    cases = [
        (
            R2Arms(arm(same, cfg.tiny_detector), arm(same, cfg.base_detector)),
            "detectors",
        ),
        (
            R2Arms(arm(same, cfg.base_detector, profile="fast"), arm(same, cfg.tiny_detector)),
            "profile",
        ),
        (
            R2Arms(arm(same, cfg.base_detector, commit="c1"), arm(same, cfg.tiny_detector)),
            "commits",
        ),
        (arms(same, r2_rows([0.1] * 19, [2.0] * 19)), "same clips"),
        (
            R2Arms(arm(same, cfg.base_detector, checks_hash="x"), arm(same, cfg.tiny_detector)),
            "checks_hash",
        ),
    ]
    for pair, match in cases:
        with pytest.raises(ExclusionInputError, match=match):
            r2_rule(dataclasses.replace(base, arms=pair), rules())


def test_two_clean_clips_in_one_scene_are_refused() -> None:
    same = r2_rows([0.1] * 20, [2.0] * 20)
    twice = dataclasses.replace(same[0], clip_id=f"{scene(0)}__other")
    with pytest.raises(ExclusionInputError, match="two clean clips"):
        r2_rule(inputs(base=(*same, twice), tiny=(*same, twice)), rules())


# Rule 1.5.


def test_the_worst_exact_lower_bound_decides() -> None:
    real = [real_fpr("physics_iq", "R1", 11, 54), real_fpr("pisa_real", "R1", 33, 360)]
    found = _one(real_fpr_rule(inputs(real=real), rules()))
    assert found.inputs["physics_iq.cp_low"] == pytest.approx(0.1064, abs=1e-3)
    assert found.inputs["pisa_real.cp_low"] == pytest.approx(0.0640, abs=1e-3)
    assert (found.excluded, found.reason) == (True, "real_fpr_high")
    assert found.inputs["pisa_real.n_clips"] == 360


def test_the_exact_bound_ignores_the_clustered_interval() -> None:
    real = [
        real_fpr("physics_iq", "R1", 5, 54, clustered_low=0.5),
        real_fpr("pisa_real", "R1", 10, 360),
    ]
    found = _one(real_fpr_rule(inputs(real=real), rules()))
    assert not found.excluded and found.inputs["physics_iq.clustered_low"] == 0.5


def _empty(dataset: str) -> RealFpr:
    found = real_fpr(dataset, "R1", 0, 54)
    return found.model_copy(update={"fpr": found.fpr.model_copy(update={"n": 0})})


def test_a_check_without_rows_is_not_decided_by_1_5() -> None:
    real = [_empty("physics_iq"), _empty("pisa_real")]
    found = _one(real_fpr_rule(inputs(real=real), rules()))
    assert (found.decided, found.excluded, found.reason) == (False, False, "no_rows")


def test_an_evaluation_of_another_artifact_or_set_is_refused() -> None:
    base = inputs()
    other = evaluation(calibration(ready=False))
    with pytest.raises(ExclusionInputError, match="scored artifact"):
        real_fpr_rule(dataclasses.replace(base, evaluation=other), rules())
    odd = evaluation(
        base.calibration, sets=(real_set("physics_iq", 54), real_set("pisa_real", 100))
    )
    with pytest.raises(ExclusionInputError, match="preregistered"):
        real_fpr_rule(dataclasses.replace(base, evaluation=odd), rules())
    old = evaluation(base.calibration, sets=())
    with pytest.raises(ExclusionInputError, match="not recorded"):
        real_fpr_rule(dataclasses.replace(base, evaluation=old), rules())
    fast = evaluation(base.calibration, profile="fast")
    with pytest.raises(ExclusionInputError, match="no perceived/precise"):
        real_fpr_rule(dataclasses.replace(base, evaluation=fast), rules())


def test_a_real_dataset_left_unscored_is_refused() -> None:
    """The worst of both datasets cannot silently become the only one scored."""
    with pytest.raises(ExclusionInputError, match=r"no real FPR on \['pisa_real'\]"):
        real_fpr_rule(inputs(real=[real_fpr("physics_iq", "R1", 11, 54)]), rules())
    with pytest.raises(ExclusionInputError, match="no real FPR"):
        real_fpr_rule(inputs(), rules())


def test_the_fallback_set_is_accepted() -> None:
    real = [real_fpr("physics_iq", "R1", 0, 54), real_fpr("pisa_real", "R1", 0, 40)]
    base = inputs(real=real)
    small = evaluation(
        base.calibration, real, sets=(real_set("physics_iq", 54), real_set("pisa_real", 40))
    )
    found = _one(real_fpr_rule(dataclasses.replace(base, evaluation=small), rules()))
    assert found.decided and not found.excluded and found.inputs["pisa_real.n_clips"] == 40


def _decided(**kw: object) -> object:
    real = [real_fpr("physics_iq", "R1", 11, 54), real_fpr("pisa_real", "R1", 0, 360)]
    cal = calibration({"R2": check_calibration(tau=1.0), "R5a": r5a_check(tau_b_low=-0.1)})
    return decide(
        inputs(cal, real=real), rules(), lambda ids: ["fall"] if "R5a" in ids else [], **kw
    )


def test_decide_unites_the_rules_and_seals_the_record() -> None:
    found = _decided()
    assert found.exclude_checks == ("R1", "R5a")
    assert found.h3_families_without_target == ("fall",)
    assert [e.rule for e in found.evidence] == ["1.1", "1.2", "1.5"]
    assert found.sealed().sha256 == found.sha256
    assert found.to_json() == _decided().to_json()
    assert found.inputs.rules_sha256 == rules().sha256()
    assert "/" not in json.dumps(found.inputs.model_dump())


def test_a_tampered_or_stale_record_is_refused(tmp_path: Path) -> None:
    found = _decided()
    path = tmp_path / "exclusions.json"
    path.write_text(found.to_json(), encoding="utf-8")
    assert load_record(path, FIXTURE_CHECKS_HASH) == found
    with pytest.raises(ExclusionInputError, match="other check parameters"):
        load_record(path, "other")
    data = json.loads(path.read_text())
    data["exclude_checks"] = ["R1"]
    with pytest.raises(ExclusionInputError, match="does not match"):
        parse_record(json.dumps(data), "edited")
    data = json.loads(found.to_json())
    data["version"] = "9.0.0"
    with pytest.raises(ExclusionInputError, match="version"):
        parse_record(json.dumps(data), "future")


def test_a_replacement_names_the_record_it_replaces() -> None:
    first = record(("R1",))
    second = _decided(replaces_sha256=first.sha256)
    assert second.replaces_sha256 == first.sha256 and second.sha256 != _decided().sha256


def test_the_markdown_lists_every_rule() -> None:
    text = record_markdown(_decided())
    assert "| 1.1 | R5a | True |" in text and "| 1.5 | R1 | True |" in text
    assert "exclude_checks: R1, R5a" in text

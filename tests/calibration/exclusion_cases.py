"""Builders of the exclusion decision (ADR 0015 1.1/1.2/1.5) shared by the calibration tests and
by the consumers' tests (H2, H3): the rules come from the real ``configs/exclusion_rules.toml``."""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from calibration_cases import row
from stratum_cases import (
    FIXTURE_CHECKS_HASH,
    artifact_of,
    check_calibration,
    dataset_ref,
    stratum_model,
)

from physverify.calibration.artifact import (
    CheckCalibration,
    DatasetRef,
    EvaluationReport,
    EvaluationStratum,
    Monotonicity,
    RealFpr,
    Weight,
)
from physverify.calibration.artifact_io import Calibration, parse_calibration
from physverify.calibration.exclusion_rules import (
    ArmRun,
    ConsumedStratum,
    DecisionInputs,
    ExclusionRules,
    R2Arms,
)
from physverify.calibration.table import CalibrationRow, Role
from physverify.schema.checks import InputLevel
from physverify.stats.intervals import Ci, Rate
from physverify.verification.exclusion_record import ExclusionRecord, InputHashes, RuleEvidence

REPO = Path(__file__).resolve().parents[2]
RULES_TOML = REPO / "configs" / "exclusion_rules.toml"
FAMILY = "f4_stack"
R2_PERT = "r2_unsupported_rest"
PRECISE = "precise"


def rules_raw() -> dict[str, object]:
    with RULES_TOML.open("rb") as handle:
        section: dict[str, object] = tomllib.load(handle)["exclusion_rules"]
    return section


def rules() -> ExclusionRules:
    return ExclusionRules.model_validate(rules_raw())


def r5a_check(
    *, weight: float = 1.0, j_low: float | None = 0.2, tau_b_low: float | None = 0.1
) -> CheckCalibration:
    base = check_calibration(tau=1.0, weight=weight)
    return base.model_copy(
        update={
            "weight": Weight(
                value=weight,
                informedness=0.4,
                informedness_ci=Ci(low=j_low, high=0.6),
                n_positive=20,
                n_negative=20,
            ),
            "monotonicity": Monotonicity(
                kendall_tau_b=0.3,
                ci=Ci(low=tau_b_low, high=0.5),
                n_pairs=40,
                n_scenes=20,
                median_delta_by_level={},
                monotone_medians=True,
                h1_pass=True,
            ),
        }
    )


def calibration(
    checks: Mapping[str, CheckCalibration] | None = None, *, ready: bool = True
) -> Calibration:
    """A perceived/precise stratum with R2 (tau 1) and R5a; ``ready``: sufficient aggregate."""
    found = checks if checks is not None else {"R2": check_calibration(tau=1.0), "R5a": r5a_check()}
    stratum = stratum_model(found, InputLevel.PERCEIVED, 0.5 if ready else None, PRECISE)
    artifact = artifact_of((stratum,), checks_hash=FIXTURE_CHECKS_HASH)
    return parse_calibration(artifact.to_json())


def consumed(cal: Calibration, problem: str | None = None) -> ConsumedStratum:
    return ConsumedStratum(cal.stratum(InputLevel.PERCEIVED, PRECISE), problem)


def scene(index: int) -> str:
    return f"{FAMILY}-s{index:03d}"


def r2_rows(clean: Sequence[float], positive: Sequence[float]) -> tuple[CalibrationRow, ...]:
    """One clean and one R2 positive clip per scene, magnitudes as given (tau is 1)."""
    rows: list[CalibrationRow] = []
    for i, value in enumerate(clean):
        rows.append(row(value, scene=scene(i), clip_id=f"{scene(i)}__none", check_id="R2"))
    for i, value in enumerate(positive):
        rows.append(
            dataclasses.replace(
                row(
                    value,
                    role=Role.POSITIVE,
                    scene=scene(i),
                    clip_id=f"{scene(i)}__{R2_PERT}_L1",
                    level=1,
                    check_id="R2",
                ),
                perturbation=R2_PERT,
            )
        )
    return tuple(rows)


def arm(
    rows: Sequence[CalibrationRow],
    detector: str,
    *,
    run_id: str | None = None,
    commit: str = "c0",
    profile: str = PRECISE,
    checks_hash: str = FIXTURE_CHECKS_HASH,
) -> ArmRun:
    return ArmRun(
        run_id=run_id or f"run-{detector}",
        report_sha256=f"sha-{detector}",
        detectors=frozenset({detector}),
        profile=profile,
        commit=commit,
        checks_hash=checks_hash,
        rows=tuple(rows),
    )


def arms(base: Sequence[CalibrationRow], tiny: Sequence[CalibrationRow]) -> R2Arms:
    cfg = rules().r2
    return R2Arms(arm(base, cfg.base_detector), arm(tiny, cfg.tiny_detector))


def real_set(name: str, n_clips: int) -> DatasetRef:
    return dataset_ref(n_clips).model_copy(update={"name": name, "split": "all"})


def real_fpr(dataset: str, check_id: str, k: int, n: int, clustered_low: float = 0.0) -> RealFpr:
    rate = Rate(value=k / n, ci=Ci(low=clustered_low, high=1.0), k=k, n=n, n_scenes=n)
    return RealFpr(dataset=dataset, check_id=check_id, fpr=rate)


def evaluation(
    cal: Calibration,
    real: Sequence[RealFpr] = (),
    sets: Sequence[DatasetRef] | None = None,
    *,
    profile: str = PRECISE,
) -> EvaluationReport:
    chosen = (real_set("physics_iq", 54), real_set("pisa_real", 360)) if sets is None else sets
    stratum = EvaluationStratum(
        level=InputLevel.PERCEIVED,
        profile=profile,
        checks=(),
        scene_kendall=(),
        real=tuple(real),
        real_clip_flags=(),
    )
    return EvaluationReport(
        created_utc="2026-10-19T00:00:00Z",
        run_id="eval-run",
        artifact_path="calibration.json",
        artifact_sha256=cal.sha256,
        artifact_run_id=cal.run_id,
        dataset=dataset_ref(),
        real_datasets=tuple(ref.name for ref in chosen),
        strata=(stratum,),
        real_sets=tuple(chosen),
    )


def inputs(
    cal: Calibration | None = None,
    *,
    base: Sequence[CalibrationRow] | None = None,
    tiny: Sequence[CalibrationRow] | None = None,
    real: Sequence[RealFpr] = (),
    problem: str | None = None,
) -> DecisionInputs:
    cal = cal or calibration()
    same = r2_rows([0.1] * 20, [2.0] * 10 + [0.5] * 10)
    return DecisionInputs(
        calibration=cal,
        consumed=consumed(cal, problem),
        evaluation=evaluation(cal, real),
        evaluation_sha256="eval-sha",
        arms=arms(base or same, tiny or same),
        checks_hash=FIXTURE_CHECKS_HASH,
    )


def record(
    exclude: Sequence[str] = (),
    *,
    artifact_sha256: str = "artifact",
    checks_hash: str = FIXTURE_CHECKS_HASH,
) -> ExclusionRecord:
    """A sealed record for consumers' tests: ``exclude`` excluded by rule 1.5."""
    evidence = tuple(
        RuleEvidence(rule="1.5", check_id=c, excluded=True, decided=True, reason="real_fpr_high")
        for c in exclude
    )
    return ExclusionRecord(
        rules_version="1",
        consumed_level="perceived",
        consumed_profile=PRECISE,
        checks_hash=checks_hash,
        inputs=InputHashes(
            artifact_sha256=artifact_sha256,
            artifact_run_id="run-a",
            evaluation_sha256="eval",
            evaluation_run_id="eval-run",
            r2_base_run="base",
            r2_base_report_sha256="b",
            r2_tiny_run="tiny",
            r2_tiny_report_sha256="t",
            rules_sha256="rules",
        ),
        evidence=evidence,
        exclude_checks=tuple(sorted(exclude)),
        h3_families_without_target=(),
    ).sealed()


def write_record(path: Path, exclude: Sequence[str] = (), **kw: str) -> ExclusionRecord:
    found = record(exclude, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(found.to_json(), encoding="utf-8")
    return found

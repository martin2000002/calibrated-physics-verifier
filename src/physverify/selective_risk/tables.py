"""The cases of the selective-risk study and the threshold sweeps over them.

The checks read cached stage outputs (V8 oracle and V0 perceived chains) and a stage missing
from the cache fails its clip, which is counted. The reference of an event is the oracle's point
verdict; on real video, where there is no oracle, every scored event is PASS."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.calibration.artifact_io import load_calibration
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.chain_variants import oracle_variant, perceived_variant
from physverify.diagnosis.end_to_end import track_of, variant_outputs, variant_results
from physverify.diagnosis.pairing import pair_results, subject_map
from physverify.diagnosis.verification_adapter import A7CheckSuite
from physverify.schema.checks import InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.selective_risk.curve import (
    Case,
    CaseCounts,
    alpha_grid,
    curve_point,
    decision_codes,
    k_of,
    paired_cases,
    reference_codes,
)
from physverify.selective_risk.fit import choose_k, curve_rows, paired_aurc_difference
from physverify.selective_risk.settings import AbstentionSettings
from physverify.verification.runner import load_thresholds
from physverify.verification.sweep import CheckUnit

DEFAULT_K = 2.0


@dataclass(frozen=True)
class Collected:
    cases: list[Case]
    counts: CaseCounts
    clips: dict[str, ClipManifest]
    failed: list[str]


ChainFor = Callable[[str], tuple[ChainRunner, DatasetSource]]
"""The cached oracle/perceived chain of a dataset, and the dataset."""


@dataclass(frozen=True)
class Session:
    chain: ChainFor
    calibration_dir: Path
    settings: AbstentionSettings
    suite: A7CheckSuite
    reference_table: dict[str, Any]
    """Oracle level: the reference is the point verdict against its raw taus."""
    thresholds: dict[str, Any]
    """Perceived level: the table every swept decision uses."""
    min_span_iou: float
    checks: tuple[CheckUnit, ...]
    couplings: frozenset[tuple[str, str]]


def level_tables(
    artifact: Path | None,
    profile: str,
    fallback: Mapping[str, Any],
    overrides: Mapping[InputLevel, Path],
    *,
    stale_hash: str | None = None,
) -> dict[InputLevel, dict[str, Any]]:
    """Per level: an explicit table file, else the projection of A8's stratum (ADR 0009 §7),
    else ``fallback`` (what ``verify`` uses) when there is no artifact."""
    for level, given in overrides.items():
        if not given.is_file():
            raise UsageError(f"no {level.value} threshold table at {given}")
    calibration = (
        load_calibration(artifact, checks_hash=stale_hash)
        if artifact is not None and artifact.is_file()
        else None
    )
    root = None if artifact is None else artifact.parent
    out: dict[InputLevel, dict[str, Any]] = {}
    for level in (InputLevel.ORACLE, InputLevel.PERCEIVED):
        path = overrides.get(level)
        if path is not None:
            out[level] = dict(load_thresholds(path, {}))
        elif calibration is not None:
            stratum = calibration.stratum(level, profile)
            out[level] = stratum.thresholds_projection(root)
        else:
            out[level] = dict(fallback)
    return out


def collect_synthetic(session: Session, subset: str) -> Collected:
    """Oracle/perceived pairs of every cached clip of ``subset`` as cases."""
    chain, source = session.chain(session.settings.dataset)
    clips = list(source.clips(subset))
    cases: list[Case] = []
    counts, failed = CaseCounts(), []
    suite = session.suite
    for clip in clips:
        try:
            v8 = variant_outputs(chain, clip, oracle_variant())
            v0 = variant_outputs(chain, clip, perceived_variant())
            oracle = variant_results(suite, clip, v8, InputLevel.ORACLE)
            perceived = variant_results(suite, clip, v0, InputLevel.PERCEIVED)
            subjects = subject_map(track_of(v0), track_of(v8))
        except (PhysverifyError, ValueError):
            failed.append(clip.clip_id)
            continue
        paired = pair_results(oracle, perceived, subjects, session.min_span_iou)
        found, seen = paired_cases(paired, session.reference_table)
        cases += found
        counts += seen
    return Collected(cases, counts, {c.clip_id: c for c in clips}, failed)


def collect_real(session: Session, dataset: str) -> Collected:
    """Perceived events of real video, all with reference PASS (real physics)."""
    chain, source = session.chain(dataset)
    clips = list(source.clips(None))
    cases: list[Case] = []
    failed = []
    for clip in clips:
        try:
            v0 = variant_outputs(chain, clip, perceived_variant())
            results = variant_results(session.suite, clip, v0, InputLevel.PERCEIVED)
        except (PhysverifyError, ValueError):
            failed.append(clip.clip_id)
            continue
        cases += [
            Case(r.clip_id, r.check_id, Verdict.PASS, r) for r in results if r.magnitude is not None
        ]
    return Collected(cases, CaseCounts(), {c.clip_id: c for c in clips}, failed)


def _grid(settings: AbstentionSettings) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    alphas = alpha_grid(settings)
    return alphas, k_of(alphas)


def require(collected: Collected, what: str) -> None:
    if not collected.cases and collected.counts.no_reference:
        raise MissingInputError(
            f"{what}: none of {collected.counts.no_reference} oracle events has a reference; "
            "the oracle table has no tau for their checks"
        )
    if not collected.cases:
        raise MissingInputError(
            f"{what}: no case ({len(collected.failed)} clips without cached chains); "
            "run `physverify extract run` first"
        )


def sweep(
    collected: Collected, thresholds: dict[str, Any], settings: AbstentionSettings
) -> dict[str, Any]:
    alphas, ks = _grid(settings)
    reference = reference_codes(collected.cases)
    codes = decision_codes(collected.cases, ks, thresholds)
    points = [curve_point(row, reference) for row in codes]
    best = choose_k(points, list(ks), settings.risk_target, settings.ci)
    return {
        "curve": curve_rows(codes, reference, list(ks), list(alphas), settings.ci),
        "best": best,
        "alpha_star": None if best is None else float(alphas[best]),
        "k_star": None if best is None else float(ks[best]),
        "codes": codes,
    }


def named_ks(fitted: float | None) -> dict[str, float]:
    return {"k_2": DEFAULT_K} if fitted is None else {"k_star": fitted, "k_2": DEFAULT_K}


def compare_tables(
    collected: Collected, calibrated: dict[str, Any], raw: dict[str, Any], s: AbstentionSettings
) -> dict[str, Any]:
    """Both tables swept on the same cases; AURC difference with a bootstrap paired by clip."""
    swept = {"raw": sweep(collected, raw, s), "calibrated": sweep(collected, calibrated, s)}
    reference = reference_codes(collected.cases)
    difference = paired_aurc_difference(
        swept["raw"]["codes"],
        swept["calibrated"]["codes"],
        reference,
        [c.clip_id for c in collected.cases],
        n_boot=s.bootstrap,
        seed=s.seed,
        ci=s.ci,
    )
    out: dict[str, Any] = {
        name: {k: v for k, v in result.items() if k != "codes"} for name, result in swept.items()
    }
    out["aurc_calibrated_minus_raw"] = difference
    out["identical_tables"] = calibrated["checks"] == raw["checks"]
    return out

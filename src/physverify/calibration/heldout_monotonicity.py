"""Held-out monotonicity: the confirmatory figure of the freeze.

Per check, over the rows with ``split != calib``: the pooled monotonicity tau_b of the
bias-corrected deltas, the informedness J of the verdicts of the frozen tau and sigma correction
and, where the held-out scenes carry families, the stratified block of the dev score on the same
flags and corrected magnitudes. Nothing is refitted and there are no folds; ``settings`` must be
the artifact's (``fit_settings``), so the bootstrap, seed and interval level are the fit's and no
parameter comes from elsewhere. A check meets it under the dev score's clause
(``calibration.clause``). Pure, no I/O."""

from __future__ import annotations

import math
from collections.abc import Sequence

from physverify.calibration.apply import policy_for
from physverify.calibration.artifact import HeldoutMonotonicityCheck, Monotonicity
from physverify.calibration.artifact_io import Stratum
from physverify.calibration.check_weights import Flag, raw_weight
from physverify.calibration.clause import clause
from physverify.calibration.dev_score import oof_flags
from physverify.calibration.monotonicity import deltas, monotonicity
from physverify.calibration.oof import check_valued, check_values
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stratified import Families, FamilyOf, Valued, family_map, unit_scores
from physverify.calibration.table import CalibrationRow
from physverify.schema.clip import Split


def heldout_h1(
    stratum: Stratum,
    rows: Sequence[CalibrationRow],
    check_ids: Sequence[str],
    settings: CalibrationSettings,
    family_of: FamilyOf,
) -> tuple[HeldoutMonotonicityCheck, ...]:
    """Checks without a held-out row are left out: only calib rows give ``()``. ``family_of``
    names the family of a scene (None: none), as for the dev score."""
    held = [r for r in rows if r.split is not Split.CALIB]
    families = family_map((r.scene_id for r in held), family_of)
    found: list[HeldoutMonotonicityCheck] = []
    for check_id in check_ids:
        mine = [r for r in held if r.check_id == check_id]
        if mine:
            found.append(_check(stratum, check_id, mine, settings, families))
    return tuple(found)


def _check(
    stratum: Stratum,
    check_id: str,
    rows: Sequence[CalibrationRow],
    settings: CalibrationSettings,
    families: Families | None,
) -> HeldoutMonotonicityCheck:
    frozen, policy = stratum.model.checks[check_id], policy_for(stratum)
    tau_b = _finite(monotonicity(deltas(rows, stratum.sigma_correction(check_id)[0]), settings))
    valued = list(check_valued(check_values(rows, frozen, policy)).values())
    return judged(check_id, tau_b, oof_flags(rows, frozen, policy), valued, families, settings)


def judged(
    check_id: str,
    tau_b: Monotonicity,
    flags: Sequence[Flag],
    valued: Sequence[Valued],
    families: Families | None,
    settings: CalibrationSettings,
) -> HeldoutMonotonicityCheck:
    """One check from its frozen verdicts (``oof_flags``) and corrected magnitudes."""
    j = raw_weight(flags, settings)
    unit = unit_scores(valued, flags, families, settings)
    met = clause(tau_b, j, unit.stratified)
    return HeldoutMonotonicityCheck(
        check_id=check_id,
        monotonicity=tau_b,
        j=j,
        meets_h1=met.meets_h1,
        meets_j=met.meets_j,
        meets_both=met.meets_both,
        auroc_readings=unit.auroc_cv,
        stratified=unit.stratified,
        meets_stratified=met.meets_stratified,
        meets_clause=met.meets_clause,
    )


def _finite(m: Monotonicity) -> Monotonicity:
    """An undefined tau_b (no pair, or no spread) is null, as the JSON writes it."""
    tau = m.kendall_tau_b
    return m if tau is None or math.isfinite(tau) else m.model_copy(update={"kendall_tau_b": None})


def n_met(checks: Sequence[HeldoutMonotonicityCheck]) -> int:
    return sum(c.counted for c in checks)

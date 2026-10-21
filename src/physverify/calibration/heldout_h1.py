"""H1 on held-out rows (plan-congelado F4, J7): the confirmatory figure of the freeze.

Per check, over the rows with ``split != calib``: D8's pooled tau_b of the bias-corrected
deltas and the informedness J of the verdicts of the frozen tau and sigma correction. Nothing is
refitted and there are no folds; ``settings`` must be the artifact's (``fit_settings``), so the
bootstrap, seed and interval level are the fit's and no parameter comes from elsewhere. A check
meets H1 when tau_b^lo > 0 and J^lo > 0. Pure, no I/O."""

from __future__ import annotations

import math
from collections.abc import Sequence

from physverify.calibration.apply import policy_for
from physverify.calibration.devscore import oof_flags
from physverify.calibration.loader import Stratum
from physverify.calibration.monotonicity import deltas, monotonicity
from physverify.calibration.schema import HeldoutH1Check, Monotonicity
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.table import CalibrationRow
from physverify.calibration.weights import raw_weight
from physverify.schema.clip import Split


def heldout_h1(
    stratum: Stratum,
    rows: Sequence[CalibrationRow],
    check_ids: Sequence[str],
    settings: CalibrationSettings,
) -> tuple[HeldoutH1Check, ...]:
    """Checks without a held-out row are left out: only calib rows give ``()``."""
    held = [r for r in rows if r.split is not Split.CALIB]
    found: list[HeldoutH1Check] = []
    for check_id in check_ids:
        mine = [r for r in held if r.check_id == check_id]
        if mine:
            found.append(_check(stratum, check_id, mine, settings))
    return tuple(found)


def _check(
    stratum: Stratum,
    check_id: str,
    rows: Sequence[CalibrationRow],
    settings: CalibrationSettings,
) -> HeldoutH1Check:
    frozen = stratum.model.checks[check_id]
    tau_b = _finite(monotonicity(deltas(rows, stratum.sigma_correction(check_id)[0]), settings))
    j = raw_weight(oof_flags(rows, frozen, policy_for(stratum)), settings)
    low = j.informedness_ci.low
    meets_h1, meets_j = tau_b.h1_pass, low is not None and low > 0.0
    return HeldoutH1Check(
        check_id=check_id,
        monotonicity=tau_b,
        j=j,
        meets_h1=meets_h1,
        meets_j=meets_j,
        meets_both=meets_h1 and meets_j,
    )


def _finite(m: Monotonicity) -> Monotonicity:
    """An undefined tau_b (no pair, or no spread) is null, as the JSON writes it."""
    tau = m.kendall_tau_b
    return m if tau is None or math.isfinite(tau) else m.model_copy(update={"kendall_tau_b": None})


def n_met(checks: Sequence[HeldoutH1Check]) -> int:
    return sum(c.meets_both for c in checks)

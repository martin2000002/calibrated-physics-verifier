"""With sigma_pair_filter = "all" the fit and the devscore report are the ones written before
the filter existed, except ``version`` and the fields added with it (``golden.same``)."""

from __future__ import annotations

import json
from typing import Any

from dev_score_golden_case import GOLDEN, fitted
from golden import same

# Fields added with the sigma-pair filter, with the value they take under "all"; any other
# difference fails.
ADDED_AT_ALL: dict[str, Any] = {
    "n_informative": None,
    "coverage_after_informative": [],
    "coverage_after_degenerate": [],
    "sigma_pair_filter": "all",
}
# Added later, not tied to the filter; the stratified fields (devscore 1.2.0) are null or
# repeat the global figures on scenes without a family, as here.
ADDED_ANY = (
    "folds_without_sigma_factor",
    "device",
    "auroc_cv_readings",
    "auroc_cv_operational",
    "stratification",
    "stratified",
    "meets_stratified",
    "meets_clause",
    "n_met_global",
    "met_global",
    "tiebreak_auroc_cv_global",
)


def without_filter_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_filter_fields(item)
            for key, item in value.items()
            if key != "version"
            and key not in ADDED_ANY
            and not (key in ADDED_AT_ALL and item == ADDED_AT_ALL[key])
        }
    if isinstance(value, list):
        return [without_filter_fields(item) for item in value]
    return value


def golden() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return loaded


def test_the_all_filter_reproduces_the_fit_before_the_filter() -> None:
    now, before = fitted(sigma_pair_filter="all"), golden()
    for part in ("two_level", "devscore"):
        assert same(without_filter_fields(now[part]), without_filter_fields(before[part])), part


def test_the_golden_case_is_sensitive_to_the_filter() -> None:
    nonzero = without_filter_fields(fitted(sigma_pair_filter="nonzero")["two_level"])
    assert not same(nonzero, without_filter_fields(golden()["two_level"]))

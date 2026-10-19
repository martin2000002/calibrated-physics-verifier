"""``[check_scope]`` for H3: one wrapper around every scorer ``build_scorer`` returns.

The excluded checks never reach the scorer, whichever it is, and their ids are part of the
thresholds stamp (``@excl:``), so scores of another scope are stale (``same_stamp``,
``frozen_stamp_problem``) even when no code changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from physverify.calibration.settings import check_registry
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.finetune.scorers._base import AggregateScore, AggregateScorer
from physverify.finetune.scorers._status import CalibrationStatusSource
from physverify.finetune.settings import FinetuneSettings
from physverify.schema.checks import CheckResult
from physverify.verification.scope import check_scope

EXCLUDED_TAG: Final = "@excl:"


def scorer_exclusion(settings: Settings, cfg: FinetuneSettings) -> frozenset[str]:
    """``exclude_checks``, plus the time-dependent ones when ``[finetune]`` excludes them."""
    scope = check_scope(settings, check_registry())
    return scope.excluded(time_dependent=cfg.exclude_time_dependent)


class ScopedScorer:
    """``inner`` without ``excluded``; its class variables are those of ``inner`` (see
    ``scoped``), so the stamp keeps the scorer's name and version."""

    name: ClassVar[str]
    version: ClassVar[str]
    calibrated: ClassVar[bool]

    def __init__(self, inner: AggregateScorer, excluded: frozenset[str]) -> None:
        self._inner, self._excluded = inner, excluded

    def unavailable(self) -> str | None:
        return self._inner.unavailable()

    def thresholds(self) -> str:
        return f"{self._inner.thresholds()}{EXCLUDED_TAG}{','.join(sorted(self._excluded))}"

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        """``only = frozenset()`` (no common check) stays empty: an abstention, never all."""
        wanted = only if only is not None else frozenset(r.check_id for r in results)
        return self._inner.score(results, only=wanted - self._excluded)

    def calibration_status(self) -> Mapping[str, Any] | None:
        inner = self._inner
        return inner.calibration_status() if isinstance(inner, CalibrationStatusSource) else None


def scoped(scorer: AggregateScorer, excluded: frozenset[str]) -> AggregateScorer:
    """``scorer`` itself when nothing is excluded, else wrapped in a ``ScopedScorer``."""
    if not excluded:
        return scorer
    kind = type(scorer)

    class Scoped(ScopedScorer):
        name = kind.name
        version = kind.version
        calibrated = kind.calibrated

    return Scoped(scorer, excluded)


def effective_targets(target_checks: Sequence[str], excluded: frozenset[str]) -> list[str]:
    return [check for check in target_checks if check not in excluded]


def family_targets(cfg: FinetuneSettings, excluded: frozenset[str]) -> dict[str, list[str]]:
    """The target checks each family keeps after the exclusion (fixed eval prompts included)."""
    families = {f.name: list(f.target_checks) for f in cfg.prompts.families}
    for prompt in cfg.prompts.fixed_eval:
        families.setdefault(prompt.family, list(prompt.target_checks))
    return {name: effective_targets(checks, excluded) for name, checks in families.items()}


def require_family_targets(cfg: FinetuneSettings, excluded: frozenset[str]) -> None:
    """``on_family_without_targets = "error"``: a family left with no target check would be
    decided only by checks foreign to its physics."""
    empty = sorted(name for name, kept in family_targets(cfg, excluded).items() if not kept)
    if empty and cfg.on_family_without_targets == "error":
        raise UsageError(
            f"families {empty} keep no target check after excluding {sorted(excluded)}; "
            'change [check_scope] or set finetune.on_family_without_targets = "keep"'
        )

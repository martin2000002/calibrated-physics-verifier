"""Read-only queries the H200 scripts ask of the checkout, one function per query.

Every failure is a ``QueryError`` (exit 2). The domain modules are imported inside each query: a
script starts in a fraction of the time the whole package takes to import."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physverify.freeze.errors import QueryError as QueryError
from physverify.freeze.files import read_json as read_json
from physverify.freeze.selection import DEVICE
from physverify.freeze.selection import PROFILE as PROFILE

if TYPE_CHECKING:
    from physverify.config.models import Settings as Settings
    from physverify.core.registry import Registry
    from physverify.datasets._base import DatasetSource, HumanLabelSource, LabelledVideo
    from physverify.human_agreement.coverage import InputGaps as InputGaps

ERROR_KEY = "error"
SHOWN_ERRORS = 10


def query_settings(configs: Path, overrides: Sequence[str], profile: str = PROFILE) -> Settings:
    """The checkout's settings; the profile also fixes the synthetic dataset version."""
    from physverify.config.loader import load_settings

    return load_settings(
        configs, environ=os.environ, profile=profile, device=DEVICE, overrides=overrides
    )


def clip_ids(source: DatasetSource, subset: str, split: str | None) -> list[str]:
    clips = source.clips(subset)
    return sorted(c.clip_id for c in clips if split is None or c.split.value == split)


def dataset_clip_ids(settings: Settings, dataset: str, subset: str, split: str | None) -> list[str]:
    from physverify.app import plugins
    from physverify.core.errors import UsageError
    from physverify.datasets._base import dataset_from

    try:
        source = dataset_from(plugins.build("datasets"), dataset, settings)
    except UsageError as exc:
        raise QueryError(str(exc)) from exc
    return clip_ids(source, subset, split)


def allowed_real_sets(settings: Settings) -> list[Mapping[str, int]]:
    """The real-set sizes the pre-registered exclusion rules allow an evaluation to report."""
    from physverify.app import plugins
    from physverify.calibration.exclusion_rules import SECTION, exclusion_rules

    rules = exclusion_rules(settings.sections.get(SECTION, {}), plugins.build("checks"))
    return list(rules.real.allowed_sets)


def run_errors(run_dir: Path) -> list[str]:
    """Refused when a check row of the run carries an input or check error: verify exits 0 on an
    uncached clip and only marks its rows abstained, with the error in ``details``."""
    from physverify.schema.checks import CheckResult
    from physverify.schema.io import read_jsonl
    from physverify.verification.runner import CHECKS_FILE

    path = run_dir / CHECKS_FILE
    try:
        rows = read_jsonl(CheckResult, path)
    except (OSError, ValueError) as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    errors = [
        f"{r.clip_id} {r.check_id}: {r.details[ERROR_KEY] or '(no message)'}"
        for r in rows
        if ERROR_KEY in r.details
    ]
    if errors:
        more = len(errors) - SHOWN_ERRORS
        shown = errors[:SHOWN_ERRORS] + ([f"... and {more} more"] if more > 0 else [])
        raise QueryError(f"{len(errors)} check rows of {path} failed:\n" + "\n".join(shown))
    return [f"{path}: {len(rows)} check rows, none with an error"]


def preregistration_field(path: Path, field: str) -> str | None:
    """One field of the pre-registration at ``path``, as stored."""
    from physverify.core.errors import PhysverifyError
    from physverify.human_agreement.prereg import Preregistration, load_preregistration

    if field not in Preregistration.model_fields:
        raise QueryError(f"no field {field!r} in a pre-registration")
    try:
        value: object = getattr(load_preregistration(path), field)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc
    return None if value is None else str(value)


def _preregistration(settings: Settings) -> Path:
    from physverify.human_agreement.settings import agreement_settings, preregistration_path

    return preregistration_path(settings, agreement_settings(settings))


def logged_tests(settings: Settings) -> int:
    """Test comparisons logged against the configured pre-registration."""
    from physverify.human_agreement.prereg import logged_comparisons

    return len(logged_comparisons(_preregistration(settings)))


def comparison_log(settings: Settings) -> Path:
    from physverify.human_agreement.prereg import comparison_log_path

    return comparison_log_path(_preregistration(settings))


@dataclass(frozen=True)
class _Plugins:
    """The ``PluginContext`` of a query: its settings and the discovered plugin registries."""

    settings: Settings

    def registry(self, kind: str) -> Registry[Any]:
        from physverify.app import plugins

        return plugins.build(kind)


def _labels(settings: Settings) -> HumanLabelSource:
    from physverify.app.human_agreement import label_source

    return label_source(_Plugins(settings))


def _rows(labels: HumanLabelSource, subset: str) -> list[LabelledVideo]:
    from physverify.core.errors import PhysverifyError

    try:
        return labels.rows(subset)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def subset_gaps(settings: Settings, subset: str) -> InputGaps:
    """The rows of ``subset`` without an object query or a video."""
    from physverify.human_agreement.coverage import dataset_gaps

    labels = _labels(settings)
    return dataset_gaps(labels, _rows(labels, subset))


def ready_clip(settings: Settings, subset: str) -> str:
    """The first clip of ``subset`` (by id) with both a video and an object query."""
    from physverify.human_agreement.coverage import dataset_tests

    labels = _labels(settings)
    has_query, has_video = dataset_tests(labels)
    for row in sorted(_rows(labels, subset), key=lambda r: r.clip_id):
        if has_video(row) and has_query(row):
            return row.clip_id
    raise QueryError(f"no clip of {subset} has both a video and an object query")

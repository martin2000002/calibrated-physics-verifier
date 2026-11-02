"""Development datasets (``calibration.development_sets``): a dataset content declared by
``name@version`` whose clips the calib-only commands read as calib clips, whatever split its
store gives them. Another version under the same name is not one."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from physverify.calibration.artifact import DatasetRef
from physverify.calibration.session import fit_clips
from physverify.calibration.settings import CalibrationSettings
from physverify.core.errors import UsageError
from physverify.datasets._base import DatasetSource, dataset_identity
from physverify.datasets._versions import versioned_name, versioned_parts
from physverify.schema.clip import ClipManifest, Split


def is_development(source: DatasetSource, settings: CalibrationSettings) -> bool:
    name = versioned_name(source)
    return name is not None and name in settings.development_sets


def may_be_development(dataset: str, settings: CalibrationSettings) -> bool:
    """Whether some declared development set carries the name ``dataset`` (no version read)."""
    return any(versioned_parts(spec)[0] == dataset for spec in settings.development_sets)


def development_clips(
    source: DatasetSource, clips: Sequence[ClipManifest], settings: CalibrationSettings
) -> list[ClipManifest]:
    """``clips`` relabelled calib when ``source`` is a development set, else unchanged."""
    if not is_development(source, settings):
        return list(clips)
    return [replace(clip, split=Split.CALIB) for clip in clips]


def require_all_clips_scorable(
    source: DatasetSource, fitted: DatasetRef, settings: CalibrationSettings
) -> None:
    """``--score-all-clips`` scores a development set with an artifact fitted on another set."""
    if not is_development(source, settings):
        named = versioned_name(source) or source.name
        raise UsageError(
            f"--score-all-clips scores a development set only; {named} is not in "
            "calibration.development_sets"
        )
    if fitted.identity is None:
        raise UsageError(
            "--score-all-clips needs an artifact that records the identity of its fit set; this "
            "one does not: refit it"
        )
    if fitted.identity == dataset_identity(source):
        raise UsageError(
            f"--score-all-clips on the artifact's own fit set ({fitted.name} {fitted.identity}) "
            "would score the clips it was fitted on"
        )


def calib_clips(
    source: DatasetSource, subset: str, wanted: Sequence[str], settings: CalibrationSettings
) -> list[ClipManifest]:
    """``fit_clips``, except that every clip of a development set is a calib clip."""
    if not is_development(source, settings):
        return fit_clips(source, subset, wanted)
    found = fit_clips(source, subset, wanted) if wanted else source.clips(subset)
    return development_clips(source, found, settings)

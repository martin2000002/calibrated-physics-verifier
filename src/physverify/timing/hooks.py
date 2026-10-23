"""No-op defaults of a timing target's lifecycle around ``TimingTarget.run``.

A target says why it cannot run here, brings its own clips, prepares untimed state once,
describes itself and releases resources. The harness calls, per target: ``unavailable`` ->
``own_clips`` (else the ``--dataset`` selection) -> ``prepare`` -> ``run`` per call -> ``close``
(always). A target inherits ``TargetHooks`` and overrides the hooks it needs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from physverify.schema.clip import ClipManifest
from physverify.timing._base import TimingContext, TimingTarget


class TargetHooks:
    def unavailable(self, ctx: TimingContext) -> str | None:
        """Reason the target cannot run in this environment (it is then skipped), or None."""
        return None

    def own_clips(self, ctx: TimingContext) -> Sequence[ClipManifest] | None:
        """Clips the target brings itself; None means the command-line selection."""
        return None

    def prepare(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> Mapping[str, float]:
        """Untimed-by-call setup, once per run; returns setup measures (e.g. a model load)."""
        return {}

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        return {}

    def close(self) -> None:
        return None


def brings_own_clips(target: type[TimingTarget]) -> bool:
    """True when the target class overrides ``own_clips`` (it needs no ``--dataset``)."""
    return getattr(target, "own_clips", TargetHooks.own_clips) is not TargetHooks.own_clips

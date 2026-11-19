"""The persisted record of a check whose input never reached it on part of a set."""

from __future__ import annotations

from physverify.core.models import ArtifactModel
from physverify.schema.checks import InputLevel


class InputGap(ArtifactModel):
    """Clips of a set whose every result of a check is the runner's input gap: the input never
    reached the check (a heavy stage cached under another device, an extraction that failed)."""

    dataset: str
    level: InputLevel
    check_id: str
    missing: int
    n: int

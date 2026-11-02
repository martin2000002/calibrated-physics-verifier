"""A check that inherits its version from R1 instead of declaring one."""

from __future__ import annotations

from typing import ClassVar

from physverify.checks.r1_interpenetration import InterpenetrationCheck


class InheritedVersionCheck(InterpenetrationCheck):
    check_id: ClassVar[str] = "R9"


PLUGINS = (InheritedVersionCheck,)

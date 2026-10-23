"""The type of the ``make_settings`` fixture: ``make_settings("section.key=value", ...,
profile=None)`` over the committed configs."""

from __future__ import annotations

from collections.abc import Callable

from physverify.config.models import Settings

SettingsFactory = Callable[..., Settings]

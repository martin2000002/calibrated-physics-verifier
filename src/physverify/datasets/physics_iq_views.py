"""``physics_iq_views``: the ``left`` and ``right`` views of Physics-IQ, a sealed real control of
the freeze.

Scenarios, takes, frame rate, annotations and bucket come from ``[physics_iq]``; this section only
adds the views. The raw directory and the fetch manifest are the plugin's own, so the development
view (``center``) and this control never share files or a stage cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.datasets._physics_iq import PhysicsIQDataset

if TYPE_CHECKING:
    from physverify.config.models import Settings

SECTION = "physics_iq_views"


class PhysicsIQViewsSettings(FrozenModel):
    views: tuple[str, ...] = Field(min_length=1)


class PhysicsIQViewsDataset(PhysicsIQDataset):
    name: ClassVar[str] = "physics_iq_views"

    def views_of(self, settings: Settings) -> tuple[str, ...]:
        return settings.section(SECTION, PhysicsIQViewsSettings).views


PLUGINS = (PhysicsIQViewsDataset,)

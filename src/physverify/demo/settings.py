"""Settings of the end-to-end demo: the ``[demo]`` section."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field

from physverify.config.models import Settings
from physverify.core.models import FrozenModel
from physverify.generation.settings import SLUG_PATTERN

SECTION: Final = "demo"


class OverlayStyle(FrozenModel):
    """Sizes are in pixels of a 480-pixel-high frame and scale with the frame height.
    ``badge_width`` is the badge panel's minimum width; ``all_checks`` badges every check, not only
    the targeted, decided and failing ones."""

    font_scale: float = Field(0.5, gt=0)
    min_font_scale: float = Field(0.35, gt=0)
    label_scale: float = Field(0.7, gt=0)
    bar_height: int = Field(26, ge=4)
    badge_height: int = Field(18, ge=4)
    badge_width: int = Field(120, ge=8)
    badge_alpha: float = Field(0.6, ge=0, le=1)
    all_checks: bool = False
    margin: int = Field(6, ge=0)
    reference_height: int = Field(480, ge=1)


class DemoSettings(FrozenModel):
    adhoc_campaign: str = Field("demo_adhoc", pattern=SLUG_PATTERN)
    out_dir: str = "demo"
    allow_provisional: bool = True
    select_run: str = "main"
    overlay: OverlayStyle = Field(default_factory=OverlayStyle)


def load_demo(settings: Settings) -> DemoSettings:
    return settings.section(SECTION, DemoSettings)


def default_out(settings: Settings) -> Path:
    """``out_dir`` under ``paths.data_dir`` unless it is absolute."""
    out = Path(load_demo(settings).out_dir)
    return out if out.is_absolute() else settings.paths.data_dir / out

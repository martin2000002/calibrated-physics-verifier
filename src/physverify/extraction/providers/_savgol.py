"""Parameters of Savitzky-Golay differentiation (not a plugin module): kinematics/savgol runs with
them, and the image-plane rest test reads the plan's window and order."""

from __future__ import annotations

from pydantic import Field

from physverify.extraction._base import ProviderParams
from physverify.physics.kinematics import SavgolSpec
from physverify.physics.row_time import odd_rows_in, reference_seconds


class SavgolParams(ProviderParams):
    window_s: float = Field(default=reference_seconds(7), gt=0.0)
    polyorder: int = Field(default=2, ge=0)

    def spec(self, row_seconds: float) -> SavgolSpec:
        """The window at ``row_seconds`` per row: odd and > polyorder at every frame rate."""
        window = odd_rows_in(self.window_s, row_seconds, self.polyorder + 1)
        return SavgolSpec(window, self.polyorder)

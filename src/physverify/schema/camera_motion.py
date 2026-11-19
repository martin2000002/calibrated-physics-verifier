"""Background image motion of a clip between consecutive sampled frames."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema._validate import ArraySpec, check_bool, check_str
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from numpy.typing import NDArray

CLIP_PERCENTILE = 90.0


@dataclass(frozen=True, eq=False)
class CameraMotion(Record):
    """Per pair of consecutive sampled frames (P,): ``pair_shift``, the camera motion of the pair
    as a fraction of the image diagonal (NaN when the pair has too little background to tell),
    ``pair_seconds``, the real time between its two frames, and ``background_fraction``.
    ``measured`` is False when the estimate failed; such a record carries no pairs."""

    estimator: str
    pair_shift: NDArray[np.float64]
    pair_seconds: NDArray[np.float64]
    background_fraction: NDArray[np.float64]
    measured: bool

    _arrays = MappingProxyType(
        {
            "pair_shift": ArraySpec("f8", ("P",)),
            "pair_seconds": ArraySpec("f8", ("P",)),
            "background_fraction": ArraySpec("f8", ("P",)),
        }
    )

    def validate(self) -> None:
        super().validate()
        check_str("estimator", self.estimator)
        check_bool("measured", self.measured)
        if not np.all(self.pair_seconds > 0):
            raise SchemaError("pair_seconds must be > 0")

    @classmethod
    def unmeasured(cls, estimator: str) -> Self:
        empty = np.zeros(0, dtype=np.float64)
        return cls(estimator, empty, empty.copy(), empty.copy(), False)

    def pair_values(self, per_second: bool) -> NDArray[np.float64]:
        """``pair_shift``, or ``pair_shift / pair_seconds`` (per second) when ``per_second``."""
        values: NDArray[np.float64] = (
            self.pair_shift / self.pair_seconds if per_second else self.pair_shift
        )
        return values

    def clip_statistics(self, per_second: bool) -> tuple[float, float] | None:
        """The 90th percentile (linear) and the maximum of ``pair_values`` over the pairs with
        background; None when the record is unmeasured or no pair has background."""
        values = self.pair_values(per_second)
        finite = values[np.isfinite(values)]
        if not self.measured or finite.size == 0:
            return None
        return float(np.percentile(finite, CLIP_PERCENTILE)), float(finite.max())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            estimator=data["estimator"],
            measured=data["measured"],
            **cls.decode_arrays(data),
        )

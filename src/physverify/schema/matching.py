"""Identity map between perceived and ground-truth object ids."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from physverify.schema._validate import (
    check_enum,
    check_int,
    check_number,
    check_str,
    check_str_tuple,
)
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


class MatchMethod(str, Enum):
    BOX_IOU = "box_iou"
    MASK_IOU = "mask_iou"


@dataclass(frozen=True, eq=False)
class IdMap(Record):
    """``pairs`` are (pred_id, gt_id, score); every id appears at most once overall."""

    pairs: tuple[tuple[str, str, float], ...]
    unmatched_pred: tuple[str, ...]
    unmatched_gt: tuple[str, ...]
    method: MatchMethod
    frames_used: tuple[int, ...]
    min_score: float

    def pred_to_gt(self) -> dict[str, str]:
        return {pred: gt for pred, gt, _ in self.pairs}

    def gt_to_pred(self) -> dict[str, str]:
        return {gt: pred for pred, gt, _ in self.pairs}

    def validate(self) -> None:
        check_enum("method", self.method, MatchMethod)
        check_str_tuple("unmatched_pred", self.unmatched_pred)
        check_str_tuple("unmatched_gt", self.unmatched_gt)
        check_number("min_score", self.min_score)
        if not isinstance(self.frames_used, tuple):
            raise SchemaError("frames_used must be a tuple of frame indices")
        for frame in self.frames_used:
            check_int("frames_used", frame, minimum=0)
        self._validate_pairs()

    def _validate_pairs(self) -> None:
        if not isinstance(self.pairs, tuple):
            raise SchemaError("pairs must be a tuple")
        for pair in self.pairs:
            if not isinstance(pair, tuple) or len(pair) != 3:
                raise SchemaError("each pair is (pred_id, gt_id, score)")
            check_str("pred_id", pair[0])
            check_str("gt_id", pair[1])
            check_number("score", pair[2])
        preds = [p for p, _, _ in self.pairs] + list(self.unmatched_pred)
        gts = [g for _, g, _ in self.pairs] + list(self.unmatched_gt)
        if len(set(preds)) != len(preds) or len(set(gts)) != len(gts):
            raise SchemaError("an id may appear only once in pairs and unmatched lists")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            pairs=tuple((str(p), str(g), s) for p, g, s in data["pairs"]),
            unmatched_pred=tuple(data["unmatched_pred"]),
            unmatched_gt=tuple(data["unmatched_gt"]),
            method=MatchMethod(data["method"]),
            frames_used=tuple(data["frames_used"]),
            min_score=data["min_score"],
        )

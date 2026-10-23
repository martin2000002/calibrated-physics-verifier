"""R2 truth: overhang of the bodies at rest in the middle frame that stand on at least one
movable body."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.frame_windows import on_top_pairs, rest_mask, solid_at
from physverify.synth.magnitudes import nanmax, overhang_on


class OverhangTruth:
    check_id: ClassVar[str] = "R2"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        mid = states.n_frames // 2
        ids = states.object_ids.astype(str).tolist()
        pairs = on_top_pairs(contacts)
        values: list[float] = []
        for i, object_id in enumerate(ids):
            lowers = sorted({low for up, low in pairs if up == object_id})
            if not lowers or all(low == ENV_ID for low in lowers) or not rest_mask(states, i)[mid]:
                continue
            solids = [
                None if low == ENV_ID else solid_at(states, ids.index(low), mid) for low in lowers
            ]
            values.append(overhang_on(solid_at(states, i, mid), solids, tol))
        return nanmax(values)


PLUGINS = (OverhangTruth,)

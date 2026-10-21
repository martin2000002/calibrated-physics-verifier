"""LIFT tables of the analytic fixture, written once by LIFT version 2 (commit 34e4e6e) into
``golden/lift_v2/<case>.npz`` and compared with version 3 under ``temporal_filter = "none"``
(it2 T12). The same module runs in both trees:

    PYTHONPATH=<old>/src:tests/fixtures:tests/extraction python tests/extraction/lift_golden_case.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fakes import affine_disparity
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip, build_clip

from physverify.config.models import RuntimeSettings
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthMap
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable

GOLDEN = Path(__file__).resolve().parent / "golden" / "lift_v2"
PERCEIVED = {Stage.TRACK: "edgetam", Stage.DEPTH: "vda_small", Stage.LIFT: "backproject"}
CASES: dict[str, dict[str, Any]] = {
    "truth_closed_form": {"noisy": False, "params": {}},
    "noisy_monte_carlo": {"noisy": True, "params": {"sigma_mode": "monte_carlo", "mc_draws": 64}},
}
IMPULSE_GAIN = 1.3


def profile() -> ErrorProfile:
    return profile_of(
        entry(Stage.DEPTH, "vda_small", "depth_rel", 0.05),
        entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.1),
        entry(Stage.LIFT, "backproject", "centroid_rel", 0.04),
        entry(Stage.LIFT, "backproject", "size_rel", 0.06),
    )


def upstream(clip: SyntheticClip, noisy: bool) -> dict[Stage, StageOutput]:
    """Truth masks with a two-row gap in the first object; with ``noisy`` an affine disparity
    whose second object is IMPULSE_GAIN times nearer on one row (a depth impulse)."""
    s = make_context(clip).sampling
    gt = clip.ground_truth
    track = gt.masks(s)
    masks = track.masks.copy()
    shown = np.flatnonzero(masks[:, 0].any(axis=(1, 2)))
    lo = int(shown[len(shown) // 2])
    masks[lo : lo + 2, 0] = False
    depth: DepthMap = gt.depth(s)
    if noisy:
        depth = affine_disparity(depth, 2.0, 0.0)
        disparity = depth.depth.copy()
        row = int(shown[1])
        disparity[row][track.masks[row, 1]] *= np.float32(IMPULSE_GAIN)
        depth = replace(depth, depth=disparity)
    return {
        Stage.TRACK: replace(track, masks=masks),
        Stage.DEPTH: depth,
        Stage.CAMERA: gt.camera(s),
    }


def table(clip: SyntheticClip, case: str, **extra: Any) -> ObjectStateTable:
    spec = CASES[case]
    names = {**dict.fromkeys(STAGE_ORDER, "ground_truth"), **PERCEIVED}
    ctx = make_context(clip, upstream=upstream(clip, spec["noisy"]), plan=names, profile=profile())
    output = run_provider(BackprojectLift, ctx, **spec["params"], **extra)
    assert isinstance(output, ObjectStateTable)
    return output


def main(out: Path = GOLDEN) -> None:
    clip = build_clip(RuntimeSettings())
    out.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        table(clip, case).save_npz(out / f"{case}.npz")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN)

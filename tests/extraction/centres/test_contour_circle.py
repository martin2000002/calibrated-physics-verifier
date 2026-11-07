"""The contour circle centre on synthetic discs and on three calibration tracks.

``golden/centres/contour_circle_tracks.npz`` holds three two-ball tracks (24 rows, 256 x 256)
of the calibration set, as masks of the ``sam2_1_hiera_tiny`` tracker (no frames, depths or
ground truth), and the centres, radii and methods the original research script ``contour.py``
gave on them. The file records that script's sha256 and the sha256 of each cached mask entry.
The tracks were picked before any centre was read: the first track, the first one where all
three methods occur and the first one with the most rows without a centre.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import disc_tracks
import numpy as np
import pytest
from support import make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.core.errors import ConfigError
from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.backprojection import object_summary
from physverify.extraction.centres._base import (
    CentreMethod,
    ImageCentres,
    build_centre,
    centre_strategies,
)
from physverify.extraction.centres.contour_circle import (
    ContourCircleCentre,
    Params,
    angular_span_deg,
    blocked,
    causal_track_radii,
    silhouette_contour,
    track_radii,
)
from physverify.extraction.lift_params import LiftParams
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "centres" / "contour_circle_tracks.npz"
REFERENCE_SCRIPT_SHA256 = "48becfae6a7d71bee77285991576eef6c9fc396d3fb463984c34bc6e438ea500"
REFERENCE_METHODS = {"circle": "circle", "centroid": "centroid", "none": "absent"}
SHAPE = (48, 48)
TRUE_CENTRE = (21.3, 24.6)
FREE_ROWS = 3
CAUSAL_CASES = 200


def disc(centre: tuple[float, float], radius: float) -> np.ndarray:
    """Pixels whose centre (col + 0.5, row + 0.5) lies within ``radius`` of ``centre`` (u, v)."""
    rows, cols = np.indices(SHAPE)
    inside: np.ndarray = (cols + 0.5 - centre[0]) ** 2 + (rows + 0.5 - centre[1]) ** 2 <= radius**2
    return inside


def angles(centre: tuple[float, float]) -> np.ndarray:
    rows, cols = np.indices(SHAPE)
    found: np.ndarray = np.degrees(np.arctan2(rows + 0.5 - centre[1], cols + 0.5 - centre[0]))
    return found


def track(own: np.ndarray, other: np.ndarray) -> np.ndarray:
    """(T, 2, h, w): ``FREE_ROWS`` rows of the whole disc alone, then ``own`` beside ``other``."""
    free = np.stack([disc(TRUE_CENTRE, 6.0), np.zeros(SHAPE, dtype=bool)])
    cut = np.stack([own, other])
    return np.stack([free] * FREE_ROWS + [cut])


def centres_of(masks: np.ndarray, kinds: tuple[PrimitiveKind, ...]) -> ImageCentres:
    found = ContourCircleCentre(Params()).image_centres(masks, kinds)
    assert found is not None
    return found


SPHERES = (PrimitiveKind.SPHERE, PrimitiveKind.SPHERE)


def half_cut_errors() -> tuple[list[float], list[float]]:
    """Fit and centroid errors (px) of radius-6 discs whose right half another mask covers, over
    sub-pixel centres."""
    fits, centroids = [], []
    for q in np.linspace(0.0, 1.0, 11):
        centre = (TRUE_CENTRE[0] + q, TRUE_CENTRE[1] + q / 3)
        side = np.abs(angles(centre)) < 90.0
        other = disc(centre, 9.0) & side
        own = disc(centre, 6.0) & ~other
        free = np.stack([disc(centre, 6.0), np.zeros(SHAPE, dtype=bool)])
        masks = np.stack([free] * FREE_ROWS + [np.stack([own, other])])
        found = centres_of(masks, SPHERES)
        assert found.method[-1, 0] == CentreMethod.CIRCLE
        fits.append(math.dist(found.uv[-1, 0], centre))
        centroids.append(math.dist(np.argwhere(own)[:, ::-1].mean(axis=0) + 0.5, centre))
    return fits, centroids


def test_a_disc_cut_over_half_its_edge_is_fitted_far_better_than_its_centroid() -> None:
    fits, centroids = half_cut_errors()
    assert max(fits) < 0.5
    assert float(np.median(fits)) < 0.5 * float(np.median(centroids))


@pytest.mark.xfail(
    strict=True,
    reason="a binary radius-6 disc carries about 0.1 px of digitisation error, even uncut",
)
def test_a_disc_cut_over_half_its_edge_is_fitted_within_five_hundredths_of_a_pixel() -> None:
    fits, _ = half_cut_errors()
    assert max(fits) < 0.05


def test_an_arc_under_ninety_degrees_falls_back_to_the_centroid() -> None:
    radius = 12.0
    own = disc(TRUE_CENTRE, radius)
    ring = disc(TRUE_CENTRE, radius + 3.0) & ~own
    other = ring & (np.abs(angles(TRUE_CENTRE)) > 30.0)
    masks = np.stack([np.stack([own, np.zeros(SHAPE, dtype=bool)])] * FREE_ROWS)
    masks = np.concatenate([masks, np.stack([own, other])[None]])
    contour = silhouette_contour(own, blocked(other, Params().exclusion_px))
    centroid = np.argwhere(own)[:, ::-1].mean(axis=0) + 0.5
    assert len(contour) >= Params().contour_min_px
    assert angular_span_deg(contour, centroid) < Params().min_arc_deg
    found = centres_of(masks, SPHERES)
    assert found.method[-1, 0] == CentreMethod.CENTROID
    np.testing.assert_allclose(found.uv[-1, 0], centroid, rtol=0, atol=1e-12)
    assert found.method[0, 0] == CentreMethod.CIRCLE


def test_the_strategy_is_registered_and_its_params_are_validated() -> None:
    assert isinstance(
        BackprojectLift(LiftParams(centre="contour_circle"), centre_strategies()).centre,
        ContourCircleCentre,
    )
    with pytest.raises(ConfigError, match=r"contour_circle.*centre_params"):
        build_centre("contour_circle", {"min_arc_deg": -1.0}, centre_strategies())


def test_two_pixels_are_absent() -> None:
    own = np.zeros(SHAPE, dtype=bool)
    own[10, 10:12] = True
    found = centres_of(track(own, np.zeros(SHAPE, dtype=bool)), SPHERES)
    assert found.method[-1, 0] == CentreMethod.ABSENT
    assert bool(np.isnan(found.uv[-1, 0]).all())
    assert found.method[-1, 1] == CentreMethod.ABSENT


def test_a_box_keeps_the_surface_median_summary() -> None:
    whole = disc(TRUE_CENTRE, 6.0)
    masks = np.stack([np.stack([whole, disc((8.0, 8.0), 5.0)])] * 2)
    kinds = [PrimitiveKind.BOX, PrimitiveKind.SPHERE]
    found = centres_of(masks, tuple(kinds))
    assert bool((found.method[:, 0] == CentreMethod.SURFACE_MEDIAN).all())
    assert bool(np.isnan(found.uv[:, 0]).all())
    assert bool((found.method[:, 1] == CentreMethod.CIRCLE).all())
    z, valid = np.full((2, *SHAPE), 3.0), np.ones((2, *SHAPE), dtype=bool)
    k = np.array([[50.0, 0.0, 24.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
    lift = BackprojectLift(LiftParams(centre="contour_circle"), centre_strategies())
    seam = cast(MaskTrack, SimpleNamespace(masks=masks))
    rows = lift._summaries(seam, z, valid, k, None, kinds, found)
    plain = object_summary(whole, z[0], valid[0], k, lift._options, None, PrimitiveKind.BOX)
    assert rows[0][0] is not None and plain is not None
    assert rows[0][0].centre.tobytes() == plain.centre.tobytes()
    sphere = rows[0][1]
    assert sphere is not None
    u = sphere.centre[0] / sphere.centre[2] * k[0, 0] + k[0, 2]
    assert u == pytest.approx(found.uv[0, 1, 0], abs=1e-9)


def golden_tracks() -> list[tuple[str, dict[str, np.ndarray]]]:
    data = np.load(GOLDEN)
    assert str(data["contour_sha256"]) == REFERENCE_SCRIPT_SHA256
    keys = sorted({name.split("_", 1)[0] for name in data.files if name[0].isdigit()})
    return [(str(data[f"{k}_clip"]), {f: data[f"{k}_{f}"] for f in FIELDS}) for k in keys]


FIELDS = ("shape", "masks", "xy", "radius_px", "method")


@pytest.mark.parametrize(("clip", "track_data"), golden_tracks())
def test_the_port_reproduces_the_reference_centres_on_calibration_tracks(
    clip: str, track_data: dict[str, np.ndarray]
) -> None:
    shape = tuple(int(n) for n in track_data["shape"])
    count = math.prod(shape)
    masks = np.unpackbits(track_data["masks"], count=count).astype(bool).reshape(shape)
    found = centres_of(masks, (PrimitiveKind.SPHERE,) * shape[1])
    expected = np.vectorize(REFERENCE_METHODS.__getitem__)(track_data["method"])
    np.testing.assert_array_equal(found.method, expected, err_msg=clip)
    np.testing.assert_allclose(found.uv, track_data["xy"], rtol=0, atol=1e-9, err_msg=clip)
    np.testing.assert_allclose(
        found.radius_px, track_data["radius_px"], rtol=0, atol=1e-9, err_msg=clip
    )


def test_the_lift_records_the_method_and_moves_only_the_sphere(
    fixture_clip: SyntheticClip,
) -> None:
    s = make_context(fixture_clip).sampling
    gt = fixture_clip.ground_truth
    upstream = {Stage.TRACK: gt.masks(s), Stage.DEPTH: gt.depth(s), Stage.CAMERA: gt.camera(s)}
    plan = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: "backproject"}
    ctx = make_context(fixture_clip, upstream=upstream, plan=plan)
    default = run_provider(BackprojectLift, ctx)
    contour = run_provider(BackprojectLift, ctx, centre="contour_circle")
    assert isinstance(default, ObjectStateTable) and isinstance(contour, ObjectStateTable)
    assert contour.centre_method is not None
    sphere = contour.primitive == PrimitiveKind.SPHERE.value
    assert bool(sphere.any()) and bool((~sphere).any())
    kept = contour.centre_method[:, ~sphere]
    assert bool((kept == CentreMethod.SURFACE_MEDIAN).all())
    np.testing.assert_array_equal(contour.position[:, ~sphere], default.position[:, ~sphere])
    absent = contour.centre_method == CentreMethod.ABSENT
    assert not bool(contour.visible[absent].any())
    placed = np.isin(contour.centre_method[:, sphere], [CentreMethod.CIRCLE, CentreMethod.CENTROID])
    assert bool(placed.any())
    assert bool((contour.visible[:, sphere] == placed).all())
    assert default.causal_centre is None and contour.causal_centre is not None
    assert bool(np.isnan(contour.causal_centre[:, ~sphere]).all())
    seen = contour.causal_centre[:, sphere]
    assert bool((np.isfinite(seen).all(axis=-1) == placed).all())
    image = contour.position[:, sphere, :2] / contour.position[:, sphere, 2:]
    np.testing.assert_allclose(seen[placed], image[placed], rtol=0, atol=0.02)


def test_the_causal_radius_reads_the_window_before_the_row_then_every_earlier_row() -> None:
    radii = [5.0, 6.0, 7.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    masks = np.stack([np.stack([disc(TRUE_CENTRE, r)]) for r in radii])
    masks[0, 0] = masks[3, 0] = False
    free = [math.sqrt(float(m.sum()) / math.pi) if m.any() else math.nan for m in masks[:, 0]]
    found = causal_track_radii(masks, 0, Params(radius_window_rows=2))
    assert math.isnan(found[0])
    assert found[2] == pytest.approx(float(np.median(free[1:3])))
    assert found[3] == pytest.approx(float(np.median(free[1:3])))
    assert found[9] == pytest.approx(free[9])


def causal_and_lookahead_unmoved(seed: int) -> tuple[bool, bool]:
    """Whether rewriting the rows after a random row leaves rows up to it unchanged in the
    causal centres and radii, and in the track's centres and radii."""
    rng = np.random.default_rng(seed)
    masks = disc_tracks.approach(rng)
    row = int(rng.integers(0, masks.shape[0] - 1))
    late = disc_tracks.rewritten_after(masks, row, rng)
    kinds = (PrimitiveKind.SPHERE,) * masks.shape[1]
    before, after = centres_of(masks, kinds), centres_of(late, kinds)
    assert before.causal_uv is not None and after.causal_uv is not None
    head = slice(0, row + 1)

    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.array_equal(a[head], b[head], equal_nan=True))

    objects = range(masks.shape[1])
    causal = same(before.causal_uv, after.causal_uv) and all(
        same(causal_track_radii(masks, n, Params()), causal_track_radii(late, n, Params()))
        for n in objects
    )
    lookahead = same(before.uv, after.uv) and all(
        same(track_radii(masks, n, Params()), track_radii(late, n, Params())) for n in objects
    )
    return causal, lookahead


@pytest.mark.parametrize("seed", range(CAUSAL_CASES))
def test_no_row_after_a_row_moves_its_causal_centre(seed: int) -> None:
    assert causal_and_lookahead_unmoved(seed)[0]


def test_the_track_radius_looks_ahead_where_the_causal_one_does_not() -> None:
    moved = sum(not causal_and_lookahead_unmoved(seed)[1] for seed in range(20))
    assert moved > 0


def test_the_causal_centre_is_the_track_centre_when_every_row_is_free_at_one_radius() -> None:
    masks = np.stack([np.stack([disc(TRUE_CENTRE, 6.0)])] * 4)
    found = centres_of(masks, (PrimitiveKind.SPHERE,))
    assert found.causal_uv is not None
    np.testing.assert_array_equal(found.causal_uv, found.uv)

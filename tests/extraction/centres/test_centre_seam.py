"""The lift's centre strategy seam: the registry, the config errors and the default path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from support import make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.app.plugins import PLUGIN_KINDS
from physverify.core.cache_keys import params_hash
from physverify.core.errors import ConfigError
from physverify.extraction._base import GROUND_TRUTH, build_provider
from physverify.extraction.backprojection import ObjectSummary, SummaryOptions, object_summary
from physverify.extraction.centres._base import (
    CENTRE_STRATEGIES,
    CentreMethod,
    ImageCentres,
    build_centre,
    centre_strategies,
)
from physverify.extraction.centres.surface_median import SurfaceMedianCentre
from physverify.extraction.lift_params import (
    GroundAnchoredLiftParams,
    LiftParams,
    SizeConstancyLiftParams,
    UnnestedLiftParams,
    summary_options,
)
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

K = np.array([[60.0, 0.0, 15.5], [0.0, 58.0, 12.0], [0.0, 0.0, 1.0]])
SHAPE = (24, 32)
RANDOM_CASES = 200
DEFAULT_PARAMS_HASHES = {
    LiftParams: "361c8bf64a9766794127aecec6c2c31f03438774e4f73f0fd93f9243d5d0de45",
    UnnestedLiftParams: "1d3dd2014fe5a0dfc47b8551a637afbb0df93425e96018f40e04dff3889bfa9f",
    GroundAnchoredLiftParams: "8853670addd52e45d5ce1c2e931f5290e0b4f4647ea7c44cc9d8ef87d2d1ccb7",
    SizeConstancyLiftParams: "7a870cf78e206b3159ac87c97f92ca7ba78fbcec1215475966b4f9c3bf71e2e0",
}
"""The params hashes before the centre strategies existed: the measured LIFT error-profile
entries carry them, so the default strategy must keep them."""


def test_the_registry_lists_both_strategies_and_the_app_reads_the_same_one() -> None:
    assert sorted(centre_strategies()) == ["contour_circle", "surface_median"]
    assert dict(PLUGIN_KINDS["centre_strategies"]()) == dict(centre_strategies())


@pytest.mark.parametrize("model", DEFAULT_PARAMS_HASHES)
def test_the_default_centre_keeps_the_params_hash(model: type[LiftParams]) -> None:
    explicit = model(centre="surface_median", centre_params={})
    for params in (model(), explicit):
        assert params_hash(params.model_dump(mode="json")) == DEFAULT_PARAMS_HASHES[model]


@pytest.mark.parametrize(
    "fields", [{"centre": "contour_circle"}, {"centre_params": {"area_min_px": 4}}]
)
def test_a_centre_off_its_default_moves_the_params_hash(fields: dict[str, object]) -> None:
    params = LiftParams.model_validate(fields)
    assert params_hash(params.model_dump(mode="json")) != DEFAULT_PARAMS_HASHES[LiftParams]
    assert LiftParams.model_validate(params.model_dump(mode="json")) == params


def test_an_unknown_centre_is_a_config_error_naming_the_known_ones() -> None:
    known = "known: contour_circle, surface_median"
    with pytest.raises(ConfigError, match=rf"'nope' is unknown; {known}"):
        BackprojectLift(LiftParams(centre="nope"), centre_strategies())


def test_the_lift_is_built_with_the_centre_registry_it_is_given() -> None:
    class MarkerCentre(SurfaceMedianCentre):
        name = "marker"

    registries = {CENTRE_STRATEGIES.name: {"marker": MarkerCentre}}
    lift = build_provider(BackprojectLift, LiftParams(centre="marker"), registries.__getitem__)
    assert isinstance(cast(BackprojectLift, lift).centre, MarkerCentre)


def test_invalid_centre_params_are_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"surface_median.*centre_params"):
        build_centre("surface_median", {"unknown": 1}, centre_strategies())


def test_the_default_lift_resolves_the_surface_median_and_records_no_method(
    fixture_clip: SyntheticClip,
) -> None:
    assert isinstance(
        BackprojectLift(LiftParams(), centre_strategies()).centre, SurfaceMedianCentre
    )
    s = make_context(fixture_clip).sampling
    gt = fixture_clip.ground_truth
    upstream = {Stage.TRACK: gt.masks(s), Stage.DEPTH: gt.depth(s), Stage.CAMERA: gt.camera(s)}
    plan = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: "backproject"}
    ctx = make_context(fixture_clip, upstream=upstream, plan=plan)
    table = run_provider(BackprojectLift, ctx)
    assert isinstance(table, ObjectStateTable)
    assert table.centre_method is None


def reference_centre(mask: np.ndarray, z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The surface median written out: X = z K^-1 [u, v, 1] over the eroded valid pixels."""
    rows, cols = np.nonzero(binary_surface(mask, valid))
    depth = z[rows, cols]
    x = (cols + 0.5 - K[0, 2]) / K[0, 0] * depth
    y = (rows + 0.5 - K[1, 2]) / K[1, 1] * depth
    found: np.ndarray = np.median(np.column_stack([x, y, depth]), axis=0)
    return found


def random_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, PrimitiveKind]:
    rng = np.random.default_rng(seed)
    noise = rng.random(SHAPE)
    centre = rng.uniform((0, 0), SHAPE)
    rows, cols = np.indices(SHAPE)
    radius = rng.uniform(1.0, 9.0)
    disc = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius**2
    mask = disc & (noise > rng.uniform(0.0, 0.3))
    z = rng.uniform(0.5, 5.0, SHAPE)
    valid = rng.random(SHAPE) > 0.1
    kind = (PrimitiveKind.SPHERE, PrimitiveKind.BOX, PrimitiveKind.UNKNOWN)[seed % 3]
    return mask, z, valid, kind


def same(a: ObjectSummary | None, b: ObjectSummary | None) -> bool:
    if a is None or b is None:
        return a is b
    return (
        a.centre.tobytes() == b.centre.tobytes()
        and a.extents.tobytes() == b.extents.tobytes()
        and (a.r_eq_px, a.n_pixels, a.view) == (b.r_eq_px, b.n_pixels, b.view)
    )


@pytest.mark.parametrize("seed", range(RANDOM_CASES))
def test_the_surface_median_path_is_bit_identical_on_random_masks(seed: int) -> None:
    mask, z, valid, kind = random_case(seed)
    options = summary_options(LiftParams(min_area_px=1))
    summary = object_summary(mask, z, valid, K, options, None, kind)
    if summary is not None:
        assert summary.centre.tobytes() == reference_centre(mask, z, valid).tobytes()
    lifted = BackprojectLift(LiftParams(min_area_px=1), centre_strategies())
    track = cast(MaskTrack, SimpleNamespace(masks=mask[None, None]))
    zs, valids = z[None], valid[None]
    plain = lifted._summaries(track, zs, valids, K, None, [kind])
    method = np.full((1, 1), CentreMethod.SURFACE_MEDIAN.value)
    tagged = ImageCentres(uv=np.full((1, 1, 2), np.nan), method=method, radius_px=np.ones((1, 1)))
    seamed = lifted._summaries(track, zs, valids, K, None, [kind], tagged)
    assert same(plain[0][0], summary)
    assert same(seamed[0][0], summary)


def test_a_given_centre_sits_on_its_ray_at_the_surface_median_depth() -> None:
    mask, z, valid, _ = random_case(7)
    options = SummaryOptions(1, (0.0, 100.0), 9, 3.0, 0.2)
    default = object_summary(mask, z, valid, K, options)
    placed = object_summary(mask, z, valid, K, options, centre_px=(14.25, 10.75))
    assert default is not None and placed is not None
    z_s = float(np.median(z[binary_surface(mask, valid)]))
    expected = [(14.25 - K[0, 2]) / K[0, 0] * z_s, (10.75 - K[1, 2]) / K[1, 1] * z_s, z_s]
    np.testing.assert_allclose(placed.centre, expected, rtol=0, atol=1e-12)
    assert (placed.r_eq_px, placed.n_pixels) == (default.r_eq_px, default.n_pixels)


def test_a_given_centre_needs_one_surface_pixel_not_min_area() -> None:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[5:7, 5:7] = True
    z, valid = np.ones(SHAPE), np.ones(SHAPE, dtype=bool)
    options = SummaryOptions(1, (0.0, 100.0), 9, 3.0, 0.2)
    assert object_summary(mask, z, valid, K, options) is None
    assert object_summary(mask, z, valid, K, options, centre_px=(6.0, 6.0)) is not None


def binary_surface(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pixels: np.ndarray = binary_erosion(mask) & valid
    return pixels if pixels.any() else mask & valid

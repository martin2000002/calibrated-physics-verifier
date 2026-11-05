import numpy as np

from physverify.extraction.contact_crops import (
    ZONE_FEATURES,
    CropSettings,
    contact_zone,
    crop_centre,
    pair_crop,
    square_crop,
    zone_features,
)


def _masks() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((40, 60), dtype=bool)
    b = np.zeros((40, 60), dtype=bool)
    a[10:30, 5:20] = True
    b[10:30, 22:40] = True
    return a, b


def test_zone_lies_between_touching_masks() -> None:
    a, b = _masks()
    zone = contact_zone(a, b, reach_px=2)
    cols = np.nonzero(zone)[1]
    assert zone.any() and cols.min() >= 18 and cols.max() <= 23
    row, col = crop_centre(a, b, zone)
    assert 19.0 <= col <= 22.0 and 15.0 <= row <= 25.0


def test_far_masks_have_no_zone_and_centre_between_centroids() -> None:
    a, b = _masks()
    zone = contact_zone(a, b, reach_px=0)
    assert not zone.any()
    _, col = crop_centre(a, b, zone)
    assert abs(col - (12.0 + 30.5) / 2) < 1e-9


def test_square_crop_has_fixed_size_at_the_border() -> None:
    image = np.arange(40 * 60 * 3, dtype=float).reshape(40, 60, 3)
    assert square_crop(image, (0.0, 0.0), 64).shape == (64, 64, 3)
    frame = np.zeros((80, 120, 3))
    a, b = _masks()
    assert pair_crop(frame, a, b, CropSettings()).shape == (64, 64, 3)


def test_zone_features_read_depth_on_core_and_ring() -> None:
    a, b = _masks()
    depth = np.full(a.shape, 2.0)
    depth[:, 21:] = 3.0
    values = zone_features(a, b, depth, np.ones(a.shape, dtype=bool), CropSettings(reach_px=2))
    named = dict(zip(ZONE_FEATURES, values, strict=True))
    assert named["overlap_px_share"] == 0.0
    assert named["zone_px_share"] > 0.0
    assert named["depth_gap_ab"] < 0.0
    invalid = zone_features(a, b, depth, np.zeros(a.shape, dtype=bool), CropSettings())
    assert np.isnan(invalid[ZONE_FEATURES.index("depth_core")])

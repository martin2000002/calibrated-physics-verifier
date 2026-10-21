"""The diagnosis metric registry: one file per metric, keyed ``<stage>/<name>``."""

from __future__ import annotations

from physverify.cli.context import PLUGIN_KINDS
from physverify.diagnosis.cases import SampledMetric
from physverify.diagnosis.error_profile_writer import PROFILE_QUANTITIES
from physverify.diagnosis.settings import MetricSettings

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
EXPECTED = {
    "detect/box",
    "detect/label",
    "track/mask",
    "depth/absrel",
    "depth/object_rel",
    "depth/object_parts",
    "camera/fov",
    "camera/gravity",
    "lift/position",
    "lift/gravity_angle",
    "scale/relative",
    "kinematics/derivatives",
    "contacts/events",
    "contacts/impacts",
}


def test_every_level_one_metric_is_registered() -> None:
    assert set(PLUGIN_KINDS["metrics"]()) == EXPECTED


def test_metrics_take_their_options_and_sample() -> None:
    for key, cls in PLUGIN_KINDS["metrics"]().items():
        metric = cls(OPTIONS)
        assert isinstance(metric, SampledMetric)
        assert f"{metric.stage.value}/{metric.name}" == key


def test_profile_quantities_come_from_registered_metrics() -> None:
    assert set(PROFILE_QUANTITIES) <= EXPECTED

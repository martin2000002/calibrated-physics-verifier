"""contacts/impacts matches impacts alone, by time, per pair."""

from __future__ import annotations

import sys
from pathlib import Path

from synthetic_clip import manifest

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_metrics_contacts import OPTIONS, record, table  # noqa: E402

from physverify.diagnosis.error_profile_writer import PROFILE_QUANTITIES  # noqa: E402
from physverify.diagnosis.metrics.contacts_impacts import ContactImpactMetric  # noqa: E402
from physverify.schema.constants import ENV_ID  # noqa: E402
from physverify.schema.contacts import ContactKind, ContactRecord  # noqa: E402

METRIC = ContactImpactMetric(OPTIONS)
CLIP = manifest()


def scores(pred: list[ContactRecord], truth: list[ContactRecord]) -> dict[str, float]:
    return METRIC.compute(table(*pred), table(*truth), CLIP)


def test_t12_one_native_frame_matches_and_a_hundredth_more_does_not() -> None:
    truth = [record(span=(10.0, 10.0))]
    assert scores([record(span=(11.0, 11.0))], truth)["recall"] == 1.0
    assert scores([record(span=(11.01, 11.01))], truth)["recall"] == 0.0
    assert scores([record(span=(8.0, 9.0))], truth)["recall"] == 1.0


def test_t12_rests_are_ignored_and_pairs_must_agree() -> None:
    rest = record(span=(0.0, 20.0), kind=ContactKind.RESTING)
    truth = [record(span=(10.0, 10.0)), rest]
    out = scores([rest, record(a="ball", b="other", span=(10.0, 10.0))], truth)
    assert (out["recall.num"], out["recall.den"], out["precision.den"]) == (0.0, 1.0, 1.0)
    swapped = record(a="box", b="ball", span=(10.2, 10.2))
    out = scores([swapped], truth)
    assert out["recall"] == out["precision"] == 1.0
    assert abs(out["t_err_frames.mean"] - 0.2) < 1e-12


def test_t12_environment_and_pairs_are_counted_apart() -> None:
    truth = [record(b=ENV_ID, span=(5.0, 5.0)), record(span=(12.0, 12.0))]
    pred = [record(b=ENV_ID, span=(5.4, 5.4)), record(b=ENV_ID, span=(20.0, 20.0))]
    out = scores(pred, truth)
    assert (out["recall_env"], out["recall_pair"]) == (1.0, 0.0)
    assert (out["precision_env.num"], out["precision_env.den"]) == (1.0, 2.0)
    assert out["precision_pair.den"] == 0.0
    assert (out["recall.num"], out["recall.den"], out["precision.den"]) == (1.0, 2.0, 2.0)


def test_t12_one_to_one() -> None:
    truth = [record(span=(10.0, 10.0))]
    out = scores([record(span=(10.0, 10.0)), record(span=(10.5, 10.5))], truth)
    assert (out["recall"], out["precision"]) == (1.0, 0.5)


def test_t12_the_metric_is_not_in_the_profile() -> None:
    assert "contacts/impacts" not in PROFILE_QUANTITIES

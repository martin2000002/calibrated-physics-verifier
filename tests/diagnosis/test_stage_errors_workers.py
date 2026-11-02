"""Level 1 measured by several worker processes equals the serial measurement."""

from __future__ import annotations

from collections.abc import Callable

from synthetic_clip import DATASET

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.error_profile_writer import StageSamples
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import measure, stage_pairs

Fingerprint = list[tuple[str, str, str, str, tuple[int, ...], bytes | tuple[float, float]]]


def fingerprint(measured: StageSamples) -> Fingerprint:
    """Every sample array's dtype, shape and bytes, and every ratio, in report order."""
    rows: Fingerprint = []
    for pair in measured:
        for clip_id, outcome in pair.outcomes.items():
            for name, values in outcome.samples.items():
                rows.append(
                    (
                        pair.metric + pair.pred_variant,
                        clip_id,
                        name,
                        values.dtype.str,
                        values.shape,
                        values.tobytes(),
                    )
                )
            for name, ratio in outcome.ratios.items():
                rows.append((pair.metric + pair.pred_variant, clip_id, name, "ratio", (), ratio))
    return rows


def test_three_workers_measure_the_same_bytes_as_one(
    make_settings: Callable[..., Settings], make_chain: Callable[..., ChainRunner]
) -> None:
    settings = make_settings(f"diagnosis.dataset={DATASET}")
    options = diagnosis_settings(settings).metrics
    metrics = {key: cls(options) for key, cls in PLUGIN_KINDS["metrics"]().items()}
    pairs = stage_pairs(diagnosis_settings(settings))
    runner = make_chain(settings)
    clips = list(runner.dataset.clips())
    assert len(clips) > 1
    parallel = measure(runner, clips, metrics, pairs, options, workers=3)
    serial = measure(runner, clips, metrics, pairs, options, workers=1)
    ids = [c.clip_id for c in clips]
    assert all(list(m.outcomes) == [i for i in ids if i in m.outcomes] for m in parallel)
    assert any(list(m.outcomes) == ids for m in parallel)
    assert fingerprint(parallel) == fingerprint(serial)
    assert len(fingerprint(serial)) > 0


def test_the_worker_count_stays_out_of_the_config_hash(
    make_settings: Callable[..., Settings],
) -> None:
    one, many = make_settings("runtime.workers=1"), make_settings("runtime.workers=48")
    assert many.runtime.workers == 48
    assert one.config_hash() == many.config_hash()
    assert "workers" not in one.runtime.model_dump()

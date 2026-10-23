"""``diagnose stages`` on calib clips only (WP-O C6): the campaign's ladder never touches eval."""

from __future__ import annotations

from collections.abc import Callable

from synthetic_clip import DATASET

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.diagnosis.context import DiagnosisContext, sampled_metrics
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import run_stage_errors
from physverify.extraction.error_profile import load_error_profile
from physverify.schema.clip import Split


def test_calib_clips_only_write_the_profile_with_an_empty_report_pass(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    settings = make_settings(f"diagnosis.dataset={DATASET}", "diagnosis.min_samples=3")
    metrics = sampled_metrics(PLUGIN_KINDS["metrics"](), diagnosis_settings(settings).metrics)
    ctx = make_diagnosis_context(settings, metrics=metrics)
    calib = [c for c in ctx.dataset().clips() if c.split is Split.CALIB]
    assert calib and len(calib) < len(ctx.dataset().clips())
    report = run_stage_errors(ctx, calib, write_profile=True)
    assert not report.failures and report.profile_written
    assert any(row.split is Split.CALIB and row.n_clips for row in report.rows)
    assert all(row.n_clips == 0 for row in report.rows if row.split is Split.EVAL)
    written = load_error_profile(settings.paths.error_profile)
    assert written is not None and written.entries

"""Level 1 with an input route: the route's substituted stages get pairs and profile entries of
their own, measured against the next rung; the main plan's pairs and entries stay as they were."""

from __future__ import annotations

from collections.abc import Callable

from synthetic_clip import DATASET

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.measurement_cases import SampledMetric
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import run_stage_errors, stage_pairs
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.verification.check_routes import check_routing

AREA_LIFT = "backproject_area_depth"
BASE = (f"diagnosis.dataset={DATASET}", "diagnosis.min_samples=3")
ROUTE = (f"verification.routes.area_depth.lift={AREA_LIFT}",)


def pairs_of(settings: Settings) -> list[tuple[Stage, str, str]]:
    pairs = stage_pairs(diagnosis_settings(settings), check_routing(settings).routes)
    return [(p.stage, p.pred.name, p.truth.name) for p in pairs]


def test_a_route_adds_one_pair_per_substituted_stage(
    make_settings: Callable[..., Settings],
) -> None:
    plain, routed = pairs_of(make_settings(*BASE)), pairs_of(make_settings(*BASE, *ROUTE))
    k = STAGE_ORDER.index(Stage.LIFT)
    assert routed == [*plain, (Stage.LIFT, f"V{k}/route=area_depth", f"V{k + 1}")]


def test_the_route_lift_gets_its_own_profile_entries(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    def profile(*overrides: str) -> dict[tuple[Stage, str, str], tuple[float, ...]]:
        settings = make_settings(*BASE, *overrides)
        options = diagnosis_settings(settings).metrics
        metrics: dict[str, SampledMetric] = {
            key: cls(options) for key, cls in PLUGIN_KINDS["metrics"]().items()
        }
        ctx = make_diagnosis_context(settings, metrics=metrics)
        report = run_stage_errors(ctx, ctx.dataset().clips(), write_profile=False)
        return {(r.stage, r.provider, r.quantity): (r.bias, r.sigma, r.n) for r in report.profile}

    plain, routed = profile(), profile(*ROUTE)
    own = {key: value for key, value in routed.items() if key[1] == AREA_LIFT}
    assert own and {stage for stage, _, _ in own} == {Stage.LIFT}
    assert {key[2] for key in own} == {
        key[2] for key in plain if key[:2] == (Stage.LIFT, "backproject")
    }
    assert {key: value for key, value in routed.items() if key[1] != AREA_LIFT} == plain

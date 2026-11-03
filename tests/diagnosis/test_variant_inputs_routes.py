"""End-to-end inputs follow the input routes: a routed check reads its route's plan, the rest
read the variant's plan, as in ``verify``."""

from __future__ import annotations

from collections.abc import Callable

from synthetic_clip import DATASET

from physverify.config.models import Settings
from physverify.diagnosis.chain_variants import oracle_variant, perceived_variant
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.end_to_end import variant_outputs
from physverify.diagnosis.verification_adapter import variant_inputs
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.stages import Stage

AREA_LIFT = "backproject_area_depth"
BASE = (f"diagnosis.dataset={DATASET}", "diagnosis.min_samples=3")
ROUTED_R3A = (
    f"verification.routes.area_depth.lift={AREA_LIFT}",
    "verification.check_routes.R3a=area_depth",
)


def lifts(ctx: DiagnosisContext, perceived: bool) -> tuple[str, str]:
    """The lift R3a and R5a read under the perceived or the oracle variant."""
    variant = perceived_variant() if perceived else oracle_variant()
    chain = ctx.chain(ctx.dataset())
    clip = next(c for c in ctx.dataset().clips() if c.split is Split.EVAL)
    level = InputLevel.PERCEIVED if perceived else InputLevel.ORACLE
    item = variant_inputs(chain, clip, variant_outputs(chain, clip, variant), level)
    found = []
    for check_id in ("R3a", "R5a"):
        inputs = item.for_check(check_id).inputs
        assert inputs is not None
        found.append(inputs.provenance[Stage.LIFT.value])
    return found[0], found[1]


def test_a_routed_check_reads_its_route_lift_in_the_perceived_variant(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    routed = make_diagnosis_context(make_settings(*BASE, *ROUTED_R3A))
    plain = make_diagnosis_context(make_settings(*BASE))
    assert lifts(routed, perceived=True) == (AREA_LIFT, "backproject")
    assert lifts(plain, perceived=True) == ("backproject", "backproject")


def test_the_oracle_variant_is_not_routed(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    ctx = make_diagnosis_context(make_settings(*BASE, *ROUTED_R3A))
    r3a, r5a = lifts(ctx, perceived=False)
    assert r3a == r5a != AREA_LIFT

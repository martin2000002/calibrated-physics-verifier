import math
from dataclasses import replace

import pytest
from pydantic import BaseModel
from settings_factory import SettingsFactory
from synthetic_clip import SyntheticFixtureDataset
from wiring import resolve_plan

from physverify.app.plugins import PLUGIN_KINDS
from physverify.extraction._base import SupportGapGate
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.providers._velocity_tests import two_sided_quantile
from physverify.schema.error_profile import SUPPORT_GAP_REL, ErrorEntry, ErrorProfile
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.verification.inputs import OracleSource, scene_gap_bound


def plan(make_settings: SettingsFactory, profile: str | None, gt: bool = False) -> ResolvedPlan:
    settings = make_settings(profile=profile)
    providers = PLUGIN_KINDS["providers"]()
    dataset = SyntheticFixtureDataset().name
    return resolve_plan(settings, providers, dataset=dataset, gt_stages=STAGE_ORDER if gt else ())


def gap_profile(lift: str) -> ErrorProfile:
    entry = ErrorEntry(
        stage=Stage.LIFT,
        provider=lift,
        provider_version="1",
        quantity=SUPPORT_GAP_REL,
        unit="",
        bias=0.03,
        sigma=0.04,
        n=3,
        residuals=(0.0, 0.03, 0.06),
    )
    return ErrorProfile(created_utc="t", run_id="r", commit="", entries=(entry,))


def test_the_bound_is_the_scene_gap_gate_over_the_lift_entry(
    make_settings: SettingsFactory,
) -> None:
    scene = plan(make_settings, "precise")
    alpha = next(s.params for s in scene.stages if s.stage is Stage.CONTACTS).alpha
    bound = scene_gap_bound(scene, gap_profile(scene.names[Stage.LIFT]))
    assert bound == pytest.approx(two_sided_quantile(alpha) * 0.05)
    assert math.isnan(scene_gap_bound(scene, None))


def test_ground_truth_is_exact_and_other_contacts_have_no_bound(
    make_settings: SettingsFactory,
) -> None:
    assert scene_gap_bound(plan(make_settings, None, gt=True), None) == 0.0
    proximity = plan(make_settings, None)
    assert math.isnan(scene_gap_bound(proximity, gap_profile(proximity.names[Stage.LIFT])))


def test_oracle_inputs_have_an_exact_gap() -> None:
    dataset = SyntheticFixtureDataset()
    (item,) = OracleSource(dataset).load(dataset.clips())
    assert item.inputs is not None and item.inputs.support_gap_bound == 0.0


class DeclaredGate(BaseModel):
    """Params of a CONTACTS provider the verifier does not know, declaring its gap gate."""

    def support_gap_z(self) -> float:
        return 3.0


def with_contacts_params(plan: ResolvedPlan, params: BaseModel) -> ResolvedPlan:
    stages = tuple(
        replace(s, params=params) if s.stage is Stage.CONTACTS else s for s in plan.stages
    )
    return replace(plan, stages=stages)


def test_the_bound_reads_the_gate_the_contacts_provider_declares(
    make_settings: SettingsFactory,
) -> None:
    scene = plan(make_settings, "precise")
    contacts = next(s.params for s in scene.stages if s.stage is Stage.CONTACTS)
    assert isinstance(contacts, SupportGapGate)
    declared = with_contacts_params(scene, DeclaredGate())
    bound = scene_gap_bound(declared, gap_profile(scene.names[Stage.LIFT]))
    assert bound == pytest.approx(3.0 * 0.05)

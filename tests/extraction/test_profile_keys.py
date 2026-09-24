"""AC20: LIFT's own error entry enters its key, and nothing outside ``profile_keys`` matters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
from fakes import FAKE
from support import (
    SettingsFactory,
    chain_keys,
    entry,
    make_context,
    plan_overrides,
    profile_of,
    run_pipeline,
    run_provider,
    truth_outputs,
    with_sigma,
)
from synthetic_clip import SyntheticClip, SyntheticFixtureDataset

from physverify.extraction._base import StageProvider
from physverify.extraction.cache import StageCache, profile_keys
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.io import stable_hash
from physverify.schema.stages import STAGE_ORDER, Stage, StageArtifactMeta

Providers = Mapping[str, type[StageProvider]]
FROM_LIFT = [Stage.LIFT, Stage.SCALE, Stage.KINEMATICS, Stage.CONTACTS]
ABOVE_LIFT = [Stage.DETECT, Stage.TRACK, Stage.DEPTH, Stage.CAMERA]
P0 = profile_of(
    entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.05),
    entry(Stage.DEPTH, "vda_small", "depth_rel", 0.1),
    entry(Stage.LIFT, "backproject", "centroid_rel", 0.1),
    entry(Stage.LIFT, "backproject", "size_rel", 0.08),
)


def own(quantity: str, profile: ErrorProfile = P0) -> ErrorProfile:
    return with_sigma(profile, (Stage.LIFT, "backproject", quantity), 0.3)


def keys_of(plan: ResolvedPlan, clip: SyntheticClip, profile: ErrorProfile) -> dict[Stage, str]:
    return chain_keys(plan, clip.manifest, gt_version="v1", profile=profile)


def changed(
    plan: ResolvedPlan, clip: SyntheticClip, a: ErrorProfile, b: ErrorProfile
) -> list[Stage]:
    before, after = keys_of(plan, clip, a), keys_of(plan, clip, b)
    return [stage for stage in STAGE_ORDER if before[stage] != after[stage]]


@pytest.mark.parametrize("quantity", ["centroid_rel", "size_rel"])
def test_the_own_lift_entry_changes_exactly_the_light_stages_from_lift(
    make_settings: SettingsFactory,
    real_providers: Providers,
    fixture_clip: SyntheticClip,
    quantity: str,
) -> None:
    fast = resolve_plan(make_settings(), real_providers, dataset="synthetic_fixture")
    assert changed(fast, fixture_clip, P0, own(quantity)) == FROM_LIFT
    lifted = resolve_plan(
        make_settings(), real_providers, dataset="synthetic_fixture", gt_stages=(Stage.LIFT,)
    )
    assert changed(lifted, fixture_clip, P0, own(quantity)) == []


def test_an_entry_of_another_lift_provider_changes_nothing(
    make_settings: SettingsFactory, real_providers: Providers, fixture_clip: SyntheticClip
) -> None:
    fast = resolve_plan(make_settings(), real_providers, dataset="synthetic_fixture")
    other = replace(P0, entries=(*P0.entries, entry(Stage.LIFT, "other", "centroid_rel", 0.9)))
    assert changed(fast, fixture_clip, P0, other) == []


def full_profile(plan: Mapping[Stage, str], provider: type[StageProvider]) -> ErrorProfile:
    """One entry per stage of ``plan`` (with ``provider`` in its own stage), plus a stranger."""
    names = {**plan, provider.stage: provider.name}
    entries = [entry(stage, name, "q", 0.1) for stage, name in names.items()]
    return profile_of(*P0.entries, *entries, entry(Stage.LIFT, "other", "centroid_rel", 0.2))


def outside_altered(profile: ErrorProfile, keys: tuple[tuple[Stage, str], ...]) -> ErrorProfile:
    entries = tuple(
        e if (e.stage, e.provider) in keys else replace(e, sigma=e.sigma * 7.0 + 1.0, bias=0.5)
        for e in profile.entries
    )
    return replace(profile, entries=entries)


def test_light_outputs_only_depend_on_their_profile_keys(
    make_settings: SettingsFactory, real_providers: Providers, fixture_clip: SyntheticClip
) -> None:
    fast = dict(resolve_plan(make_settings(), real_providers, dataset="synthetic_fixture").names)
    upstream = truth_outputs(fixture_clip)
    light = [p for p in real_providers.values() if not p.heavy and not p.is_ground_truth]
    assert {p.stage for p in light} >= {Stage.CAMERA, Stage.LIFT, Stage.SCALE, Stage.KINEMATICS}
    for provider in light:
        plan = {**fast, provider.stage: provider.name}
        profile = full_profile(fast, provider)
        altered = outside_altered(profile, profile_keys(provider, plan))
        outputs = [
            run_provider(
                provider, make_context(fixture_clip, upstream=upstream, plan=plan, profile=p)
            )
            for p in (profile, altered)
        ]
        hashes = [stable_hash(output.to_dict()) for output in outputs]
        assert hashes[0] == hashes[1], provider.name


def test_a_pipeline_run_with_a_new_own_entry_recomputes_from_lift(
    make_settings: SettingsFactory, providers: Providers, fixture_dataset: SyntheticFixtureDataset
) -> None:
    settings = make_settings(*plan_overrides(detect=FAKE, track=FAKE, depth=FAKE))
    p0 = profile_of(
        entry(Stage.TRACK, FAKE, "centroid_px_rel", 0.05),
        entry(Stage.DEPTH, FAKE, "depth_rel", 0.1),
        entry(Stage.LIFT, "backproject", "centroid_rel", 0.1),
    )
    first = run_pipeline(settings, providers, fixture_dataset, profile=p0)
    second = run_pipeline(settings, providers, fixture_dataset, profile=own("centroid_rel", p0))
    assert first.failures == second.failures == ()
    assert [event.stage for event in second.events if not event.cached] == FROM_LIFT
    clip_id = first.events[0].clip_id
    cache = StageCache(settings.paths.cache_dir)

    def lift_meta(report_keys: Mapping[tuple[str, Stage], str]) -> str:
        key = report_keys[(clip_id, Stage.LIFT)]
        path = cache.directory(fixture_dataset.name, clip_id, Stage.LIFT, "backproject", key)
        return StageArtifactMeta.load_json(path / "meta.json").error_profile_hash

    assert lift_meta(first.keys) != lift_meta(second.keys)
    assert all(first.keys[(clip_id, s)] == second.keys[(clip_id, s)] for s in ABOVE_LIFT)

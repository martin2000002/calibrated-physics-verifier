"""Resolve the active profile into one provider and validated parameters per stage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.settings import SECTION, ExtractionSettings
from physverify.schema.clip import SamplingSpec
from physverify.schema.errors import SchemaError
from physverify.schema.stages import STAGE_ORDER, Stage, check_upstream_order

SETTINGS_SECTIONS_ATTR = "settings_sections"
"""Optional provider ClassVar ``{param_field: (settings_section, model)}``: the plan fills each
field from that configuration section, so the section enters ``params`` and the cache key."""

PROVIDER_PARAMS_ATTR = "provider_params"
"""Optional provider ClassVar ``{param_field: (stage, provider_name)}``: the plan fills each field
with that provider's params as ``[extraction.providers.<stage>.<name>]`` configures them, so a
provider that runs another as a fallback runs it as configured and that configuration enters
``params`` and the cache key."""

PERCEIVED_PARAMS_ATTR = "perceived_params"
"""Optional ground-truth provider ClassVar ``{param_field: read(perceived_provider)}``: when the
truth stands in for the configured perceived provider, the plan fills each field from that
provider's class, so the truth matches what it replaces (e.g. metric depth for a metric model)."""

PerceivedParams = Mapping[str, Callable[[type[StageProvider]], object]]


@dataclass(frozen=True)
class StagePlan:
    stage: Stage
    provider: type[StageProvider]
    params: BaseModel

    @property
    def name(self) -> str:
        return self.provider.name


@dataclass(frozen=True)
class ResolvedPlan:
    profile: str
    stages: tuple[StagePlan, ...]
    sampling: SamplingSpec

    @property
    def names(self) -> Mapping[Stage, str]:
        return {step.stage: step.provider.name for step in self.stages}

    def ground_truth_stages(self) -> tuple[Stage, ...]:
        return tuple(step.stage for step in self.stages if step.provider.is_ground_truth)

    def step(self, stage: Stage) -> StagePlan:
        return next(step for step in self.stages if step.stage == stage)


def settings_sections(provider: type[StageProvider]) -> Mapping[str, tuple[str, type[BaseModel]]]:
    sections: Mapping[str, tuple[str, type[BaseModel]]] = getattr(
        provider, SETTINGS_SECTIONS_ATTR, {}
    )
    return sections


def provider_params(provider: type[StageProvider]) -> Mapping[str, tuple[Stage, str]]:
    fields: Mapping[str, tuple[Stage, str]] = getattr(provider, PROVIDER_PARAMS_ATTR, {})
    return fields


def perceived_params(provider: type[StageProvider]) -> PerceivedParams:
    reads: PerceivedParams = getattr(provider, PERCEIVED_PARAMS_ATTR, {})
    return reads


def resolve_plan(
    settings: Settings,
    providers: Mapping[str, type[StageProvider]],
    *,
    dataset: str,
    occlusion: Mapping[str, float],
    gt_stages: tuple[Stage, ...] = (),
    fps: float | None = None,
    max_side: int | None = None,
) -> ResolvedPlan:
    config = settings.section(SECTION, ExtractionSettings)
    steps = tuple(
        _stage_plan(stage, config, providers, settings, stage in gt_stages) for stage in STAGE_ORDER
    )
    sampling = SamplingSpec(
        target_fps=fps if fps is not None else config.sampling.target_fps.get(dataset),
        max_side=max_side if max_side is not None else config.sampling.max_side,
        start=config.sampling.start,
        stop=config.sampling.stop,
    )
    check_gap_threshold(steps, occlusion)
    return ResolvedPlan(profile=settings.profile, stages=steps, sampling=sampling)


GAP_THRESHOLD_FIELD = "gap_occluded_at"
CHECK_THRESHOLD_FIELD = "occluded_at"


def check_gap_threshold(steps: tuple[StagePlan, ...], occlusion: Mapping[str, float]) -> None:
    """LIFT's gap occlusion estimate and every check that reads occlusion must split "hidden"
    from "missing" at the same share in the effective configuration (``--set`` included);
    otherwise a perceived gap could count as missing for one and as explained for the other.
    ``occlusion`` maps each check that reads occlusion to its effective ``occluded_at``."""
    gap = next(
        (getattr(s.params, GAP_THRESHOLD_FIELD, None) for s in steps if s.stage is Stage.LIFT),
        None,
    )
    if gap is None:
        return
    disagree = {check_id: value for check_id, value in occlusion.items() if value != gap}
    if disagree:
        raise ConfigError(
            f"LIFT {GAP_THRESHOLD_FIELD} = {gap} but the checks use {CHECK_THRESHOLD_FIELD} "
            f"{disagree}; set both to the same value"
        )


def _stage_plan(
    stage: Stage,
    config: ExtractionSettings,
    providers: Mapping[str, type[StageProvider]],
    settings: Settings,
    ground_truth: bool,
) -> StagePlan:
    if stage not in config.plan:
        raise ConfigError(f"[extraction.plan] has no provider for stage {stage.value!r}")
    name = GROUND_TRUTH if ground_truth else config.plan[stage]
    key = f"{stage.value}/{name}"
    provider = _provider(providers, stage, name)
    try:
        check_upstream_order(stage, provider.requires)
    except SchemaError as exc:
        raise ConfigError(f"{key}: {exc}") from exc
    raw = dict(config.providers.get(stage, {}).get(name, {}))
    reads = perceived_params(provider)
    if ground_truth and reads and config.plan[stage] != GROUND_TRUTH:
        _mirror(reads, key, raw, _provider(providers, stage, config.plan[stage]))
    _fill_provider_params(provider, key, raw, config, providers, settings)
    return StagePlan(stage, provider, _params(provider, key, raw, settings))


def _fill_provider_params(
    provider: type[StageProvider],
    key: str,
    raw: dict[str, Any],
    config: ExtractionSettings,
    providers: Mapping[str, type[StageProvider]],
    settings: Settings,
) -> None:
    for field, (stage, name) in provider_params(provider).items():
        table = f"[extraction.providers.{stage.value}.{name}]"
        if field in raw:
            raise ConfigError(f"{key}: {field!r} comes from {table} and cannot be overridden")
        other = _provider(providers, stage, name)
        other_key = f"{stage.value}/{name}"
        other_raw = dict(config.providers.get(stage, {}).get(name, {}))
        _fill_provider_params(other, other_key, other_raw, config, providers, settings)
        raw[field] = _params(other, other_key, other_raw, settings)


def _provider(
    providers: Mapping[str, type[StageProvider]], stage: Stage, name: str
) -> type[StageProvider]:
    key = f"{stage.value}/{name}"
    if key not in providers:
        known = sorted(k for k in providers if k.startswith(f"{stage.value}/"))
        raise ConfigError(f"unknown provider {key!r}; available for this stage: {known}")
    return providers[key]


def _mirror(
    reads: PerceivedParams, key: str, raw: dict[str, Any], perceived: type[StageProvider]
) -> None:
    for field, read in reads.items():
        if field in raw:
            raise ConfigError(f"{key}: {field!r} follows the perceived provider it replaces")
        raw[field] = read(perceived)


def _params(
    provider: type[StageProvider], key: str, raw: dict[str, Any], settings: Settings
) -> BaseModel:
    for field, (section, model) in settings_sections(provider).items():
        if field in raw:
            raise ConfigError(f"{key}: {field!r} comes from [{section}] and cannot be overridden")
        if section not in settings.sections:
            raise ConfigError(f"{key} needs the configuration section [{section}]")
        raw[field] = settings.section(section, model)
    try:
        return provider.Params.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"[extraction.providers.{key.replace('/', '.')}] is invalid:\n{exc}"
        ) from exc

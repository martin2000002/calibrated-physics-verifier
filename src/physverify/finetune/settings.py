"""Settings of the preference fine-tuning cycle: the ``[finetune]`` section and the prompt set it
names in ``[prompt_sets]``.

Generator parameters are not here: they are read from ``[timing.generation]``, so the cost
measured by the timing harness is exactly the cost of these videos.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from physverify.config.models import Settings, validated
from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel
from physverify.generation.settings import SLUG_PATTERN

SECTION = "finetune"
PROMPT_SETS_SECTION = "prompt_sets"
LIBRARY_SECTION = "prompt_library"
LIBRARY_KEY = "library"
"""A ``[prompt_sets.<name>]`` table with this key renders its families from ``[prompt_library]``."""
SURFACE_TOKEN = "{surface}"
PLANE_TOKEN = "{plane}"
PromptSplit = Literal["train", "dev", "eval"]
SPLITS: tuple[PromptSplit, ...] = ("train", "dev", "eval")
TEMPLATE_ID_PATTERN = r"^[a-z0-9_]+$"
FIXED_ID_PATTERN = r"^[a-z0-9_-]+$"
EXTRA_KEY = "extra_train"
EXTRA_TAG_PATTERN = r"^extra:[a-z0-9_.-]+$"


class QueryTemplate(FrozenModel):
    """A detection query whose phrase may name template slots, e.g. ``"{primitive}"``."""

    phrase: str = Field(min_length=1)
    expected_count: int | None = Field(None, ge=1)


class TemplateSpec(FrozenModel):
    """``holdout`` templates only ever produce ``eval`` prompts (tagged ``unseen_template``)."""

    id: str = Field(pattern=TEMPLATE_ID_PATTERN)
    text: str = Field(min_length=1)
    queries: list[QueryTemplate] = Field(min_length=1)
    holdout: bool = False


class FamilySpec(FrozenModel):
    name: str = Field(pattern=SLUG_PATTERN)
    target_checks: list[str] = Field(min_length=1)
    templates: list[TemplateSpec] = Field(min_length=2)
    weight: float = Field(1.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> FamilySpec:
        ids = [template.id for template in self.templates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"family {self.name}: template ids must be unique")
        holdout = sum(template.holdout for template in self.templates)
        if not 1 <= holdout < len(self.templates):
            raise ValueError(
                f"family {self.name}: needs at least one holdout and one seen template"
            )
        return self


class FixedPrompt(FrozenModel):
    id: str = Field(pattern=FIXED_ID_PATTERN)
    family: str = Field(pattern=SLUG_PATTERN)
    text: str = Field(min_length=1)
    queries: list[QueryTemplate] = Field(min_length=1)
    target_checks: list[str] = Field(min_length=1)


def _check_families(families: list[FamilySpec], slots: Mapping[str, list[str]]) -> None:
    names = [family.name for family in families]
    if len(set(names)) != len(names):
        raise ValueError("family names must be unique")
    if any(not values or len(set(values)) != len(values) for values in slots.values()):
        raise ValueError("every slot needs distinct values")


class ExtraTrainSettings(FrozenModel):
    """Train prompts appended after a built set: per family, the first ``n`` (so ``n`` is per
    family) of its seen-template expansion over ``slots`` permuted with ``seed``; never ``val``."""

    seed: int
    n: int = Field(ge=1)
    tag: str = Field(pattern=EXTRA_TAG_PATTERN)
    families: list[FamilySpec] = Field(min_length=1)
    slots: dict[str, list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ExtraTrainSettings:
        _check_families(self.families, self.slots)
        return self


class PromptSetSettings(FrozenModel):
    seed: int
    n_train: int = Field(ge=0)
    n_dev: int = Field(ge=0)
    n_eval: int = Field(ge=0)
    eval_unseen_fraction: float = Field(0.5, ge=0, le=1)
    val_fraction: float = Field(0.1, ge=0, lt=1)
    families: list[FamilySpec] = Field(min_length=1)
    slots: dict[str, list[str]] = Field(min_length=1)
    unscaled_queries: list[str] = Field(default_factory=list)
    fixed_eval: list[FixedPrompt] = Field(default_factory=list)
    domain: str | None = None
    """A ``[prompt_domains]`` rule every expansion and fixed prompt must satisfy."""
    extra_train: ExtraTrainSettings | None = None

    @model_validator(mode="after")
    def _check(self) -> PromptSetSettings:
        _check_families(self.families, self.slots)
        return self


class CandidateSettings(FrozenModel):
    """Candidates per prompt: ``k_train`` for pairs, ``k_eval`` (base, best-of-K baseline) and
    ``k_eval_finetuned`` with seeds k = 0..K-1 paired with the base, ``k_dev`` per checkpoint."""

    k_train: int = Field(4, ge=2)
    k_eval: int = Field(8, ge=1)
    k_eval_finetuned: int = Field(4, ge=1)
    k_dev: int = Field(2, ge=1)
    base_seed: int = Field(ge=0)
    store_latents: bool = True


class PairSettings(FrozenModel):
    min_margin: float = Field(ge=0)
    sigma_k: float = Field(1.0, ge=0)
    max_pairs_per_prompt: int = Field(1, ge=1)
    min_motion: float = Field(ge=0)
    min_motion_ratio: float = Field(0.5, ge=0)
    min_coverage: float = Field(0.0, ge=0, le=1)
    require_object_count: bool = True
    weight_by_margin: bool = False
    margin_ref: float = Field(0.5, gt=0)


class LoraSettings(FrozenModel):
    rank: int = Field(32, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.0, ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)


class TrainSettings(FrozenModel):
    beta: float = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_accum: int = Field(ge=1)
    epochs: float = Field(gt=0)
    max_steps: int | None = Field(None, ge=1)
    # The schedule is a fraction of the run's optimizer steps, resolved from the pair count.
    warmup_fraction: float = Field(0.1, ge=0, lt=1)
    checkpoint_fraction: float = Field(0.2, gt=0, le=1)
    keep_last: int | None = Field(None, ge=1)
    grad_clip: float = Field(gt=0)
    seed: int = 0
    gradient_checkpointing: bool = True
    beta_grid: list[float] = Field(default_factory=lambda: [100.0, 500.0, 2500.0, 5000.0])
    beta_grid_fraction: float = Field(0.25, gt=0, le=1)
    beta_kl_factor: float = Field(2.0, ge=1)


class SelectSettings(FrozenModel):
    checkpoints: int = Field(3, ge=1)
    max_abstention_rise: float = Field(0.10, ge=0)
    max_motion_drop: float = Field(0.20, ge=0, le=1)
    ci_z: float = Field(1.96, gt=0)


class StopRuleSettings(FrozenModel):
    """The pilot's stop rule: too few pairs, too many abstentions or too few better winners on
    inspection mean the generator comparison is not evaluable with this verifier. ``enforce =
    false`` only reports it."""

    enforce: bool = True
    min_pairs_per_prompt: float = Field(0.3, ge=0)
    min_pairs: int | None = Field(None, ge=1)
    """When set, the yield condition is this count of fit pairs instead of the rate above."""
    max_candidate_abstention: float = Field(0.5, ge=0, le=1)
    inspected: int = Field(10, ge=1)
    min_better_fraction: float = Field(0.6, ge=0, le=1)
    seed: int = 0


class FinetuneSettings(FrozenModel):
    campaign: str = Field(pattern=SLUG_PATTERN)
    scorer: str = "calibrated"
    exclude_time_dependent: bool = True
    """Leave the ``[check_scope]`` time-dependent checks out of every scorer."""
    on_family_without_targets: Literal["error", "keep"] = "error"
    """A family whose ``target_checks`` are all excluded: refuse, or keep it scored by others."""
    prompt_set: str = Field(pattern=SLUG_PATTERN)
    prompts: PromptSetSettings
    """Resolved from ``[prompt_sets.<prompt_set>]`` by ``load_finetune``."""
    candidates: CandidateSettings
    pairs: PairSettings
    lora: LoraSettings
    train: TrainSettings
    select: SelectSettings = Field(default_factory=SelectSettings)
    stop_rule: StopRuleSettings = Field(default_factory=StopRuleSettings)


class SurfaceSpec(FrozenModel):
    """A support surface: the slot its rendered templates name, its values, and the word that
    replaces ``{plane}``."""

    slot: str = Field(pattern=TEMPLATE_ID_PATTERN)
    plane: str = Field(min_length=1)
    values: list[str] = Field(min_length=1)


class LibraryFamily(FrozenModel):
    """Shapes are templates whose text names ``{surface}`` and ``{plane}``."""

    name: str = Field(pattern=SLUG_PATTERN)
    target_checks: list[str] = Field(min_length=1)
    shapes: list[TemplateSpec] = Field(min_length=1)
    weight: float = Field(1.0, gt=0)


class SceneSpec(FrozenModel):
    """A fixed prompt whose ``{surface}`` is the value ``support`` gives each surface."""

    id: str = Field(pattern=FIXED_ID_PATTERN)
    family: str = Field(pattern=SLUG_PATTERN)
    text: str = Field(min_length=1)
    queries: list[QueryTemplate] = Field(min_length=1)
    support: dict[str, str] = Field(min_length=1)


class PromptLibrary(FrozenModel):
    """Every shape and scene declared once; a set selects them and renders them per surface."""

    surfaces: dict[str, SurfaceSpec] = Field(min_length=1)
    slots: dict[str, list[str]] = Field(min_length=1)
    families: list[LibraryFamily] = Field(min_length=1)
    scenes: list[SceneSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> PromptLibrary:
        shapes = [shape.id for family in self.families for shape in family.shapes]
        if len(set(shapes)) != len(shapes):
            raise ValueError("shape ids must be unique across families")
        if any(set(scene.support) - set(self.surfaces) for scene in self.scenes):
            raise ValueError("a scene supports a surface the library does not declare")
        return self


class LibrarySelection(FrozenModel):
    """What a set takes from the library: families by name less the seen shapes it drops,
    surfaces, slots narrowed to a subset of the library's values, and fixed scenes rendered on
    ``scene_surface``."""

    families: list[str] = Field(min_length=1)
    drop_shapes: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(min_length=1)
    slots: dict[str, list[str]] = Field(default_factory=dict)
    scenes: list[str] = Field(default_factory=list)
    scene_surface: str | None = None


class ExtraSelection(LibrarySelection):
    """An ``extra_train`` table: a selection with the seed, ``n`` and tag of its prompts."""

    seed: int
    n: int
    tag: str
    scenes: list[str] = Field(default_factory=list, max_length=0)


def render(text: str, surface: SurfaceSpec, value: str) -> str:
    return text.replace(SURFACE_TOKEN, value).replace(PLANE_TOKEN, surface.plane)


def _require_known(kind: str, wanted: Collection[str], known: Collection[str]) -> None:
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise ConfigError(f"unknown {kind} {unknown}; the library declares {sorted(known)}")


def _selected(library: PromptLibrary, selection: LibrarySelection) -> list[LibraryFamily]:
    """The selected families, in library order, without the dropped shapes."""
    return [
        family.model_copy(
            update={"shapes": [s for s in family.shapes if s.id not in selection.drop_shapes]}
        )
        for family in library.families
        if family.name in selection.families
    ]


def _check_selection(library: PromptLibrary, selection: LibrarySelection) -> None:
    _require_known("families", selection.families, [family.name for family in library.families])
    chosen = (family for family in library.families if family.name in selection.families)
    seen = [shape.id for family in chosen for shape in family.shapes if not shape.holdout]
    _require_known("seen shapes to drop", selection.drop_shapes, seen)
    _require_known("surfaces", selection.surfaces, library.surfaces)
    _require_known("slots", selection.slots, library.slots)
    _require_known("scenes", selection.scenes, [scene.id for scene in library.scenes])
    for name, values in selection.slots.items():
        _require_known(f"values of slot {name}", values, library.slots[name])
    if selection.scenes and selection.scene_surface not in selection.surfaces:
        raise ConfigError("fixed scenes need a scene_surface among the selected surfaces")


def _rendered_family(family: LibraryFamily, surfaces: Mapping[str, SurfaceSpec]) -> dict[str, Any]:
    """Every shape on every surface, as ``<shape>_<surface>`` templates."""
    templates = [
        {
            **shape.model_dump(),
            "id": f"{shape.id}_{name}",
            "text": render(shape.text, surface, "{" + surface.slot + "}"),
        }
        for shape in family.shapes
        for name, surface in surfaces.items()
    ]
    return {**family.model_dump(exclude={"shapes"}), "templates": templates}


def _fixed_scenes(
    library: PromptLibrary, selection: LibrarySelection, surfaces: Mapping[str, SurfaceSpec]
) -> list[dict[str, Any]]:
    """The selected scenes on ``scene_surface``, with their family's target checks."""
    targets = {family.name: family.target_checks for family in library.families}
    scenes = {scene.id: scene for scene in library.scenes}
    fixed = []
    for scene in (scenes[name] for name in selection.scenes):
        where = str(selection.scene_surface)
        if where not in scene.support:
            raise ConfigError(f"scene {scene.id} declares no support on {where}")
        text = render(scene.text, surfaces[where], scene.support[where])
        body = scene.model_dump(exclude={"support", "text"})
        fixed.append({**body, "text": text, "target_checks": targets[scene.family]})
    return fixed


def _rendered(library: PromptLibrary, selection: LibrarySelection) -> dict[str, Any]:
    """The selected families on the selected surfaces, and the slots they expand over."""
    surfaces = {name: library.surfaces[name] for name in selection.surfaces}
    families = [_rendered_family(family, surfaces) for family in _selected(library, selection)]
    slots = {**library.slots, **selection.slots}
    slots.update({surface.slot: surface.values for surface in surfaces.values()})
    return {"families": families, "slots": slots}


def _extra(library: PromptLibrary, table: Any, section: str) -> dict[str, Any]:
    selection = validated(ExtraSelection, table, section)
    _check_selection(library, selection)
    return {**selection.model_dump(include={"seed", "n", "tag"}), **_rendered(library, selection)}


def compose(library: PromptLibrary, table: Mapping[str, Any], section: str) -> PromptSetSettings:
    """The set a ``library`` table selects, with its other keys taken as they are; an
    ``extra_train`` table is a selection of its own."""
    selection = validated(LibrarySelection, table[LIBRARY_KEY], f"{section}.{LIBRARY_KEY}")
    _check_selection(library, selection)
    surfaces = {name: library.surfaces[name] for name in selection.surfaces}
    body = {key: value for key, value in table.items() if key != LIBRARY_KEY}
    if EXTRA_KEY in table:
        body[EXTRA_KEY] = _extra(library, table[EXTRA_KEY], f"{section}.{EXTRA_KEY}")
    fixed = _fixed_scenes(library, selection, surfaces)
    composed = {**body, **_rendered(library, selection), "fixed_eval": fixed}
    return validated(PromptSetSettings, composed, section)


def load_prompt_set(settings: Settings, name: str) -> PromptSetSettings:
    """``[prompt_sets.<name>]``: declared whole, or selected from ``[prompt_library]``."""
    sets = settings.sections.get(PROMPT_SETS_SECTION, {})
    if name not in sets:
        raise ConfigError(f"unknown prompt set {name!r}; known: {sorted(sets)}")
    section, table = f"{PROMPT_SETS_SECTION}.{name}", sets[name]
    if LIBRARY_KEY in table:
        return compose(settings.section(LIBRARY_SECTION, PromptLibrary), table, section)
    return validated(PromptSetSettings, table, section)


def load_finetune(settings: Settings) -> FinetuneSettings:
    table = settings.sections.get(SECTION, {})
    if "prompts" in table:
        raise ConfigError(
            f"[{SECTION}] names its prompts with prompt_set; set them in "
            f"[{PROMPT_SETS_SECTION}.<name>], not in {SECTION}.prompts"
        )
    name = table.get("prompt_set")
    if not isinstance(name, str):
        raise ConfigError(
            f"[{SECTION}] needs prompt_set, the name of a [{PROMPT_SETS_SECTION}] table"
        )
    prompts = load_prompt_set(settings, name)
    return validated(FinetuneSettings, {**table, "prompts": prompts}, SECTION)

"""Stage cache keys and the on-disk stage cache (ADR 0005, 3.3).

Every stage key is computed here, so the provider's own profile entry (``reads_own_error``)
can never be left out of a key by one caller and included by another.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel

from physverify.core.hashing import cache_key
from physverify.extraction._base import StageProvider, error_profile_keys
from physverify.extraction.plan import StagePlan
from physverify.schema.clip import FrameSampling
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.errors import SchemaError
from physverify.schema.outputs import STAGE_OUTPUT_TYPES, StageOutput
from physverify.schema.stages import Stage, StageArtifactMeta

READS_OWN_ERROR_ATTR = "reads_own_error"
OUTPUT_STEM = "output"
META_FILE = "meta.json"
PARAMS_FILE = "params.json"

OUTPUT_FORMAT: Mapping[Stage, Literal["npz", "json"]] = MappingProxyType(
    {stage: "json" if stage in (Stage.SCALE, Stage.CONTACTS) else "npz" for stage in Stage}
)


def reads_own_error(provider: type[StageProvider]) -> bool:
    return getattr(provider, READS_OWN_ERROR_ATTR, False) is True and not provider.is_ground_truth


def profile_keys(
    provider: type[StageProvider], plan: Mapping[Stage, str]
) -> tuple[tuple[Stage, str], ...]:
    """Profile entries that can change this provider's output."""
    keys = error_profile_keys(provider.error_profile_reads, plan)
    if reads_own_error(provider):
        keys += ((provider.stage, plan[provider.stage]),)
    return keys


def stage_key(
    step: StagePlan,
    *,
    clip_id: str,
    sampling: FrameSampling,
    upstream: Mapping[Stage, str],
    plan: Mapping[Stage, str],
    gt_version: str | None,
    error_profile: ErrorProfile | None,
    device: str,
) -> str:
    provider = step.provider
    return cache_key(
        clip_id=clip_id,
        stage=step.stage,
        provider=provider.name,
        version=provider.version,
        params=step.params.model_dump(mode="json"),
        sampling=sampling,
        requires=provider.requires,
        upstream=upstream,
        is_ground_truth=provider.is_ground_truth,
        gt_version=gt_version,
        error_profile=error_profile,
        error_profile_keys=profile_keys(provider, plan),
        heavy=provider.heavy,
        device=device,
    )


class StageCache:
    """``root/<dataset>/<clip_id>/<stage>/<provider>-<key[:12]>/{output.*,meta.json,params.json}``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def directory(self, dataset: str, clip_id: str, stage: Stage, provider: str, key: str) -> Path:
        return self.root / dataset / clip_id / stage.value / f"{provider}-{key[:12]}"

    def _meta(self, directory: Path, key: str) -> StageArtifactMeta | None:
        path = directory / META_FILE
        if not path.is_file():
            return None
        try:
            meta = StageArtifactMeta.load_json(path)
        except (SchemaError, ValueError, KeyError, OSError):
            return None
        return meta if meta.cache_key == key else None

    def contains(self, dataset: str, clip_id: str, stage: Stage, provider: str, key: str) -> bool:
        """True when a complete entry with this exact key exists (reads only meta.json)."""
        directory = self.directory(dataset, clip_id, stage, provider, key)
        output = directory / f"{OUTPUT_STEM}.{OUTPUT_FORMAT[stage]}"
        return output.is_file() and self._meta(directory, key) is not None

    def load(
        self, dataset: str, clip_id: str, stage: Stage, provider: str, key: str
    ) -> StageOutput | None:
        directory = self.directory(dataset, clip_id, stage, provider, key)
        output = directory / f"{OUTPUT_STEM}.{OUTPUT_FORMAT[stage]}"
        if not output.is_file() or self._meta(directory, key) is None:
            return None
        record_type = STAGE_OUTPUT_TYPES[stage]
        loaded = (
            record_type.load_json(output)
            if OUTPUT_FORMAT[stage] == "json"
            else record_type.load_npz(output)
        )
        return cast(StageOutput, loaded)

    def store(
        self,
        dataset: str,
        clip_id: str,
        stage: Stage,
        output: StageOutput,
        meta: StageArtifactMeta,
        params: BaseModel,
    ) -> Path:
        """Write into ``<dir>.tmp-<pid>`` and rename, so a crash never leaves a half entry."""
        final = self.directory(dataset, clip_id, stage, meta.provider, meta.cache_key)
        staging = final.with_name(f"{final.name}.tmp-{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        target = staging / f"{OUTPUT_STEM}.{OUTPUT_FORMAT[stage]}"
        if OUTPUT_FORMAT[stage] == "json":
            output.save_json(target)
        else:
            output.save_npz(target)
        meta.save_json(staging / META_FILE)
        (staging / PARAMS_FILE).write_text(
            json.dumps(params.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
        )
        shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        return final

"""Candidate generation with the timed Wan2.1 call (ADR 0012, 3.5 and 4.2).

Every generator parameter comes from ``[timing.generation]`` through ``call_kwargs`` (D3). One
call per candidate (B = 1) with the seed of D5; the mp4, the final latents (float32, the dtype of
the denoising loop) and the meta are written in that order, so an interrupted run resumes at the
first candidate without a meta.
"""

from __future__ import annotations

import hashlib
import importlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from physverify.core.errors import PhysverifyError, UsageError
from physverify.diagnosis.atomic_io import write_atomic
from physverify.finetune.records import (
    BASE_TAG,
    CandidateRecord,
    PromptRecord,
    read_record,
    write_record,
)
from physverify.finetune.seeds import WHOLE, Shard, candidate_id, candidate_seed
from physverify.finetune.settings import CandidateSettings, PromptSplit
from physverify.finetune.store import CampaignStore
from physverify.finetune.wan_lora import load_adapter
from physverify.schema.io import stable_hash
from physverify.timing.generation import call_kwargs, is_out_of_memory, torch_generator
from physverify.timing.mp4 import to_uint8, write_mp4
from physverify.timing.settings import GenerationSettings
from physverify.timing.targets.wan21_generate import batch_frames

BATCH_SIZE: Final = 1
CAPTURED: Final = ["latents", "prompt_embeds"]
LATENTS_KEY: Final = "latents"
EMBEDS_KEY: Final = "prompt_embeds"
ADAPTER_HASH_CHARS: Final = 12
HASH_EXCLUDED: Final = frozenset({"prompts", "subset", "batch_sizes"})
SHOWN_CONFLICTS: Final = 5

Pipeline = Any
PipeFactory = Callable[[], Pipeline]
GeneratorFactory = Callable[[str, int], Any]
AdapterLoader = Callable[[Pipeline, Path], None]
TensorSaver = Callable[[Mapping[str, Any], Path], None]


class GenerationError(PhysverifyError):
    pass


class LatentCapture:
    """``callback_on_step_end``: copies ``prompt_embeds`` at step 0 and ``latents`` at the last
    step (after ``scheduler.step``, so x0 in the transformer's normalized space). Returns ``{}``:
    the pipeline's tensors are never modified."""

    def __init__(self, num_steps: int) -> None:
        self._last = num_steps - 1
        self.latents: Any = None
        self.prompt_embeds: Any = None

    def __call__(
        self, pipe: Any, step: int, timestep: Any, kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        if step == 0 and EMBEDS_KEY in kwargs:
            self.prompt_embeds = kwargs[EMBEDS_KEY].detach().cpu()
        if step == self._last and LATENTS_KEY in kwargs:
            self.latents = kwargs[LATENTS_KEY].detach().cpu()
        return {}


def save_safetensors(tensors: Mapping[str, Any], path: Path) -> None:
    """Atomic ``safetensors`` write of contiguous CPU tensors."""
    save_file = importlib.import_module("safetensors.torch").save_file
    contiguous = {k: v.contiguous() for k, v in tensors.items()}
    write_atomic(path, lambda temp: save_file(contiguous, str(temp)))


def default_adapter_loader(pipe: Pipeline, adapter: Path) -> None:
    load_adapter(pipe.transformer, adapter)


def adapter_tag(adapter: Path) -> str:
    """``lora-<sha12>`` over the adapter's files, so one checkpoint always has one tag."""
    digest = hashlib.sha256()
    files = sorted(p for p in adapter.rglob("*") if p.is_file()) if adapter.is_dir() else [adapter]
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"lora-{digest.hexdigest()[:ADAPTER_HASH_CHARS]}"


def config_hash(
    gen: GenerationSettings, prompt: PromptRecord, seed: int, model_tag: str, store_latents: bool
) -> str:
    params = gen.model_dump(mode="json", exclude=set(HASH_EXCLUDED))
    return stable_hash(
        {
            "generation": params,
            "text": prompt.text,
            "seed": seed,
            "model_tag": model_tag,
            "latents": store_latents,
        }
    )


@dataclass(frozen=True)
class GenerationReport:
    generated: list[str] = field(default_factory=list)
    manifest: Path | None = None


class CandidateGenerator:
    def __init__(
        self,
        store: CampaignStore,
        gen: GenerationSettings,
        cand: CandidateSettings,
        *,
        pipe_factory: PipeFactory,
        device: str,
        generator_factory: GeneratorFactory = torch_generator,
        adapter: Path | None = None,
        adapter_loader: AdapterLoader = default_adapter_loader,
        tensor_saver: TensorSaver = save_safetensors,
        clock: Callable[[], float] = time.perf_counter,
        commit: str = "",
    ) -> None:
        self._store = store
        self._gen = gen
        self._cand = cand
        self._pipe_factory = pipe_factory
        self._device = device
        self._generator = generator_factory
        self._adapter = adapter
        self._adapter_loader = adapter_loader
        self._save = tensor_saver
        self._clock = clock
        self._commit = commit
        self._pipe: Pipeline = None
        self.model_tag = BASE_TAG if adapter is None else adapter_tag(adapter)

    def k_for(self, split: PromptSplit, finetuned: bool) -> int:
        if split == "train":
            return self._cand.k_train
        if split == "dev":
            return self._cand.k_dev
        return self._cand.k_eval_finetuned if finetuned else self._cand.k_eval

    def plan(
        self,
        prompts: Iterable[PromptRecord],
        shard: Shard = WHOLE,
        k: int | None = None,
    ) -> list[CandidateRecord]:
        """Candidates still to generate, in prompt order then k; complete ones are skipped.

        Raises UsageError when a stored candidate has another ``config_hash``."""
        records, conflicts = [], []
        for prompt in prompts:
            count = k if k is not None else self.k_for(prompt.split, self._adapter is not None)
            for index in range(count):
                record = self.record(prompt, index)
                if not shard.owns(record.candidate_id):
                    continue
                if self._stored_hash(record) not in (None, record.config_hash):
                    conflicts.append(record.candidate_id)
                elif not self._store.is_complete(record):
                    records.append(record)
        if conflicts:
            raise UsageError(
                f"{len(conflicts)} stored candidates of {self.model_tag} were generated with "
                f"another configuration (e.g. {conflicts[:SHOWN_CONFLICTS]}); regenerating them "
                "would leave their scores and extraction stale: use a new finetune.campaign"
            )
        return records

    def _stored_hash(self, record: CandidateRecord) -> str | None:
        meta = self._store.meta_path(record.model_tag, record.prompt_id, record.k)
        return read_record(meta, CandidateRecord).config_hash if meta.is_file() else None

    def record(self, prompt: PromptRecord, k: int, gen_s: float | None = None) -> CandidateRecord:
        seed = candidate_seed(self._cand.base_seed, prompt.prompt_id, k)
        tag = self.model_tag
        latents = self._store.latents_path(tag, prompt.prompt_id, k)
        return CandidateRecord(
            candidate_id=candidate_id(prompt.prompt_id, k),
            prompt_id=prompt.prompt_id,
            split=prompt.split,
            k=k,
            seed=seed,
            model_tag=tag,
            video_path=str(self._store.video_path(tag, prompt.prompt_id, k)),
            latents_path=str(latents) if self._cand.store_latents else None,
            embeds_path=str(self._store.embeds_path(prompt.prompt_id)),
            gen_s=gen_s,
            commit=self._commit,
            config_hash=config_hash(self._gen, prompt, seed, tag, self._cand.store_latents),
        )

    def run(
        self, plan: Sequence[CandidateRecord], prompts: Mapping[str, PromptRecord]
    ) -> GenerationReport:
        generated: list[str] = []
        manifest = None
        for record in plan:
            self._one(record, prompts[record.prompt_id])
            generated.append(record.candidate_id)
            manifest = self._store.write_manifest(self.model_tag)
        return GenerationReport(generated, manifest)

    def kwargs(
        self, prompt: PromptRecord, record: CandidateRecord, capture: LatentCapture
    ) -> dict[str, Any]:
        return {
            **call_kwargs(prompt.text, self._gen, BATCH_SIZE),
            "generator": self._generator(self._device, record.seed),
            "callback_on_step_end": capture,
            "callback_on_step_end_tensor_inputs": list(CAPTURED),
        }

    def _pipeline(self) -> Pipeline:
        if self._pipe is None:
            self._pipe = self._pipe_factory()
            if self._adapter is not None:
                self._adapter_loader(self._pipe, self._adapter)
        return self._pipe

    def _one(self, record: CandidateRecord, prompt: PromptRecord) -> None:
        pipe = self._pipeline()
        capture = LatentCapture(self._gen.num_inference_steps)
        start = self._clock()
        try:
            output = pipe(**self.kwargs(prompt, record, capture))
        except Exception as exc:
            if is_out_of_memory(exc):
                raise GenerationError(f"{record.candidate_id}: out of memory at B = 1") from exc
            raise
        gen_s = self._clock() - start
        frames = batch_frames(output, BATCH_SIZE)[0]
        write_mp4(to_uint8(frames), Path(record.video_path), self._gen.fps)
        self._save_tensors(record, capture)
        done = record.model_copy(update={"gen_s": gen_s})
        write_record(self._store.meta_path(record.model_tag, record.prompt_id, record.k), done)

    def _save_tensors(self, record: CandidateRecord, capture: LatentCapture) -> None:
        if record.latents_path is not None:
            if capture.latents is None:
                raise GenerationError(f"{record.candidate_id}: the pipeline gave no final latents")
            self._save({LATENTS_KEY: capture.latents.float()}, Path(record.latents_path))
        embeds = Path(record.embeds_path)
        if not embeds.is_file():
            if capture.prompt_embeds is None:
                raise GenerationError(f"{record.candidate_id}: the pipeline gave no prompt_embeds")
            self._save({EMBEDS_KEY: capture.prompt_embeds}, embeds)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import safetensors.torch
import torch
from pydantic import BaseModel
from safetensors.torch import load_file

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets.local_video import QueryEntry
from physverify.finetune.candidates import (
    CandidateGenerator,
    LatentCapture,
    adapter_tag,
    save_safetensors,
)
from physverify.finetune.layout import CampaignStore
from physverify.finetune.records import BASE_TAG, CandidateRecord, PromptRecord, read_record
from physverify.finetune.settings import FinetuneSettings, load_finetune
from physverify.generation.settings import GenerationSettings
from physverify.generation.wan21 import call_kwargs
from physverify.timing.settings import load_timing

STEPS = 3
GENERATOR_FIELDS = (
    "num_inference_steps",
    "guidance_scale",
    "num_frames",
    "height",
    "width",
    "flow_shift",
    "negative_prompt",
)


class GenerationInterruptedError(Exception):
    pass


class FakePipe:
    """Calls the step callback like ``WanPipeline`` and returns tiny float frames."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_after = fail_after
        self.adapters: list[Path] = []

    def __call__(self, **kwargs: Any) -> Any:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise GenerationInterruptedError
        self.calls.append(kwargs)
        latents = torch.full((1, 2, 2, 4, 4), float(len(self.calls)), dtype=torch.bfloat16)
        embeds = torch.ones((1, 5, 8), dtype=torch.bfloat16)
        for step in range(kwargs["num_inference_steps"]):
            back = kwargs["callback_on_step_end"](
                self, step, 1000 - step, {"latents": latents, "prompt_embeds": embeds}
            )
            assert back == {}
        return SimpleNamespace(frames=np.full((1, 3, 16, 16, 3), 0.5, dtype=np.float32))


def _prompt(i: int) -> PromptRecord:
    return PromptRecord(
        prompt_id=f"fall-{i:04d}",
        text=f"prompt {i}",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fall_drop",
        target_checks=["R5a"],
        split="train",
    )


@pytest.fixture
def gen(settings: Settings) -> GenerationSettings:
    return load_timing(settings).generation.model_copy(update={"num_inference_steps": STEPS})


@pytest.fixture
def cfg(settings: Settings) -> FinetuneSettings:
    return load_finetune(settings)


@pytest.fixture
def store(tmp_path: Path) -> CampaignStore:
    return CampaignStore(root=tmp_path / "ft" / "c", raw_root=tmp_path / "raw", campaign="c")


def _generator(
    store: CampaignStore, gen: GenerationSettings, cfg: FinetuneSettings, pipe: FakePipe, **kw: Any
) -> CandidateGenerator:
    return CandidateGenerator(
        store,
        gen,
        cfg.candidates,
        pipe_factory=lambda: pipe,
        device="cpu",
        generator_factory=lambda device, seed: ("gen", device, seed),
        **kw,
    )


def _run(generator: CandidateGenerator, prompts: list[PromptRecord]) -> list[CandidateRecord]:
    plan = generator.plan(prompts)
    generator.run(plan, {p.prompt_id: p for p in prompts})
    return plan


def test_one_call_per_candidate_with_the_timing_kwargs(
    store: CampaignStore, gen: GenerationSettings, cfg: FinetuneSettings
) -> None:
    pipe = FakePipe()
    prompts = [_prompt(i) for i in range(3)]
    plan = _run(_generator(store, gen, cfg, pipe), prompts)
    k = cfg.candidates.k_train
    assert len(pipe.calls) == len(plan) == 3 * k
    first = pipe.calls[0]
    expected = call_kwargs(prompts[0].text, gen, 1)
    assert {key: first[key] for key in expected} == expected
    assert first["generator"] == ("gen", "cpu", plan[0].seed)
    assert first["callback_on_step_end_tensor_inputs"] == ["latents", "prompt_embeds"]
    record = read_record(store.meta_path(BASE_TAG, prompts[0].prompt_id, 0), CandidateRecord)
    assert Path(record.video_path).is_file()
    latents = load_file(str(record.latents_path))["latents"]
    assert latents.dtype == torch.float32
    assert Path(record.embeds_path).is_file()
    assert (store.video_dir(BASE_TAG) / "manifest.json").is_file()


def test_an_interrupted_run_resumes_without_redoing_candidates(
    store: CampaignStore, gen: GenerationSettings, cfg: FinetuneSettings
) -> None:
    prompts = [_prompt(i) for i in range(3)]
    total = 3 * cfg.candidates.k_train
    cut = 5
    with pytest.raises(GenerationInterruptedError):
        _run(_generator(store, gen, cfg, FakePipe(fail_after=cut)), prompts)
    metas = sorted(store.root.rglob("*.meta.json"))
    assert len(metas) == cut
    before = {p: p.read_bytes() for p in metas}
    pipe = FakePipe()
    _run(_generator(store, gen, cfg, pipe), prompts)
    assert len(pipe.calls) == total - cut
    assert {p: p.read_bytes() for p in metas} == before
    assert _generator(store, gen, cfg, FakePipe()).plan(prompts) == []


def test_another_generator_config_is_refused_over_stored_candidates(
    store: CampaignStore, gen: GenerationSettings, cfg: FinetuneSettings
) -> None:
    prompts = [_prompt(0)]
    _run(_generator(store, gen, cfg, FakePipe()), prompts)
    stored = {p: p.read_bytes() for p in sorted(store.root.rglob("*")) if p.is_file()}
    videos = {p: p.read_bytes() for p in sorted(store.video_dir(BASE_TAG).glob("*.mp4"))}
    changed = gen.model_copy(update={"guidance_scale": gen.guidance_scale + 1})
    pipe = FakePipe()
    with pytest.raises(UsageError, match=f"another campaign than {store.campaign} "):
        _run(_generator(store, changed, cfg, pipe), prompts)
    assert pipe.calls == []
    assert {p: p.read_bytes() for p in sorted(store.root.rglob("*")) if p.is_file()} == stored
    assert {p: p.read_bytes() for p in sorted(store.video_dir(BASE_TAG).glob("*.mp4"))} == videos


def test_a_new_prompt_still_generates_beside_a_matching_campaign(
    store: CampaignStore, gen: GenerationSettings, cfg: FinetuneSettings
) -> None:
    _run(_generator(store, gen, cfg, FakePipe()), [_prompt(0)])
    plan = _generator(store, gen, cfg, FakePipe()).plan([_prompt(0), _prompt(1)])
    assert {r.prompt_id for r in plan} == {_prompt(1).prompt_id}


def test_tensor_writes_use_a_unique_temp_file_and_leave_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temps: list[str] = []
    real = safetensors.torch.save_file

    def recording(tensors: Any, filename: str) -> None:
        temps.append(filename)
        real(tensors, filename)

    monkeypatch.setattr(safetensors.torch, "save_file", recording)
    path = tmp_path / "embeds" / "p.safetensors"
    save_safetensors({"x": torch.ones(2)}, path)
    save_safetensors({"x": torch.zeros(2)}, path)
    assert len(set(temps)) == 2 and str(path) not in temps
    assert torch.equal(load_file(str(path))["x"], torch.zeros(2))
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_latent_capture_keeps_embeds_at_step_zero_and_latents_at_the_last() -> None:
    capture = LatentCapture(2)
    first, last = torch.zeros(1), torch.ones(1)
    assert capture(None, 0, 0, {"latents": first, "prompt_embeds": first}) == {}
    assert capture(None, 1, 0, {"latents": last}) == {}
    assert torch.equal(capture.latents, last)
    assert torch.equal(capture.prompt_embeds, first)


def test_an_adapter_gets_its_own_tag_and_finetuned_k(
    store: CampaignStore, gen: GenerationSettings, cfg: FinetuneSettings, tmp_path: Path
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.safetensors").write_bytes(b"abc")
    loaded: list[Path] = []
    pipe = FakePipe()
    generator = _generator(
        store, gen, cfg, pipe, adapter=adapter, adapter_loader=lambda p, a: loaded.append(a)
    )
    assert generator.model_tag == adapter_tag(adapter)
    assert generator.model_tag.startswith("lora-")
    assert generator.k_for("eval", True) == cfg.candidates.k_eval_finetuned
    eval_prompt = _prompt(1).model_copy(update={"split": "eval"})
    _run(generator, [eval_prompt])
    assert loaded == [adapter]
    assert len(pipe.calls) == cfg.candidates.k_eval_finetuned
    base = _generator(store, gen, cfg, FakePipe()).record(eval_prompt, 0)
    assert store.candidates(generator.model_tag)[0].seed == base.seed


def _field_names(model: type[BaseModel]) -> set[str]:
    names: set[str] = set()
    for name, info in model.model_fields.items():
        names.add(name)
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            names |= _field_names(annotation)
    return names


def test_generator_parameters_are_only_in_timing(repo_root: Path) -> None:
    text = (repo_root / "configs" / "finetune.toml").read_text(encoding="utf-8")
    assert not [name for name in GENERATOR_FIELDS if f"{name} =" in text]
    assert not set(GENERATOR_FIELDS) & _field_names(FinetuneSettings)

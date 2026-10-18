"""Wan2.1 transformer with a LoRA adapter as an ``AdapterModel`` (H200 only; ADR 0012, 3.8).

The base weights stay in the generator dtype and frozen; the LoRA weights train in float32. The
reference model is the same transformer with the adapter disabled, so no second copy is loaded.
The adapter is saved with ``save_lora_adapter``, the format ``load_lora_adapter`` reads at
generation time.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from physverify.extraction.weights import hub_offline
from physverify.finetune.settings import LoraSettings
from physverify.timing.settings import GenerationSettings

TRANSFORMER: Final = "transformer"
ADAPTER_NAME: Final = "default"
LORA_KEY: Final = "lora_"


class WanAdapterModel:
    def __init__(self, transformer: Any, compute_dtype: Any) -> None:
        self._transformer = transformer
        self._dtype = compute_dtype

    def predict(self, x_t: Any, t: Any, cond: Any) -> Any:
        out = self._transformer(
            hidden_states=x_t.to(self._dtype),
            timestep=t,
            encoder_hidden_states=cond.to(self._dtype),
            return_dict=False,
        )
        return out[0]

    @contextmanager
    def reference(self) -> Iterator[None]:
        self._transformer.disable_adapters()
        try:
            yield
        finally:
            self._transformer.enable_adapters()

    def trainable_parameters(self) -> list[Any]:
        return [p for p in self._transformer.parameters() if p.requires_grad]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._transformer.save_lora_adapter(str(directory), adapter_name=ADAPTER_NAME)

    def load(self, directory: Path) -> None:
        peft = importlib.import_module("peft")
        safetensors = importlib.import_module("safetensors.torch")
        files = sorted(directory.glob("*.safetensors"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one adapter file in {directory}, found {files}")
        state = safetensors.load_file(str(files[0]))
        state = {key.removeprefix(f"{TRANSFORMER}."): value for key, value in state.items()}
        result = peft.set_peft_model_state_dict(self._transformer, state, adapter_name=ADAPTER_NAME)
        check_lora_load(result, ADAPTER_NAME)


def check_lora_load(result: Any, adapter_name: str) -> None:
    """``set_peft_model_state_dict`` loads with ``strict=False``: every LoRA key of the adapter
    must have been filled and no saved key may be left over, or the run would resume silently
    from a partly initialised adapter."""
    marker = f".{adapter_name}."
    missing = [k for k in result.missing_keys if LORA_KEY in k and marker in k]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            f"adapter state does not match the model: {len(missing)} LoRA keys missing "
            f"{missing[:3]}, {len(unexpected)} unexpected {unexpected[:3]}"
        )


def lora_config(lora: LoraSettings) -> Any:
    peft = importlib.import_module("peft")
    return peft.LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        init_lora_weights=True,
    )


def load_wan_adapter_model(
    snapshot: Path,
    gen: GenerationSettings,
    lora: LoraSettings,
    device: str,
    seed: int,
    checkpointing: bool = True,
) -> WanAdapterModel:
    diffusers = importlib.import_module("diffusers")
    torch = importlib.import_module("torch")
    dtype = getattr(torch, gen.dtype)
    with hub_offline():
        transformer = diffusers.WanTransformer3DModel.from_pretrained(
            snapshot, subfolder=TRANSFORMER, torch_dtype=dtype
        )
    transformer.requires_grad_(False)
    torch.manual_seed(seed)  # the LoRA A init is random; the same seed for every beta of the grid
    transformer.add_adapter(lora_config(lora), adapter_name=ADAPTER_NAME)
    for parameter in transformer.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.train()
    return WanAdapterModel(transformer.to(device), dtype)

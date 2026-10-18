from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from physverify.finetune.settings import LoraSettings
from physverify.finetune.wan_lora import (
    ADAPTER_FILE,
    ADAPTER_NAME,
    LORA_KEY,
    WanAdapterModel,
    check_lora_load,
    load_adapter,
    lora_config,
)

LORA = f"blocks.0.attn1.to_q.lora_A.{ADAPTER_NAME}.weight"


def test_base_keys_missing_from_the_adapter_file_are_fine() -> None:
    check_lora_load(
        SimpleNamespace(missing_keys=["blocks.0.attn1.to_q.weight"], unexpected_keys=[]),
        ADAPTER_NAME,
    )


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(missing_keys=[LORA], unexpected_keys=[]),
        SimpleNamespace(missing_keys=[], unexpected_keys=["blocks.0.attn1.to_q.lora_A.weight"]),
    ],
)
def test_a_missing_or_unexpected_lora_key_is_an_error(result: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="does not match"):
        check_lora_load(result, ADAPTER_NAME)


TARGETS = ["attn1.to_q", "attn1.to_v"]


def _tiny_transformer() -> Any:
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    return diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=32,
    )


def _trained_adapter(directory: Path) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    model = _tiny_transformer()
    model.add_adapter(lora_config(LoraSettings(rank=4, alpha=4, target_modules=TARGETS)))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if LORA_KEY in name:
                parameter.normal_()
    WanAdapterModel(model, torch.float32).save(directory)
    return _lora_state(model)


def _lora_state(model: Any) -> dict[str, Any]:
    peft = pytest.importorskip("peft")
    return dict(peft.get_peft_model_state_dict(model, adapter_name=ADAPTER_NAME))


def test_a_saved_adapter_loads_back_into_a_fresh_transformer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    saved = _trained_adapter(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [ADAPTER_FILE]
    fresh = _tiny_transformer()
    load_adapter(fresh, tmp_path)
    loaded = _lora_state(fresh)
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[key], value) for key, value in saved.items())


def test_an_adapter_with_a_key_the_model_lacks_is_an_error(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    _trained_adapter(tmp_path)
    path = tmp_path / ADAPTER_FILE
    state = pytest.importorskip("safetensors.torch").load_file(str(path))
    state["blocks.9.attn1.to_q.lora_A.weight"] = torch.zeros(4, 16)
    save_file(state, str(path))
    with pytest.raises(ValueError, match="does not match"):
        load_adapter(_tiny_transformer(), tmp_path)


def test_a_directory_without_the_adapter_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=ADAPTER_FILE):
        load_adapter(object(), tmp_path)

from __future__ import annotations

from types import SimpleNamespace

import pytest

from physverify.finetune.wan_lora import ADAPTER_NAME, check_lora_load

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

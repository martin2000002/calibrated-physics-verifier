"""The H3 eval candidate identity is pinned: a change here regenerates every stored video."""

from __future__ import annotations

from pathlib import Path

from physverify.config.loader import load_settings
from physverify.datasets.local_video import QueryEntry
from physverify.finetune.candidates import config_hash
from physverify.finetune.records import BASE_TAG, PromptRecord
from physverify.finetune.seeds import candidate_id, candidate_seed
from physverify.finetune.settings import generator_settings, load_finetune
from physverify.generator_comparison.settings import load_h3

PINNED_HASH = "d7962fa12ea64a05d6beed0ba5dc45538ca22bb31059069ccc76deee0031702a"


def test_the_base_eval_candidate_identity_is_unchanged(
    config_dir: Path, environ: dict[str, str]
) -> None:
    settings = load_settings(config_dir, environ=environ, profile="precise")
    prompt = PromptRecord(
        prompt_id="fall-0000",
        text="a red ball falls onto a wooden table",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fall_drop",
        target_checks=["R5a"],
        split="eval",
    )
    seed = candidate_seed(load_finetune(settings).candidates.base_seed, prompt.prompt_id, 3)
    latents = load_h3(settings).eval_store_latents
    digest = config_hash(generator_settings(settings), prompt, seed, BASE_TAG, latents)
    assert candidate_id(prompt.prompt_id, 3) == "fall-0000__k03"
    assert (seed, latents, digest) == (1524816330, False, PINNED_HASH)

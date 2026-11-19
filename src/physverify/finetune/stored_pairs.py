"""Latent pairs of the campaign store, read on demand as trainer batches."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physverify.finetune.records import CandidateRecord, PairRecord
from physverify.finetune.trainer import PairBatch

TensorLoader = Callable[[Path], Mapping[str, Any]]


def load_safetensors(path: Path) -> Mapping[str, Any]:
    load_file = importlib.import_module("safetensors.torch").load_file
    loaded: Mapping[str, Any] = load_file(str(path))
    return loaded


@dataclass(frozen=True)
class StoredPair:
    pair_id: str
    latents_w: Path
    latents_l: Path
    embeds: Path
    weight: float


class StoredPairSource:
    """Pairs read from the campaign store on demand (latents float32, embeds as generated)."""

    def __init__(
        self,
        pairs: Sequence[StoredPair],
        device: str,
        dtype: Any = None,
        loader: TensorLoader = load_safetensors,
    ) -> None:
        self._pairs = list(pairs)
        self._device = device
        self._dtype = dtype
        self._load = loader

    def __len__(self) -> int:
        return len(self._pairs)

    def get(self, index: int) -> PairBatch:
        pair = self._pairs[index]
        return PairBatch(
            pair_id=pair.pair_id,
            x0_w=self._tensor(pair.latents_w, "latents"),
            x0_l=self._tensor(pair.latents_l, "latents"),
            cond=self._tensor(pair.embeds, "prompt_embeds"),
            weight=pair.weight,
        )

    def _tensor(self, path: Path, key: str) -> Any:
        tensor = self._load(path)[key].to(self._device)
        return tensor if self._dtype is None else tensor.to(self._dtype)


def stored_pairs(
    pairs: Sequence[PairRecord], candidates: Sequence[CandidateRecord]
) -> list[StoredPair]:
    """The latent files of each pair; the embeddings are the prompt's (shared by both)."""
    by_id = {c.candidate_id: c for c in candidates}
    stored = []
    for pair in pairs:
        winner, loser = by_id.get(pair.winner), by_id.get(pair.loser)
        if winner is None or loser is None:
            raise ValueError(f"{pair.pair_id}: a member has no candidate record")
        if winner.latents_path is None or loser.latents_path is None:
            raise ValueError(f"{pair.pair_id}: latents were not stored (store_latents = false)")
        stored.append(
            StoredPair(
                pair.pair_id,
                Path(winner.latents_path),
                Path(loser.latents_path),
                Path(winner.embeds_path),
                pair.weight,
            )
        )
    return stored

"""Small helpers shared by several providers (not a plugin module)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.core.models import FrozenModel
from physverify.extraction._base import StageContext
from physverify.schema.clip import FrameSampling


class ProviderParams(FrozenModel):
    """Base of every provider's parameters, which its cache key covers."""


def expect_params[P: BaseModel](params: BaseModel, model: type[P]) -> P:
    if not isinstance(params, model):
        raise TypeError(f"expected {model.__name__}, got {type(params).__name__}")
    return params


def label_candidates(label: str) -> tuple[str, ...]:
    """The whole label first, then each of its words (a query phrase like "red ball")."""
    text = label.strip().lower()
    return (text, *text.replace("_", " ").split())


def lookup_label[V](label: str, table: Mapping[str, V]) -> V | None:
    """Exact match of the label, else of its first word that is a key of ``table``."""
    for candidate in label_candidates(label):
        if candidate in table:
            return table[candidate]
    return None


def centred_intrinsics(focal: float, sampling: FrameSampling) -> NDArray[np.float64]:
    """Pinhole K in processing pixels with square pixels and the principal point at the centre."""
    return np.array(
        [
            [focal, 0.0, sampling.width / 2.0],
            [0.0, focal, sampling.height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def set_torch_threads(threads: int) -> None:
    """Bound torch's intra-op pool to ``runtime.threads`` (torch is imported lazily)."""
    import torch

    torch.set_num_threads(threads)


class LazyModel[M]:
    """A heavy provider's model, loaded by the first ``get`` and kept by the provider instance.

    D2: the pipeline keeps one instance per stage for every clip and drops it when the stage
    ends, so the weights load once per stage with the first context's device and paths."""

    def __init__(self, loader: Callable[[StageContext], M]) -> None:
        self._loader = loader
        self._model: M | None = None

    def get(self, ctx: StageContext) -> M:
        if self._model is None:
            self._model = self._loader(ctx)
        return self._model

"""A small regression network: an MLP on dense features plus a linear head on token features,
summed into one output, with inputs standardised by stored statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Standardiser:
    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, x: FloatArray) -> Standardiser:
        """Column mean and std; a constant or all-NaN column keeps scale 1 and mean 0."""
        with np.errstate(invalid="ignore"):
            finite = np.isfinite(x)
            count = finite.sum(axis=0)
            filled = np.where(finite, x, 0.0)
            mean = np.divide(filled.sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
            spread = np.where(finite, x - mean, 0.0)
            var = np.divide(
                (spread**2).sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0
            )
        scale = np.sqrt(var)
        return cls(mean, np.where(scale > 0.0, scale, 1.0))

    def apply(self, x: FloatArray) -> FloatArray:
        """Standardised ``x``; a missing (NaN) value becomes the mean, 0."""
        z = (x - self.mean) / self.scale
        return np.where(np.isfinite(z), z, 0.0)


@dataclass(frozen=True)
class TwoHeadWeights:
    """``hidden``: (W, b) per hidden layer (ReLU); ``out``: (W, b) of the MLP's linear output;
    ``token``: (w, b) of the linear head on the tokens."""

    hidden: tuple[tuple[FloatArray, FloatArray], ...]
    out: tuple[FloatArray, FloatArray]
    token: tuple[FloatArray, FloatArray]


def hidden_activations(weights: TwoHeadWeights, dense: FloatArray) -> list[FloatArray]:
    """Input followed by each hidden layer's ReLU output."""
    layers = [dense]
    for w, b in weights.hidden:
        layers.append(np.maximum(layers[-1] @ w + b, 0.0))
    return layers


def predict(weights: TwoHeadWeights, dense: FloatArray, tokens: FloatArray) -> FloatArray:
    """(n,) outputs for standardised (n, d) dense and (n, k) token features."""
    w, b = weights.out
    mlp = hidden_activations(weights, dense)[-1] @ w + b
    tw, tb = weights.token
    return np.asarray((mlp + tokens @ tw + tb).reshape(-1), dtype=np.float64)

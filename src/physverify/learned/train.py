"""Regression training of a learned variant's artifact, with early stopping on the internal
validation split only.

The split is by scene: a scene is validation when a hash of its id falls in the first
``validation_share`` of the range, so the clean clip, the positive and the decoy of a scene
always land together. Loss: mean squared error plus ``weight_decay`` times the squared weights;
Adam on shuffled mini-batches. Training stops after ``patience`` epochs without a better
validation loss, or at ``max_epochs``, and keeps the best epoch's weights. The artifact's sigma
is the validation residual rms of those weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from physverify.checks.variants._learned import LearnedArtifact
from physverify.learned.arms import uniform
from physverify.learned.samples import SampleSet
from physverify.stats.mlp import Standardiser, TwoHeadWeights, hidden_activations, predict

FloatArray = NDArray[np.float64]
SPLIT_SALT = "validation"


@dataclass(frozen=True)
class TrainSettings:
    hidden: tuple[int, ...] = (32,)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 25
    validation_share: float = 0.2
    seed: int = 0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8


@dataclass(frozen=True)
class TrainingLog:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int
    n_train: int
    n_validation: int


@dataclass(frozen=True)
class Trained:
    artifact: LearnedArtifact
    log: TrainingLog
    validation: SampleSet


def is_validation(scene_id: str, share: float) -> bool:
    return uniform(SPLIT_SALT, scene_id) < share


def split(samples: SampleSet, share: float) -> tuple[SampleSet, SampleSet]:
    held = [is_validation(s.scene_id, share) for s in samples.samples]
    return samples.subset([not h for h in held]), samples.subset(held)


@dataclass
class _Adam:
    betas: tuple[float, float]
    eps: float
    rate: float
    m: list[FloatArray] = field(default_factory=list)
    v: list[FloatArray] = field(default_factory=list)
    step: int = 0

    def update(self, params: list[FloatArray], grads: Sequence[FloatArray]) -> None:
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.step += 1
        b1, b2 = self.betas
        for p, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat, v_hat = m / (1 - b1**self.step), v / (1 - b2**self.step)
            p -= self.rate * m_hat / (np.sqrt(v_hat) + self.eps)


def _flat(weights: TwoHeadWeights) -> list[FloatArray]:
    hidden = [a for pair in weights.hidden for a in pair]
    return [*hidden, *weights.out, *weights.token]


def _weights(params: Sequence[FloatArray], layers: int) -> TwoHeadWeights:
    hidden = tuple((params[2 * i], params[2 * i + 1]) for i in range(layers))
    rest = params[2 * layers :]
    return TwoHeadWeights(hidden, (rest[0], rest[1]), (rest[2], rest[3]))


def initial_weights(n_dense: int, n_tokens: int, settings: TrainSettings) -> TwoHeadWeights:
    """He-normal hidden layers, zero output and token heads (the start predicts 0)."""
    rng = np.random.default_rng(settings.seed)
    sizes = (n_dense, *settings.hidden)
    hidden = tuple(
        (rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)), np.zeros(b)) for a, b in pairwise(sizes)
    )
    return TwoHeadWeights(
        hidden, (np.zeros((sizes[-1], 1)), np.zeros(1)), (np.zeros((n_tokens, 1)), np.zeros(1))
    )


def gradients(
    weights: TwoHeadWeights, dense: FloatArray, tokens: FloatArray, y: FloatArray, decay: float
) -> list[FloatArray]:
    """Gradient of mean squared error + ``decay`` * sum of squared weight matrices."""
    layers = hidden_activations(weights, dense)
    residual = predict(weights, dense, tokens) - y
    dy = (2.0 / len(y)) * residual[:, None]
    out_w, _ = weights.out
    token_w, _ = weights.token
    grads = [
        layers[-1].T @ dy + 2 * decay * out_w,
        dy.sum(axis=0),
        tokens.T @ dy + 2 * decay * token_w,
        dy.sum(axis=0),
    ]
    upstream = dy @ out_w.T
    hidden_grads: list[FloatArray] = []
    for index in reversed(range(len(weights.hidden))):
        w, _ = weights.hidden[index]
        local = upstream * (layers[index + 1] > 0.0)
        hidden_grads[:0] = [layers[index].T @ local + 2 * decay * w, local.sum(axis=0)]
        upstream = local @ w.T
    return [*hidden_grads, *grads]


def _loss(weights: TwoHeadWeights, dense: FloatArray, tokens: FloatArray, y: FloatArray) -> float:
    return float(np.mean((predict(weights, dense, tokens) - y) ** 2))


def train(
    samples: SampleSet,
    settings: TrainSettings,
    identity: tuple[str, str, str],
    training_seeds: frozenset[int],
) -> Trained:
    """``identity``: (check_id, unit, encoder_sha256) the artifact is stamped with."""
    fit_set, held = split(samples, settings.validation_share)
    dense, tokens, y = fit_set.arrays()
    v_dense, v_tokens, v_y = held.arrays()
    dense_std, token_std = Standardiser.fit(dense), Standardiser.fit(tokens)
    x, t = dense_std.apply(dense), token_std.apply(tokens)
    vx, vt = dense_std.apply(v_dense), token_std.apply(v_tokens)
    weights = initial_weights(x.shape[1], t.shape[1], settings)
    params = [p.copy() for p in _flat(weights)]
    adam = _Adam(settings.adam_betas, settings.adam_eps, settings.learning_rate)
    rng = np.random.default_rng(settings.seed + 1)
    best = (np.inf, 0, [p.copy() for p in params])
    train_loss, val_loss = [], []
    for epoch in range(settings.max_epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), settings.batch_size):
            batch = order[start : start + settings.batch_size]
            current = _weights(params, len(settings.hidden))
            adam.update(
                params, gradients(current, x[batch], t[batch], y[batch], settings.weight_decay)
            )
        current = _weights(params, len(settings.hidden))
        train_loss.append(_loss(current, x, t, y))
        val_loss.append(_loss(current, vx, vt, v_y))
        if val_loss[-1] < best[0]:
            best = (val_loss[-1], epoch, [p.copy() for p in params])
        elif epoch - best[1] >= settings.patience:
            break
    final = _weights(best[2], len(settings.hidden))
    check_id, unit, encoder = identity
    artifact = LearnedArtifact(
        check_id,
        unit,
        samples.dense_columns,
        samples.token_columns,
        final,
        dense_std,
        token_std,
        float(np.sqrt(best[0])),
        encoder,
        training_seeds,
        settings.validation_share,
    )
    log = TrainingLog(tuple(train_loss), tuple(val_loss), best[1], len(y), len(v_y))
    return Trained(artifact, log, held)

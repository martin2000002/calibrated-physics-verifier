"""Flow-DPO objective and the rectified-flow noising it trains on (ADR 0012, 3.8 and 4.4).

torch is imported inside the functions. Conventions match ``FlowMatchEulerDiscreteScheduler``:
``x_t = (1 - sigma) x0 + sigma eps``, velocity target ``v = eps - x0``, timestep ``t = 1000 sigma``,
with the shifted schedule ``sigma = s u / (1 + (s - 1) u)``, ``u ~ U(0, 1)``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

TIMESTEP_SCALE: Final = 1000.0


def _torch() -> Any:
    return importlib.import_module("torch")


@dataclass(frozen=True)
class DpoStats:
    loss: float
    implicit_acc: float
    reward_margin: float
    kl_proxy: float


def per_sample_error(prediction: Any, target: Any) -> Any:
    """Mean squared error per sample in float32, shape ``(B,)``."""
    diff = prediction.float() - target.float()
    return diff.pow(2).flatten(1).mean(dim=1)


def flow_dpo_loss(
    err_w: Any, err_ref_w: Any, err_l: Any, err_ref_l: Any, beta: float, weight: Any = None
) -> tuple[Any, DpoStats]:
    """``inside = -beta/2 [(err_w - err_ref_w) - (err_l - err_ref_l)]``,
    ``loss = mean(-logsigmoid(inside) * weight)``; equal learner and reference give ln 2."""
    torch = _torch()
    delta_w = err_w - err_ref_w
    delta_l = err_l - err_ref_l
    inside = -0.5 * beta * (delta_w - delta_l)
    losses = -torch.nn.functional.logsigmoid(inside)
    if weight is not None:
        losses = losses * weight
    loss = losses.mean()
    with torch.no_grad():
        stats = DpoStats(
            loss=float(loss.detach()),
            implicit_acc=float((inside > 0).float().mean()),
            reward_margin=float(inside.mean()),
            kl_proxy=float((delta_w + delta_l).mean()),
        )
    return loss, stats


def shifted_sigma(u: float | np.ndarray, shift: float) -> Any:
    return shift * u / (1.0 + (shift - 1.0) * u)


@dataclass(frozen=True)
class NoiseDraw:
    sigma: float
    seed: int


@dataclass(frozen=True)
class ShiftedFlowNoise:
    """Deterministic per ``key``: the same key always draws the same ``sigma`` and ``eps``, which
    winner and loser share (different noise per member invites reward hacking)."""

    shift: float
    seed: int = 0

    def draw(self, key: int) -> NoiseDraw:
        state = np.random.SeedSequence([self.seed, key]).generate_state(2)
        u = np.random.default_rng(int(state[0])).random()
        return NoiseDraw(float(shifted_sigma(u, self.shift)), int(state[1]))

    def epsilon(self, draw: NoiseDraw, like: Any) -> Any:
        torch = _torch()
        generator = torch.Generator(device="cpu").manual_seed(draw.seed)
        eps = torch.randn(like.shape, generator=generator, dtype=torch.float32)
        return eps.to(device=like.device, dtype=like.dtype)

    def quantiles(self, levels: Sequence[float]) -> dict[str, float]:
        """sigma is monotone in u, so its quantiles are the shifted quantiles of U(0, 1)."""
        return {f"q{round(100 * q):02d}": float(shifted_sigma(q, self.shift)) for q in levels}


def noised(x0: Any, eps: Any, sigma: float) -> tuple[Any, Any, Any]:
    """``(x_t, v_target, t)`` for a batch sharing one ``sigma``."""
    torch = _torch()
    x_t = (1.0 - sigma) * x0 + sigma * eps
    t = torch.full((x0.shape[0],), TIMESTEP_SCALE * sigma, device=x0.device)
    return x_t, eps - x0, t

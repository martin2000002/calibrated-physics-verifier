from __future__ import annotations

import math

import pytest
import torch

from physverify.finetune.dpo import (
    ShiftedFlowNoise,
    flow_dpo_loss,
    noised,
    per_sample_error,
    shifted_sigma,
)


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_the_loss_matches_a_hand_computation() -> None:
    loss, stats = flow_dpo_loss(_t(0.5), _t(0.7), _t(0.9), _t(0.6), beta=2.0)
    inside = -0.5 * 2.0 * ((0.5 - 0.7) - (0.9 - 0.6))
    assert inside == pytest.approx(0.5)
    assert float(loss) == pytest.approx(math.log1p(math.exp(-inside)), abs=1e-6)
    assert stats.implicit_acc == 1.0
    assert stats.reward_margin == pytest.approx(0.5)
    assert stats.kl_proxy == pytest.approx(0.1)
    weighted, _ = flow_dpo_loss(_t(0.5), _t(0.7), _t(0.9), _t(0.6), 2.0, _t(0.5))
    assert float(weighted) == pytest.approx(0.5 * float(loss), abs=1e-9)


def test_a_learner_equal_to_the_reference_gives_ln2() -> None:
    err = _t(0.3, 1.2, 0.01)
    loss, stats = flow_dpo_loss(err, err.clone(), err * 2, err * 2, beta=500.0)
    assert float(loss) == pytest.approx(math.log(2), abs=1e-9)
    assert stats.implicit_acc == 0.0 and stats.reward_margin == 0.0


def test_the_gradient_lowers_the_winner_error_and_raises_the_loser_error() -> None:
    err_w = _t(0.4).requires_grad_()
    err_l = _t(0.4).requires_grad_()
    loss, _ = flow_dpo_loss(err_w, _t(0.4), err_l, _t(0.4), beta=10.0)
    loss.backward()
    assert err_w.grad is not None and err_l.grad is not None
    assert float(err_w.grad) > 0 > float(err_l.grad)


def test_noising_follows_the_flow_match_conventions() -> None:
    x0, eps = torch.full((2, 3), 2.0), torch.full((2, 3), -1.0)
    x_t, v, t = noised(x0, eps, 0.25)
    assert torch.allclose(x_t, torch.full((2, 3), 0.75 * 2.0 - 0.25))
    assert torch.allclose(v, torch.full((2, 3), -3.0))
    assert torch.allclose(t, torch.full((2,), 250.0))
    assert torch.allclose(per_sample_error(v, torch.zeros(2, 3)), torch.full((2,), 9.0))


def test_the_shifted_schedule_and_its_quantiles() -> None:
    assert shifted_sigma(0.5, 8.0) == pytest.approx(4.0 / 4.5)
    assert shifted_sigma(0.0, 8.0) == 0.0 and shifted_sigma(1.0, 8.0) == 1.0
    noise = ShiftedFlowNoise(8.0)
    q = noise.quantiles((0.1, 0.5, 0.9))
    assert list(q) == ["q10", "q50", "q90"]
    assert q["q50"] == pytest.approx(4.0 / 4.5)
    assert q["q10"] < q["q50"] < q["q90"]


def test_draws_are_keyed_and_deterministic() -> None:
    noise = ShiftedFlowNoise(8.0, seed=3)
    assert noise.draw(7) == ShiftedFlowNoise(8.0, seed=3).draw(7)
    assert noise.draw(7) != noise.draw(8)
    assert 0.0 < noise.draw(7).sigma < 1.0
    like = torch.zeros(1, 4)
    assert torch.equal(noise.epsilon(noise.draw(7), like), noise.epsilon(noise.draw(7), like))

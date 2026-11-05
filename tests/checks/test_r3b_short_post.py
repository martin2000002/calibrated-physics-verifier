import math
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest

import physverify.checks.variants.r3b_image_relspeed as relspeed
from physverify.checks.variants._image_impulse import line_fit
from physverify.checks.variants.r3b_image_relspeed import RelSpeedCheck, RelSpeedParams
from physverify.schema.checks import AbstainReason, InputLevel
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage

TRUTH = {Stage.CONTACTS.value: GROUND_TRUTH, Stage.LIFT.value: GROUND_TRUTH}
G2 = RelSpeedParams(post_min_rows=2)
EXPECTED = 0.5 * math.log1p(0.2)


def only(case: sc.Case, params: RelSpeedParams) -> Any:
    (result,) = RelSpeedCheck(params).evaluate(case.inputs(InputLevel.PERCEIVED, TRUTH))
    return result


def cut_after(rows: int, gain: float = 0.2) -> sc.Case:
    """The clip ends so that ``rows`` rows lie beyond the gap after the impact."""
    return sc.collision(gain=gain, frames=math.floor(sc.EVENT_FRAME + 1.0) + rows + 1)


def noisy(case: sc.Case) -> sc.Case:
    return sc.jittered(case, 1e-3, seed=5)


@pytest.mark.parametrize("rows", [1, 2])
def test_the_default_gate_abstains_on_a_short_after_side(rows: int) -> None:
    assert only(cut_after(rows), RelSpeedParams()).abstain_reason is AbstainReason.TOO_FEW_FRAMES


def test_two_point_reads_two_after_rows_and_refuses_one() -> None:
    result = only(cut_after(2), G2)
    assert result.magnitude == pytest.approx(EXPECTED, abs=1e-12)
    assert result.details["rows_after"] == 2
    assert result.details["short_post_fit"] == "two_point"
    assert only(cut_after(1), G2).abstain_reason is AbstainReason.TOO_FEW_FRAMES


def test_a_noisy_two_row_after_side_has_a_finite_sigma() -> None:
    result = only(noisy(cut_after(2)), G2)
    assert result.magnitude == pytest.approx(EXPECTED, abs=0.1)
    assert 0.0 < result.sigma < 0.5


def test_the_default_is_three_after_rows() -> None:
    assert RelSpeedParams() == RelSpeedParams(post_min_rows=3)


@pytest.mark.parametrize("case", [sc.collision(gain=0.2), noisy(sc.collision(gain=0.2))])
def test_a_full_after_side_is_read_bit_for_bit_as_by_the_default(case: sc.Case) -> None:
    mine, default = only(case, G2), only(case, RelSpeedParams())
    assert (mine.magnitude, mine.sigma, mine.details) == (
        default.magnitude,
        default.sigma,
        default.details,
    )
    assert "short_post_fit" not in mine.details


def _spy_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, list[Any]], dict[str, np.random.Generator]]:
    """Records the generator made for each stream and which sigma read which stream."""
    made: dict[str, np.random.Generator] = {}
    seen: dict[str, list[Any]] = {"boot_sigma": [], "two_point_sigma": []}

    def rng(seed: int, clip_id: str, stream: str) -> np.random.Generator:
        made[stream] = np.random.default_rng(seed)
        return made[stream]

    def spy(name: str) -> Any:
        real = getattr(relspeed, name)

        def wrapped(*args: Any) -> Any:
            seen[name].append(next(k for k, g in made.items() if g is args[3]))
            return real(*args)

        return wrapped

    monkeypatch.setattr(relspeed, "check_rng", rng)
    for name in seen:
        monkeypatch.setattr(relspeed, name, spy(name))
    return seen, made


def test_a_short_after_side_draws_only_from_its_own_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, made = _spy_streams(monkeypatch)
    only(noisy(cut_after(2)), G2)
    assert seen == {"boot_sigma": [], "two_point_sigma": [relspeed.SHORT_STREAM]}
    fresh = np.random.default_rng(G2.seed).bit_generator.state
    assert made[RelSpeedCheck.check_id].bit_generator.state == fresh
    assert made[relspeed.SHORT_STREAM].bit_generator.state != fresh


def test_a_full_after_side_draws_from_the_check_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    seen, _ = _spy_streams(monkeypatch)
    only(noisy(sc.collision(gain=0.2)), G2)
    assert seen == {"boot_sigma": [RelSpeedCheck.check_id], "two_point_sigma": []}
    assert RelSpeedCheck.check_id != relspeed.SHORT_STREAM


def test_two_point_sigma_resamples_the_before_residuals_inflated_by_two_parameters() -> None:
    t = [np.arange(-5.0, -2.0), np.array([3.0, 4.0])]
    y = [
        np.array([[5.0, 0.1], [4.1, -0.2], [2.9, 0.15]]),
        np.array([[1.0, 0.05], [1.9, -0.1]]),
    ]
    params = RelSpeedParams(boot_draws=40)
    sigma, finite = relspeed.two_point_sigma(t, y, 0.0, np.random.default_rng(3), params)

    fits = [line_fit(ti, yi, 0.0) for ti, yi in zip(t, y, strict=True)]
    base = [f.intercept + f.slope * ti[:, None] for f, ti in zip(fits, t, strict=True)]
    pool = (y[0] - base[0]) * math.sqrt(3 / (3 - 2))
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(params.boot_draws):
        vm, vp = (
            line_fit(ti, bi + pool[rng.integers(0, 3, ti.shape[0])], 0.0).slope
            for ti, bi in zip(t, base, strict=True)
        )
        draws.append(relspeed.log_ratio(vm, vp))
    found = np.asarray(draws)[np.isfinite(draws)]
    assert finite == found.size == params.boot_draws
    assert sigma == pytest.approx(float(found.std(ddof=1)), rel=1e-12)


def test_the_params_refuse_an_unreadable_after_side() -> None:
    with pytest.raises(ValueError, match="post_min_rows"):
        RelSpeedParams(post_min_rows=4)
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        RelSpeedParams(post_min_rows=1)

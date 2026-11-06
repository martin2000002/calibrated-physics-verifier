"""R3b image_relspeed: the log ratio of separation to approach speed of a two-body impact, read on
the relative image track, with no depth of either body."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field, model_validator

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._collision_windows import no_contacts
from physverify.checks._params import CommonParams, as_params
from physverify.checks._uncertainty import check_rng
from physverify.checks.r3b_collision_energy_gain import EnergyGainCheck, external_contacts
from physverify.checks.variants._image_impulse import (
    BoolArray,
    Event,
    FloatArray,
    Gates,
    IntArray,
    as_record,
    detect,
    env_records,
    free_rows,
    image_track,
    line_fit,
    truth_events,
    walk,
)
from physverify.checks.variants.event_gates._base import EventGate, build_gates, event_gates
from physverify.checks.variants.event_scopes._base import (
    EventScope,
    ScopeLimits,
    SidedEvent,
    build_scopes,
    event_scopes,
    failed_scope,
)
from physverify.physics.tracks import native_span
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

SHORT_STREAM = "R3b/short_post"


class RelSpeedParams(CommonParams):
    """Rows of each side, the image impulse detector's gates and the residual bootstrap;
    ``same_t_frames`` is how close (native frames) a true record's middle is to an event's t to be
    that event's own record. None of the default R3b's windows or gates applies here; of the
    common fields only ``seed`` is read (``mc_draws`` stays as every check's Params has it).

    ``post_min_rows`` < ``side_min_rows`` admits a short after side (the rows a floor bounce
    leaves), read by ``two_point_sigma``. ``event_gates`` names the gates (``event_gates/``) a
    detected event must pass; ``event_scopes`` the scopes (``event_scopes/``) a measured event
    must lie in, else it is NOT_APPLICABLE; none of either by default."""

    gap_frames: float = Field(default=1.0, ge=0.0)
    bound_margin_rows: int = Field(default=1, ge=0)
    side_min_rows: int = Field(default=3, ge=3)
    side_max_rows: int = Field(default=6, ge=3)
    event_z_min: float = Field(default=5.0, ge=0.0)
    speed_z_min: float = Field(default=2.0, ge=0.0)
    kappa: float = Field(default=1.5, gt=0.0)
    cross_rows: float = Field(default=1.0, ge=0.0)
    merge_rows: int = Field(default=2, ge=0)
    radius_rows: int = Field(default=2, ge=0)
    same_t_frames: float = Field(default=1e-9, gt=0.0)
    boot_draws: int = Field(default=200, ge=2)
    boot_min_finite: float = Field(default=0.5, gt=0.0, le=1.0)
    post_min_rows: int = Field(default=3, ge=2)
    event_gates: tuple[str, ...] = ()
    event_scopes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _post_within_side(self) -> RelSpeedParams:
        if self.post_min_rows > self.side_min_rows:
            raise ValueError("post_min_rows must not exceed side_min_rows")
        return self

    def gates(self) -> Gates:
        return Gates(
            side_min=self.side_min_rows,
            side_max=self.side_max_rows,
            z_min=self.event_z_min,
            speed_z_min=self.speed_z_min,
            kappa=self.kappa,
            cross_rows=self.cross_rows,
            merge_rows=self.merge_rows,
            radius_rows=self.radius_rows,
        )

    def scope_limits(self) -> ScopeLimits:
        return ScopeLimits(
            side_min_rows=self.side_min_rows,
            side_max_rows=self.side_max_rows,
            gap_frames=self.gap_frames,
            bound_margin_rows=self.bound_margin_rows,
        )


def normal_speeds(vm: FloatArray, vp: FloatArray) -> tuple[float, float, FloatArray] | None:
    """(c-, c+, a): side slopes along the unit jump a; None unless approach then separation."""
    jump = vp - vm
    size = float(np.linalg.norm(jump))
    if size == 0.0:
        return None
    a = jump / size
    cm, cp = float(vm @ a), float(vp @ a)
    return (cm, cp, a) if cm < 0.0 < cp else None


def log_ratio(vm: FloatArray, vp: FloatArray) -> float:
    found = normal_speeds(vm, vp)
    return math.nan if found is None else math.log(found[1] / -found[0])


def boot_sigma(
    t: Sequence[FloatArray],
    y: Sequence[FloatArray],
    t0: float,
    rng: np.random.Generator,
    params: RelSpeedParams,
) -> tuple[float, int]:
    """Residual bootstrap of s over both sides, residuals inflated by sqrt(n / (n - 2)); NaN
    when fewer than ``boot_min_finite`` of the draws give a ratio."""
    base = _fitted(t, y, t0)
    resid = [
        (yi - bi) * math.sqrt(yi.shape[0] / (yi.shape[0] - 2))
        for yi, bi in zip(y, base, strict=True)
    ]
    draws = []
    for _ in range(params.boot_draws):
        slopes = [
            line_fit(ti, bi + ri[rng.integers(0, ri.shape[0], ri.shape[0])], t0).slope
            for ti, bi, ri in zip(t, base, resid, strict=True)
        ]
        draws.append(log_ratio(slopes[0], slopes[1]))
    return _spread(draws, params)


def _fitted(t: Sequence[FloatArray], y: Sequence[FloatArray], t0: float) -> list[FloatArray]:
    fits = [line_fit(ti, yi, t0) for ti, yi in zip(t, y, strict=True)]
    return [f.intercept + f.slope * (ti - t0)[:, None] for f, ti in zip(fits, t, strict=True)]


def _spread(draws: Sequence[float], params: RelSpeedParams) -> tuple[float, int]:
    found = np.asarray(draws)[np.isfinite(draws)]
    if found.size < params.boot_min_finite * params.boot_draws:
        return math.nan, int(found.size)
    return float(found.std(ddof=1)), int(found.size)


def two_point_sigma(
    t: Sequence[FloatArray],
    y: Sequence[FloatArray],
    t0: float,
    rng: np.random.Generator,
    params: RelSpeedParams,
) -> tuple[float, int]:
    """``boot_sigma`` with both sides drawing from the before side's inflated residuals: a short
    after side leaves no residual of its own (none with 2 rows), so the noise is taken as
    stationary across the impact."""
    base = _fitted(t, y, t0)
    n = y[0].shape[0]
    pool = (y[0] - base[0]) * math.sqrt(n / (n - 2))
    draws = []
    for _ in range(params.boot_draws):
        slopes = [
            line_fit(ti, bi + pool[rng.integers(0, n, ti.shape[0])], t0).slope
            for ti, bi in zip(t, base, strict=True)
        ]
        draws.append(log_ratio(slopes[0], slopes[1]))
    return _spread(draws, params)


class RelSpeedCheck:
    """R3b image_relspeed [log_normal_speed_ratio]: per impact, s = ln(c+ / -c-).

    Under weak perspective about the pair's mean depth, d' = (f / Z) [I | -u] v_rel = A v_rel for
    the relative image track d = u_a - u_b. The impulse changes v_rel only along the contact
    normal n and leaves its tangential part unchanged, so the slope jump of d lies along A n.
    With n in the image plane, projecting each side's slope on the unit jump gives
    (f / Z) v_rel . n on both sides and c+ / (-c-) is the normal restitution, f / Z cancelled.
    An out-of-plane normal and a depth difference between the bodies break that.

    Impacts are the object-object IMPACT records of ground-truth contacts (by provenance);
    otherwise the image impulse events of the received track, cut at the ENV records. Each side
    is the ``side_max_rows`` rows nearest the event beyond ``gap_frames``, outside the windows of
    the bodies' other records. sigma is the residual bootstrap of s (``boot_sigma``)."""

    check_id: ClassVar[str] = "R3b"
    variant: ClassVar[str] = "image_relspeed"
    version: ClassVar[str] = "2"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    time_dependent: ClassVar[bool] = False
    magnitude_unit: ClassVar[str] = "log_normal_speed_ratio"
    replaced_unit: ClassVar[str] = EnergyGainCheck.magnitude_unit
    Params: ClassVar[type[BaseModel]] = RelSpeedParams

    def __init__(
        self,
        params: BaseModel,
        gate_registry: Mapping[str, type[EventGate]] | None = None,
        scope_registry: Mapping[str, type[EventScope]] | None = None,
    ) -> None:
        self.params = as_params(RelSpeedParams, params)
        registry = event_gates() if gate_registry is None else gate_registry
        self.event_gates = build_gates(self.params.event_gates, registry)
        scopes = event_scopes() if scope_registry is None else scope_registry
        self.event_scopes = build_scopes(self.params.event_scopes, scopes)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        if inputs.contacts is None:
            return no_contacts(type(self), inputs)
        truth = _truth_contacts(inputs)
        found = (
            truth_events(inputs.contacts)
            if truth
            else detect(inputs.states, inputs.contacts, self.params.gates(), self.event_gates)
        )
        clip_id, seed = inputs.clip.clip_id, self.params.seed
        rngs = (check_rng(seed, clip_id, self.check_id), check_rng(seed, clip_id, SHORT_STREAM))
        tol = self.params.same_t_frames
        measured = [
            self._event(inputs, e, _bounds(inputs.contacts, found, e, truth, tol), rngs)
            for e in found
        ]
        return [self._scoped(inputs, found, result, sided) for result, sided in measured]

    def _scoped(
        self, inputs: CheckInputs, found: Sequence[Event], result: CheckResult, sided: SidedEvent
    ) -> CheckResult:
        """``result``, or NOT_APPLICABLE naming the first scope its event lies outside of and the
        predicates of it the event fails. Every event is measured first, so the scopes leave the
        bootstrap streams of the others as they are."""
        if not self.event_scopes:
            return result
        others = [x for x in found if x is not sided.event]
        limits = self.params.scope_limits()
        outside = failed_scope(self.event_scopes, inputs.states, sided, others, limits)
        if outside is None:
            return result
        details = {
            **result.details,
            "not_applicable": outside.scope,
            "not_applicable_predicates": list(outside.predicates),
        }
        reason = AbstainReason.NOT_APPLICABLE
        return abstain(type(self), inputs, result.subject, reason, result.frame_span, details)

    def _event(
        self,
        inputs: CheckInputs,
        e: Event,
        bounds: Sequence[ContactRecord],
        rngs: tuple[np.random.Generator, np.random.Generator],
    ) -> tuple[CheckResult, SidedEvent]:
        """The event's result and its side rows (none when another body touches it). A short
        after side draws from its own stream, so the full-sided events of a clip consume
        ``rngs[0]`` exactly as with the default gate."""
        states, p = inputs.states, self.params
        rng, short_rng = rngs
        subject = (e.a, e.b)
        env = ContactTable(
            states.clip_id,
            states.frame,
            states.unit,
            states.source_key,
            tuple(env_records(inputs.contacts)),
        )
        external = external_contacts(states, env, as_record(e))
        if external:
            scope = {"external_contacts": external, "t_e": e.t}
            reason = AbstainReason.OUT_OF_SCOPE
            return abstain(type(self), inputs, subject, reason, None, scope), SidedEvent(e, (), ())
        ia, ib = states.object_index(e.a), states.object_index(e.b)
        d = image_track(states, ia) - image_track(states, ib)
        fi = np.asarray(states.frame_index, dtype=np.int64)
        margin = p.bound_margin_rows
        free = free_rows(fi, bounds, e.a, margin) & free_rows(fi, bounds, e.b, margin)
        ok = np.all(np.isfinite(d), axis=1) & free
        left, right = self._side_rows(fi, ok, e.t)
        sided = SidedEvent(e, tuple(left), tuple(right))
        details: dict[str, Any] = {
            "t_e": e.t,
            "rows_before": len(left),
            "rows_after": len(right),
            "event_z": e.z,
        }
        if len(left) < p.side_min_rows or len(right) < p.post_min_rows:
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, subject, reason, None, details), sided
        span = native_span(fi, left[0], right[-1])
        t = [fi[left].astype(np.float64), fi[right].astype(np.float64)]
        y = [d[left], d[right]]
        vm, vp = (line_fit(ti, yi, e.t).slope for ti, yi in zip(t, y, strict=True))
        speeds = normal_speeds(vm, vp)
        details |= {"s_full": _full(vm, vp)}
        short = len(right) < p.side_min_rows
        if short:
            details |= {"short_post_fit": "two_point"}
        if speeds is None:
            reason = AbstainReason.ILL_CONDITIONED
            return abstain(type(self), inputs, subject, reason, span, details), sided
        cm, cp, a = speeds
        if short:
            sigma, finite = two_point_sigma(t, y, e.t, short_rng, p)
        else:
            sigma, finite = boot_sigma(t, y, e.t, rng, p)
        details |= {"c_minus": cm, "c_plus": cp, "jump_dir": a.tolist(), "boot_finite": finite}
        if _truth_contacts(inputs) and _truth_lift(inputs) and e.normal is not None:
            details |= depth_view(states, (ia, ib), (left, right), e)
        magnitude = math.log(cp / -cm)
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details), sided

    def _side_rows(self, fi: IntArray, ok: BoolArray, t_e: float) -> tuple[list[int], list[int]]:
        """Rows nearest the event with |f - t_e| > ``gap_frames``, walked outwards while ``ok``."""
        gap, limit = self.params.gap_frames, self.params.side_max_rows
        before = np.flatnonzero(fi < t_e - gap)
        after = np.flatnonzero(fi > t_e + gap)
        left = walk(ok, int(before[-1]), -1, limit) if before.size else []
        right = walk(ok, int(after[0]), 1, limit) if after.size else []
        return left, right


def _truth_contacts(inputs: CheckInputs) -> bool:
    return inputs.provenance.get(Stage.CONTACTS.value) == GROUND_TRUTH


def _truth_lift(inputs: CheckInputs) -> bool:
    return inputs.provenance.get(Stage.LIFT.value) == GROUND_TRUTH


def _bounds(
    contacts: ContactTable, found: Sequence[Event], e: Event, truth: bool, tol: float
) -> list[ContactRecord]:
    """Every record but the event's own: the true table, or the ENV records plus the events."""
    if truth:
        return [r for r in contacts.records if r.object_b == ENV_ID or not _same(r, e, tol)]
    return env_records(contacts) + [as_record(x) for x in found if x is not e]


def _same(r: ContactRecord, e: Event, tol: float) -> bool:
    pair = sorted((r.object_a, r.object_b)) == [e.a, e.b]
    return pair and abs(0.5 * (r.t_start + r.t_end) - e.t) < tol


def _full(vm: FloatArray, vp: FloatArray) -> float:
    nm, nps = float(np.linalg.norm(vm)), float(np.linalg.norm(vp))
    return math.log(nps / nm) if nm > 0.0 and nps > 0.0 else math.nan


def depth_view(
    states: ObjectStateTable,
    indices: tuple[int, int],
    rows: tuple[list[int], list[int]],
    e: Event,
) -> dict[str, float]:
    """Ground truth only: the same ratio from the 3D relative positions along the true normal,
    and the normal's tilt out of the image plane, phi = asin |n . r|, r the unit ray to the
    contact."""
    assert e.normal is not None and e.point is not None
    fi = np.asarray(states.frame_index, dtype=np.float64)
    rel = np.asarray(states.position[:, indices[0]] - states.position[:, indices[1]])
    n = e.normal / np.linalg.norm(e.normal)
    vm, vp = (line_fit(fi[r], rel[r], e.t).slope for r in rows)
    cm, cp = float(vm @ n), float(vp @ n)
    s3 = math.log(cp / -cm) if cm < 0.0 < cp else math.nan
    ray = e.point / np.linalg.norm(e.point)
    return {"s_3d": s3, "phi_deg": math.degrees(math.asin(min(1.0, abs(float(n @ ray)))))}


PLUGINS = (RelSpeedCheck,)

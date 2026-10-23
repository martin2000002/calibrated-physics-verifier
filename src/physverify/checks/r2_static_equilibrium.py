"""R2 static equilibrium of resting bodies, one linear program per component (ADR 0006 3.4)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._params import CommonParams
from physverify.checks._results import abstain, unscored
from physverify.checks._uncertainty import check_rng, mc_sigma
from physverify.physics.equilibrium import (
    EquilibriumProblem,
    EquilibriumSettings,
    SupportContact,
    horizontal_norm_bias,
    solve_equilibrium,
)
from physverify.physics.measures.support_overhang import overhang_fraction
from physverify.physics.tracks import (
    body_length,
    contact_rows,
    native_span,
    rest_mask,
    segments,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class EquilibriumParams(CommonParams):
    mc_draws: int = Field(default=64, ge=2)
    friction: float = Field(default=0.6, ge=0.0)
    cone_edges: int = Field(default=4, ge=3)
    mass_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    rest_speed: float = Field(default=0.1, gt=0.0)
    min_rest_frames: int = Field(default=3, ge=1)
    agent_labels: tuple[str, ...] = ("person", "hand", "arm", "animal", "dog", "cat")
    norm_edges: int = Field(default=8, ge=3)


@dataclass(frozen=True)
class _Event:
    """One component of resting bodies over a run of rows with a constant resting set."""

    start: int
    stop: int
    bodies: tuple[int, ...]  # table indices, sorted by object id
    support: tuple[ContactRecord, ...]  # RESTING records inside the component or with ENV
    excluded: tuple[ContactRecord, ...]  # other records touching the component at the row

    @property
    def row(self) -> int:
        return (self.start + self.stop - 1) // 2


class EquilibriumCheck:
    """R2 [residual_over_weight]: minimal force and torque imbalance over the total weight."""

    check_id: ClassVar[str] = "R2"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    magnitude_unit: ClassVar[str] = "residual_over_weight"
    Params: ClassVar[type[BaseModel]] = EquilibriumParams

    def __init__(self, params: BaseModel) -> None:
        self.params = (
            params
            if isinstance(params, EquilibriumParams)
            else EquilibriumParams.model_validate(params.model_dump(exclude_unset=True))
        )

    @property
    def settings(self) -> EquilibriumSettings:
        p = self.params
        return EquilibriumSettings(p.friction, p.cone_edges, p.mass_floor, p.norm_edges)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        states, contacts = inputs.states, inputs.contacts
        if contacts is None:
            subject = tuple(sorted(str(i) for i in states.object_ids))
            details = {"missing": "contacts"}
            return [
                abstain(type(self), inputs, subject, AbstainReason.MISSING_INPUT, None, details)
            ]
        resting = self._resting(states)
        return [
            self._event(inputs, event)
            for start, stop in _runs(resting)
            for event in _components(states, contacts, resting, start, stop)
        ]

    def _resting(self, states: ObjectStateTable) -> BoolArray:
        """Rows x bodies at rest for at least ``min_rest_frames`` consecutive rows."""
        resting = np.zeros(states.visible.shape, dtype=np.bool_)
        for i in range(states.n_objects):
            mask = rest_mask(states, i, self.params.rest_speed)
            for start, stop in segments(mask, self.params.min_rest_frames):
                resting[start:stop, i] = True
        return resting

    def _event(self, inputs: CheckInputs, event: _Event) -> CheckResult:
        states = inputs.states
        fi = states.frame_index
        subject = tuple(str(states.object_ids[i]) for i in event.bodies)
        span = native_span(fi, event.start, event.stop - 1)
        details: dict[str, Any] = {
            "rest_rows": [event.start, event.stop - 1],
            "row": event.row,
            "frame": int(fi[event.row]),
            "excluded_contacts": [_describe(r) for r in event.excluded],
        }
        reason = self._abstain_reason(states, event, details)
        if reason is not None:
            return abstain(type(self), inputs, subject, reason, span, details)
        problem = _problem(states, event)
        solution = solve_equilibrium(problem, self.settings)
        if solution.status != 0:
            details |= {"lp_status": solution.status, "lp_message": solution.message}
            return abstain(
                type(self), inputs, subject, AbstainReason.ILL_CONDITIONED, span, details
            )
        details |= {
            "masses": dict(zip(subject, solution.masses.tolist(), strict=True)),
            "body_residuals": dict(zip(subject, solution.body_residuals.tolist(), strict=True)),
            "overhang": _overhangs(states, event, subject),
            "norm_edges": self.params.norm_edges,
            "horizontal_norm_bias": horizontal_norm_bias(self.params.norm_edges),
        }
        sigma = self._sigma(inputs, event, problem)
        return unscored(type(self), inputs, subject, solution.residual, sigma, span, details)

    def _abstain_reason(
        self, states: ObjectStateTable, event: _Event, details: dict[str, Any]
    ) -> AbstainReason | None:
        if not bool(np.all(np.isfinite(states.gravity_dir))):
            details["missing"] = "gravity_dir"
            return AbstainReason.MISSING_INPUT
        centroids = states.position[event.row, list(event.bodies)]
        if not bool(np.all(np.isfinite(centroids))):
            details["missing"] = "position"
            return AbstainReason.MISSING_INPUT
        records = event.support + event.excluded
        involved = {str(states.object_ids[i]) for i in event.bodies}
        involved |= {r.object_a for r in records} | {r.object_b for r in records}
        labels = {str(states.labels[states.object_index(o)]) for o in involved - {ENV_ID}}
        agents = sorted(labels & set(self.params.agent_labels))
        if agents:
            details["agents"] = agents
            return AbstainReason.OUT_OF_SCOPE
        if any(r.manifold is ManifoldKind.POINT for r in event.support):
            return AbstainReason.CONTACT_GEOMETRY_UNKNOWN
        return None

    def _sigma(self, inputs: CheckInputs, event: _Event, problem: EquilibriumProblem) -> float:
        """Monte Carlo over the centroids and the contact points (sigma of object a)."""
        states = inputs.states
        sigmas = {"centroids": states.position_sigma[event.row, list(event.bodies)]}
        for k, record in enumerate(event.support):
            index = states.object_index(record.object_a)
            point_sigma = states.position_sigma[event.row, index]
            sigmas[f"points{k}"] = np.broadcast_to(point_sigma, record.points.shape).copy()
        if not all(bool(np.all(np.isfinite(s))) for s in sigmas.values()):
            return math.nan
        settings = self.settings

        def model(noise: Mapping[str, FloatArray]) -> FloatArray:
            draws = noise["centroids"].shape[0]
            return np.array(
                [
                    solve_equilibrium(_perturbed(problem, noise, d), settings).residual
                    for d in range(draws)
                ]
            )

        rng = check_rng(self.params.seed, inputs.clip.clip_id, self.check_id)
        return mc_sigma(model, sigmas, self.params.mc_draws, rng)


def _runs(resting: BoolArray) -> list[tuple[int, int]]:
    """Maximal [start, stop) row runs with a constant, non-empty set of resting bodies."""
    runs: list[tuple[int, int]] = []
    start = 0
    for row in range(1, resting.shape[0] + 1):
        if row == resting.shape[0] or not np.array_equal(resting[row], resting[start]):
            if bool(resting[start].any()):
                runs.append((start, row))
            start = row
    return runs


def _components(
    states: ObjectStateTable,
    contacts: ContactTable,
    resting: BoolArray,
    start: int,
    stop: int,
) -> list[_Event]:
    """Connected groups of resting bodies linked by RESTING records active at the central row."""
    row = (start + stop - 1) // 2
    fi = states.frame_index
    members = {str(states.object_ids[i]) for i in np.flatnonzero(resting[start])}
    active = [r for r in contacts.records if bool(contact_rows(r, fi)[row])]
    links = [r for r in active if _is_support(r, members)]
    parent = {m: m for m in members}

    def root(node: str) -> str:
        while parent[node] != node:
            node = parent[node]
        return node

    for record in links:
        if record.object_b != ENV_ID:
            parent[root(record.object_a)] = root(record.object_b)
    groups: dict[str, list[str]] = {}
    for member in sorted(members):
        groups.setdefault(root(member), []).append(member)
    return [_group_event(states, active, links, group, start, stop) for group in groups.values()]


def _is_support(record: ContactRecord, members: set[str]) -> bool:
    ends_resting = record.object_a in members and record.object_b in members | {ENV_ID}
    return record.kind is ContactKind.RESTING and ends_resting


def _group_event(
    states: ObjectStateTable,
    active: Sequence[ContactRecord],
    links: Sequence[ContactRecord],
    group: list[str],
    start: int,
    stop: int,
) -> _Event:
    names = set(group)
    support = tuple(r for r in links if r.object_a in names)
    used = {id(r) for r in support}
    excluded = tuple(r for r in active if id(r) not in used and names & {r.object_a, r.object_b})
    bodies = tuple(states.object_index(name) for name in group)
    return _Event(start, stop, bodies, support, excluded)


def _problem(states: ObjectStateTable, event: _Event) -> EquilibriumProblem:
    local = {str(states.object_ids[i]): k for k, i in enumerate(event.bodies)}
    contacts = tuple(
        SupportContact(
            body_a=local[r.object_a],
            body_b=None if r.object_b == ENV_ID else local[r.object_b],
            points=np.asarray(r.points, dtype=np.float64),
            normal=np.asarray(r.normal, dtype=np.float64),
        )
        for r in event.support
    )
    return EquilibriumProblem(
        centroids=np.asarray(states.position[event.row, list(event.bodies)], dtype=np.float64),
        lengths=np.array([body_length(states, i) for i in event.bodies]),
        gravity_dir=np.asarray(states.gravity_dir, dtype=np.float64),
        contacts=contacts,
    )


def _perturbed(
    problem: EquilibriumProblem, noise: Mapping[str, FloatArray], draw: int
) -> EquilibriumProblem:
    contacts = tuple(
        SupportContact(c.body_a, c.body_b, c.points + noise[f"points{k}"][draw], c.normal)
        for k, c in enumerate(problem.contacts)
    )
    centroids = problem.centroids + noise["centroids"][draw]
    return EquilibriumProblem(centroids, problem.lengths, problem.gravity_dir, contacts)


def _overhangs(
    states: ObjectStateTable, event: _Event, subject: tuple[str, ...]
) -> dict[str, float | None]:
    """Signed CM overhang of each body over the points of the contacts holding it from below."""
    gravity = np.asarray(states.gravity_dir, dtype=np.float64)
    result: dict[str, float | None] = {}
    for name, index in zip(subject, event.bodies, strict=True):
        points = [
            r.points
            for r in event.support
            if (r.object_a == name and float(r.normal @ gravity) < 0.0)
            or (r.object_b == name and float(r.normal @ gravity) > 0.0)
        ]
        size = states.size[:, index]
        known = np.all(np.isfinite(size), axis=-1)
        if not points or not bool(known.any()):
            result[name] = None
            continue
        com = states.position[event.row, index]
        extents = np.median(size[known], axis=0)
        result[name] = overhang_fraction(com, np.concatenate(points), gravity, extents)
    return result


def _describe(record: ContactRecord) -> str:
    return f"{record.object_a}|{record.object_b}|{record.kind.value}"


PLUGINS = (EquilibriumCheck,)

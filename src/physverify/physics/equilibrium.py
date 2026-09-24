"""Static equilibrium of resting bodies as a linear program (ADR 0006 section 3.1).

Unknowns: masses m_i, friction-cone coefficients lambda_klp >= 0 and per-body slacks. The
equations of body i are written in the gravity basis (u, v, g) with (u, v) = plane_basis(g):

    force:  sum_kl s_ik f_kl + m_i g = r_F,i
    torque: sum_kl s_ik (p_kl - c_i) x f_kl / L_i = r_T,i

with s_ik = +1 on ``body_a`` and -1 on ``body_b`` (third law), unit total weight (g = 1) and the
weight acting at the centroid. The vertical parts of r_F and r_T are penalized exactly
(|r| = s+ + s-); the horizontal parts with the polygonal norm n_K(r) = max_j e_j . r over K
directions e_j = (cos 2 pi j / K, sin 2 pi j / K), which keeps the program linear and satisfies
cos(pi / K) |r| <= n_K(r) <= |r|. Splitting along gravity makes a floating body cost exactly its
weight in any camera frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from physverify.physics.geometry import as_vector, plane_basis

FloatArray = NDArray[np.float64]

# Slack layout per body: vertical force (+, -), horizontal force bound, vertical torque (+, -),
# horizontal torque bound.
_SLACKS = 6
NON_FINITE_STATUS = 4


@dataclass(frozen=True, eq=False)
class SupportContact:
    body_a: int
    body_b: int | None  # None = environment, which has no equation
    points: FloatArray  # (K, 3)
    normal: FloatArray  # (3,), from b towards a


@dataclass(frozen=True, eq=False)
class EquilibriumProblem:
    centroids: FloatArray  # (N, 3)
    lengths: FloatArray  # (N,) L_i, normalizes the torque
    gravity_dir: FloatArray  # (3,)
    contacts: tuple[SupportContact, ...]


@dataclass(frozen=True)
class EquilibriumSettings:
    friction: float
    cone_edges: int
    mass_floor: float
    norm_edges: int = 8


@dataclass(frozen=True, eq=False)
class EquilibriumSolution:
    residual: float  # sum of slacks, total weight = 1
    masses: FloatArray  # (N,)
    body_residuals: FloatArray  # (N,)
    status: int
    message: str


def horizontal_norm_bias(norm_edges: int) -> float:
    """Worst relative underestimate of a horizontal residual by the K-direction norm."""
    return 1.0 / math.cos(math.pi / norm_edges) - 1.0


def cone_edges(normal: ArrayLike, friction: float, n_edges: int) -> FloatArray:
    """Edges d_p = n + mu (cos t_p u + sin t_p v), t_p = 2 pi p / P, of the linearized cone."""
    n = as_vector(normal)
    n = n / np.linalg.norm(n)
    u, v = plane_basis(n)
    theta = 2.0 * np.pi * np.arange(n_edges) / n_edges
    tangents = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    edges: FloatArray = n + friction * tangents
    return edges


def solve_equilibrium(
    problem: EquilibriumProblem, settings: EquilibriumSettings
) -> EquilibriumSolution:
    n_bodies = problem.centroids.shape[0]
    if not _finite(problem):
        return _failed(n_bodies, NON_FINITE_STATUS, "non-finite problem data")
    gravity = problem.gravity_dir / np.linalg.norm(problem.gravity_dir)
    basis = np.stack([*plane_basis(gravity), gravity])
    force, torque = _columns(problem, settings, basis)
    n_lambda = force.shape[2]
    n_vars = n_bodies + n_lambda + _SLACKS * n_bodies
    slack0 = n_bodies + n_lambda
    a_eq, b_eq = _equalities(force, torque, n_vars, slack0)
    a_ub = _norm_bounds(force, torque, n_vars, slack0, settings.norm_edges)
    cost = np.zeros(n_vars)
    cost[slack0:] = 1.0
    bounds = [(settings.mass_floor / n_bodies, None)] * n_bodies + [(0.0, None)] * (
        n_vars - n_bodies
    )
    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.zeros(a_ub.shape[0]),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if result.status != 0:
        return _failed(n_bodies, int(result.status), str(result.message))
    x = np.asarray(result.x, dtype=np.float64)
    per_body = x[slack0:].reshape(n_bodies, _SLACKS).sum(axis=1)
    return EquilibriumSolution(
        residual=float(per_body.sum()),
        masses=x[:n_bodies].copy(),
        body_residuals=per_body,
        status=0,
        message=str(result.message),
    )


def _finite(problem: EquilibriumProblem) -> bool:
    arrays = [problem.centroids, problem.lengths, problem.gravity_dir]
    arrays += [c.points for c in problem.contacts] + [c.normal for c in problem.contacts]
    finite = all(bool(np.all(np.isfinite(a))) for a in arrays)
    return finite and bool(np.all(problem.lengths > 0)) and problem.centroids.shape[0] > 0


def _failed(n_bodies: int, status: int, message: str) -> EquilibriumSolution:
    nan = np.full(n_bodies, math.nan)
    return EquilibriumSolution(math.nan, nan, nan.copy(), status, message)


def _columns(
    problem: EquilibriumProblem, settings: EquilibriumSettings, basis: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Per-body force and torque (in the gravity basis) of every cone coefficient: (N, 3, M)."""
    n_bodies = problem.centroids.shape[0]
    force_blocks: list[FloatArray] = []
    torque_blocks: list[FloatArray] = []
    for contact in problem.contacts:
        edges = cone_edges(contact.normal, settings.friction, settings.cone_edges)
        for point in np.asarray(contact.points, dtype=np.float64).reshape(-1, 3):
            f = np.zeros((n_bodies, 3, edges.shape[0]))
            tau = np.zeros((n_bodies, 3, edges.shape[0]))
            for body, sign in ((contact.body_a, 1.0), (contact.body_b, -1.0)):
                if body is None:
                    continue
                arm = point - problem.centroids[body]
                f[body] = sign * basis @ edges.T
                tau[body] = sign * basis @ np.cross(arm, edges).T / problem.lengths[body]
            force_blocks.append(f)
            torque_blocks.append(tau)
    if not force_blocks:
        empty = np.zeros((n_bodies, 3, 0))
        return empty, empty.copy()
    return np.concatenate(force_blocks, axis=2), np.concatenate(torque_blocks, axis=2)


def _equalities(
    force: FloatArray, torque: FloatArray, n_vars: int, slack0: int
) -> tuple[FloatArray, FloatArray]:
    """Vertical force and torque of each body equal their +/- slacks; masses sum to 1."""
    n_bodies, _, n_lambda = force.shape
    rows = np.zeros((2 * n_bodies + 1, n_vars))
    lam = slice(n_bodies, n_bodies + n_lambda)
    for i in range(n_bodies):
        base = slack0 + _SLACKS * i
        rows[2 * i, i] = 1.0  # weight m_i g, vertical component = m_i
        rows[2 * i, lam] = force[i, 2]
        rows[2 * i, base : base + 2] = (-1.0, 1.0)
        rows[2 * i + 1, lam] = torque[i, 2]
        rows[2 * i + 1, base + 3 : base + 5] = (-1.0, 1.0)
    rows[-1, :n_bodies] = 1.0
    rhs = np.zeros(rows.shape[0])
    rhs[-1] = 1.0
    return rows, rhs


def _norm_bounds(
    force: FloatArray, torque: FloatArray, n_vars: int, slack0: int, norm_edges: int
) -> FloatArray:
    """e_j . (r_u, r_v) - t <= 0 for the horizontal force and torque of each body."""
    n_bodies, _, n_lambda = force.shape
    theta = 2.0 * np.pi * np.arange(norm_edges) / norm_edges
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    lam = slice(n_bodies, n_bodies + n_lambda)
    blocks = []
    for i in range(n_bodies):
        base = slack0 + _SLACKS * i
        for columns, bound in ((force[i, :2], base + 2), (torque[i, :2], base + 5)):
            block = np.zeros((norm_edges, n_vars))
            block[:, lam] = directions @ columns
            block[:, bound] = -1.0
            blocks.append(block)
    return np.vstack(blocks)


"""Shared helpers for Kinova Cartesian MPC examples."""

import numpy as np
import pinocchio as pin

from splineopt.bspline import BSpline

import scipy.sparse as sp
from scipy.optimize import linprog

from examples.utils.kinova_helpers import (
    Q_MIN, Q_MAX, V_MIN, V_MAX, A_MIN, A_MAX,
)

DURATION = 10.0

START = np.array([
    0.00, 0.45, 0.15, -1.35, -0.15, 0.75, 0.10,
])

A_REF = np.array([
    0.65, 0.60, -0.55, 0.70, 0.80, -0.45, 0.75,
])

B_REF = np.array([
    -0.25, 0.25, 0.30, -0.22, -0.35, 0.28, -0.30,
])


def make_spline(horizon, ndof, n_control_points):
    """Create a cubic, open-clamped MPC B-spline."""
    degree = 3

    inner = np.linspace(
        0.0,
        horizon,
        n_control_points - degree + 1,
    )[1:-1]

    knots = np.r_[
        np.zeros(degree + 1),
        inner,
        np.full(degree + 1, horizon),
    ]

    return BSpline(
        knots=knots,
        degree=degree,
        n_dof=ndof,
    )


def pose_from_theta(robot, theta):
    """End-effector pose from joint coordinates."""
    q = robot.joint_positions_to_configuration(theta)
    return robot.frame_pose(q)


def pose_errors(actual, desired):
    """Position error [m] and orientation error [rad]."""
    position_error = np.linalg.norm(
        actual.translation - desired.translation
    )

    orientation_error = np.linalg.norm(
        pin.log3(actual.rotation.T @ desired.rotation)
    )

    return float(position_error), float(orientation_error)


def enforce_initial_state_on_guess(
    spline, control_points, q0, v0
):
    """Adjust the first two control points to enforce q(0), v(0).

    Requires an open-clamped B-spline. The remaining control
    points are unchanged; feasibility is enforced by the solver.
    """
    c = np.asarray(control_points, dtype=float).reshape(
        spline.n_control_points,
        spline.n_dof,
    ).copy()

    p = spline.degree
    h = spline.knots[p + 1] - spline.knots[0]

    if h <= 0:
        raise ValueError(
            "Expected a positive first B-spline segment"
        )

    c[0] = q0
    c[1] = q0 + (h / p) * v0

    return c


def max_linear_violation(problem, control_points):
    """Maximum violation of the linear MPC constraints."""
    A, lower, upper = problem.linear_constraints()

    values = np.asarray(
        A @ np.asarray(control_points).reshape(-1)
    ).reshape(-1)

    return float(max(
        np.max(lower - values),
        np.max(values - upper),
        0.0,
    ))


# ============================================================
# Feasible Cartesian MPC reference
# ============================================================

def reference_state(t):
    """Analytically bounded two-harmonic joint trajectory."""
    t = float(np.clip(t, 0.0, DURATION))
    omega = 2.0 * np.pi / DURATION

    w1 = omega * t
    w2 = 2.0 * omega * t

    q = (
        START
        + 0.5 * A_REF * (1.0 - np.cos(w1))
        + 0.5 * B_REF * (1.0 - np.cos(w2))
    )

    v = (
        0.5 * omega * A_REF * np.sin(w1)
        + omega * B_REF * np.sin(w2)
    )

    a = (
        0.5 * omega**2 * A_REF * np.cos(w1)
        + 2.0 * omega**2 * B_REF * np.cos(w2)
    )

    return q, v, a


def reference_pose(robot, t):
    """Cartesian pose corresponding to the joint reference."""
    q, _, _ = reference_state(t)
    return pose_from_theta(robot, q)


# ============================================================
# Feasibility diagnostics
# ============================================================

def bound_violation(values, lower, upper):
    """Maximum violation of lower and upper bounds."""
    values = np.asarray(values)

    return float(max(
        0.0,
        np.max(lower - values),
        np.max(values - upper),
    ))


def validate_reference():
    """Check analytical and sampled feasibility of the reference."""
    omega = 2.0 * np.pi / DURATION

    # Conservative analytical bounds
    q_lower = (
        START
        + np.minimum(A_REF, 0.0)
        + np.minimum(B_REF, 0.0)
    )
    q_upper = (
        START
        + np.maximum(A_REF, 0.0)
        + np.maximum(B_REF, 0.0)
    )

    v_bound = omega * (
        0.5 * np.abs(A_REF) + np.abs(B_REF)
    )
    a_bound = omega**2 * (
        0.5 * np.abs(A_REF) + 2.0 * np.abs(B_REF)
    )

    analytical_ok = (
        np.all(q_lower >= Q_MIN)
        and np.all(q_upper <= Q_MAX)
        and np.all(v_bound <= V_MAX)
        and np.all(a_bound <= A_MAX)
    )

    # Dense numerical verification
    times = np.linspace(0.0, DURATION, 2001)
    states = [reference_state(t) for t in times]

    q = np.array([state[0] for state in states])
    v = np.array([state[1] for state in states])
    a = np.array([state[2] for state in states])

    violations = {
        "q": bound_violation(q, Q_MIN, Q_MAX),
        "v": bound_violation(v, V_MIN, V_MAX),
        "a": bound_violation(a, A_MIN, A_MAX),
    }

    print("\n" + "=" * 70)
    print("REFERENCE FEASIBILITY")
    print("=" * 70)
    print(f"Duration: {DURATION:.2f} s")
    print(
        "Velocity usage (analytical bound): "
        f"{np.max(v_bound / V_MAX):.3f}"
    )
    print(
        "Acceleration usage (analytical bound): "
        f"{np.max(a_bound / A_MAX):.3f}"
    )
    print(
        f"Violations: q={violations['q']:.3e}, "
        f"v={violations['v']:.3e}, "
        f"a={violations['a']:.3e}"
    )

    if not analytical_ok:
        raise RuntimeError("Analytical reference is not feasible.")

    if max(violations.values()) > 1e-10:
        raise RuntimeError("Reference feasibility verification failed.")

    print("Reference is analytically feasible.")
    print("=" * 70)


def check_mpc_feasibility(problem):
    """Check linear MPC feasibility using HiGHS (no tracking cost).

    This diagnostic is excluded from MPC solve-time measurements.
    """
    A, lower, upper = problem.linear_constraints()
    A = sp.csr_matrix(A)

    equality = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isclose(lower, upper, atol=1e-12, rtol=0)
    )

    upper_mask = np.isfinite(upper) & ~equality
    lower_mask = np.isfinite(lower) & ~equality

    A_ub = sp.vstack([
        A[upper_mask],
        -A[lower_mask],
    ], format="csr")

    b_ub = np.r_[
        upper[upper_mask],
        -lower[lower_mask],
    ]

    return linprog(
        c=np.zeros(A.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A[equality],
        b_eq=lower[equality],
        bounds=[(None, None)] * A.shape[1],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8,
        },
    )

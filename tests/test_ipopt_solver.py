import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from splineopt.ipopt_solver import (
    IpoptSolver,
)


CYIPOPT_AVAILABLE = (
    importlib.util.find_spec(
        "cyipopt"
    )
    is not None
)


requires_ipopt = pytest.mark.skipif(
    not CYIPOPT_AVAILABLE,
    reason="CyIpopt is not installed.",
)


class DummySpline:
    """Minimal spline description required by the solver."""

    n_control_points = 1
    n_dof = 2


class QuadraticConstrainedProblem:
    """
    Solve

        minimize

            (x - 1)^2
            + (y - 2)^2

        subject to

            x + y = 3

    Exact optimum:

        x = 1
        y = 2
    """

    def __init__(
        self,
    ):
        self.spline = DummySpline()

    # =========================================================
    # Objective
    # =========================================================

    def objective(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        x = c[0]
        y = c[1]

        return float(
            (x - 1.0) ** 2
            + (y - 2.0) ** 2
        )

    def gradient(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        x = c[0]
        y = c[1]

        return np.array([
            2.0 * (x - 1.0),
            2.0 * (y - 2.0),
        ])

    # =========================================================
    # Constraints
    # =========================================================

    def linear_constraints(
        self,
    ):
        A = sp.csc_matrix(
            np.array([
                [1.0, 1.0]
            ])
        )

        # Equality:
        #
        #     x + y = 3
        #
        l = np.array([
            3.0
        ])

        u = np.array([
            3.0
        ])

        return A, l, u

    # =========================================================
    # Metrics
    # =========================================================

    def error_metrics(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        error = np.linalg.norm(
            c
            - np.array([
                1.0,
                2.0,
            ])
        )

        return {
            "position_rms": error,
            "position_max": error,
            "orientation_rms": 0.0,
            "orientation_max": 0.0,
        }


# ============================================================
# Parameter validation
# ============================================================


def test_ipopt_rejects_negative_tolerance():

    with pytest.raises(
        ValueError,
        match="tolerance",
    ):
        IpoptSolver(
            tolerance=-1e-6,
        )


def test_ipopt_rejects_zero_iterations():

    with pytest.raises(
        ValueError,
        match="max_iterations",
    ):
        IpoptSolver(
            max_iterations=0,
        )


# ============================================================
# Basic solve
# ============================================================


@requires_ipopt
def test_ipopt_solves_constrained_quadratic():

    problem = (
        QuadraticConstrainedProblem()
    )

    solver = IpoptSolver(
        tolerance=1e-9,
        max_iterations=100,
    )

    initial = np.array([
        [0.0, 3.0]
    ])

    result = solver.solve(
        problem=problem,
        initial_control_points=initial,
    )

    assert result.success

    np.testing.assert_allclose(
        result.control_points,
        np.array([
            [1.0, 2.0]
        ]),
        atol=1e-6,
    )


# ============================================================
# Objective
# ============================================================


@requires_ipopt
def test_ipopt_reduces_objective():

    problem = (
        QuadraticConstrainedProblem()
    )

    solver = IpoptSolver(
        tolerance=1e-9,
    )

    initial = np.array([
        [0.0, 3.0]
    ])

    initial_cost = (
        problem.objective(
            initial
        )
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=initial,
    )

    assert (
        result.objective
        < initial_cost
    )

    assert (
        result.objective
        < 1e-10
    )


# ============================================================
# Constraint satisfaction
# ============================================================


@requires_ipopt
def test_ipopt_solution_satisfies_constraints():

    problem = (
        QuadraticConstrainedProblem()
    )

    solver = IpoptSolver(
        tolerance=1e-9,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=np.array([
            [0.0, 3.0]
        ]),
    )

    c = (
        result.control_points
        .reshape(-1)
    )

    A, l, u = (
        problem.linear_constraints()
    )

    constraint_value = np.asarray(
        A @ c
    ).reshape(-1)

    np.testing.assert_allclose(
        constraint_value,
        l,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        constraint_value,
        u,
        atol=1e-7,
    )


# ============================================================
# Result structure
# ============================================================


@requires_ipopt
def test_ipopt_returns_valid_result_information():

    problem = (
        QuadraticConstrainedProblem()
    )

    solver = IpoptSolver(
        tolerance=1e-9,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=np.array([
            [0.0, 3.0]
        ]),
    )

    assert result.status in {
        0,
        1,
    }

    assert isinstance(
        result.status_message,
        str,
    )

    assert result.solve_time >= 0.0

    assert result.iterations >= 0

    assert isinstance(
        result.raw_info,
        dict,
    )


# ============================================================
# Initial point validation
# ============================================================


@requires_ipopt
def test_ipopt_rejects_wrong_initial_size():

    problem = (
        QuadraticConstrainedProblem()
    )

    solver = IpoptSolver()

    wrong_initial = np.array([
        [0.0, 1.0, 2.0]
    ])

    with pytest.raises(
        ValueError,
        match="expected",
    ):
        solver.solve(
            problem=problem,
            initial_control_points=(
                wrong_initial
            ),
        )


def finite_difference_gradient(
    function,
    x,
    epsilon=1e-6,
):
    """Central finite-difference approximation of a gradient."""

    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    gradient = np.zeros_like(x)

    for i in range(x.size):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        gradient[i] = (
            function(x_plus)
            - function(x_minus)
        ) / (
            2.0 * epsilon
        )

    return gradient


def test_cartesian_problem_gradient_matches_finite_difference(
    cartesian_problem,
):
    problem = cartesian_problem

    n_variables = (
        problem.spline.n_control_points
        * problem.spline.n_dof
    )

    rng = np.random.default_rng(42)

    c = rng.normal(
        scale=0.1,
        size=n_variables,
    )

    analytical = problem.gradient(c)

    numerical = finite_difference_gradient(
        problem.objective,
        c,
        epsilon=1e-6,
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )


def test_cartesian_problem_gradient_matches_finite_difference(
    cartesian_problem,
    initial_control_points,
):
    problem = cartesian_problem

    c = np.asarray(
        initial_control_points,
        dtype=float,
    ).reshape(-1)

    analytical = problem.gradient(c)

    numerical = finite_difference_gradient(
        problem.objective,
        c,
        epsilon=1e-6,
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )


def test_cartesian_problem_gradient_shape(
    cartesian_problem,
    initial_control_points,
):
    c = np.asarray(
        initial_control_points,
        dtype=float,
    ).reshape(-1)

    gradient = (
        cartesian_problem.gradient(c)
    )

    assert gradient.shape == c.shape
    assert np.all(np.isfinite(gradient))

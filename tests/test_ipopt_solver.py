import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from splineopt.bspline import BSpline
from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
)
from splineopt.ipopt_solver import (
    IpoptSolver,
    _CyIpoptAdapter,
)
from splineopt.robotics.robot import Robot


class NonlinearConstrainedProblem:
    """
    Small analytical problem with one linear and one nonlinear
    constraint.

    Variables:

        x, y

    Linear constraint:

        x + y = 3

    Nonlinear constraint:

        x^2 + y^2 >= 2
    """

    def __init__(
        self,
    ):
        self.spline = DummySpline()

    def objective(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        x, y = c

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

        x, y = c

        return np.array([
            2.0 * (x - 1.0),
            2.0 * (y - 2.0),
        ])

    def linear_constraints(
        self,
    ):
        A = sp.csc_matrix([
            [1.0, 1.0],
        ])

        l = np.array([
            3.0
        ])

        u = np.array([
            3.0
        ])

        return A, l, u

    def nonlinear_constraints(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        x, y = c

        return np.array([
            x**2 + y**2
        ])

    def nonlinear_constraints_jacobian(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        x, y = c

        return sp.csc_matrix([
            [
                2.0 * x,
                2.0 * y,
            ]
        ])

    def nonlinear_constraint_bounds(
        self,
    ):
        return (
            np.array([
                2.0
            ]),
            np.array([
                np.inf
            ]),
        )

    def error_metrics(
        self,
        c,
    ):
        return {
            "position_rms": 0.0,
            "position_max": 0.0,
            "orientation_rms": 0.0,
            "orientation_max": 0.0,
        }
# ============================================================
# IPOPT availability
# ============================================================


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


# ============================================================
# Simple quadratic test problem
# ============================================================


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
# Kinova Cartesian problem fixtures
# ============================================================


ROOT = Path(
    __file__
).resolve().parents[1]


URDF_PATH = (
    ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


Q_MIN = np.full(
    7,
    -10.0,
)

Q_MAX = np.full(
    7,
    10.0,
)

V_MIN = np.full(
    7,
    -10.0,
)

V_MAX = np.full(
    7,
    10.0,
)

A_MIN = np.full(
    7,
    -10.0,
)

A_MAX = np.full(
    7,
    10.0,
)


@pytest.fixture
def kinova_case():
    """
    Small reachable Cartesian tracking problem used to validate
    CartesianTrajectoryProblem.gradient().
    """

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame=(
            "end_effector_link"
        ),
    )

    knots = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        3.0,
        3.0,
        3.0,
    ])

    spline = BSpline(
        knots=knots,
        degree=3,
        n_dof=robot.nv,
    )

    q0 = np.array([
        0.0,
        0.4,
        0.1,
        -1.2,
        0.0,
        0.8,
        0.0,
    ])

    v0 = np.zeros(
        robot.nv
    )

    a0 = np.zeros(
        robot.nv
    )

    initial_control_points = np.tile(
        q0,
        (
            spline.n_control_points,
            1,
        ),
    )

    # --------------------------------------------------------
    # Reachable Cartesian reference
    # --------------------------------------------------------

    target_control_points = (
        initial_control_points.copy()
    )

    target_control_points[
        3:,
        0,
    ] += 0.15

    target_control_points[
        3:,
        2,
    ] -= 0.10

    target_control_points[
        3:,
        4,
    ] += 0.05

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        20,
    )

    target_poses = []

    c_target = (
        target_control_points
        .reshape(-1)
    )

    for t in tracking_times:

        A = spline.evaluation_matrix(
            t,
            derivative_order=0,
        )

        theta = A @ c_target

        q_pin = (
            robot
            .joint_positions_to_configuration(
                theta
            )
        )

        target_poses.append(
            robot.frame_pose(
                q_pin
            )
        )

    return {
        "robot": robot,
        "spline": spline,
        "q0": q0,
        "v0": v0,
        "a0": a0,
        "tracking_times": (
            tracking_times
        ),
        "target_poses": (
            target_poses
        ),
        "initial_control_points": (
            initial_control_points
        ),
    }


@pytest.fixture
def cartesian_problem(
    kinova_case,
):
    return CartesianTrajectoryProblem(
        robot=kinova_case[
            "robot"
        ],
        spline=kinova_case[
            "spline"
        ],

        tracking_times=kinova_case[
            "tracking_times"
        ],

        target_poses=kinova_case[
            "target_poses"
        ],

        q0=kinova_case[
            "q0"
        ],
        v0=kinova_case[
            "v0"
        ],
        a0=kinova_case[
            "a0"
        ],

        q_min=Q_MIN,
        q_max=Q_MAX,

        v_min=V_MIN,
        v_max=V_MAX,

        a_min=A_MIN,
        a_max=A_MAX,

        orientation_scale=0.10,
        lambda_smooth=1e-5,
        lambda_terminal=0.0,
    )


@pytest.fixture
def initial_control_points(
    kinova_case,
):
    return kinova_case[
        "initial_control_points"
    ].copy()


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


# ============================================================
# Finite-difference helper
# ============================================================


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

    gradient = np.zeros_like(
        x
    )

    for i in range(
        x.size
    ):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        gradient[i] = (
            function(
                x_plus
            )
            - function(
                x_minus
            )
        ) / (
            2.0
            * epsilon
        )

    return gradient


# ============================================================
# CartesianTrajectoryProblem gradient
# ============================================================


def test_cartesian_problem_gradient_matches_finite_difference(
    cartesian_problem,
    initial_control_points,
):
    """
    Check the analytical gradient of the complete nonlinear
    Cartesian objective against central finite differences.
    """

    problem = (
        cartesian_problem
    )

    c = np.asarray(
        initial_control_points,
        dtype=float,
    ).reshape(-1)

    analytical = (
        problem.gradient(
            c
        )
    )

    numerical = (
        finite_difference_gradient(
            problem.objective,
            c,
            epsilon=1e-6,
        )
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
    """
    The analytical gradient must contain exactly one value
    per B-spline optimization variable.
    """

    c = np.asarray(
        initial_control_points,
        dtype=float,
    ).reshape(-1)

    gradient = (
        cartesian_problem.gradient(
            c
        )
    )

    assert (
        gradient.shape
        == c.shape
    )

    assert np.all(
        np.isfinite(
            gradient
        )
    )


def test_ipopt_adapter_combines_linear_and_nonlinear_constraints():

    problem = (
        NonlinearConstrainedProblem()
    )

    adapter = (
        _CyIpoptAdapter(
            problem
        )
    )

    x = np.array([
        1.0,
        2.0,
    ])

    values = (
        adapter.constraints(
            x
        )
    )

    # Linear:
    #
    #     x + y = 3
    #
    # Nonlinear:
    #
    #     x^2 + y^2 = 5

    np.testing.assert_allclose(
        values,
        np.array([
            3.0,
            5.0,
        ]),
    )

    np.testing.assert_allclose(
        adapter.constraint_lower,
        np.array([
            3.0,
            2.0,
        ]),
    )

    assert (
        adapter.constraint_upper[0]
        == 3.0
    )

    assert np.isposinf(
        adapter.constraint_upper[1]
    )


def test_ipopt_adapter_nonlinear_jacobian_values():

    problem = (
        NonlinearConstrainedProblem()
    )

    adapter = (
        _CyIpoptAdapter(
            problem
        )
    )

    x = np.array([
        1.0,
        2.0,
    ])

    rows, cols = (
        adapter.jacobianstructure()
    )

    values = (
        adapter.jacobian(
            x
        )
    )

    dense = np.zeros(
        (
            2,
            2,
        )
    )

    for row, col, value in zip(
        rows,
        cols,
        values,
    ):
        dense[
            row,
            col,
        ] += value

    expected = np.array([
        # Linear:
        #
        # x + y
        [1.0, 1.0],

        # Nonlinear:
        #
        # x^2 + y^2
        [2.0, 4.0],
    ])

    np.testing.assert_allclose(
        dense,
        expected,
    )


def test_ipopt_adapter_nonlinear_jacobian_matches_finite_difference():

    problem = (
        NonlinearConstrainedProblem()
    )

    adapter = (
        _CyIpoptAdapter(
            problem
        )
    )

    x = np.array([
        0.8,
        2.2,
    ])

    epsilon = 1e-7

    numerical = np.empty(
        (
            2,
            2,
        )
    )

    for j in range(
        2
    ):

        direction = np.zeros(
            2
        )

        direction[j] = 1.0

        g_plus = (
            adapter.constraints(
                x
                + epsilon
                * direction
            )
        )

        g_minus = (
            adapter.constraints(
                x
                - epsilon
                * direction
            )
        )

        numerical[:, j] = (
            g_plus
            - g_minus
        ) / (
            2.0
            * epsilon
        )

    rows, cols = (
        adapter.jacobianstructure()
    )

    values = (
        adapter.jacobian(
            x
        )
    )

    analytical = np.zeros(
        (
            2,
            2,
        )
    )

    for row, col, value in zip(
        rows,
        cols,
        values,
    ):
        analytical[
            row,
            col,
        ] += value

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


@requires_ipopt
def test_ipopt_solves_problem_with_nonlinear_constraint():

    problem = (
        NonlinearConstrainedProblem()
    )

    solver = IpoptSolver(
        tolerance=1e-9,
        max_iterations=100,
    )

    initial = np.array([
        [0.5, 2.5]
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

    c = (
        result.control_points
        .reshape(-1)
    )

    nonlinear_value = (
        problem
        .nonlinear_constraints(
            c
        )[0]
    )

    assert (
        nonlinear_value
        >= 2.0 - 1e-7
    )

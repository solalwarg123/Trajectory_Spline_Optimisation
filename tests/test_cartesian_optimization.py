import numpy as np
import pytest

from pathlib import Path

from splineopt.bspline import BSpline
from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
)
from splineopt.optimizer import (
    GaussNewtonSolver,
)
from splineopt.robotics.robot import Robot


# ============================================================
# Robot
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


# ============================================================
# Limits
# ============================================================

Q_MIN = np.full(7, -10.0)
Q_MAX = np.full(7, 10.0)

V_MIN = np.full(7, -10.0)
V_MAX = np.full(7, 10.0)

A_MIN = np.full(7, -10.0)
A_MAX = np.full(7, 10.0)


# ============================================================
# Fixture
# ============================================================

@pytest.fixture
def kinova_case():

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
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

    v0 = np.zeros(robot.nv)
    a0 = np.zeros(robot.nv)

    initial_control_points = np.tile(
        q0,
        (
            spline.n_control_points,
            1,
        ),
    )

    # --------------------------------------------------------
    # Reachable reference trajectory
    # --------------------------------------------------------

    target_control_points = (
        initial_control_points.copy()
    )

    target_control_points[3:, 0] += 0.15
    target_control_points[3:, 2] -= 0.10
    target_control_points[3:, 4] += 0.05

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        20,
    )

    target_poses = []

    c_target = target_control_points.reshape(-1)

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
        "tracking_times": tracking_times,
        "target_poses": target_poses,
        "initial_control_points":
            initial_control_points,
    }


# ============================================================
# Helper
# ============================================================

def make_problem(
    kinova_case,
    *,
    orientation_scale=0.10,
    lambda_smooth=1e-5,
    lambda_terminal=0.0,
):

    return CartesianTrajectoryProblem(
        robot=kinova_case["robot"],
        spline=kinova_case["spline"],

        tracking_times=(
            kinova_case["tracking_times"]
        ),

        target_poses=(
            kinova_case["target_poses"]
        ),

        q0=kinova_case["q0"],
        v0=kinova_case["v0"],
        a0=kinova_case["a0"],

        q_min=Q_MIN,
        q_max=Q_MAX,

        v_min=V_MIN,
        v_max=V_MAX,

        a_min=A_MIN,
        a_max=A_MAX,

        orientation_scale=orientation_scale,
        lambda_smooth=lambda_smooth,
        lambda_terminal=lambda_terminal,
    )


# ============================================================
# Problem tests
# ============================================================


def test_problem_linear_constraints_are_feasible(
    kinova_case,
):
    """
    The initial trajectory must satisfy all linear constraints.
    """

    problem = make_problem(
        kinova_case
    )

    c = (
        kinova_case[
            "initial_control_points"
        ]
        .reshape(-1)
    )

    A, l, u = (
        problem.linear_constraints()
    )

    values = A @ c

    assert np.all(
        values >= l - 1e-9
    )

    assert np.all(
        values <= u + 1e-9
    )


def test_problem_error_metrics_are_valid(
    kinova_case,
):
    """
    The target control points are not stored directly, so this
    test evaluates the initial trajectory only as a sanity check
    on the returned metrics.
    """

    problem = make_problem(
        kinova_case
    )

    metrics = problem.error_metrics(
        kinova_case[
            "initial_control_points"
        ]
    )

    assert metrics["position_rms"] >= 0.0
    assert metrics["position_max"] >= 0.0

    assert metrics["orientation_rms"] >= 0.0
    assert metrics["orientation_max"] >= 0.0


# ============================================================
# Validation tests
# ============================================================

def test_problem_rejects_negative_orientation_scale(
    kinova_case,
):
    with pytest.raises(ValueError):

        make_problem(
            kinova_case,
            orientation_scale=-0.1,
        )


def test_problem_rejects_negative_smoothness(
    kinova_case,
):
    with pytest.raises(ValueError):

        make_problem(
            kinova_case,
            lambda_smooth=-1.0,
        )


def test_problem_rejects_unsorted_tracking_times(
    kinova_case,
):
    times = (
        kinova_case[
            "tracking_times"
        ]
        .copy()
    )

    times[5], times[6] = (
        times[6],
        times[5],
    )

    with pytest.raises(ValueError):

        CartesianTrajectoryProblem(
            robot=kinova_case["robot"],
            spline=kinova_case["spline"],

            tracking_times=times,

            target_poses=(
                kinova_case[
                    "target_poses"
                ]
            ),

            q0=kinova_case["q0"],
            v0=kinova_case["v0"],
            a0=kinova_case["a0"],

            q_min=Q_MIN,
            q_max=Q_MAX,

            v_min=V_MIN,
            v_max=V_MAX,

            a_min=A_MIN,
            a_max=A_MAX,
        )


# ============================================================
# End-to-end solve
# ============================================================

def test_gauss_newton_solver_reduces_objective(
    kinova_case,
):
    """
    The new solver must reduce the true nonlinear objective.
    """

    problem = make_problem(
        kinova_case,
        orientation_scale=0.10,
        lambda_smooth=1e-5,
    )

    solver = GaussNewtonSolver(
        damping=1e-3,
        max_iterations=10,
    )

    initial_control_points = (
        kinova_case[
            "initial_control_points"
        ]
    )

    initial_cost = problem.objective(
        initial_control_points
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=(
            initial_control_points
        ),
    )

    final_cost = problem.objective(
        result.control_points
    )

    assert final_cost < initial_cost


def test_gauss_newton_solver_returns_feasible_solution(
    kinova_case,
):
    """
    The final trajectory returned by the solver must satisfy
    the linear trajectory constraints.
    """

    problem = make_problem(
        kinova_case,
        orientation_scale=0.10,
        lambda_smooth=1e-5,
    )

    solver = GaussNewtonSolver(
        damping=1e-3,
        max_iterations=10,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=(
            kinova_case[
                "initial_control_points"
            ]
        ),
    )

    c = (
        result.control_points
        .reshape(-1)
    )

    A, l, u = (
        problem.linear_constraints()
    )

    values = A @ c

    assert np.all(
        values >= l - 1e-7
    )

    assert np.all(
        values <= u + 1e-7
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "orientation_scale",
        "lambda_smooth",
        "lambda_terminal",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_invalid_cost_parameters_are_rejected(
    kinova_case,
    parameter,
    invalid_value,
):
    """Non-finite cost parameters must be rejected."""

    with pytest.raises(ValueError, match=parameter):
        make_problem(
            kinova_case,
            **{parameter: invalid_value},
        )


@pytest.mark.parametrize("orientation_scale", [0.0, 0.1])
def test_cartesian_residual_jacobian_finite_difference(
    kinova_case,
    orientation_scale,
):
    problem = make_problem(
        kinova_case,
        orientation_scale=orientation_scale,
        lambda_smooth=1e-5,
    )

    c = kinova_case["initial_control_points"].reshape(-1).copy()

    # Perturb the initial trajectory to avoid a trivial configuration.
    rng = np.random.default_rng(42)
    c += rng.normal(scale=0.01, size=c.shape)

    residual, J = problem.residual_and_jacobian(c)

    rng = np.random.default_rng(123)
    direction = rng.normal(size=c.size)
    direction /= np.linalg.norm(direction)

    eps = 1e-6

    r_plus = problem.residual(c + eps * direction)
    r_minus = problem.residual(c - eps * direction)

    numerical = (r_plus - r_minus) / (2.0 * eps)
    analytical = J @ direction

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-3,
        atol=1e-5,
    )


@pytest.mark.parametrize("orientation_scale", [0.0, 0.1])
def test_cartesian_objective_gradient_finite_difference(
    kinova_case,
    orientation_scale,
):
    problem = make_problem(
        kinova_case,
        orientation_scale=orientation_scale,
        lambda_smooth=1e-2,
        lambda_terminal=1.0,
    )

    c = kinova_case["initial_control_points"].reshape(-1).copy()

    rng = np.random.default_rng(42)
    c += rng.normal(scale=0.01, size=c.shape)

    analytical = problem.gradient(c)

    direction = rng.normal(size=c.size)
    direction /= np.linalg.norm(direction)

    eps = 1e-6

    numerical = (
        problem.objective(c + eps * direction)
        - problem.objective(c - eps * direction)
    ) / (2.0 * eps)

    np.testing.assert_allclose(
        analytical @ direction,
        numerical,
        rtol=1e-3,
        atol=1e-5,
    )


def test_gauss_newton_preserves_exact_reference(kinova_case):
    case = kinova_case
    problem = make_problem(
        case,
        orientation_scale=0.1,
        lambda_smooth=0.0,
    )

    # An unchanged initial trajectory is not the target in this fixture.
    # Construct a new problem whose reference matches that trajectory.
    initial_cp = case["initial_control_points"].copy()

    q_pin = case["robot"].joint_positions_to_configuration(case["q0"])
    target_pose = case["robot"].frame_pose(q_pin)

    problem.update(
        case["q0"],
        case["v0"],
        [target_pose.copy() for _ in case["tracking_times"]],
    )

    initial_cost = problem.objective(initial_cp)

    solver = GaussNewtonSolver(
        damping=1e-3,
        max_iterations=5,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=initial_cp,
    )

    final_cost = problem.objective(result.control_points)

    assert initial_cost < 1e-10
    assert final_cost < 1e-10

    np.testing.assert_allclose(
        result.control_points,
        initial_cp,
        atol=1e-5,
    )

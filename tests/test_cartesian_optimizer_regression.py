from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from splineopt.bspline import BSpline
from splineopt.optimizer import CartesianTrajectoryOptimizer
from splineopt.robotics import Robot


# =====================================================================
# Test configuration
# =====================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


Q_MIN = np.array([
    -np.inf,
    -2.24,
    -np.inf,
    -2.57,
    -np.inf,
    -2.09,
    -np.inf,
])

Q_MAX = np.array([
    np.inf,
    2.24,
    np.inf,
    2.57,
    np.inf,
    2.09,
    np.inf,
])

V_MAX = np.full(7, 0.8727)
V_MIN = -V_MAX

A_MAX = np.array([
    1.0,
    1.0,
    1.0,
    1.0,
    10.0,
    10.0,
    10.0,
])

A_MIN = -A_MAX


# =====================================================================
# Helpers
# =====================================================================


def make_test_spline():
    """
    Create a small cubic open-clamped 7-DoF B-spline.

    The horizon is deliberately long enough that the target trajectory
    easily satisfies the project velocity and acceleration limits.
    """

    degree = 3
    n_control_points = 8

    t_start = 0.0
    t_end = 4.0

    n_internal_knots = (
        n_control_points
        - degree
        - 1
    )

    internal_knots = np.linspace(
        t_start,
        t_end,
        n_internal_knots + 2,
    )[1:-1]

    knots = np.concatenate([
        np.full(
            degree + 1,
            t_start,
        ),
        internal_knots,
        np.full(
            degree + 1,
            t_end,
        ),
    ])

    return BSpline(
        knots=knots,
        degree=degree,
        n_dof=7,
    )


def make_optimizer(
    robot,
    spline,
    orientation_scale,
):
    return CartesianTrajectoryOptimizer(
        robot=robot,
        spline=spline,
        lambda_smooth=1e-5,
        damping=1e-3,
        orientation_scale=orientation_scale,
    )


def assert_linear_constraints_feasible(
    A,
    l,
    u,
    x,
    tol=2e-3,
):
    """
    Check OSQP-form constraints:

        l <= A x <= u

    Infinite bounds are ignored.
    """

    values = np.asarray(
        A @ np.asarray(x).reshape(-1)
    ).reshape(-1)

    finite_lower = np.isfinite(l)
    finite_upper = np.isfinite(u)

    assert np.all(
        values[finite_lower]
        >= l[finite_lower] - tol
    )

    assert np.all(
        values[finite_upper]
        <= u[finite_upper] + tol
    )


# =====================================================================
# Shared Kinova test problem
# =====================================================================


@pytest.fixture(scope="module")
def kinova_case():
    """
    Build a small deterministic trajectory-optimization problem.

    The initial trajectory is constant, therefore:

        q(t0)    = theta0
        qdot(t0) = 0
        qddot(t0)= 0

    The target trajectory is another feasible B-spline obtained by
    progressively moving the last control points.
    """

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    spline = make_test_spline()

    theta0 = np.array([
        0.00,
        0.30,
        0.00,
        -0.50,
        0.00,
        0.40,
        0.00,
    ])

    initial_control_points = np.tile(
        theta0,
        (
            spline.n_control_points,
            1,
        ),
    )

    # Keep the first three control points unchanged so that the target
    # also starts with zero velocity and acceleration.
    progression = np.array([
        0.0,
        0.0,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ])

    delta = np.array([
        0.20,
        0.12,
        -0.16,
        0.10,
        0.12,
        -0.08,
        0.15,
    ])

    target_control_points = (
        initial_control_points
        + progression[:, None]
        * delta[None, :]
    )

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        21,
    )

    target_poses = []

    for t in tracking_times:

        theta_target = spline.position(
            target_control_points,
            t,
        )

        q_pin = (
            robot
            .joint_positions_to_configuration(
                theta_target
            )
        )

        pose = robot.frame_pose(
            q_pin
        )

        # Explicit copy so that no Pinocchio internal data can be reused
        # or modified by a later forward-kinematics call.
        target_poses.append(
            pin.SE3(
                pose.rotation.copy(),
                pose.translation.copy(),
            )
        )

    target_positions = np.array([
        pose.translation.copy()
        for pose in target_poses
    ])

    return {
        "robot": robot,
        "spline": spline,
        "tracking_times": tracking_times,
        "target_poses": target_poses,
        "target_positions": target_positions,
        "initial_control_points": initial_control_points,
        "target_control_points": target_control_points,
        "q0": theta0,
        "v0": np.zeros(7),
        "a0": np.zeros(7),
    }


# =====================================================================
# Tests
# =====================================================================


def test_zero_orientation_scale_is_allowed(
    kinova_case,
):
    """
    orientation_scale = 0 must select exact position-only tracking.

    Negative values remain invalid.
    """

    robot = kinova_case["robot"]
    spline = kinova_case["spline"]

    optimizer = make_optimizer(
        robot,
        spline,
        orientation_scale=0.0,
    )

    assert optimizer.orientation_scale == 0.0

    with pytest.raises(ValueError):
        make_optimizer(
            robot,
            spline,
            orientation_scale=-0.1,
        )


def test_position_only_dispatch_matches_legacy_position_path(
    kinova_case,
):
    """
    With orientation_scale = 0, the new generic tracking dispatcher
    must reproduce exactly the previous position-only implementation.
    """

    optimizer = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.0,
    )

    control_points = (
        kinova_case["initial_control_points"]
    )

    times = kinova_case["tracking_times"]

    target_poses = (
        kinova_case["target_poses"]
    )

    target_positions = (
        kinova_case["target_positions"]
    )

    residual_old, jacobian_old = (
        optimizer
        .weighted_residual_and_jacobian(
            control_points=control_points,
            tracking_times=times,
            target_positions=target_positions,
        )
    )

    residual_new, jacobian_new = (
        optimizer
        .weighted_tracking_residual_and_jacobian(
            control_points=control_points,
            tracking_times=times,
            target_poses=target_poses,
        )
    )

    np.testing.assert_allclose(
        residual_new,
        residual_old,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        jacobian_new,
        jacobian_old,
        rtol=1e-12,
        atol=1e-12,
    )


def test_orientation_scale_only_scales_rotational_terms(
    kinova_case,
):
    """
    Doubling orientation_scale must:

      - leave translational residual/Jacobian rows unchanged;
      - double rotational residual/Jacobian rows.
    """

    optimizer_01 = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.10,
    )

    optimizer_02 = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.20,
    )

    control_points = (
        kinova_case["initial_control_points"]
    )

    times = kinova_case["tracking_times"]

    target_poses = (
        kinova_case["target_poses"]
    )

    residual_01, jacobian_01 = (
        optimizer_01
        .weighted_pose_residual_and_jacobian(
            control_points=control_points,
            tracking_times=times,
            target_poses=target_poses,
        )
    )

    residual_02, jacobian_02 = (
        optimizer_02
        .weighted_pose_residual_and_jacobian(
            control_points=control_points,
            tracking_times=times,
            target_poses=target_poses,
        )
    )

    n_samples = len(times)

    residual_01 = residual_01.reshape(
        n_samples,
        6,
    )

    residual_02 = residual_02.reshape(
        n_samples,
        6,
    )

    jacobian_01 = jacobian_01.reshape(
        n_samples,
        6,
        -1,
    )

    jacobian_02 = jacobian_02.reshape(
        n_samples,
        6,
        -1,
    )

    # Translation
    np.testing.assert_allclose(
        residual_02[:, :3],
        residual_01[:, :3],
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        jacobian_02[:, :3, :],
        jacobian_01[:, :3, :],
        rtol=1e-10,
        atol=1e-12,
    )

    # Rotation
    np.testing.assert_allclose(
        residual_02[:, 3:],
        2.0 * residual_01[:, 3:],
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        jacobian_02[:, 3:, :],
        2.0 * jacobian_01[:, 3:, :],
        rtol=1e-10,
        atol=1e-12,
    )


def test_objective_matches_weighted_residual_plus_smoothness(
    kinova_case,
):
    """
    Verify that objective() is exactly:

        ||r_w(c)||² + lambda * c.T R2 c
    """

    optimizer = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.10,
    )

    control_points = (
        kinova_case["initial_control_points"]
    )

    times = kinova_case["tracking_times"]
    target_poses = kinova_case["target_poses"]

    residual, _ = (
        optimizer
        .weighted_tracking_residual_and_jacobian(
            control_points=control_points,
            tracking_times=times,
            target_poses=target_poses,
        )
    )

    c = np.asarray(
        control_points,
        dtype=float,
    ).reshape(-1)

    expected = float(
        residual @ residual
        + optimizer.lambda_smooth
        * c
        @ optimizer.R2
        @ c
    )

    actual = optimizer.objective(
        control_points=control_points,
        tracking_times=times,
        target_poses=target_poses,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_gauss_newton_qp_solution_respects_constraints(
    kinova_case,
):
    """
    A full constrained Gauss-Newton QP step must satisfy all linear
    B-spline position, velocity, acceleration and initial constraints.
    """

    optimizer = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.10,
    )

    kwargs = {
        "current_control_points":
            kinova_case[
                "initial_control_points"
            ],
        "tracking_times":
            kinova_case[
                "tracking_times"
            ],
        "target_poses":
            kinova_case[
                "target_poses"
            ],
        "q0":
            kinova_case["q0"],
        "v0":
            kinova_case["v0"],
        "a0":
            kinova_case["a0"],
        "q_min":
            Q_MIN,
        "q_max":
            Q_MAX,
        "v_min":
            V_MIN,
        "v_max":
            V_MAX,
        "a_min":
            A_MIN,
        "a_max":
            A_MAX,
    }

    _, _, A, l, u = (
        optimizer
        .build_gauss_newton_qp(
            **kwargs
        )
    )

    c_qp, osqp_result = (
        optimizer
        .gauss_newton_step(
            **kwargs,
            verbose=False,
        )
    )

    assert (
        osqp_result
        .info
        .status
        .lower()
        .startswith("solved")
    )

    assert_linear_constraints_feasible(
        A,
        l,
        u,
        c_qp,
    )


def test_first_gauss_newton_step_decreases_true_objective(
    kinova_case,
):
    """
    On this small well-conditioned problem, the first full GN step is
    expected to decrease the true nonlinear objective.
    """

    optimizer = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.10,
    )

    initial_control_points = (
        kinova_case["initial_control_points"]
    )

    times = kinova_case["tracking_times"]
    target_poses = kinova_case["target_poses"]

    cost_before = optimizer.objective(
        control_points=initial_control_points,
        tracking_times=times,
        target_poses=target_poses,
    )

    c_qp, _ = (
        optimizer
        .gauss_newton_step(
            current_control_points=initial_control_points,
            tracking_times=times,
            target_poses=target_poses,
            q0=kinova_case["q0"],
            v0=kinova_case["v0"],
            a0=kinova_case["a0"],
            q_min=Q_MIN,
            q_max=Q_MAX,
            v_min=V_MIN,
            v_max=V_MAX,
            a_min=A_MIN,
            a_max=A_MAX,
            verbose=False,
        )
    )

    cost_after = optimizer.objective(
        control_points=c_qp,
        tracking_times=times,
        target_poses=target_poses,
    )

    assert cost_after < cost_before


def test_short_full_pose_solve_improves_objective_and_stays_feasible(
    kinova_case,
):
    """
    Small end-to-end regression test for the current solve() pipeline.

    We intentionally do not require a specific final status because
    stopping/globalization will be modified in the next development
    step.
    """

    optimizer = make_optimizer(
        kinova_case["robot"],
        kinova_case["spline"],
        orientation_scale=0.10,
    )

    initial_control_points = (
        kinova_case["initial_control_points"]
    )

    times = kinova_case["tracking_times"]
    target_poses = kinova_case["target_poses"]

    initial_cost = optimizer.objective(
        control_points=initial_control_points,
        tracking_times=times,
        target_poses=target_poses,
    )

    result = optimizer.solve(
        tracking_times=times,
        target_poses=target_poses,
        initial_control_points=initial_control_points,
        q0=kinova_case["q0"],
        v0=kinova_case["v0"],
        a0=kinova_case["a0"],
        q_min=Q_MIN,
        q_max=Q_MAX,
        v_min=V_MIN,
        v_max=V_MAX,
        a_min=A_MIN,
        a_max=A_MAX,
        max_iterations=3,
        verbose=False,
    )

    final_control_points = (
        result.control_points
    )

    final_cost = optimizer.objective(
        control_points=final_control_points,
        tracking_times=times,
        target_poses=target_poses,
    )

    assert final_cost < initial_cost
    assert len(result.history) >= 1

    # Rebuild the constraint matrix. The constraints themselves do not
    # depend on the current control-point values.
    _, _, A, l, u = (
        optimizer
        .build_gauss_newton_qp(
            current_control_points=final_control_points,
            tracking_times=times,
            target_poses=target_poses,
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
    )

    assert_linear_constraints_feasible(
        A,
        l,
        u,
        final_control_points,
    )

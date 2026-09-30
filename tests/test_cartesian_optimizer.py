from pathlib import Path

import numpy as np

from splineopt.bspline import BSpline
from splineopt.optimizer import (
    CartesianTrajectoryOptimizer,
)
from splineopt.robotics import Robot


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


def test_cartesian_control_point_jacobian():

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0,
        2.0,
        3.0, 3.0, 3.0, 3.0,
    ])

    spline = BSpline(
        knots=knots,
        degree=3,
        n_dof=robot.nv,
    )

    optimizer = CartesianTrajectoryOptimizer(
        robot=robot,
        spline=spline,
    )

    rng = np.random.default_rng(4)

    control_points = (
        0.15
        * rng.standard_normal(
            (
                spline.n_control_points,
                spline.n_dof,
            )
        )
    )

    times = np.array([
        0.25,
        0.8,
        1.4,
        2.2,
        2.75,
    ])

    target_positions = np.zeros(
        (len(times), 3)
    )

    residual, J = (
        optimizer.residual_and_jacobian(
            control_points,
            times,
            target_positions,
        )
    )

    direction = rng.standard_normal(
        control_points.size
    )

    direction /= np.linalg.norm(
        direction
    )

    c = control_points.reshape(-1)

    epsilon = 1e-7

    residual_plus, _ = (
        optimizer.residual_and_jacobian(
            c + epsilon * direction,
            times,
            target_positions,
        )
    )

    residual_minus, _ = (
        optimizer.residual_and_jacobian(
            c - epsilon * direction,
            times,
            target_positions,
        )
    )

    numerical = (
        residual_plus
        - residual_minus
    ) / (2.0 * epsilon)

    analytical = J @ direction

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=2e-4,
        atol=2e-6,
    )


def test_gauss_newton_step_reduces_cartesian_cost():
    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0,
        2.0,
        3.0, 3.0, 3.0, 3.0,
    ])

    spline = BSpline(
        knots=knots,
        degree=3,
        n_dof=robot.nv,
    )

    optimizer = CartesianTrajectoryOptimizer(
        robot=robot,
        spline=spline,
        lambda_smooth=1e-6,
        damping=1e-3,
    )

    n_cp = spline.n_control_points

    # Initial feasible trajectory:
    # all control points equal to the same joint configuration.

    theta_start = np.array([
        0.0,
        0.4,
        0.1,
        -1.2,
        0.0,
        0.8,
        0.0,
    ])

    control_points = np.tile(
        theta_start,
        (n_cp, 1),
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        15,
    )

    # Build a reachable Cartesian target from a slightly
    # modified joint trajectory.

    target_control_points = (
        control_points.copy()
    )

    # Keep the first control points unchanged so that
    # initial conditions remain compatible.

    target_control_points[3:, 0] += 0.15
    target_control_points[3:, 2] -= 0.10

    target_positions = []

    for t in times:

        A_k = spline.evaluation_matrix(
            t,
            derivative_order=0,
        )

        theta = (
            A_k
            @ target_control_points.reshape(-1)
        )

        q_pin = (
            robot.joint_positions_to_configuration(
                theta
            )
        )

        target_positions.append(
            robot.frame_position(q_pin)
        )

    target_positions = np.asarray(
        target_positions
    )

    initial_cost = optimizer.objective(
        control_points,
        times,
        target_positions,
    )

    c_qp, _ = optimizer.gauss_newton_step(
        current_control_points=control_points,
        tracking_times=times,
        target_poses=target_positions,
        q0=theta_start,
        v0=np.zeros(robot.nv),
        q_min=np.full(robot.nv, -10.0),
        q_max=np.full(robot.nv, 10.0),
        v_min=np.full(robot.nv, -10.0),
        v_max=np.full(robot.nv, 10.0),
        a_min=np.full(robot.nv, -10.0),
        a_max=np.full(robot.nv, 10.0),
    )

    new_cost = optimizer.objective(
        c_qp,
        times,
        target_positions,
    )

    assert new_cost < initial_cost


def test_cartesian_gauss_newton_converges():
    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0,
        2.0,
        3.0, 3.0, 3.0, 3.0,
    ])

    spline = BSpline(
        knots=knots,
        degree=3,
        n_dof=robot.nv,
    )

    optimizer = CartesianTrajectoryOptimizer(
        robot=robot,
        spline=spline,
        lambda_smooth=1e-6,
        damping=1e-3,
    )

    theta_start = np.array([
        0.0,
        0.4,
        0.1,
        -1.2,
        0.0,
        0.8,
        0.0,
    ])

    control_points = np.tile(
        theta_start,
        (spline.n_control_points, 1),
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        20,
    )

    # ---------------------------------------------------------
    # Generate reachable Cartesian reference
    # ---------------------------------------------------------

    target_control_points = (
        control_points.copy()
    )

    target_control_points[3:, 0] += 0.15
    target_control_points[3:, 2] -= 0.10

    target_positions = []

    c_target = target_control_points.reshape(-1)

    for t in times:

        A_k = spline.evaluation_matrix(
            t,
            derivative_order=0,
        )

        theta = A_k @ c_target

        q_pin = (
            robot.joint_positions_to_configuration(
                theta
            )
        )

        target_positions.append(
            robot.frame_position(q_pin)
        )

    target_positions = np.asarray(
        target_positions
    )

    # ---------------------------------------------------------
    # Initial error
    # ---------------------------------------------------------

    initial_metrics = (
        optimizer.cartesian_error_metrics(
            control_points,
            times,
            target_positions,
        )
    )

    # ---------------------------------------------------------
    # Optimize
    # ---------------------------------------------------------

    result = optimizer.solve(
        tracking_times=times,
        target_poses=target_positions,
        initial_control_points=control_points,
        q0=theta_start,
        v0=np.zeros(robot.nv),
        q_min=np.full(robot.nv, -10.0),
        q_max=np.full(robot.nv, 10.0),
        v_min=np.full(robot.nv, -10.0),
        v_max=np.full(robot.nv, 10.0),
        a_min=np.full(robot.nv, -10.0),
        a_max=np.full(robot.nv, 10.0),
        max_iterations=20,
        verbose=True
    )

    final_metrics = (
        optimizer.cartesian_error_metrics(
            result.control_points,
            times,
            target_positions,
        )
    )

    assert len(result.history) > 0

    assert (
        final_metrics["rms"]
        < initial_metrics["rms"]
    )

    # Require a meaningful improvement,
    # not just numerical noise.

    assert (
        final_metrics["rms"]
        < 0.5 * initial_metrics["rms"]
    )


def test_pose_control_point_jacobian():

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0,
        2.0,
        3.0, 3.0, 3.0, 3.0,
    ])

    spline = BSpline(
        knots=knots,
        degree=3,
        n_dof=robot.nv,
    )

    optimizer = CartesianTrajectoryOptimizer(
        robot=robot,
        spline=spline,
        orientation_scale=0.1,
    )

    rng = np.random.default_rng(8)

    control_points = (
        0.15
        * rng.standard_normal(
            (
                spline.n_control_points,
                spline.n_dof,
            )
        )
    )

    times = np.array([
        0.25,
        0.8,
        1.4,
        2.2,
        2.75,
    ])

    # ---------------------------------------------------------
    # Generate non-zero SE(3) targets from another trajectory
    # ---------------------------------------------------------

    target_control_points = (
        control_points.copy()
    )

    target_control_points[2:, 0] += 0.08
    target_control_points[2:, 2] -= 0.06
    target_control_points[2:, 4] += 0.05

    c_target = (
        target_control_points.reshape(-1)
    )

    target_poses = []

    for t in times:

        A_k = spline.evaluation_matrix(
            t,
            derivative_order=0,
        )

        theta_target = (
            A_k @ c_target
        )

        q_target = (
            robot
            .joint_positions_to_configuration(
                theta_target
            )
        )

        target_poses.append(
            robot.frame_pose(
                q_target
            )
        )

    # ---------------------------------------------------------
    # Analytical Jacobian
    # ---------------------------------------------------------

    c = control_points.reshape(-1)

    residual, J = (
        optimizer.pose_residual_and_jacobian(
            c,
            times,
            target_poses,
        )
    )

    # ---------------------------------------------------------
    # Directional finite difference
    # ---------------------------------------------------------

    direction = rng.standard_normal(
        c.size
    )

    direction /= np.linalg.norm(
        direction
    )

    epsilon = 1e-7

    residual_plus, _ = (
        optimizer.pose_residual_and_jacobian(
            c + epsilon * direction,
            times,
            target_poses,
        )
    )

    residual_minus, _ = (
        optimizer.pose_residual_and_jacobian(
            c - epsilon * direction,
            times,
            target_poses,
        )
    )

    numerical = (
        residual_plus
        - residual_minus
    ) / (2.0 * epsilon)

    analytical = (
        J @ direction
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=3e-4,
        atol=3e-6,
    )

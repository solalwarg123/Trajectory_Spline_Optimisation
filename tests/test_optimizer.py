import numpy as np

from splineopt.bspline import BSpline
from splineopt.joint_optimizer import JointTrajectoryOptimizer

import pytest


def make_test_spline():
    """
    Cubic open clamped B-spline on [0, 3].

    Degree:
        p = 3

    Number of control points:
        Nc = len(knots) - p - 1 = 6
    """

    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0,
        2.0,
        3.0, 3.0, 3.0, 3.0,
    ])

    return BSpline(
        knots=knots,
        degree=3,
        n_dof=1,
    )


def test_qp_dimensions():

    spline = make_test_spline()

    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=1e-2,
    )

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        10,
    )

    q_ref = np.zeros((
        len(tracking_times),
        1,
    ))

    P, g, A, l, u = optimizer.build_qp(
        tracking_times=tracking_times,
        q_ref=q_ref,
        q0=np.array([0.0]),
        v0=np.array([0.0]),
        q_min=np.array([-1.0]),
        q_max=np.array([1.0]),
        v_min=np.array([-2.0]),
        v_max=np.array([2.0]),
        a_min=np.array([-5.0]),
        a_max=np.array([5.0]),
    )

    n_variables = (
        spline.n_control_points
        * spline.n_dof
    )

    # For Nc = 6 and nq = 1:
    #
    # initial conditions:
    #     q(0) = q0      -> 1 row
    #     v(0) = v0      -> 1 row
    #
    # position:
    #     I              -> 6 rows
    #
    # velocity:
    #     D1             -> 5 rows
    #
    # acceleration:
    #     D2             -> 4 rows
    #
    # total = 2 + 6 + 5 + 4 = 17

    n_constraints = (
        2 * spline.n_dof
        + spline.n_control_points * spline.n_dof
        + (spline.n_control_points - 1) * spline.n_dof
        + (spline.n_control_points - 2) * spline.n_dof
    )

    assert P.shape == (
        n_variables,
        n_variables,
    )

    assert g.shape == (
        n_variables,
    )

    assert A.shape == (
        n_constraints,
        n_variables,
    )

    assert l.shape == (
        n_constraints,
    )

    assert u.shape == (
        n_constraints,
    )


def test_constant_reference():

    spline = make_test_spline()

    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=1e-2,
    )

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        20,
    )

    q_target = 0.2

    q_ref = np.full(
        (
            len(tracking_times),
            1,
        ),
        q_target,
    )

    control_points, result = optimizer.solve(
        tracking_times=tracking_times,
        q_ref=q_ref,
        q0=np.array([q_target]),
        v0=np.array([0.0]),
        q_min=np.array([-1.0]),
        q_max=np.array([1.0]),
        v_min=np.array([-2.0]),
        v_max=np.array([2.0]),
        a_min=np.array([-5.0]),
        a_max=np.array([5.0]),
    )

    assert result.info.status.lower().startswith(
        "solved"
    )

    # A constant B-spline trajectory should have all
    # control points equal to the constant target.

    np.testing.assert_allclose(
        control_points,
        q_target,
        atol=1e-4,
    )

    # Verify the trajectory itself.

    test_times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    positions = np.array([
        spline.position(
            control_points,
            t,
        )[0]
        for t in test_times
    ])

    velocities = np.array([
        spline.velocity(
            control_points,
            t,
        )[0]
        for t in test_times
    ])

    accelerations = np.array([
        spline.acceleration(
            control_points,
            t,
        )[0]
        for t in test_times
    ])

    np.testing.assert_allclose(
        positions,
        q_target,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        velocities,
        0.0,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        accelerations,
        0.0,
        atol=1e-4,
    )


def test_aggressive_reference_respects_constraints():

    spline = make_test_spline()

    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=1e-3,
    )

    # --------------------------------------------------------
    # Aggressive nominal trajectory
    # --------------------------------------------------------

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    q_target = 0.8
    rise_time = 0.5

    q_ref = np.zeros((
        len(tracking_times),
        1,
    ))

    rising = tracking_times <= rise_time

    q_ref[rising, 0] = (
        0.5
        * q_target
        * (
            1.0
            - np.cos(
                np.pi
                * tracking_times[rising]
                / rise_time
            )
        )
    )

    q_ref[~rising, 0] = q_target

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    q_min = np.array([-1.0])
    q_max = np.array([1.0])

    v_min = np.array([-0.5])
    v_max = np.array([0.5])

    a_min = np.array([-1.0])
    a_max = np.array([1.0])

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    control_points, result = optimizer.solve(
        tracking_times=tracking_times,
        q_ref=q_ref,
        q0=np.array([0.0]),
        v0=np.array([0.0]),
        q_min=q_min,
        q_max=q_max,
        v_min=v_min,
        v_max=v_max,
        a_min=a_min,
        a_max=a_max,
    )

    assert result.info.status.lower().startswith(
        "solved"
    )

    c = control_points.reshape(-1)

    # --------------------------------------------------------
    # Check control-point constraints
    # --------------------------------------------------------

    D1 = spline.derivative_control_matrix(
        order=1
    )

    D2 = spline.derivative_control_matrix(
        order=2
    )

    velocity_control_points = D1 @ c
    acceleration_control_points = D2 @ c

    assert np.all(
        control_points >= q_min - 1e-5
    )

    assert np.all(
        control_points <= q_max + 1e-5
    )

    assert np.all(
        velocity_control_points
        >= v_min - 1e-5
    )

    assert np.all(
        velocity_control_points
        <= v_max + 1e-5
    )

    assert np.all(
        acceleration_control_points
        >= a_min - 1e-5
    )

    assert np.all(
        acceleration_control_points
        <= a_max + 1e-5
    )

    # --------------------------------------------------------
    # Dense continuous-time verification
    # --------------------------------------------------------

    test_times = np.linspace(
        spline.t_start,
        spline.t_end,
        1000,
    )

    positions = np.array([
        spline.position(
            control_points,
            t,
        )[0]
        for t in test_times
    ])

    velocities = np.array([
        spline.velocity(
            control_points,
            t,
        )[0]
        for t in test_times
    ])

    accelerations = np.array([
        spline.acceleration(
            control_points,
            t,
        )[0]
        for t in test_times
    ])

    assert np.max(positions) <= (
        q_max[0] + 1e-5
    )

    assert np.min(positions) >= (
        q_min[0] - 1e-5
    )

    assert np.max(velocities) <= (
        v_max[0] + 1e-5
    )

    assert np.min(velocities) >= (
        v_min[0] - 1e-5
    )

    assert np.max(accelerations) <= (
        a_max[0] + 1e-5
    )

    assert np.min(accelerations) >= (
        a_min[0] - 1e-5
    )

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    np.testing.assert_allclose(
        spline.position(
            control_points,
            spline.t_start,
        ),
        np.array([0.0]),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        spline.velocity(
            control_points,
            spline.t_start,
        ),
        np.array([0.0]),
        atol=1e-5,
    )

    # --------------------------------------------------------
    # The optimized trajectory cannot reproduce the
    # aggressive nominal trajectory exactly.
    # --------------------------------------------------------

    optimized_at_tracking_times = np.array([
        spline.position(
            control_points,
            t,
        )[0]
        for t in tracking_times
    ])

    tracking_error = (
        optimized_at_tracking_times
        - q_ref[:, 0]
    )

    rms_tracking_error = np.sqrt(
        np.mean(tracking_error ** 2)
    )

    assert rms_tracking_error > 1e-3


def test_optional_boundary_conditions():
    spline = make_test_spline()
    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=1e-2,
    )

    tracking_times = np.linspace(
        spline.t_start, spline.t_end, 30
    )
    q_ref = np.full((len(tracking_times), 1), 0.2)

    a0 = np.array([0.0])
    vf = np.array([0.0])

    control_points, result = optimizer.solve(
        tracking_times=tracking_times,
        q_ref=q_ref,
        q0=np.array([0.0]),
        v0=np.array([0.0]),
        a0=a0,
        vf=vf,
        q_min=np.array([-1.0]),
        q_max=np.array([1.0]),
        v_min=np.array([-2.0]),
        v_max=np.array([2.0]),
        a_min=np.array([-5.0]),
        a_max=np.array([5.0]),
    )

    assert result.info.status.lower().startswith("solved")

    np.testing.assert_allclose(
        spline.acceleration(control_points, spline.t_start),
        a0,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        spline.velocity(control_points, spline.t_end),
        vf,
        atol=1e-5,
    )


def test_infeasible_joint_qp():
    spline = make_test_spline()
    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=1e-2,
    )

    tracking_times = np.linspace(
        spline.t_start, spline.t_end, 20
    )
    q_ref = np.zeros((len(tracking_times), 1))

    # Initial position is incompatible with joint position bounds.
    with pytest.raises(RuntimeError, match="OSQP failed"):
        optimizer.solve(
            tracking_times=tracking_times,
            q_ref=q_ref,
            q0=np.array([2.0]),
            v0=np.array([0.0]),
            q_min=np.array([-1.0]),
            q_max=np.array([1.0]),
            v_min=np.array([-2.0]),
            v_max=np.array([2.0]),
            a_min=np.array([-5.0]),
            a_max=np.array([5.0]),
        )

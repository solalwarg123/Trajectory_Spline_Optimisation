from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from splineopt.robotics import Robot


# ---------------------------------------------------------------------
# Robot fixture
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


@pytest.fixture
def robot():
    return Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_non_singular_configuration(robot):
    """
    Move away from the neutral configuration to obtain a more
    generic starting posture.
    """

    q0 = robot.neutral()

    v = np.zeros(robot.nv)

    values = np.array([
        0.20,
        -0.30,
        0.15,
        -0.25,
        0.20,
        0.15,
        -0.10,
    ])

    v[:min(robot.nv, len(values))] = values[:robot.nv]

    return robot.integrate(
        q0,
        v,
    )


# ---------------------------------------------------------------------
# Already at target
# ---------------------------------------------------------------------

def test_ik_already_at_target(robot):

    q = make_non_singular_configuration(
        robot
    )

    target_pose = robot.frame_pose(
        q
    )

    result = robot.ik(
        target_pose,
        q0=q,
    )

    assert result.success
    assert result.iterations == 0

    assert result.error_norm < 1e-10

    np.testing.assert_allclose(
        result.q,
        q,
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Reach known reachable pose
# ---------------------------------------------------------------------

def test_ik_reaches_known_pose(robot):

    # -------------------------------------------------------------
    # Initial configuration
    # -------------------------------------------------------------

    q_start = make_non_singular_configuration(
        robot
    )

    # -------------------------------------------------------------
    # Construct another reachable configuration
    # -------------------------------------------------------------

    delta = np.zeros(robot.nv)

    values = np.array([
        0.08,
        -0.10,
        0.06,
        0.08,
        -0.05,
        0.07,
        0.04,
    ])

    delta[:min(robot.nv, len(values))] = values[:robot.nv]

    q_target = robot.integrate(
        q_start,
        delta,
    )

    # -------------------------------------------------------------
    # Generate target pose using FK
    # -------------------------------------------------------------

    target_pose = robot.frame_pose(
        q_target
    )

    # -------------------------------------------------------------
    # Solve IK
    # -------------------------------------------------------------

    result = robot.ik(
        target_pose,
        q0=q_start,
        tolerance=1e-5,
        max_iterations=500,
        step_size=0.1,
        damping=1e-6,
    )

    assert result.success

    # -------------------------------------------------------------
    # Compare resulting Cartesian pose
    # -------------------------------------------------------------

    achieved_pose = robot.frame_pose(
        result.q
    )

    pose_error = pin.log6(
        achieved_pose.actInv(
            target_pose
        )
    ).vector

    assert np.linalg.norm(
        pose_error
    ) < 1e-5


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------

def test_ik_history(robot):

    q_start = make_non_singular_configuration(
        robot
    )

    delta = np.zeros(robot.nv)
    delta[0] = 0.05
    delta[2] = -0.05
    delta[4] = 0.04

    q_target = robot.integrate(
        q_start,
        delta,
    )

    target_pose = robot.frame_pose(
        q_target
    )

    result = robot.ik(
        target_pose,
        q0=q_start,
    )

    assert result.success

    assert result.q_history.ndim == 2

    assert result.q_history.shape[1] == robot.nq

    assert len(result.error_history) >= 1

    # The final error must be much smaller than the initial one.
    assert (
        result.error_history[-1]
        <
        result.error_history[0]
    )
    assert len(result.q_history) == result.iterations + 1
    assert len(result.error_history) == result.iterations + 1

    np.testing.assert_allclose(
        result.q_history[-1],
        result.q,
    )


def test_ik_trajectory(robot):

    # -------------------------------------------------------------
    # Build a known reachable joint trajectory
    # -------------------------------------------------------------

    n_samples = 20

    s = np.linspace(
        0.0,
        1.0,
        n_samples,
    )

    theta = np.zeros(
        (n_samples, robot.nv)
    )

    theta[:, 0] = 0.20 + 0.15 * s
    theta[:, 1] = -0.40 + 0.10 * s
    theta[:, 2] = 0.15 * np.sin(np.pi * s)
    theta[:, 3] = -0.50 + 0.20 * s
    theta[:, 4] = 0.10 * np.sin(2 * np.pi * s)
    theta[:, 5] = 0.30 - 0.10 * s
    theta[:, 6] = 0.10 * s

    q_known = (
        robot.joint_trajectory_to_configurations(
            theta
        )
    )

    # -------------------------------------------------------------
    # Use FK to generate reachable Cartesian targets
    # -------------------------------------------------------------

    targets = [
        robot.frame_pose(q)
        for q in q_known
    ]

    # -------------------------------------------------------------
    # Sequential IK
    # -------------------------------------------------------------

    result = robot.ik_trajectory(
        targets,
        q0=q_known[0],
        tolerance=1e-5,
        max_iterations=500,
        step_size=0.1,
        damping=1e-6,
    )

    assert result.success

    assert result.q_pin.shape == (
        n_samples,
        robot.nq,
    )

    assert result.theta.shape == (
        n_samples,
        robot.nv,
    )

    # -------------------------------------------------------------
    # Verify every achieved Cartesian pose
    # -------------------------------------------------------------

    for q, target in zip(
        result.q_pin,
        targets,
    ):

        achieved = robot.frame_pose(q)

        error = pin.log6(
            achieved.actInv(target)
        ).vector

        assert np.linalg.norm(error) < 1e-5


def test_ik_non_convergence(robot):
    q_start = make_non_singular_configuration(robot)

    delta = np.zeros(robot.nv)
    delta[0] = 0.2
    delta[1] = -0.2

    q_target = robot.integrate(q_start, delta)
    target_pose = robot.frame_pose(q_target)

    result = robot.ik(
        target_pose,
        q0=q_start,
        tolerance=1e-8,
        max_iterations=0,
    )

    assert not result.success
    assert result.iterations == 0
    assert result.error_norm > 1e-8

    assert result.q_history.shape == (1, robot.nq)
    assert result.error_history.shape == (1,)

    np.testing.assert_allclose(result.q, q_start)


def test_ik_trajectory_failure_handling(robot):
    q_start = make_non_singular_configuration(robot)

    delta = np.zeros(robot.nv)
    delta[0] = 0.2
    delta[1] = -0.2

    q_target = robot.integrate(q_start, delta)

    targets = [
        robot.frame_pose(q_start),
        robot.frame_pose(q_target),
    ]

    # Continue and report the unsuccessful solve.
    result = robot.ik_trajectory(
        targets,
        q0=q_start,
        max_iterations=0,
        tolerance=1e-8,
        stop_on_failure=False,
    )

    assert not result.success
    np.testing.assert_array_equal(
        result.success_mask,
        [True, False],
    )
    assert result.q_pin.shape == (2, robot.nq)
    assert result.theta.shape == (2, robot.nv)
    assert len(result.results) == 2

    # Alternatively, stop immediately on failure.
    with pytest.raises(
        RuntimeError,
        match="IK failed at trajectory sample 1",
    ):
        robot.ik_trajectory(
            targets,
            q0=q_start,
            max_iterations=0,
            tolerance=1e-8,
            stop_on_failure=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": 0.0},
        {"tolerance": np.nan},
        {"max_iterations": -1},
        {"max_iterations": 2.5},
        {"step_size": 0.0},
        {"step_size": np.inf},
        {"damping": -1.0},
        {"damping": 0.0},
        {"max_step_norm": 0.0},
    ],
)
def test_ik_rejects_invalid_parameters(robot, kwargs):
    target_pose = robot.frame_pose(robot.neutral())

    with pytest.raises(ValueError):
        robot.ik(
            target_pose,
            **kwargs,
        )

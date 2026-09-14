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

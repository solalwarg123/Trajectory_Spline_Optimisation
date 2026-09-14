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
# Model loading
# ---------------------------------------------------------------------

def test_robot_loads(robot):
    assert robot.model is not None
    assert robot.data is not None

    assert robot.nq > 0
    assert robot.nv > 0


def test_end_effector_exists(robot):
    assert "end_effector_link" in robot.frame_names

    frame_id = robot.frame_id()

    assert frame_id == robot.model.getFrameId(
        "end_effector_link"
    )


def test_joint_names(robot):
    # Kinova Gen3 is a 7-DoF robot.
    assert len(robot.joint_names) == 7


# ---------------------------------------------------------------------
# Neutral configuration
# ---------------------------------------------------------------------

def test_neutral_configuration_dimension(robot):
    q = robot.neutral()

    assert isinstance(q, np.ndarray)
    assert q.shape == (robot.nq,)
    assert np.all(np.isfinite(q))


# ---------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------

def test_integrate_dimension(robot):
    q = robot.neutral()

    v = np.zeros(robot.nv)
    v[0] = 0.2

    q_new = robot.integrate(
        q,
        v,
        dt=0.1,
    )

    assert q_new.shape == (robot.nq,)
    assert np.all(np.isfinite(q_new))


def test_integrate_matches_pinocchio_difference(robot):
    """
    Check that Robot.integrate() correctly moves along the
    Pinocchio configuration manifold.
    """

    q = robot.neutral()

    v = np.zeros(robot.nv)
    v[0] = 0.3
    v[1] = -0.2

    dt = 0.05

    q_new = robot.integrate(
        q,
        v,
        dt=dt,
    )

    # difference(q, q_new) lives in tangent space
    dq = pin.difference(
        robot.model,
        q,
        q_new,
    )

    np.testing.assert_allclose(
        dq,
        dt * v,
        atol=1e-10,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------

def test_frame_pose(robot):
    q = robot.neutral()

    pose = robot.frame_pose(q)

    assert isinstance(pose, pin.SE3)

    assert pose.translation.shape == (3,)
    assert pose.rotation.shape == (3, 3)

    assert np.all(np.isfinite(pose.translation))
    assert np.all(np.isfinite(pose.rotation))


def test_frame_rotation_is_valid(robot):
    """
    Check that the rotation returned by FK is a valid
    rotation matrix.
    """

    q = robot.neutral()

    R = robot.frame_pose(q).rotation

    np.testing.assert_allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-10,
    )

    np.testing.assert_allclose(
        np.linalg.det(R),
        1.0,
        atol=1e-10,
    )


def test_frame_position(robot):
    q = robot.neutral()

    position = robot.frame_position(q)

    assert position.shape == (3,)
    assert np.all(np.isfinite(position))

    pose = robot.frame_pose(q)

    np.testing.assert_allclose(
        position,
        pose.translation,
    )


# ---------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------

def test_frame_jacobian_dimension(robot):
    q = robot.neutral()

    J = robot.frame_jacobian(q)

    assert J.shape == (6, robot.nv)
    assert np.all(np.isfinite(J))


def test_position_jacobian_finite_difference(robot):
    """
    Validate the translational part of the analytical Jacobian
    against a numerical finite-difference approximation.

    For a tangent-space direction v:

        dp/dt ≈ [p(integrate(q, eps*v)) - p(q)] / eps

    which should match:

        J_position(q) @ v
    """

    q0 = robot.neutral()

    # -------------------------------------------------------------
    # Move away from the neutral configuration
    # -------------------------------------------------------------

    offset = np.zeros(robot.nv)

    values = [
        0.20,
        -0.15,
        0.10,
        0.25,
        -0.20,
        0.10,
        0.15,
    ]

    offset[:min(robot.nv, len(values))] = values[:robot.nv]

    q = robot.integrate(
        q0,
        offset,
    )

    # -------------------------------------------------------------
    # Direction in tangent space
    # -------------------------------------------------------------

    direction = np.zeros(robot.nv)

    direction_values = [
        0.3,
        -0.2,
        0.15,
        0.1,
        -0.25,
        0.2,
        -0.1,
    ]

    direction[:min(robot.nv, len(direction_values))] = (
        direction_values[:robot.nv]
    )

    # -------------------------------------------------------------
    # Analytical derivative
    # -------------------------------------------------------------

    J = robot.frame_jacobian(q)

    # Pinocchio spatial vectors are:
    #
    # [linear velocity]
    # [angular velocity]
    #
    # so the first 3 rows correspond to translation.
    J_position = J[:3, :]

    analytical = J_position @ direction

    # -------------------------------------------------------------
    # Numerical derivative
    # -------------------------------------------------------------

    epsilon = 1e-7

    p0 = robot.frame_position(q)

    q_epsilon = robot.integrate(
        q,
        direction,
        dt=epsilon,
    )

    p_epsilon = robot.frame_position(
        q_epsilon
    )

    numerical = (
        p_epsilon - p0
    ) / epsilon

    # -------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def test_invalid_configuration_dimension(robot):
    q_wrong = np.zeros(robot.nq + 1)

    with pytest.raises(ValueError):
        robot.frame_pose(q_wrong)


def test_invalid_velocity_dimension(robot):
    q = robot.neutral()

    v_wrong = np.zeros(robot.nv + 1)

    with pytest.raises(ValueError):
        robot.integrate(
            q,
            v_wrong,
        )


def test_invalid_frame(robot):
    q = robot.neutral()

    with pytest.raises(ValueError):
        robot.frame_pose(
            q,
            frame="this_frame_does_not_exist",
        )

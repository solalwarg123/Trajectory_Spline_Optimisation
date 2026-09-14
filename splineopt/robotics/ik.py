from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class IKResult:
    """
    Result of an inverse kinematics solve.
    """

    q: np.ndarray
    success: bool
    error_norm: float
    iterations: int

    # Useful later for visualization / debugging
    q_history: np.ndarray
    error_history: np.ndarray


@dataclass
class IKTrajectoryResult:
    """
    Result of sequential inverse kinematics over a Cartesian trajectory.
    """

    q_pin: np.ndarray
    theta: np.ndarray

    success: bool
    success_mask: np.ndarray

    error_norms: np.ndarray
    iterations: np.ndarray

    results: list


def solve_pose_ik(
    robot,
    target_pose,
    q0=None,
    frame=None,
    tolerance=1e-4,
    max_iterations=500,
    step_size=0.1,
    damping=1e-6,
    max_step_norm=1.0,
):
    """
    Solve full SE(3) inverse kinematics.

    The target contains both:
        - position
        - orientation

    The algorithm is a damped iterative Jacobian IK based on
    Pinocchio's SE(3) logarithm.

    Parameters
    ----------
    robot:
        Robot instance.

    target_pose:
        Desired frame pose as a pin.SE3.

    q0:
        Initial Pinocchio configuration.
        If None, the neutral configuration is used.

    frame:
        Frame to control.
        If None, Robot.end_effector_frame is used.

    tolerance:
        Convergence tolerance on the SE(3) error norm.

    max_iterations:
        Maximum number of IK iterations.

    step_size:
        Integration step.

    damping:
        Damping coefficient used in the pseudo-inverse.

    max_step_norm:
        Maximum norm of the tangent-space update before applying
        step_size. Set to None to disable clipping.
    """

    if not isinstance(target_pose, pin.SE3):
        raise TypeError(
            "target_pose must be a pinocchio.SE3 object."
        )

    # ------------------------------------------------------------------
    # Initial configuration
    # ------------------------------------------------------------------

    if q0 is None:
        q = robot.neutral()
    else:
        q = np.asarray(
            q0,
            dtype=float,
        ).reshape(-1)

        if q.shape != (robot.nq,):
            raise ValueError(
                f"Expected q0 shape ({robot.nq},), "
                f"got {q.shape}."
            )

        q = q.copy()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    q_history = [q.copy()]
    error_history = []

    # ------------------------------------------------------------------
    # Iterative IK
    # ------------------------------------------------------------------

    for iteration in range(max_iterations + 1):

        # --------------------------------------------------------------
        # Current end-effector pose
        # --------------------------------------------------------------

        current_pose = robot.frame_pose(
            q,
            frame=frame,
        )

        # Transform from current frame to desired frame
        current_to_target = current_pose.actInv(
            target_pose
        )

        # --------------------------------------------------------------
        # SE(3) error
        # --------------------------------------------------------------
        #
        # 6D error in the tangent space of SE(3).
        #
        # It contains both translation and rotation error.
        # --------------------------------------------------------------

        error = pin.log6(
            current_to_target
        ).vector

        error_norm = np.linalg.norm(
            error
        )

        error_history.append(
            error_norm
        )

        # --------------------------------------------------------------
        # Convergence
        # --------------------------------------------------------------

        if error_norm < tolerance:

            return IKResult(
                q=q.copy(),
                success=True,
                error_norm=error_norm,
                iterations=iteration,
                q_history=np.asarray(q_history),
                error_history=np.asarray(error_history),
            )

        # --------------------------------------------------------------
        # Maximum iterations reached
        # --------------------------------------------------------------

        if iteration == max_iterations:
            break

        # --------------------------------------------------------------
        # Frame Jacobian
        # --------------------------------------------------------------
        #
        # The SE(3) error is expressed in the local frame, therefore
        # we compute the frame Jacobian in LOCAL coordinates.
        # --------------------------------------------------------------

        J = robot.frame_jacobian(
            q,
            frame=frame,
            reference_frame=pin.ReferenceFrame.LOCAL,
        )

        # --------------------------------------------------------------
        # Jacobian of the SE(3) logarithmic error
        # --------------------------------------------------------------
        #
        # This is the same correction used in Pinocchio's official
        # iterative IK example.
        # --------------------------------------------------------------

        J_error = (
            -pin.Jlog6(
                current_to_target.inverse()
            )
            @ J
        )

        # --------------------------------------------------------------
        # Damped least squares
        # --------------------------------------------------------------
        #
        # dq = -J^T (J J^T + lambda I)^-1 e
        # --------------------------------------------------------------

        JJt = (
            J_error
            @ J_error.T
            + damping * np.eye(6)
        )

        dq = (
            -J_error.T
            @ np.linalg.solve(
                JJt,
                error,
            )
        )

        # --------------------------------------------------------------
        # Limit very large updates
        # --------------------------------------------------------------

        if max_step_norm is not None:

            dq_norm = np.linalg.norm(
                dq
            )

            if dq_norm > max_step_norm:
                dq *= (
                    max_step_norm
                    / dq_norm
                )

        # --------------------------------------------------------------
        # Update configuration
        # --------------------------------------------------------------

        q = robot.integrate(
            q,
            dq,
            dt=step_size,
        )

        q_history.append(
            q.copy()
        )

    # ------------------------------------------------------------------
    # No convergence
    # ------------------------------------------------------------------

    return IKResult(
        q=q.copy(),
        success=False,
        error_norm=error_norm,
        iterations=max_iterations,
        q_history=np.asarray(q_history),
        error_history=np.asarray(error_history),
    )


def solve_pose_ik_trajectory(
    robot,
    target_poses,
    q0=None,
    frame=None,
    stop_on_failure=True,
    **ik_kwargs,
):
    """
    Solve a sequence of SE(3) inverse kinematics problems.

    Each solution is used as the initial configuration of the next IK.
    This encourages continuity of the joint trajectory and is especially
    important for redundant robots such as the 7-DoF Kinova Gen3.
    """

    target_poses = list(
        target_poses
    )

    if len(target_poses) == 0:
        raise ValueError(
            "target_poses cannot be empty."
        )

    # ------------------------------------------------------------------
    # Initial configuration
    # ------------------------------------------------------------------

    if q0 is None:
        q_current = robot.neutral()
    else:
        q_current = np.asarray(
            q0,
            dtype=float,
        ).reshape(-1)

        if q_current.shape != (robot.nq,):
            raise ValueError(
                f"Expected q0 shape ({robot.nq},), "
                f"got {q_current.shape}."
            )

        q_current = q_current.copy()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    results = []
    q_sequence = []

    success_mask = []
    error_norms = []
    iterations = []

    # ------------------------------------------------------------------
    # Sequential IK
    # ------------------------------------------------------------------

    for k, target_pose in enumerate(
        target_poses
    ):

        result = solve_pose_ik(
            robot=robot,
            target_pose=target_pose,
            q0=q_current,
            frame=frame,
            **ik_kwargs,
        )

        results.append(result)

        success_mask.append(
            result.success
        )

        error_norms.append(
            result.error_norm
        )

        iterations.append(
            result.iterations
        )

        if not result.success:

            if stop_on_failure:
                raise RuntimeError(
                    f"IK failed at trajectory sample {k}. "
                    f"Final error: {result.error_norm:.3e}"
                )

        # Warm-start next target from previous solution.
        q_current = result.q.copy()

        q_sequence.append(
            q_current
        )

    # ------------------------------------------------------------------
    # Convert Pinocchio configurations to continuous joint coordinates
    # ------------------------------------------------------------------

    q_sequence = np.asarray(
        q_sequence
    )

    theta = (
        robot.configuration_trajectory_to_joint_positions(
            q_sequence
        )
    )

    success_mask = np.asarray(
        success_mask,
        dtype=bool,
    )

    return IKTrajectoryResult(
        q_pin=q_sequence,
        theta=theta,
        success=bool(np.all(success_mask)),
        success_mask=success_mask,
        error_norms=np.asarray(error_norms),
        iterations=np.asarray(iterations),
        results=results,
    )

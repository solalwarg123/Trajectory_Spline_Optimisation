from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class IKResult:
    """
    Result of a single SE(3) inverse kinematics solve.

    Attributes:
        q: Final Pinocchio configuration, shape (nq,).
        success: Whether the pose error reached the tolerance.
        error_norm: Final norm of the 6D logarithmic pose error.
        iterations: Number of configuration updates performed.
        q_history: Configurations visited, including the initial one.
        error_history: Pose error norm at each evaluated configuration.
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
    Result of sequential inverse kinematics over Cartesian target poses.

    Attributes:
        q_pin: Sequence of Pinocchio configurations, shape (N, nq).
        theta: Continuous joint coordinates, shape (N, nv).
        success: True if all individual IK solves succeeded.
        success_mask: Per-target success flags.
        error_norms: Final pose errors for each target.
        iterations: Iteration counts for each target.
        results: Individual IKResult objects, including their histories.
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
        Positive damping coefficient used in the damped
        least-squares normal equations.

    max_step_norm:
        Maximum norm of the tangent-space update before applying
        step_size. Set to None to disable clipping.
    """

    if not isinstance(target_pose, pin.SE3):
        raise TypeError(
            "target_pose must be a pinocchio.SE3 object."
        )

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    if not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer.")

    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative.")

    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive.")

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive.")

    if max_step_norm is not None:
        if not np.isfinite(max_step_norm) or max_step_norm <= 0.0:
            raise ValueError(
                "max_step_norm must be None or finite and positive."
            )

    # Initialize the Pinocchio configuration

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
        if not np.all(np.isfinite(q)):
            raise ValueError("q0 must contain finite values.")

        q = q.copy()

    # Store configurations and errors for diagnostics

    q_history = [q.copy()]
    error_history = []

    # Iterative damped SE(3) inverse kinematics

    for iteration in range(max_iterations + 1):

        # Compute the current pose and target transform

        current_pose = robot.frame_pose(
            q,
            frame=frame,
        )

        # Transform from current frame to desired frame
        current_to_target = current_pose.actInv(
            target_pose
        )

        # Logarithmic pose error in the local frame

        error = pin.log6(
            current_to_target
        ).vector

        error_norm = np.linalg.norm(
            error
        )

        error_history.append(
            error_norm
        )

        # Stop when the pose error is sufficiently small

        if error_norm < tolerance:

            return IKResult(
                q=q.copy(),
                success=True,
                error_norm=error_norm,
                iterations=iteration,
                q_history=np.asarray(q_history),
                error_history=np.asarray(error_history),
            )

        # Evaluate the final error without taking another step

        if iteration == max_iterations:
            break

        # Express the frame Jacobian in LOCAL coordinates

        J = robot.frame_jacobian(
            q,
            frame=frame,
            reference_frame=pin.ReferenceFrame.LOCAL,
        )

        # Correct the Jacobian for the SE(3) logarithmic error

        J_error = (
            -pin.Jlog6(
                current_to_target.inverse()
            )
            @ J
        )

        # Compute the damped least-squares update

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

        # Clip excessively large tangent-space updates

        if max_step_norm is not None:

            dq_norm = np.linalg.norm(
                dq
            )

            if dq_norm > max_step_norm:
                dq *= (
                    max_step_norm
                    / dq_norm
                )

        # Update configuration

        q = robot.integrate(
            q,
            dq,
            dt=step_size,
        )

        q_history.append(
            q.copy()
        )

    # Return the final configuration and diagnostics

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
    Solve inverse kinematics sequentially over Cartesian target poses.

    Each IK solution initializes the next solve, encouraging continuity
    of the joint trajectory, especially for redundant manipulators.

    The resulting Pinocchio configurations are converted to continuous
    joint coordinates using the Robot interface.

    Parameters
    ----------
    robot:
        Robot instance providing kinematics and configuration conversions.
    target_poses:
        Non-empty sequence of desired pin.SE3 poses.
    q0:
        Optional initial Pinocchio configuration, shape (nq,).
        Defaults to the neutral configuration.
    frame:
        Controlled frame. Defaults to the robot end-effector frame.
    stop_on_failure:
        If True, raise RuntimeError on the first unsuccessful IK solve.
        If False, continue using the last returned configuration.
    **ik_kwargs:
        Additional parameters forwarded to solve_pose_ik().

    Returns
    -------
    IKTrajectoryResult:
        Pinocchio configurations, continuous joint coordinates,
        success flags, pose errors, iteration counts and individual results.

    Raises
    ------
    ValueError:
        If the target sequence is empty or q0 is invalid.
    RuntimeError:
        If an IK solve fails and stop_on_failure is True.
    """

    target_poses = list(
        target_poses
    )

    if len(target_poses) == 0:
        raise ValueError(
            "target_poses cannot be empty."
        )

    # Initialize the configuration for sequential IK

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
        if not np.all(np.isfinite(q_current)):
            raise ValueError("q0 must contain finite values.")

        q_current = q_current.copy()

    # Store individual solutions and convergence diagnostics

    results = []
    q_sequence = []

    success_mask = []
    error_norms = []
    iterations = []

    # Warm-start each IK from the previous solution

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

    # Convert configurations to continuous joint coordinates

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

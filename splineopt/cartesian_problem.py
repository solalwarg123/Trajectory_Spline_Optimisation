"""Cartesian B-spline trajectory optimization problem.

This module defines the mathematical optimization problem.

It contains:
    - Cartesian tracking residuals,
    - analytical Jacobians,
    - objective terms,
    - trajectory constraints,
    - error metrics.

It deliberately contains no optimization algorithm.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from splineopt.constraints import build_joint_constraints


def _trapezoidal_weights(times):
    """Return trapezoidal quadrature weights."""

    times = np.asarray(
        times,
        dtype=float,
    ).reshape(-1)

    if times.size == 0:
        raise ValueError(
            "times cannot be empty."
        )

    if times.size == 1:
        return np.ones(1)

    dt = np.diff(times)

    if np.any(dt <= 0.0):
        raise ValueError(
            "times must be strictly increasing."
        )

    weights = np.empty_like(times)

    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]

    if times.size > 2:
        weights[1:-1] = 0.5 * (
            times[2:] - times[:-2]
        )

    return weights


class CartesianTrajectoryProblem:
    """Cartesian trajectory optimization problem.

    The optimization variable is the flattened vector of
    B-spline joint-space control points c.

    The problem defines

        minimize J(c)

        subject to

            l <= A c <= u.

    The solver is intentionally kept separate.
    """

    def __init__(
        self,
        robot,
        spline,
        tracking_times,
        target_poses,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        *,
        a0=None,
        vf=None,
        orientation_scale=0.0,
        lambda_smooth=1e-5,
        lambda_terminal=0.0,
    ):
        self.robot = robot
        self.spline = spline

        # -----------------------------------------------------
        # Cartesian reference
        # -----------------------------------------------------

        self.tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        self.target_poses = list(
            target_poses
        )

        # -----------------------------------------------------
        # Boundary conditions
        # -----------------------------------------------------

        self.q0 = np.asarray(
            q0,
            dtype=float,
        ).reshape(-1)

        self.v0 = np.asarray(
            v0,
            dtype=float,
        ).reshape(-1)

        self.a0 = (
            None
            if a0 is None
            else np.asarray(
                a0,
                dtype=float,
            ).reshape(-1)
        )

        self.vf = (
            None
            if vf is None
            else np.asarray(
                vf,
                dtype=float,
            ).reshape(-1)
        )

        # -----------------------------------------------------
        # Joint limits
        # -----------------------------------------------------

        self.q_min = np.asarray(
            q_min,
            dtype=float,
        ).reshape(-1)

        self.q_max = np.asarray(
            q_max,
            dtype=float,
        ).reshape(-1)

        self.v_min = np.asarray(
            v_min,
            dtype=float,
        ).reshape(-1)

        self.v_max = np.asarray(
            v_max,
            dtype=float,
        ).reshape(-1)

        self.a_min = np.asarray(
            a_min,
            dtype=float,
        ).reshape(-1)

        self.a_max = np.asarray(
            a_max,
            dtype=float,
        ).reshape(-1)

        # -----------------------------------------------------
        # Cost parameters
        # -----------------------------------------------------

        self.orientation_scale = float(
            orientation_scale
        )

        self.lambda_smooth = float(
            lambda_smooth
        )

        self.lambda_terminal = float(
            lambda_terminal
        )

        self.R2 = (
            self.spline.smoothness_matrix(
                derivative_order=2
            )
        )

        self._validate()

    # =========================================================
    # Validation
    # =========================================================

    def _validate(self):
        n_dof = self.spline.n_dof

        if n_dof != self.robot.nv:
            raise ValueError(
                "spline.n_dof must match robot.nv."
            )

        if self.tracking_times.size == 0:
            raise ValueError(
                "tracking_times must not be empty."
            )

        if np.any(
            np.diff(self.tracking_times) <= 0.0
        ):
            raise ValueError(
                "tracking_times must be strictly increasing."
            )

        if len(self.target_poses) != len(
            self.tracking_times
        ):
            raise ValueError(
                "target_poses and tracking_times "
                "must have the same length."
            )

        if np.any(
            self.tracking_times
            < self.spline.t_start
        ) or np.any(
            self.tracking_times
            > self.spline.t_end
        ):
            raise ValueError(
                "All tracking_times must lie "
                "inside the spline domain."
            )

        # The canonical Cartesian reference is a sequence
        # of poses, even for position-only tracking.
        for pose in self.target_poses:
            if (
                not hasattr(pose, "translation")
                or not hasattr(pose, "rotation")
            ):
                raise TypeError(
                    "target_poses must contain "
                    "SE(3)-like poses."
                )

        vectors = {
            "q0": self.q0,
            "v0": self.v0,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "a_min": self.a_min,
            "a_max": self.a_max,
        }

        if self.a0 is not None:
            vectors["a0"] = self.a0

        if self.vf is not None:
            vectors["vf"] = self.vf

        for name, value in vectors.items():
            if value.size != n_dof:
                raise ValueError(
                    f"{name} must contain "
                    f"{n_dof} values, "
                    f"got {value.size}."
                )

        if np.any(self.q_min > self.q_max):
            raise ValueError(
                "q_min must be <= q_max."
            )

        if np.any(self.v_min > self.v_max):
            raise ValueError(
                "v_min must be <= v_max."
            )

        if np.any(self.a_min > self.a_max):
            raise ValueError(
                "a_min must be <= a_max."
            )

        if self.orientation_scale < 0.0:
            raise ValueError(
                "orientation_scale must be non-negative."
            )

        if self.lambda_smooth < 0.0:
            raise ValueError(
                "lambda_smooth must be non-negative."
            )

        if self.lambda_terminal < 0.0:
            raise ValueError(
                "lambda_terminal must be non-negative."
            )

    # =========================================================
    # Constraints
    # =========================================================

    def linear_constraints(self):
        """Return the linear constraints l <= A c <= u."""

        return build_joint_constraints(
            spline=self.spline,
            q0=self.q0,
            v0=self.v0,
            q_min=self.q_min,
            q_max=self.q_max,
            v_min=self.v_min,
            v_max=self.v_max,
            a_min=self.a_min,
            a_max=self.a_max,
            a0=self.a0,
            vf=self.vf,
        )

    # =========================================================
    # Position-only tracking
    # =========================================================

    def position_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_positions=None,
    ):
        """Cartesian position residual and Jacobian."""

        if tracking_times is None:
            tracking_times = self.tracking_times

            if target_positions is None:
                target_positions = np.asarray([
                    pose.translation.copy()
                    for pose in self.target_poses
                ])

        else:
            tracking_times = np.asarray(
                tracking_times,
                dtype=float,
            ).reshape(-1)

            if target_positions is None:
                raise ValueError(
                    "target_positions must be provided "
                    "when custom tracking_times are used."
                )

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        target_positions = np.asarray(
            target_positions,
            dtype=float,
        )

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        n_variables = (
            self.spline.n_control_points
            * self.spline.n_dof
        )

        if c.size != n_variables:
            raise ValueError(
                f"Expected {n_variables} "
                "control-point values, "
                f"got {c.size}."
            )

        residual = np.empty(
            3 * len(tracking_times)
        )

        jacobian = np.empty(
            (
                3 * len(tracking_times),
                n_variables,
            )
        )

        for k, t in enumerate(
            tracking_times
        ):
            A_k = (
                self.spline.evaluation_matrix(
                    t,
                    derivative_order=0,
                )
            )

            theta = A_k @ c

            q_pin = (
                self.robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            position = (
                self.robot.frame_position(
                    q_pin
                )
            )

            J_frame = (
                self.robot.frame_jacobian(
                    q_pin
                )
            )

            J_position = J_frame[:3, :]

            J_control_points = (
                J_position @ A_k
            )

            row = slice(
                3 * k,
                3 * (k + 1),
            )

            residual[row] = (
                position
                - target_positions[k]
            )

            jacobian[row, :] = (
                J_control_points
            )

        return residual, jacobian

    def weighted_position_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_positions=None,
    ):
        """Time-weighted position residual."""

        if tracking_times is None:
            tracking_times = (
                self.tracking_times
            )

        residual, jacobian = (
            self.position_residual_and_jacobian(
                control_points=control_points,
                tracking_times=tracking_times,
                target_positions=target_positions,
            )
        )

        weights = _trapezoidal_weights(
            tracking_times
        )

        sqrt_weights = np.repeat(
            np.sqrt(weights),
            3,
        )

        return (
            sqrt_weights * residual,
            sqrt_weights[:, None]
            * jacobian,
        )

    # =========================================================
    # Full SE(3) tracking
    # =========================================================

    def pose_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_poses=None,
    ):
        """Full SE(3) residual and analytical Jacobian."""

        if tracking_times is None:
            tracking_times = (
                self.tracking_times
            )

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        if target_poses is None:
            target_poses = (
                self.target_poses
            )

        if len(target_poses) != len(
            tracking_times
        ):
            raise ValueError(
                "target_poses must have the same "
                "length as tracking_times."
            )

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        n_variables = (
            self.spline.n_control_points
            * self.spline.n_dof
        )

        if c.size != n_variables:
            raise ValueError(
                f"Expected {n_variables} "
                "control-point values, "
                f"got {c.size}."
            )

        residual = np.empty(
            6 * len(tracking_times)
        )

        jacobian = np.empty(
            (
                6 * len(tracking_times),
                n_variables,
            )
        )

        for k, t in enumerate(
            tracking_times
        ):
            A_k = (
                self.spline.evaluation_matrix(
                    t,
                    derivative_order=0,
                )
            )

            theta = A_k @ c

            q_pin = (
                self.robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            current_pose = (
                self.robot.frame_pose(
                    q_pin
                )
            )

            target_pose = target_poses[k]

            error_pose = (
                current_pose.actInv(
                    target_pose
                )
            )

            error = pin.log6(
                error_pose
            ).vector

            J_frame = (
                self.robot.frame_jacobian(
                    q_pin,
                    reference_frame=pin.LOCAL,
                )
            )

            J_error_theta = (
                -pin.Jlog6(
                    error_pose.inverse()
                )
                @ J_frame
            )

            J_error_control_points = (
                J_error_theta @ A_k
            )

            row = slice(
                6 * k,
                6 * (k + 1),
            )

            residual[row] = error

            jacobian[row, :] = (
                J_error_control_points
            )

        return residual, jacobian

    def weighted_pose_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_poses=None,
    ):
        """Apply spatial and temporal weighting."""

        if tracking_times is None:
            tracking_times = (
                self.tracking_times
            )

        residual, jacobian = (
            self.pose_residual_and_jacobian(
                control_points=control_points,
                tracking_times=tracking_times,
                target_poses=target_poses,
            )
        )

        weights = _trapezoidal_weights(
            tracking_times
        )

        spatial_scale = np.array(
            [
                1.0,
                1.0,
                1.0,
                self.orientation_scale,
                self.orientation_scale,
                self.orientation_scale,
            ]
        )

        sample_scale = np.tile(
            spatial_scale,
            len(tracking_times),
        )

        time_scale = np.repeat(
            np.sqrt(weights),
            6,
        )

        total_scale = (
            sample_scale
            * time_scale
        )

        return (
            total_scale * residual,
            total_scale[:, None]
            * jacobian,
        )

    # =========================================================
    # Optimization residual
    # =========================================================

    def tracking_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_poses=None,
    ):
        """Tracking residual selected by orientation_scale."""

        if tracking_times is None:
            tracking_times = (
                self.tracking_times
            )

        if target_poses is None:
            target_poses = (
                self.target_poses
            )

        if self.orientation_scale == 0.0:
            target_positions = np.asarray(
                [
                    pose.translation.copy()
                    for pose in target_poses
                ]
            )

            return (
                self.weighted_position_residual_and_jacobian(
                    control_points=control_points,
                    tracking_times=tracking_times,
                    target_positions=target_positions,
                )
            )

        return (
            self.weighted_pose_residual_and_jacobian(
                control_points=control_points,
                tracking_times=tracking_times,
                target_poses=target_poses,
            )
        )

    def residual_and_jacobian(
        self,
        control_points,
    ):
        """Complete least-squares residual used for optimization."""

        residual, jacobian = (
            self.tracking_residual_and_jacobian(
                control_points
            )
        )

        if self.lambda_terminal == 0.0:
            return residual, jacobian

        terminal_residual, terminal_jacobian = (
            self.tracking_residual_and_jacobian(
                control_points=control_points,
                tracking_times=np.array(
                    [self.tracking_times[-1]]
                ),
                target_poses=[
                    self.target_poses[-1]
                ],
            )
        )

        terminal_scale = np.sqrt(
            self.lambda_terminal
        )

        residual = np.concatenate(
            [
                residual,
                terminal_scale
                * terminal_residual,
            ]
        )

        jacobian = np.vstack(
            [
                jacobian,
                terminal_scale
                * terminal_jacobian,
            ]
        )

        return residual, jacobian

    # =========================================================
    # Objective
    # =========================================================

    def objective(
        self,
        control_points,
    ):
        """Evaluate the true nonlinear objective."""

        residual, _ = (
            self.residual_and_jacobian(
                control_points
            )
        )

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        tracking_cost = (
            residual @ residual
        )

        smoothness_cost = (
            self.lambda_smooth
            * c
            @ self.R2
            @ c
        )

        return float(
            tracking_cost
            + smoothness_cost
        )

    def gradient(
        self,
        control_points,
    ):
        """Return the gradient of the complete nonlinear objective."""

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        residual, jacobian = (
            self.residual_and_jacobian(
                c
            )
        )

        tracking_gradient = (
            2.0
            * jacobian.T
            @ residual
        )

        smoothness_gradient = (
            2.0
            * self.lambda_smooth
            * self.R2
            @ c
        )

        return (
            tracking_gradient
            + smoothness_gradient
        )

    # =========================================================
    # Metrics
    # =========================================================

    def error_metrics(
        self,
        control_points,
    ):
        """Return physically interpretable pose errors."""

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        position_errors = []
        orientation_errors = []

        for t, target_pose in zip(
            self.tracking_times,
            self.target_poses,
        ):
            A_k = (
                self.spline.evaluation_matrix(
                    t,
                    derivative_order=0,
                )
            )

            theta = A_k @ c

            q_pin = (
                self.robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            current_pose = (
                self.robot.frame_pose(
                    q_pin
                )
            )

            position_error = np.linalg.norm(
                current_pose.translation
                - target_pose.translation
            )

            rotation_error = (
                current_pose.rotation.T
                @ target_pose.rotation
            )

            orientation_error = (
                np.linalg.norm(
                    pin.log3(
                        rotation_error
                    )
                )
            )

            position_errors.append(
                position_error
            )

            orientation_errors.append(
                orientation_error
            )

        position_errors = np.asarray(
            position_errors
        )

        orientation_errors = np.asarray(
            orientation_errors
        )

        return {
            "position_rms": float(
                np.sqrt(
                    np.mean(
                        position_errors ** 2
                    )
                )
            ),
            "position_max": float(
                np.max(position_errors)
            ),
            "orientation_rms": float(
                np.sqrt(
                    np.mean(
                        orientation_errors ** 2
                    )
                )
            ),
            "orientation_max": float(
                np.max(orientation_errors)
            ),
        }

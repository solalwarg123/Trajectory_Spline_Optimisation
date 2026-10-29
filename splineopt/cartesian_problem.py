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
import scipy.sparse as sp

from splineopt.constraints import build_joint_constraints


def _trapezoidal_weights(times):
    """Return trapezoidal quadrature weights."""

    times = np.asarray(
        times,
        dtype=float,
    ).reshape(-1)

    if times.size == 0:
        raise ValueError("times cannot be empty.")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite.")

    if times.size == 1:
        return np.ones(1)

    dt = np.diff(times)

    if np.any(dt <= 0.0):
        raise ValueError("times must be strictly increasing.")

    weights = np.empty_like(times)

    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]

    if times.size > 2:
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])

    return weights


class CartesianTrajectoryProblem:
    """
    Cartesian B-spline trajectory optimization problem.

    The decision variables are the joint-space B-spline control
    points, flattened into a vector c.

    The objective combines:
        - Cartesian position or SE(3) tracking,
        - joint acceleration regularization,
        - an optional terminal tracking penalty.

    Tracking residuals are weighted by temporal quadrature.
    Joint position, velocity, acceleration and boundary conditions
    are expressed as linear constraints.

    Optional capsule-sphere collision constraints are nonlinear
    and evaluated at discrete collocation times.

    The problem provides both residual-only evaluations and
    residuals with analytical Jacobians, allowing optimizers to
    avoid unnecessary derivative computations.

    Fixed spline evaluation matrices, quadrature weights and
    constraint matrices are cached at initialization. The update()
    method replaces the MPC initial state and Cartesian targets
    without rebuilding those caches.

    This class defines the optimization problem only. Numerical
    solution algorithms are implemented separately.
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
        collision_constraints=None,
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

        self.target_poses = list(target_poses)

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

        self.orientation_scale = float(orientation_scale)

        self.lambda_smooth = float(lambda_smooth)

        self.lambda_terminal = float(lambda_terminal)

        self.R2 = self.spline.smoothness_matrix(derivative_order=2)

        self.collision_constraints = collision_constraints

        self._validate()

        # ---------------------------------------------------------
        # Fixed-horizon caches
        #
        # Computed once because spline knots, tracking times and
        # cost weights remain unchanged during MPC updates.
        #
        # _A_tracking: B-spline evaluation matrices
        # _tracking_weights: temporal quadrature weights
        # _position_total_scale: weighted position residual scaling
        # _pose_total_scale: weighted SE(3) residual scaling
        # _A_terminal: evaluation at the final tracking time
        #
        # Target positions are updated separately in update().
        # ---------------------------------------------------------

        self.n_variables = self.spline.n_control_points * self.spline.n_dof

        self._A_tracking = [
            self.spline.evaluation_matrix(
                t,
                derivative_order=0,
            )
            for t in self.tracking_times
        ]

        self._tracking_weights = _trapezoidal_weights(self.tracking_times)

        self._pose_spatial_scale = np.array(
            [
                1.0,
                1.0,
                1.0,
                self.orientation_scale,
                self.orientation_scale,
                self.orientation_scale,
            ],
            dtype=float,
        )

        self._pose_total_scale = np.tile(
            self._pose_spatial_scale,
            len(self.tracking_times),
        ) * np.repeat(
            np.sqrt(self._tracking_weights),
            6,
        )

        self._position_total_scale = np.repeat(
            np.sqrt(self._tracking_weights),
            3,
        )

        self._target_positions = np.asarray(
            [pose.translation.copy() for pose in self.target_poses],
            dtype=float,
        )

        # Terminal sample is already the last tracking sample.
        self._A_terminal = self._A_tracking[-1]

        # -----------------------------------------------------
        # Static linear constraints
        # -----------------------------------------------------

        (
            self._A_static,
            self._l_static,
            self._u_static,
        ) = build_joint_constraints(
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
    # Validation
    # =========================================================

    def _validate(self):
        n_dof = self.spline.n_dof

        if n_dof != self.robot.nv:
            raise ValueError("spline.n_dof must match robot.nv.")

        if self.tracking_times.size == 0:
            raise ValueError("tracking_times must not be empty.")

        if not np.all(np.isfinite(self.tracking_times)):
            raise ValueError("tracking_times must be finite.")

        if np.any(np.diff(self.tracking_times) <= 0.0):
            raise ValueError("tracking_times must be strictly increasing.")

        if len(self.target_poses) != len(self.tracking_times):
            raise ValueError(
                "target_poses and tracking_times " "must have the same length."
            )

        if np.any(self.tracking_times < self.spline.t_start) or np.any(
            self.tracking_times > self.spline.t_end
        ):
            raise ValueError(
                "All tracking_times must lie " "inside the spline domain.")

        # The canonical Cartesian reference is a sequence
        # of poses, even for position-only tracking.

        for pose in self.target_poses:
            if not hasattr(pose, "translation") or not hasattr(pose, "rotation"):
                raise TypeError("target_poses must contain SE(3)-like poses.")

            translation = np.asarray(pose.translation)
            rotation = np.asarray(pose.rotation)

            if translation.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError("Invalid target pose dimensions.")

            if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
                raise ValueError("Target poses must contain finite values.")

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
                    f"{name} must contain " f"{n_dof} values, " f"got {value.size}."
                )

        for name in ("q0", "v0", "a0", "vf"):
            value = getattr(self, name)
            if value is not None and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")

        if np.any(self.q_min > self.q_max):
            raise ValueError("q_min must be <= q_max.")

        if np.any(self.v_min > self.v_max):
            raise ValueError("v_min must be <= v_max.")

        if np.any(self.a_min > self.a_max):
            raise ValueError("a_min must be <= a_max.")

        for name, value in (
            ("orientation_scale", self.orientation_scale),
            ("lambda_smooth", self.lambda_smooth),
            ("lambda_terminal", self.lambda_terminal),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def _control_vector(self, control_points):
        """Return a flattened control-point vector of the expected size."""
        c = np.asarray(control_points, dtype=float).reshape(-1)

        if c.size != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} control-point values, " f"got {c.size}."
            )

        return c

    # =========================================================
    # Problem update (MPC)
    # =========================================================

    def update(self, q0, v0, target_poses):
        """
        Update the initial state and Cartesian references for MPC.

        Only time-varying data is modified:
            - Initial joint position q0 and velocity v0.
            - Corresponding linear equality constraint bounds.
            - Target SE(3) poses and cached target positions.

        The spline, tracking grid, quadrature weights, smoothness
        matrix and constraint matrix remain unchanged.

        Inputs are validated before modifying the problem. Target
        poses are copied to prevent external mutation.

        The MPC horizon retains the same time grid and duration.
        """

        n_dof = self.spline.n_dof

        q0 = np.asarray(q0, dtype=float).reshape(-1).copy()
        v0 = np.asarray(v0, dtype=float).reshape(-1).copy()
        target_poses = list(target_poses)

        # Validate before modifying the problem
        if q0.size != n_dof or v0.size != n_dof:
            raise ValueError(f"q0 and v0 must contain {n_dof} values.")

        if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(v0)):
            raise ValueError("q0 and v0 must be finite.")

        if len(target_poses) != len(self.tracking_times):
            raise ValueError(
                "target_poses and tracking_times " "must have the same length."
            )

        for pose in target_poses:
            if not (hasattr(pose, "translation") and hasattr(pose, "rotation")):
                raise TypeError("target_poses must contain SE(3)-like poses.")

            translation = np.asarray(pose.translation)
            rotation = np.asarray(pose.rotation)

            if translation.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError(
                    "Target poses must have translation shape (3,) "
                    "and rotation shape (3, 3)."
                )

            if not np.all(np.isfinite(translation)) or not np.all(
                np.isfinite(rotation)
            ):
                raise ValueError("Target poses must contain finite values.")

        # Copy poses to avoid external mutation of cached data
        target_poses = [
            pin.SE3(
                np.asarray(pose.rotation).copy(),
                np.asarray(pose.translation).copy(),
            )
            for pose in target_poses
        ]

        target_positions = np.asarray(
            [pose.translation.copy() for pose in target_poses]
        )

        # Update initial conditions
        self.q0 = q0
        self.v0 = v0

        # The first two blocks of the joint constraints
        # correspond to q(0) = q0 and dq(0) = v0.
        self._l_static[:n_dof] = q0
        self._u_static[:n_dof] = q0

        self._l_static[n_dof: 2 * n_dof] = v0
        self._u_static[n_dof: 2 * n_dof] = v0

        # Update Cartesian reference
        self.target_poses = target_poses
        self._target_positions = target_positions

    # =========================================================
    # Constraints
    # =========================================================

    def linear_constraints(self):
        """Return the linear constraints l <= A c <= u."""

        return (
            self._A_static,
            self._l_static,
            self._u_static,
        )

    def linearized_constraints(
        self,
        control_points,
    ):
        """Return constraints linearized around the current trajectory.

        Static joint-space constraints are always included.

        If collision constraints are enabled, their first-order
        approximation around `control_points` is appended.

        Returns
        -------
        A, l, u :
            OSQP constraint matrices such that

                l <= A c <= u.

        Collision constraints are linearized around the supplied
        control points and are local first-order approximations.
        """

        A, l, u = self.linear_constraints()

        collision = self.collision_constraints

        # Fast path: exactly the previous problem when
        # collisions are not used.
        if collision is None or not collision.has_constraints:
            return A, l, u

        H_col, h_col = collision.linearize(control_points)

        A = sp.vstack(
            [
                A,
                H_col,
            ],
            format="csc",
        )

        l = np.concatenate(
            [
                l,
                h_col,
            ]
        )

        u = np.concatenate(
            [
                u,
                np.full(
                    h_col.size,
                    np.inf,
                ),
            ]
        )

        return A, l, u

    # =========================================================
    # Position-only tracking
    # =========================================================

    def position_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_positions=None,
    ):
        """
        Compute Cartesian position tracking residuals and Jacobian.

        At each tracking time t_k, the joint configuration is:

            q_k = A_k @ c

        The position residual is:

            r_k = p(q_k) - p_target,k

        Its Jacobian with respect to the B-spline control points is:

            J_k = J_position(q_k) @ A_k

        Args:
            control_points: Joint-space B-spline control points.
            tracking_times: Optional custom evaluation times.
            target_positions: Optional Cartesian reference positions,
                shape (n_times, 3).

        Returns:
            residual: Unweighted position errors, shape (3*n_times,).
            jacobian: Analytical Jacobian, shape
                (3*n_times, n_variables).

        When no custom tracking data is provided, cached spline
        evaluation matrices and reference positions are reused.
        """

        use_cached_tracking = tracking_times is None and target_positions is None

        if use_cached_tracking:

            tracking_times = self.tracking_times

            target_positions = self._target_positions

            A_matrices = self._A_tracking

        else:

            if tracking_times is None:
                tracking_times = self.tracking_times

            tracking_times = np.asarray(
                tracking_times,
                dtype=float,
            ).reshape(-1)

            if target_positions is None:
                raise ValueError(
                    "target_positions must be provided "
                    "when custom tracking_times are used."
                )

            target_positions = np.asarray(
                target_positions,
                dtype=float,
            )

            A_matrices = [
                self.spline.evaluation_matrix(
                    t,
                    derivative_order=0,
                )
                for t in tracking_times
            ]

        c = self._control_vector(control_points)

        residual = np.empty(3 * len(tracking_times))

        jacobian = np.empty(
            (
                3 * len(tracking_times),
                self.n_variables,
            )
        )

        for k, A_k in enumerate(A_matrices):

            theta = A_k @ c

            q_pin = self.robot.joint_positions_to_configuration(theta)

            position = self.robot.frame_position(q_pin)

            J_frame = self.robot.frame_jacobian(q_pin)

            J_position = J_frame[:3, :]

            J_control_points = J_position @ A_k

            row = slice(
                3 * k,
                3 * (k + 1),
            )

            residual[row] = position - target_positions[k]

            jacobian[row, :] = J_control_points

        return (
            residual,
            jacobian,
        )

    def weighted_position_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_positions=None,
    ):
        """
        Apply trapezoidal time-integration weights to position tracking.

        Each residual and its Jacobian are multiplied by sqrt(w_k),
        where w_k is the quadrature weight at tracking time t_k.

        Consequently, the squared residual norm approximates:

            integral ||p(q(t)) - p_target(t)||² dt

        The cached scaling vector is reused for the standard tracking
        grid. Custom evaluation times require new quadrature weights.

        Returns:
            Weighted residual and analytical Jacobian.
        """

        use_cache = tracking_times is None and target_positions is None

        residual, jacobian = self.position_residual_and_jacobian(
            control_points=control_points,
            tracking_times=tracking_times,
            target_positions=target_positions,
        )

        if use_cache:
            scale = self._position_total_scale
        else:
            times = self.tracking_times if tracking_times is None else tracking_times
            weights = _trapezoidal_weights(times)
            scale = np.repeat(np.sqrt(weights), 3)

        return scale * residual, scale[:, None] * jacobian

    # =========================================================
    # Full SE(3) tracking
    # =========================================================

    def pose_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_poses=None,
    ):
        """
        Compute full SE(3) tracking residuals and analytical Jacobian.

        For each tracking configuration q_k, define the relative pose:

            T_error = T_current(q_k)^(-1) @ T_target,k

        The 6D tracking residual is expressed in the local frame:

            r_k = log6(T_error)

        The Jacobian with respect to joint coordinates is:

            J_r = -Jlog6(T_error^(-1)) @ J_frame_LOCAL

        Using the B-spline mapping q_k = A_k @ c:

            J_k = J_r @ A_k

        The residual contains three translational and three
        rotational components in Pinocchio's SE(3) tangent convention.

        Args:
            control_points: Joint-space B-spline control points.
            tracking_times: Optional custom evaluation times.
            target_poses: Optional SE(3) reference poses.

        Returns:
            residual: Unweighted SE(3) errors, shape (6*n_times,).
            jacobian: Analytical Jacobian, shape
                (6*n_times, n_variables).

        Cached spline evaluation matrices and target poses are reused
        when no custom tracking data is provided.
        """

        use_cached_tracking = tracking_times is None and target_poses is None

        if use_cached_tracking:

            tracking_times = self.tracking_times

            target_poses = self.target_poses

            A_matrices = self._A_tracking

        else:

            if tracking_times is None:
                tracking_times = self.tracking_times

            tracking_times = np.asarray(
                tracking_times,
                dtype=float,
            ).reshape(-1)

            if target_poses is None:
                target_poses = self.target_poses

            A_matrices = [
                self.spline.evaluation_matrix(
                    t,
                    derivative_order=0,
                )
                for t in tracking_times
            ]

        if len(target_poses) != len(tracking_times):
            raise ValueError(
                "target_poses must have the same " "length as tracking_times."
            )

        c = self._control_vector(control_points)

        residual = np.empty(6 * len(tracking_times))

        jacobian = np.empty(
            (
                6 * len(tracking_times),
                self.n_variables,
            )
        )

        for k, A_k in enumerate(A_matrices):

            theta = A_k @ c

            q_pin = self.robot.joint_positions_to_configuration(theta)

            current_pose = self.robot.frame_pose(q_pin)

            target_pose = target_poses[k]

            error_pose = current_pose.actInv(target_pose)

            error = pin.log6(error_pose).vector

            J_frame = self.robot.frame_jacobian(
                q_pin,
                reference_frame=pin.LOCAL,
            )

            J_error_theta = -pin.Jlog6(error_pose.inverse()) @ J_frame

            J_error_control_points = J_error_theta @ A_k

            row = slice(
                6 * k,
                6 * (k + 1),
            )

            residual[row] = error

            jacobian[row, :] = J_error_control_points

        return (
            residual,
            jacobian,
        )

    def weighted_pose_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_poses=None,
    ):
        """
        Apply spatial and temporal weights to SE(3) tracking.

        The six residual components are scaled by:

            S = diag(1, 1, 1, s, s, s)

        where s = orientation_scale.

        Each sample is additionally multiplied by sqrt(w_k), with
        w_k the trapezoidal quadrature weight.

        This balances translation and rotation tracking in the
        least-squares objective while approximating time integration.

        Both residuals and Jacobians receive the same scaling.
        The standard tracking grid uses cached scaling factors.

        Returns:
            Weighted SE(3) residual and analytical Jacobian.
        """

        use_cache = tracking_times is None and target_poses is None

        residual, jacobian = self.pose_residual_and_jacobian(
            control_points=control_points,
            tracking_times=tracking_times,
            target_poses=target_poses,
        )

        if use_cache:
            scale = self._pose_total_scale
        else:
            times = self.tracking_times if tracking_times is None else tracking_times
            weights = _trapezoidal_weights(times)

            scale = np.tile(self._pose_spatial_scale, len(weights)) * np.repeat(
                np.sqrt(weights), 6
            )

        return scale * residual, scale[:, None] * jacobian

    # =========================================================
    # Optimization residual
    # =========================================================

    def position_residual(
        self,
        control_points,
    ):
        """Return weighted position tracking residuals without computing Jacobians."""

        c = self._control_vector(control_points)

        residual = np.empty(3 * len(self.tracking_times))

        for k, A_k in enumerate(self._A_tracking):

            theta = A_k @ c

            q_pin = self.robot.joint_positions_to_configuration(theta)

            position = self.robot.frame_position(q_pin)

            row = slice(
                3 * k,
                3 * (k + 1),
            )

            residual[row] = position - self._target_positions[k]

        return self._position_total_scale * residual

    def pose_residual(
        self,
        control_points,
    ):
        """Return weighted SE(3) tracking residuals without computing Jacobians."""

        c = self._control_vector(control_points)

        residual = np.empty(6 * len(self.tracking_times))

        for k, (
            A_k,
            target_pose,
        ) in enumerate(
            zip(
                self._A_tracking,
                self.target_poses,
            )
        ):

            theta = A_k @ c

            q_pin = self.robot.joint_positions_to_configuration(theta)

            current_pose = self.robot.frame_pose(q_pin)

            error_pose = current_pose.actInv(target_pose)

            error = pin.log6(error_pose).vector

            row = slice(
                6 * k,
                6 * (k + 1),
            )

            residual[row] = error

        return self._pose_total_scale * residual

    def tracking_residual(
        self,
        control_points,
    ):
        """Select the residual-only tracking path.

        Position-only tracking is used when orientation_scale is zero.
        Otherwise, full SE(3) tracking is used.

        This avoids Jacobian computations during objective evaluations
        and line searches.
        """

        if self.orientation_scale == 0.0:
            return self.position_residual(control_points)

        return self.pose_residual(control_points)

    def tracking_residual_and_jacobian(
        self,
        control_points,
        tracking_times=None,
        target_poses=None,
    ):
        """Tracking residual selected by orientation_scale."""

        use_cached_tracking = tracking_times is None and target_poses is None

        # ---------------------------------------------------------
        # Standard optimization path:
        # keep arguments as None so downstream functions use
        # cached tracking matrices, targets and weights.
        # ---------------------------------------------------------

        if use_cached_tracking:

            if self.orientation_scale == 0.0:

                return self.weighted_position_residual_and_jacobian(
                    control_points=control_points,
                )

            return self.weighted_pose_residual_and_jacobian(
                control_points=control_points,
            )

        # ---------------------------------------------------------
        # Custom tracking grid
        # ---------------------------------------------------------

        if tracking_times is None:
            tracking_times = self.tracking_times

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        if target_poses is None:
            target_poses = self.target_poses

        if self.orientation_scale == 0.0:

            target_positions = np.asarray(
                [pose.translation.copy() for pose in target_poses],
                dtype=float,
            )

            return self.weighted_position_residual_and_jacobian(
                control_points=control_points,
                tracking_times=tracking_times,
                target_positions=target_positions,
            )

        return self.weighted_pose_residual_and_jacobian(
            control_points=control_points,
            tracking_times=tracking_times,
            target_poses=target_poses,
        )

    def terminal_residual_and_jacobian(
        self,
        control_points,
    ):
        """
        Compute the terminal tracking residual and analytical Jacobian.

        The final tracking sample is penalized separately from the
        time-integrated tracking objective:

            cost_terminal = lambda_terminal * ||e(T)||²

        The returned residual and Jacobian are multiplied by
        sqrt(lambda_terminal), so their squared norm contributes
        the desired terminal cost.

        The error uses position-only or full SE(3) tracking depending
        on orientation_scale, consistently with the main tracking
        objective.

        The cached final B-spline evaluation matrix is reused.

        Returns:
            terminal_residual: Weighted terminal tracking error.
            terminal_jacobian: Derivative with respect to control
                points, shape (3 or 6, n_variables).
        """

        c = self._control_vector(control_points)

        A_k = self._A_terminal

        theta = A_k @ c

        q_pin = self.robot.joint_positions_to_configuration(theta)

        terminal_scale = np.sqrt(self.lambda_terminal)

        if self.orientation_scale == 0.0:

            position = self.robot.frame_position(q_pin)

            J_frame = self.robot.frame_jacobian(q_pin)

            residual = position - self._target_positions[-1]

            jacobian = J_frame[:3, :] @ A_k

        else:

            current_pose = self.robot.frame_pose(q_pin)

            target_pose = self.target_poses[-1]

            error_pose = current_pose.actInv(target_pose)

            error = pin.log6(error_pose).vector

            J_frame = self.robot.frame_jacobian(
                q_pin,
                reference_frame=pin.LOCAL,
            )

            J_error_theta = -pin.Jlog6(error_pose.inverse()) @ J_frame

            residual = self._pose_spatial_scale * error

            jacobian = self._pose_spatial_scale[:,
                                                None] * (J_error_theta @ A_k)

        return (
            terminal_scale * residual,
            terminal_scale * jacobian,
        )

    def residual_and_jacobian(
        self,
        control_points,
    ):
        """Assemble the complete least-squares residual and Jacobian.

        The result contains the weighted tracking residual and its
        analytical Jacobian, optionally followed by the terminal
        tracking contribution.

        The acceleration smoothness term is handled separately
        through R2 and is not included in these residuals.

        Returns:
            residual: Complete least-squares residual vector.
            jacobian: Corresponding matrix with n_variables columns.
        """

        residual, jacobian = self.tracking_residual_and_jacobian(
            control_points)

        if self.lambda_terminal == 0.0:
            return (
                residual,
                jacobian,
            )

        (
            terminal_residual,
            terminal_jacobian,
        ) = self.terminal_residual_and_jacobian(control_points)

        return (
            np.concatenate(
                [
                    residual,
                    terminal_residual,
                ]
            ),
            np.vstack(
                [
                    jacobian,
                    terminal_jacobian,
                ]
            ),
        )

    def residual(
        self,
        control_points,
    ):
        """Assemble the complete least-squares residual without Jacobians.

        Concatenates time-weighted Cartesian tracking residuals and,
        when enabled, the terminal tracking residual.

        Used by objective() to avoid derivative computations.
        """

        residual = self.tracking_residual(control_points)

        if self.lambda_terminal == 0.0:
            return residual

        c = self._control_vector(control_points)

        theta_terminal = self._A_terminal @ c

        q_pin_terminal = self.robot.joint_positions_to_configuration(
            theta_terminal)

        terminal_scale = np.sqrt(self.lambda_terminal)

        if self.orientation_scale == 0.0:

            terminal_position = self.robot.frame_position(q_pin_terminal)

            terminal_residual = terminal_position - self._target_positions[-1]

        else:

            terminal_pose = self.robot.frame_pose(q_pin_terminal)

            error_pose = terminal_pose.actInv(self.target_poses[-1])

            terminal_residual = self._pose_spatial_scale * \
                pin.log6(error_pose).vector

        return np.concatenate(
            [
                residual,
                terminal_scale * terminal_residual,
            ]
        )

    # =========================================================
    # Objective
    # =========================================================

    def objective(
        self,
        control_points,
    ):
        """
        Evaluate the complete nonlinear trajectory objective.

        The objective is:

            F(c) = ||r(c)||² + lambda_smooth * c.T @ R2 @ c

        where r(c) contains the time-weighted Cartesian tracking
        residuals and, if enabled, the terminal tracking residual.

        R2 is the B-spline acceleration smoothness matrix, so:

            c.T @ R2 @ c = integral ||q_ddot(t)||² dt

        Collision constraints are not penalized in this objective.
        They are handled separately as feasibility constraints.

        Returns:
            Scalar objective value.
        """

        residual = self.residual(control_points)

        c = self._control_vector(control_points)

        tracking_cost = residual @ residual

        smoothness_cost = self.lambda_smooth * c @ self.R2 @ c

        return float(tracking_cost + smoothness_cost)

    def gradient(
        self,
        control_points,
    ):
        """
        Compute the analytical gradient of the nonlinear objective.

        For the complete least-squares residual r(c), with
        Jacobian J(c), the gradient is:

            grad F(c) = 2 * J.T @ r + 2 * lambda_smooth * R2 @ c

        The residual Jacobian includes Cartesian tracking and the
        optional terminal penalty.

        Returns:
            Gradient vector, shape (n_variables,).
        """

        c = self._control_vector(control_points)

        residual, jacobian = self.residual_and_jacobian(c)

        tracking_gradient = 2.0 * jacobian.T @ residual

        smoothness_gradient = 2.0 * self.lambda_smooth * self.R2 @ c

        return tracking_gradient + smoothness_gradient

    # =========================================================
    # Metrics
    # =========================================================

    def error_metrics(
        self,
        control_points,
    ):
        """
        Compute physically interpretable Cartesian tracking errors.

        Position errors are Euclidean distances in meters.

        Orientation errors are SO(3) geodesic rotation angles in
        radians, computed from the relative rotation logarithm.

        Unlike optimization residuals, these metrics are unweighted:
        they are independent of orientation_scale, temporal quadrature
        weights and the terminal cost.

        Returns:
            Dictionary containing:
                position_rms: RMS position error, meters.
                position_max: Maximum position error, meters.
                orientation_rms: RMS orientation error, radians.
                orientation_max: Maximum orientation error, radians.
        """

        c = self._control_vector(control_points)

        position_errors = []
        orientation_errors = []

        for A_k, target_pose in zip(
            self._A_tracking,
            self.target_poses,
        ):
            theta = A_k @ c

            q_pin = self.robot.joint_positions_to_configuration(theta)

            current_pose = self.robot.frame_pose(q_pin)

            position_error = np.linalg.norm(
                current_pose.translation - target_pose.translation
            )

            rotation_error = current_pose.rotation.T @ target_pose.rotation

            orientation_error = np.linalg.norm(pin.log3(rotation_error))

            position_errors.append(position_error)

            orientation_errors.append(orientation_error)

        position_errors = np.asarray(position_errors)

        orientation_errors = np.asarray(orientation_errors)

        return {
            "position_rms": float(np.sqrt(np.mean(position_errors**2))),
            "position_max": float(np.max(position_errors)),
            "orientation_rms": float(np.sqrt(np.mean(orientation_errors**2))),
            "orientation_max": float(np.max(orientation_errors)),
        }

    def collision_distances(
        self,
        control_points,
    ):
        """Return nonlinear collision distances at collocations."""

        if (
            self.collision_constraints is None
            or not self.collision_constraints.has_constraints
        ):
            return np.empty(
                (0, 0),
                dtype=float,
            )

        return self.collision_constraints.distances(control_points)

    def minimum_collision_distance(
        self,
        control_points,
    ):
        """Minimum nonlinear capsule-obstacle distance."""

        if (
            self.collision_constraints is None
            or not self.collision_constraints.has_constraints
        ):
            return np.inf

        return self.collision_constraints.minimum_distance(control_points)

    def collision_feasible(
        self,
        control_points,
        tolerance=1e-9,
    ):
        """Check nonlinear collision feasibility."""

        if (
            self.collision_constraints is None
            or not self.collision_constraints.has_constraints
        ):
            return True

        return self.collision_constraints.is_feasible(
            control_points,
            tolerance=tolerance,
        )

    def nonlinear_constraints(
        self,
        control_points,
    ):
        """
        Evaluate nonlinear constraints.

        Currently these are the capsule-obstacle signed distances.
        """

        collision = self.collision_constraints

        if collision is None or not collision.has_constraints:
            return np.empty(
                0,
                dtype=float,
            )

        return np.asarray(
            collision.distances(control_points),
            dtype=float,
        ).reshape(-1)

    def nonlinear_constraints_jacobian(
        self,
        control_points,
    ):
        """
        Jacobian of nonlinear constraints with respect to the
        stacked B-spline control-point vector.
        """

        collision = self.collision_constraints

        n_variables = self.spline.n_control_points * self.spline.n_dof

        if collision is None or not collision.has_constraints:
            return sp.csc_matrix(
                (
                    0,
                    n_variables,
                )
            )

        _, jacobian = collision.values_and_jacobian(control_points)

        return jacobian

    def nonlinear_constraint_bounds(
        self,
    ):
        """
        Bounds associated with nonlinear constraints.

        Collision avoidance is

            d(c) >= safety_distance.
        """

        collision = self.collision_constraints

        if collision is None or not collision.has_constraints:
            return (
                np.empty(
                    0,
                    dtype=float,
                ),
                np.empty(
                    0,
                    dtype=float,
                ),
            )

        n_constraints = collision.n_constraints

        lower = np.full(
            n_constraints,
            collision.safety_distance,
            dtype=float,
        )

        upper = np.full(
            n_constraints,
            np.inf,
            dtype=float,
        )

        return lower, upper

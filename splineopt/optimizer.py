import numpy as np
import scipy.sparse as sp
import osqp
from dataclasses import dataclass, field
import pinocchio as pin


@dataclass
@dataclass
class CartesianIteration:
    iteration: int
    objective: float

    position_rms: float
    position_max: float

    orientation_rms: float
    orientation_max: float

    step_norm: float
    alpha: float

    osqp_status: str
    osqp_solve_time: float


@dataclass
class CartesianOptimizationResult:
    control_points: np.ndarray
    success: bool
    status: str
    iterations: int
    history: list = field(default_factory=list)


def _trapezoidal_weights(times):
    """
    Trapezoidal quadrature weights such that

        sum_k w_k f(t_k)

    approximates

        integral f(t) dt.
    """

    times = np.asarray(
        times,
        dtype=float,
    ).reshape(-1)

    if times.size == 0:
        raise ValueError("times cannot be empty.")

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


def _build_joint_constraints(
    spline,
    q0,
    v0,
    q_min,
    q_max,
    v_min,
    v_max,
    a_min,
    a_max,
    a0=None,
):
    """
    Build the linear B-spline constraints

        l <= A c <= u.
    """

    n_dof = spline.n_dof
    n_control_points = spline.n_control_points
    n_variables = n_control_points * n_dof

    def dof_vector(value, name):
        value = np.asarray(value, dtype=float).reshape(-1)

        if value.size != n_dof:
            raise ValueError(
                f"{name} must contain {n_dof} values, "
                f"got {value.size}."
            )

        return value

    # Initial conditions
    t0 = spline.t_start

    equality_matrices = [
        spline.evaluation_matrix(t0, derivative_order=0),
        spline.evaluation_matrix(t0, derivative_order=1),
    ]

    equality_values = [
        dof_vector(q0, "q0"),
        dof_vector(v0, "v0"),
    ]

    if a0 is not None:
        equality_matrices.append(
            spline.evaluation_matrix(
                t0,
                derivative_order=2,
            )
        )
        equality_values.append(
            dof_vector(a0, "a0")
        )

    A_eq = sp.csc_matrix(
        np.vstack(equality_matrices)
    )

    b_eq = np.concatenate(
        equality_values
    )

    # Position
    q_min = dof_vector(q_min, "q_min")
    q_max = dof_vector(q_max, "q_max")

    I = sp.eye(
        n_variables,
        format="csc",
    )

    l_position = np.tile(
        q_min,
        n_control_points,
    )

    u_position = np.tile(
        q_max,
        n_control_points,
    )

    # Velocity
    v_min = dof_vector(v_min, "v_min")
    v_max = dof_vector(v_max, "v_max")

    D1 = sp.csc_matrix(
        spline.derivative_control_matrix(order=1)
    )

    l_velocity = np.tile(
        v_min,
        n_control_points - 1,
    )

    u_velocity = np.tile(
        v_max,
        n_control_points - 1,
    )

    # Acceleration
    a_min = dof_vector(a_min, "a_min")
    a_max = dof_vector(a_max, "a_max")

    D2 = sp.csc_matrix(
        spline.derivative_control_matrix(order=2)
    )

    l_acceleration = np.tile(
        a_min,
        n_control_points - 2,
    )

    u_acceleration = np.tile(
        a_max,
        n_control_points - 2,
    )

    A = sp.vstack(
        [
            A_eq,
            I,
            D1,
            D2,
        ],
        format="csc",
    )

    l = np.concatenate(
        [
            b_eq,
            l_position,
            l_velocity,
            l_acceleration,
        ]
    )

    u = np.concatenate(
        [
            b_eq,
            u_position,
            u_velocity,
            u_acceleration,
        ]
    )

    return A, l, u


class JointTrajectoryOptimizer:
    """
    Joint-space B-spline trajectory optimizer.

    Solve the quadratic program

        minimize
            ||A_track c - b_track||^2
            + lambda_smooth * c.T @ R2 @ c

        subject to
            q(t0)    = q0
            qdot(t0) = v0

            q_min <= c    <= q_max
            v_min <= D1 c <= v_max
            a_min <= D2 c <= a_max

    Optionally:

            qddot(t0) = a0

    The QP is written in the OSQP form

        minimize
            1/2 c.T @ P @ c + g.T @ c

        subject to
            l <= A @ c <= u
    """

    def __init__(
        self,
        spline,
        lambda_smooth=1.0,
    ):
        self.spline = spline
        self.lambda_smooth = float(lambda_smooth)

        if self.lambda_smooth < 0:
            raise ValueError(
                "lambda_smooth must be non-negative."
            )

    # ========================================================
    # QP construction
    # ========================================================

    def build_constraints(
        self,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        a0=None,
    ):
        return _build_joint_constraints(
            spline=self.spline,
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
        )

    def build_qp(
        self,
        tracking_times,
        q_ref,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        a0=None,
    ):
        """
        Build the QP matrices

            P, g, A, l, u

        for OSQP.

        Parameters
        ----------
        tracking_times :
            Times at which the optimized trajectory is compared
            with the nominal trajectory.

        q_ref :
            Reference joint positions at tracking_times.
            Shape:

                (n_tracking_times, n_dof)

        q0 :
            Initial joint position.

        v0 :
            Initial joint velocity.

        q_min, q_max :
            Joint position limits.

        v_min, v_max :
            Joint velocity limits.

        a_min, a_max :
            Joint acceleration limits.

        a0 :
            Optional initial joint acceleration.
        """

        spline = self.spline

        n_dof = spline.n_dof
        n_control_points = spline.n_control_points

        n_variables = (
            n_control_points
            * n_dof
        )

        # ====================================================
        # Inputs
        # ====================================================

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        q_ref = np.asarray(
            q_ref,
            dtype=float,
        )

        n_tracking_times = len(
            tracking_times
        )

        if q_ref.shape != (
            n_tracking_times,
            n_dof,
        ):
            raise ValueError(
                "q_ref must have shape "
                f"({n_tracking_times}, {n_dof}), "
                f"got {q_ref.shape}."
            )

        if n_tracking_times == 0:
            raise ValueError(
                "tracking_times cannot be empty."
            )

        if np.any(
            tracking_times < spline.t_start
        ) or np.any(
            tracking_times > spline.t_end
        ):
            raise ValueError(
                "All tracking_times must lie inside "
                "the spline domain."
            )

        # ====================================================
        # Tracking matrix
        #
        # J_track =
        #
        #     ||A_track c - b_track||^2
        #
        # ====================================================

        A_track = np.vstack([
            spline.evaluation_matrix(
                t,
                derivative_order=0,
            )
            for t in tracking_times
        ])

        b_track = q_ref.reshape(-1)

        # ====================================================
        # Smoothness matrix
        #
        # J_smooth = c.T R2 c
        #
        # ====================================================

        R2 = spline.smoothness_matrix(
            derivative_order=2
        )

        # ====================================================
        # Quadratic objective
        #
        # J =
        #   ||A_track c - b_track||^2
        #   + lambda c.T R2 c
        #
        # OSQP expects:
        #
        #   1/2 c.T P c + g.T c
        #
        # Therefore:
        #
        #   P =
        #       2 (
        #           A_track.T A_track
        #           + lambda R2
        #         )
        #
        #   g =
        #       -2 A_track.T b_track
        #
        # ====================================================

        P_dense = 2.0 * (
            A_track.T @ A_track
            + self.lambda_smooth * R2
        )

        # Avoid small numerical asymmetries
        P_dense = 0.5 * (
            P_dense
            + P_dense.T
        )

        g = (
            -2.0
            * A_track.T
            @ b_track
        )

        # OSQP only needs the upper triangular part of P
        P = sp.triu(
            sp.csc_matrix(P_dense),
            format="csc",
        )

        A, l, u = self.build_constraints(
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
        )

        return P, g, A, l, u

    # ========================================================
    # Solve
    # ========================================================

    def solve(
        self,
        tracking_times,
        q_ref,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        a0=None,
        verbose=False,
    ):
        """
        Build and solve the joint-space trajectory QP.

        Returns
        -------
        control_points :
            Optimal control points with shape

                (n_control_points, n_dof)

        result :
            Complete OSQP result object.
        """

        P, g, A, l, u = self.build_qp(
            tracking_times=tracking_times,
            q_ref=q_ref,
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
        )

        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=g,
            A=A,
            l=l,
            u=u,
            verbose=verbose,
            polishing=True,
        )

        result = solver.solve()

        if not result.info.status.lower().startswith(
            "solved"
        ):
            raise RuntimeError(
                "OSQP failed with status: "
                f"{result.info.status}"
            )

        control_points = result.x.reshape(
            self.spline.n_control_points,
            self.spline.n_dof,
        )

        return control_points, result

    # ========================================================
    # Utilities
    # ========================================================

    def _dof_vector(
        self,
        value,
        name,
    ):
        """
        Convert an input to a vector of shape (n_dof,).
        """

        value = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if value.size != self.spline.n_dof:
            raise ValueError(
                f"{name} must contain "
                f"{self.spline.n_dof} values, "
                f"got {value.size}."
            )

        return value


class CartesianTrajectoryOptimizer:
    """
    Cartesian B-spline trajectory optimizer.

    First implementation: Cartesian position tracking only.

    The optimization variables are the B-spline control points c.
    """

    def __init__(
        self,
        robot,
        spline,
        lambda_smooth=1e-5,
        damping=1e-4,
        orientation_scale=0.10,
    ):
        self.robot = robot
        self.spline = spline

        self.lambda_smooth = float(
            lambda_smooth
        )

        self.damping = float(
            damping
        )

        self.orientation_scale = float(
            orientation_scale
        )

        if self.orientation_scale < 0.0:
            raise ValueError(
                "orientation_scale must be non negative."
            )

        if spline.n_dof != robot.nv:
            raise ValueError(
                "spline.n_dof must match robot.nv."
            )

        if self.lambda_smooth < 0.0:
            raise ValueError(
                "lambda_smooth must be non-negative."
            )

        if self.damping < 0.0:
            raise ValueError(
                "damping must be non-negative."
            )

        self.R2 = spline.smoothness_matrix(
            derivative_order=2
        )

    def residual_and_jacobian(
        self,
        control_points,
        tracking_times,
        target_positions,
    ):
        """
        Compute Cartesian position residual and its analytical
        Jacobian with respect to the B-spline control points.

        For each time t_k:

            theta_k = A_k c

            r_k = p(theta_k) - p_ref,k

            dr_k/dc = J_p(theta_k) A_k
        """

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        target_positions = np.asarray(
            target_positions,
            dtype=float,
        )

        if target_positions.shape != (
            len(tracking_times),
            3,
        ):
            raise ValueError(
                "target_positions must have shape "
                f"({len(tracking_times)}, 3)."
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
                f"Expected {n_variables} control-point values, "
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

            # B-spline:
            #
            # theta(t) = A_k c

            A_k = self.spline.evaluation_matrix(
                t,
                derivative_order=0,
            )

            theta = A_k @ c

            # Pinocchio configuration

            q_pin = (
                self.robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            # End-effector position

            position = (
                self.robot.frame_position(
                    q_pin
                )
            )

            # Pinocchio frame Jacobian
            #
            # LOCAL_WORLD_ALIGNED is the default in Robot,
            # so the translational rows correspond to the
            # derivative of the world-frame position.

            J_frame = (
                self.robot.frame_jacobian(
                    q_pin
                )
            )

            J_position = J_frame[:3, :]

            # Chain rule:
            #
            # dp/dc = dp/dtheta * dtheta/dc
            #
            #       = J_position @ A_k

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

    def weighted_residual_and_jacobian(
        self,
        control_points,
        tracking_times,
        target_positions,
    ):
        """
        Apply trapezoidal quadrature weights so that the
        least-squares objective approximates a time integral.
        """

        residual, jacobian = (
            self.residual_and_jacobian(
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

        residual_weighted = (
            sqrt_weights
            * residual
        )

        jacobian_weighted = (
            sqrt_weights[:, None]
            * jacobian
        )

        return (
            residual_weighted,
            jacobian_weighted,
        )

    def weighted_tracking_residual_and_jacobian(
        self,
        control_points,
        tracking_times,
        target_poses,
    ):
        """
        Tracking residual used by the optimizer.

        orientation_scale == 0:
            exact Cartesian position-only tracking.

        orientation_scale > 0:
            full SE(3) pose tracking.
        """

        if self.orientation_scale == 0.0:

            target_positions = np.array([
                pose.translation.copy()
                for pose in target_poses
            ])

            return self.weighted_residual_and_jacobian(
                control_points=control_points,
                tracking_times=tracking_times,
                target_positions=target_positions,
            )

        return self.weighted_pose_residual_and_jacobian(
            control_points=control_points,
            tracking_times=tracking_times,
            target_poses=target_poses,
        )

    def objective(
        self,
        control_points,
        tracking_times,
        target_poses,
    ):
        """
        Evaluate the true nonlinear pose-tracking objective.

            integral (
                ||translation error||²
                +
                orientation_scale² ||rotation error||²
            ) dt

            + lambda_smooth c.T R2 c
        """

        residual, _ = (
            self.weighted_tracking_residual_and_jacobian(
                control_points=control_points,
                tracking_times=tracking_times,
                target_poses=target_poses,
            )
        )

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        pose_cost = (
            residual @ residual
        )

        smoothness_cost = (
            self.lambda_smooth
            * c
            @ self.R2
            @ c
        )

        return float(
            pose_cost
            + smoothness_cost
        )

    def cartesian_error_metrics(
        self,
        control_points,
        tracking_times,
        target_positions,
    ):
        residual, _ = self.residual_and_jacobian(
            control_points=control_points,
            tracking_times=tracking_times,
            target_positions=target_positions,
        )

        errors = np.linalg.norm(
            residual.reshape(-1, 3),
            axis=1,
        )

        return {
            "rms": float(
                np.sqrt(
                    np.mean(errors ** 2)
                )
            ),
            "max": float(
                np.max(errors)
            ),
        }

    def build_gauss_newton_qp(
        self,
        current_control_points,
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
        a0=None,
    ):
        """
        Build one constrained Gauss-Newton QP.

        Around c_i:

            r(c) ~= r_i + J_i (c - c_i)

        We solve

            min_c
                ||r_i + J_i(c - c_i)||²
                + lambda c.T R2 c
                + damping ||c - c_i||²

        under the original linear B-spline constraints.
        """

        c_i = np.asarray(
            current_control_points,
            dtype=float,
        ).reshape(-1)

        # Weighted residual and Jacobian:
        #
        # ||r_w||² approximates
        #
        # integral ||r(t)||² dt

        r, J = (
            self.weighted_tracking_residual_and_jacobian(
                control_points=c_i,
                tracking_times=tracking_times,
                target_poses=target_poses,
            )
        )

        # Linearized residual:
        #
        # r_i + J_i(c - c_i)
        #
        # = J_i c + d

        d = r - J @ c_i

        # Gauss-Newton Hessian approximation

        H = (
            J.T @ J
            + self.lambda_smooth * self.R2
            + self.damping * np.eye(c_i.size)
        )

        # OSQP uses
        #
        # 1/2 c.T P c + g.T c

        P = 2.0 * H

        # Numerical symmetrization
        P = 0.5 * (
            P + P.T
        )

        # Linear term:
        #
        # 2 J.T d
        #
        # plus damping
        #
        # -2 mu c_i

        g = 2.0 * (
            J.T @ d
            - self.damping * c_i
        )

        P = sp.triu(
            sp.csc_matrix(P),
            format="csc",
        )

        A, l, u = _build_joint_constraints(
            spline=self.spline,
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
        )

        return P, g, A, l, u

    def gauss_newton_step(
        self,
        current_control_points,
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
        a0=None,
        verbose=False,
    ):
        """
        Solve one constrained Gauss-Newton QP.

        Returns the full-step QP solution c_qp.
        """

        P, g, A, l, u = (
            self.build_gauss_newton_qp(
                current_control_points=current_control_points,
                tracking_times=tracking_times,
                target_poses=target_poses,
                q0=q0,
                v0=v0,
                q_min=q_min,
                q_max=q_max,
                v_min=v_min,
                v_max=v_max,
                a_min=a_min,
                a_max=a_max,
                a0=a0,
            )
        )

        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=g,
            A=A,
            l=l,
            u=u,
            verbose=verbose,
            polishing=True,
        )

        result = solver.solve()

        if not result.info.status.lower().startswith(
            "solved"
        ):
            raise RuntimeError(
                "Gauss-Newton QP failed: "
                f"{result.info.status}"
            )

        return (
            np.asarray(
                result.x,
                dtype=float,
            ),
            result,
        )

    def pose_error_metrics(
        self,
        control_points,
        tracking_times,
        target_poses,
    ):
        """
        Report physically interpretable pose errors:

            position RMS / max [m]
            orientation RMS / max [rad]

        These metrics are unweighted and are therefore independent
        of orientation_scale.
        """

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        position_errors = []
        orientation_errors = []

        for t, target_pose in zip(
            tracking_times,
            target_poses,
        ):
            A_k = self.spline.evaluation_matrix(
                t,
                derivative_order=0,
            )

            theta = A_k @ c

            q_pin = (
                self.robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            current_pose = self.robot.frame_pose(
                q_pin
            )

            # Euclidean translation error
            position_error = np.linalg.norm(
                current_pose.translation
                - target_pose.translation
            )

            # Geodesic orientation error
            rotation_error = (
                current_pose.rotation.T
                @ target_pose.rotation
            )

            orientation_error = np.linalg.norm(
                pin.log3(
                    rotation_error
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
                np.max(
                    position_errors
                )
            ),
            "orientation_rms": float(
                np.sqrt(
                    np.mean(
                        orientation_errors ** 2
                    )
                )
            ),
            "orientation_max": float(
                np.max(
                    orientation_errors
                )
            ),
        }

    def solve(
        self,
        tracking_times,
        target_poses,
        initial_control_points,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        a0=None,
        max_iterations=20,
        step_tolerance=1e-6,
        cost_tolerance=1e-8,
        line_search_shrink=0.5,
        min_alpha=1e-4,
        verbose=False,
    ):
        """
        Solve the nonlinear Cartesian trajectory optimization problem
        with constrained Gauss-Newton iterations.

        Each iteration:

            1. linearizes the Cartesian tracking residual;
            2. solves the constrained Gauss-Newton QP;
            3. performs a backtracking line search;
            4. accepts the step if the true nonlinear objective decreases.
        """

        c = np.asarray(
            initial_control_points,
            dtype=float,
        ).reshape(-1)

        history = []

        current_cost = self.objective(
            c,
            tracking_times,
            target_poses,
        )

        if verbose:
            metrics = self.pose_error_metrics(
                c,
                tracking_times,
                target_poses,
            )

            print(
                f"Initial: "
                f"J={current_cost:.6e}  "
                f"pos RMS={1e3 * metrics['position_rms']:.3f} mm  "
                f"pos max={1e3 * metrics['position_max']:.3f} mm  "
                f"ori RMS={np.degrees(metrics['orientation_rms']):.3f} deg  "
                f"ori max={np.degrees(metrics['orientation_max']):.3f} deg"
            )

        for iteration in range(max_iterations):

            # -----------------------------------------------------
            # Solve local constrained Gauss-Newton model
            # -----------------------------------------------------

            c_qp, osqp_result = self.gauss_newton_step(
                current_control_points=c,
                tracking_times=tracking_times,
                target_poses=target_poses,
                q0=q0,
                v0=v0,
                q_min=q_min,
                q_max=q_max,
                v_min=v_min,
                v_max=v_max,
                a_min=a_min,
                a_max=a_max,
                a0=a0,
                verbose=False,
            )

            direction = c_qp - c

            full_step_norm = np.linalg.norm(
                direction
            )

            # -----------------------------------------------------
            # QP proposes essentially no movement
            # -----------------------------------------------------

            if full_step_norm <= (
                step_tolerance
                * (1.0 + np.linalg.norm(c))
            ):
                return CartesianOptimizationResult(
                    control_points=c.reshape(
                        self.spline.n_control_points,
                        self.spline.n_dof,
                    ),
                    success=True,
                    status="step_tolerance",
                    iterations=iteration,
                    history=history,
                )

            # -----------------------------------------------------
            # Backtracking line search
            #
            # Both c and c_qp satisfy the linear constraints.
            # Since the feasible set is convex:
            #
            # c + alpha (c_qp - c)
            #
            # remains feasible for alpha in [0, 1].
            # -----------------------------------------------------

            alpha = 1.0
            accepted = False

            while alpha >= min_alpha:

                candidate = (
                    c
                    + alpha * direction
                )

                candidate_cost = self.objective(
                    candidate,
                    tracking_times,
                    target_poses,
                )

                if candidate_cost < current_cost:
                    accepted = True
                    break

                alpha *= line_search_shrink

            if not accepted:
                return CartesianOptimizationResult(
                    control_points=c.reshape(
                        self.spline.n_control_points,
                        self.spline.n_dof,
                    ),
                    success=False,
                    status="line_search_failed",
                    iterations=iteration,
                    history=history,
                )

            # -----------------------------------------------------
            # Accept step
            # -----------------------------------------------------

            previous_cost = current_cost

            c = candidate
            current_cost = candidate_cost

            step_norm = (
                alpha * full_step_norm
            )

            metrics = self.pose_error_metrics(
                c,
                tracking_times,
                target_poses,
            )

            history.append(
                CartesianIteration(
                    iteration=iteration,
                    objective=current_cost,

                    position_rms=metrics[
                        "position_rms"
                    ],
                    position_max=metrics[
                        "position_max"
                    ],

                    orientation_rms=metrics[
                        "orientation_rms"
                    ],
                    orientation_max=metrics[
                        "orientation_max"
                    ],

                    step_norm=step_norm,
                    alpha=alpha,

                    osqp_status=(
                        osqp_result.info.status
                    ),

                    osqp_solve_time=float(
                        osqp_result.info.solve_time
                    ),
                )
            )

            if verbose:
                print(
                    f"[{iteration:02d}] "
                    f"J={current_cost:.6e}  "
                    f"pos RMS="
                    f"{1e3 * metrics['position_rms']:.3f} mm  "
                    f"ori RMS="
                    f"{np.degrees(metrics['orientation_rms']):.3f} deg  "
                    f"|step|={step_norm:.3e}  "
                    f"alpha={alpha:.3f}"
                )

            # -----------------------------------------------------
            # Convergence based on accepted step
            # -----------------------------------------------------

            if step_norm <= (
                step_tolerance
                * (1.0 + np.linalg.norm(c))
            ):
                return CartesianOptimizationResult(
                    control_points=c.reshape(
                        self.spline.n_control_points,
                        self.spline.n_dof,
                    ),
                    success=True,
                    status="step_tolerance",
                    iterations=iteration + 1,
                    history=history,
                )

            # -----------------------------------------------------
            # Convergence based on relative objective decrease
            # -----------------------------------------------------

            relative_decrease = (
                previous_cost
                - current_cost
            ) / max(
                abs(previous_cost),
                1e-12,
            )

            if relative_decrease <= cost_tolerance:
                return CartesianOptimizationResult(
                    control_points=c.reshape(
                        self.spline.n_control_points,
                        self.spline.n_dof,
                    ),
                    success=True,
                    status="cost_tolerance",
                    iterations=iteration + 1,
                    history=history,
                )

        return CartesianOptimizationResult(
            control_points=c.reshape(
                self.spline.n_control_points,
                self.spline.n_dof,
            ),
            success=False,
            status="max_iterations",
            iterations=max_iterations,
            history=history,
        )

    def pose_residual_and_jacobian(
        self,
        control_points,
        tracking_times,
        target_poses,
    ):
        """
        Compute the full SE(3) pose residual and its analytical
        Jacobian with respect to the B-spline control points.

        For each sample:

            theta_k = A_k c

            M_error = M(theta_k)^(-1) M_ref,k

            e_k = log6(M_error)

        and

            de_k / dc
                = de_k / dtheta_k * A_k.

        Pinocchio's LOCAL frame Jacobian is used because the
        SE(3) log residual is expressed in the local tangent space.
        """

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        if len(target_poses) != len(tracking_times):
            raise ValueError(
                "target_poses must have the same length "
                "as tracking_times."
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
                f"Expected {n_variables} control-point values, "
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

            # ---------------------------------------------------------
            # B-spline:
            #
            # theta(t) = A_k c
            # ---------------------------------------------------------

            A_k = self.spline.evaluation_matrix(
                t,
                derivative_order=0,
            )

            theta = A_k @ c

            q_pin = (
                self.robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            # ---------------------------------------------------------
            # Current and desired end-effector poses
            # ---------------------------------------------------------

            current_pose = self.robot.frame_pose(
                q_pin
            )

            target_pose = target_poses[k]

            # Pose of the target expressed in the current
            # end-effector frame:
            #
            #     M_error = M_current^{-1} M_target
            # ---------------------------------------------------------

            error_pose = current_pose.actInv(
                target_pose
            )

            error = pin.log6(
                error_pose
            ).vector

            # ---------------------------------------------------------
            # LOCAL geometric frame Jacobian
            # ---------------------------------------------------------

            J_frame = self.robot.frame_jacobian(
                q_pin,
                reference_frame=pin.LOCAL,
            )

            # ---------------------------------------------------------
            # Derivative of
            #
            # log6(M_current^{-1} M_target)
            #
            # with respect to joint tangent coordinates.
            #
            # This is the same differential used in a standard
            # Pinocchio SE(3) IK formulation.
            # ---------------------------------------------------------

            J_error_theta = (
                -pin.Jlog6(
                    error_pose.inverse()
                )
                @ J_frame
            )

            # Chain rule:
            #
            # de / dc
            #     = de / dtheta
            #       dtheta / dc
            # ---------------------------------------------------------

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
        tracking_times,
        target_poses,
    ):
        """
        Apply spatial weighting and trapezoidal time quadrature.

        orientation_scale has units of meters / radian.

        The resulting least-squares norm behaves approximately as

            ||translation error||²
            +
            orientation_scale² ||rotation error||².
        """

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

        spatial_scale = np.array([
            1.0,
            1.0,
            1.0,
            self.orientation_scale,
            self.orientation_scale,
            self.orientation_scale,
        ])

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
            total_scale[:, None] * jacobian,
        )

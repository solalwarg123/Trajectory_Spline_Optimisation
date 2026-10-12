import numpy as np
import scipy.sparse as sp
import osqp
from dataclasses import dataclass, field
from splineopt.constraints import build_joint_constraints


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
        vf=None,
    ):
        return build_joint_constraints(
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
            vf=vf,
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
        vf=None,
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
            vf=vf
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
        vf=None,
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
            vf=vf
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


class GaussNewtonSolver:
    """Constrained Gauss-Newton solver.

    This class knows how to solve a trajectory optimization
    problem but does not define the trajectory problem itself.

    The problem provides:
        - residuals and Jacobians,
        - the nonlinear objective,
        - linear constraints,
        - error metrics.

    The solver provides:
        - Gauss-Newton quadratic models,
        - OSQP solves,
        - backtracking line search,
        - convergence logic.
    """

    def __init__(
        self,
        *,
        damping=1e-4,
        damping_mode="static",
        damping_increase=2.0,
        damping_decrease=2.0,
        rho_low=0.25,
        rho_high=0.75,
        min_damping=1e-10,
        max_damping=1e6,
        max_damping_retries=2,
        max_iterations=20,
        step_tolerance=1e-6,
        cost_tolerance=1e-8,
        line_search_shrink=0.5,
        min_alpha=1e-4,
        collision_slack_weight=1e4,
        collision_merit_weight=1e4,
    ):
        self.damping = float(
            damping
        )

        self.damping_mode = str(
            damping_mode
        )

        self.damping_increase = float(
            damping_increase
        )

        self.damping_decrease = float(
            damping_decrease
        )

        self.rho_low = float(
            rho_low
        )

        self.rho_high = float(
            rho_high
        )

        self.min_damping = float(
            min_damping
        )

        self.max_damping = float(
            max_damping
        )

        self.max_damping_retries = int(
            max_damping_retries
        )

        self.max_iterations = int(
            max_iterations
        )

        self.step_tolerance = float(
            step_tolerance
        )

        self.cost_tolerance = float(
            cost_tolerance
        )

        self.line_search_shrink = float(
            line_search_shrink
        )

        self.min_alpha = float(
            min_alpha
        )
        self.collision_slack_weight = float(
            collision_slack_weight
        )
        self.collision_merit_weight = float(
            collision_merit_weight
        )

        self._validate_parameters()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_parameters(self):

        if self.damping < 0.0:
            raise ValueError(
                "damping must be non-negative."
            )

        if self.damping_mode not in {
            "static",
            "adaptive",
        }:
            raise ValueError(
                "damping_mode must be "
                "'static' or 'adaptive'."
            )

        if self.damping_increase <= 1.0:
            raise ValueError(
                "damping_increase must be > 1."
            )

        if self.damping_decrease <= 1.0:
            raise ValueError(
                "damping_decrease must be > 1."
            )

        if not (
            0.0
            <= self.rho_low
            < self.rho_high
        ):
            raise ValueError(
                "rho thresholds must satisfy "
                "0 <= rho_low < rho_high."
            )

        if self.min_damping <= 0.0:
            raise ValueError(
                "min_damping must be positive."
            )

        if self.max_damping < self.min_damping:
            raise ValueError(
                "max_damping must be >= min_damping."
            )

        if self.max_damping_retries < 0:
            raise ValueError(
                "max_damping_retries must be non-negative."
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive."
            )

        if self.step_tolerance < 0.0:
            raise ValueError(
                "step_tolerance must be non-negative."
            )

        if self.cost_tolerance < 0.0:
            raise ValueError(
                "cost_tolerance must be non-negative."
            )

        if not (
            0.0
            < self.line_search_shrink
            < 1.0
        ):
            raise ValueError(
                "line_search_shrink must be in (0, 1)."
            )

        if not (
            0.0
            < self.min_alpha
            <= 1.0
        ):
            raise ValueError(
                "min_alpha must be in (0, 1]."
            )
        if self.collision_slack_weight <= 0.0:
            raise ValueError(
                "collision_slack_weight must be positive."
            )
        if self.collision_merit_weight <= 0.0:
            raise ValueError(
                "collision_merit_weight must be positive."
            )

    # =========================================================
    # Gauss-Newton QP
    # =========================================================
    def _collision_constraints(
        self,
        problem,
    ):
        """
        Return active collision constraints, or None.

        This centralizes collision detection so that a problem
        without collisions follows exactly the historical solver path.
        """

        collision = getattr(
            problem,
            "collision_constraints",
            None,
        )

        if (
            collision is None
            or not collision.has_constraints
        ):
            return None

        return collision

    def _merit_value(
        self,
        problem,
        control_points,
    ):
        """
        Merit function used by the line search.

        Without collisions:

            Phi(c) = J(c)

        exactly as before.

        With collisions:

            Phi(c)
            =
            J(c)
            +
            lambda_merit
            * sum(max(0, d_safe - d_i(c))^2)
        """

        objective = float(
            problem.objective(
                control_points
            )
        )

        collision = (
            self._collision_constraints(
                problem
            )
        )

        # -----------------------------------------------------
        # Historical behaviour:
        #
        # no collision constraint -> merit == objective.
        # -----------------------------------------------------

        if collision is None:
            return objective

        distances = (
            collision.distances(
                control_points
            )
            .reshape(-1)
        )

        violations = np.maximum(
            0.0,
            collision.safety_distance
            - distances,
        )

        violation_cost = float(
            violations
            @ violations
        )

        return float(
            objective
            + self.collision_merit_weight
            * violation_cost
        )

    def _collision_feasible(
        self,
        problem,
        control_points,
        *,
        tolerance=1e-6,
    ):
        """
        Return True when there are no active collision constraints,
        or when all collision constraints are satisfied.

        This preserves the historical behaviour when collisions are
        absent while preventing convergence from being reported with
        positive collision slacks.
        """

        collision = (
            self._collision_constraints(
                problem
            )
        )

        if collision is None:
            return True

        return bool(
            collision.is_feasible(
                control_points,
                tolerance=tolerance,
            )
        )

    def build_gauss_newton_qp(
        self,
        problem,
        current_control_points,
        *,
        damping=None,
    ):
        """Build one constrained Gauss-Newton QP.

        The damped model is used to compute the step.

        The undamped Gauss-Newton model is returned separately
        and is used to evaluate the gain ratio rho.
        """

        c_i = np.asarray(
            current_control_points,
            dtype=float,
        ).reshape(-1)

        if damping is None:
            damping = self.damping

        damping = float(damping)

        if damping < 0.0:
            raise ValueError(
                "damping must be non-negative."
            )

        # ---------------------------------------------------------
        # Residual and Jacobian at current point
        # ---------------------------------------------------------

        r, J = problem.residual_and_jacobian(
            c_i
        )

        # Linearized residual:
        #
        # r(c) ~= r_i + J_i (c - c_i)
        #      = J_i c + d
        #
        d = r - J @ c_i

        # =========================================================
        # Undamped Gauss-Newton model
        #
        # Used only to evaluate rho.
        # =========================================================

        H_model = (
            J.T @ J
            + problem.lambda_smooth
            * problem.R2
        )

        P_model = 2.0 * H_model

        P_model = 0.5 * (
            P_model
            + P_model.T
        )

        g_model = 2.0 * (
            J.T @ d
        )

        # =========================================================
        # Damped QP
        #
        # Used by OSQP to compute the step.
        # =========================================================

        H_qp = (
            H_model
            + damping
            * np.eye(c_i.size)
        )

        P_qp = 2.0 * H_qp

        P_qp = 0.5 * (
            P_qp
            + P_qp.T
        )

        g_qp = 2.0 * (
            J.T @ d
            - damping * c_i
        )

        # OSQP only needs upper triangular P.
        P_osqp = sp.triu(
            sp.csc_matrix(P_qp),
            format="csc",
        )

        # =========================================================
        # Constraints
        # =========================================================
        #
        # Two distinct paths are intentionally kept:
        #
        #   1) no collision constraints:
        #
        #          historical QP in c only
        #
        #   2) collision constraints:
        #
        #          augmented QP in z = [c, s]
        #
        #      with collision slacks
        #
        #          H_col c + s >= h_col
        #          s >= 0
        #
        # =========================================================

        collision = (
            self._collision_constraints(
                problem
            )
        )

        # =========================================================
        # Historical path: NO COLLISIONS
        # =========================================================

        if collision is None:

            if hasattr(
                problem,
                "linearized_constraints",
            ):
                A, l, u = (
                    problem.linearized_constraints(
                        c_i
                    )
                )

            else:
                A, l, u = (
                    problem.linear_constraints()
                )

            return (
                P_osqp,
                g_qp,
                A,
                l,
                u,
                P_model,
                g_model,
            )
        # =========================================================
        # Collision-aware path
        # =========================================================

        A_static, l_static, u_static = (
            problem.linear_constraints()
        )

        A_static = sp.csc_matrix(
            A_static
        )

        l_static = np.asarray(
            l_static,
            dtype=float,
        ).reshape(-1)

        u_static = np.asarray(
            u_static,
            dtype=float,
        ).reshape(-1)

        H_col, h_col = (
            collision.linearize(
                c_i
            )
        )

        H_col = sp.csc_matrix(
            H_col
        )

        h_col = np.asarray(
            h_col,
            dtype=float,
        ).reshape(-1)

        n_control_variables = (
            c_i.size
        )

        n_slacks = (
            h_col.size
        )

        # Defensive fallback.
        #
        # In principle _collision_constraints() already guarantees
        # that n_slacks > 0, but keeping this branch makes the
        # behaviour robust to custom collision implementations.

        if n_slacks == 0:
            return (
                P_osqp,
                g_qp,
                A_static,
                l_static,
                u_static,
                P_model,
                g_model,
            )

        # ---------------------------------------------------------
        # Augmented objective
        #
        # z = [c, s]
        #
        # J_QP(z)
        # =
        # J_GN(c)
        # + lambda_slack ||s||^2
        #
        # OSQP uses:
        #
        #     1/2 z.T P z + g.T z
        #
        # hence:
        #
        #     P_ss = 2 lambda_slack I.
        # ---------------------------------------------------------

        P_slack = (
            2.0
            * self.collision_slack_weight
            * sp.eye(
                n_slacks,
                format="csc",
            )
        )

        P_augmented = sp.block_diag(
            (
                P_osqp,
                P_slack,
            ),
            format="csc",
        )

        g_augmented = np.concatenate(
            [
                g_qp,
                np.zeros(
                    n_slacks,
                    dtype=float,
                ),
            ]
        )

        # ---------------------------------------------------------
        # Static trajectory constraints
        #
        #     l_static <= A_static c <= u_static
        #
        # become
        #
        #     [A_static  0] [c] .
        #                     [s]
        # ---------------------------------------------------------

        A_static_augmented = sp.hstack(
            [
                A_static,
                sp.csc_matrix(
                    (
                        A_static.shape[0],
                        n_slacks,
                    )
                ),
            ],
            format="csc",
        )

        # ---------------------------------------------------------
        # Linearized collision constraints
        #
        #     H_col c + s >= h_col
        # ---------------------------------------------------------

        A_collision_augmented = sp.hstack(
            [
                H_col,
                sp.eye(
                    n_slacks,
                    format="csc",
                ),
            ],
            format="csc",
        )

        # ---------------------------------------------------------
        # Slack non-negativity
        #
        #     s >= 0
        # ---------------------------------------------------------

        A_slack_nonnegative = sp.hstack(
            [
                sp.csc_matrix(
                    (
                        n_slacks,
                        n_control_variables,
                    )
                ),
                sp.eye(
                    n_slacks,
                    format="csc",
                ),
            ],
            format="csc",
        )

        # ---------------------------------------------------------
        # Complete augmented constraint matrix
        # ---------------------------------------------------------

        A_augmented = sp.vstack(
            [
                A_static_augmented,
                A_collision_augmented,
                A_slack_nonnegative,
            ],
            format="csc",
        )

        lower_augmented = np.concatenate(
            [
                l_static,
                h_col,
                np.zeros(
                    n_slacks,
                    dtype=float,
                ),
            ]
        )

        upper_augmented = np.concatenate(
            [
                u_static,
                np.full(
                    n_slacks,
                    np.inf,
                    dtype=float,
                ),
                np.full(
                    n_slacks,
                    np.inf,
                    dtype=float,
                ),
            ]
        )

        return (
            P_augmented,
            g_augmented,
            A_augmented,
            lower_augmented,
            upper_augmented,
            P_model,
            g_model,
        )

    def _qp_model_value(
        self,
        P,
        g,
        control_points,
    ):
        """
        Evaluate the undamped quadratic Gauss-Newton model

            1/2 c.T P c + g.T c.

        This model is used only to compute the gain ratio rho.
        """

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        return float(
            0.5
            * c
            @ P
            @ c
            + g
            @ c
        )

    def _gain_ratio(
        self,
        problem,
        current_control_points,
        candidate_control_points,
        P,
        g,
    ):
        """
        Compare the actual decrease in the nonlinear objective
        with the decrease predicted by the local undamped
        Gauss-Newton quadratic model.

        The collision-aware merit function is intentionally not used
        here. Collision restoration is handled by the line search;
        rho is kept as a measure of the quality of the local tracking
        model, preserving the historical adaptive-damping logic.
        """

        c = np.asarray(
            current_control_points,
            dtype=float,
        ).reshape(-1)

        candidate = np.asarray(
            candidate_control_points,
            dtype=float,
        ).reshape(-1)

        current_cost = float(
            problem.objective(
                c
            )
        )

        candidate_cost = float(
            problem.objective(
                candidate
            )
        )

        actual_reduction = (
            current_cost
            - candidate_cost
        )

        predicted_reduction = (
            self._qp_model_value(
                P,
                g,
                c,
            )
            - self._qp_model_value(
                P,
                g,
                candidate,
            )
        )

        if predicted_reduction <= 0.0:
            return -np.inf

        return float(
            actual_reduction
            / predicted_reduction
        )
    # =========================================================
    # One Gauss-Newton / SQP step
    # =========================================================

    def gauss_newton_step(
        self,
        problem,
        current_control_points,
        *,
        damping=None,
        verbose=False,
    ):
        """
        Solve one constrained Gauss-Newton / SQP QP.

        Without collision constraints, the QP variable is simply c
        and the behaviour is identical to the historical solver.

        With collision constraints, OSQP internally solves in

            z = [c, s]

        where s are non-negative collision slacks.

        Only c is returned to the outer nonlinear solver.
        """

        c_current = np.asarray(
            current_control_points,
            dtype=float,
        ).reshape(-1)

        (
            P_osqp,
            g_qp,
            A,
            l,
            u,
            P_model,
            g_model,
        ) = self.build_gauss_newton_qp(
            problem=problem,
            current_control_points=(
                c_current
            ),
            damping=damping,
        )

        solver = osqp.OSQP()

        solver.setup(
            P=P_osqp,
            q=g_qp,
            A=A,
            l=l,
            u=u,
            verbose=verbose,
            polishing=True,
        )

        result = solver.solve()

        if not (
            result.info.status
            .lower()
            .startswith("solved")
        ):
            raise RuntimeError(
                "Gauss-Newton QP failed: "
                f"{result.info.status}"
            )

        solution = np.asarray(
            result.x,
            dtype=float,
        ).reshape(-1)

        n_control_variables = (
            c_current.size
        )

        if (
            solution.size
            < n_control_variables
        ):
            raise RuntimeError(
                "OSQP returned fewer variables "
                "than expected."
            )

        # -----------------------------------------------------
        # The first variables are always the B-spline control
        # points. Any remaining variables are collision slacks.
        # -----------------------------------------------------

        c_qp = solution[
            :n_control_variables
        ]

        self._last_collision_slacks = (
            solution[
                n_control_variables:
            ].copy()
        )

        return (
            c_qp,
            result,
            P_model,
            g_model,
        )

    # =========================================================
    # Solve
    # =========================================================

    def solve(
        self,
        problem,
        initial_control_points,
        *,
        verbose=False,
    ):
        """
        Solve the nonlinear trajectory optimization problem.

        Without active collision constraints, the solver follows the
        historical Gauss-Newton behaviour:

            - QP in the B-spline control points c,
            - objective-based line search,
            - objective-based convergence.

        With active collision constraints:

            - collision slacks are introduced inside the QP,
            - the line search uses a collision-aware merit function,
            - convergence is accepted only if collision constraints
              are satisfied.
        """

        # =========================================================
        # Initial state
        # =========================================================

        c = np.asarray(
            initial_control_points,
            dtype=float,
        ).reshape(-1)

        history = []

        current_cost = float(
            problem.objective(
                c
            )
        )

        # ---------------------------------------------------------
        # Current damping value
        #
        # self.damping is the configured initial damping.
        # mu is local to this solve and may evolve in adaptive mode.
        # ---------------------------------------------------------

        mu = float(
            np.clip(
                self.damping,
                self.min_damping,
                self.max_damping,
            )
        )

        # =========================================================
        # Initial reporting
        # =========================================================

        if verbose:

            metrics = (
                problem.error_metrics(
                    c
                )
            )

            collision = (
                self._collision_constraints(
                    problem
                )
            )

            if collision is None:

                min_collision_distance = (
                    None
                )

            else:

                min_collision_distance = (
                    collision.minimum_distance(
                        c
                    )
                )

            message = (
                f"Initial: "
                f"J={current_cost:.6e}  "
                f"pos RMS="
                f"{1e3 * metrics['position_rms']:.3f} mm  "
                f"pos max="
                f"{1e3 * metrics['position_max']:.3f} mm  "
                f"ori RMS="
                f"{np.degrees(metrics['orientation_rms']):.3f} deg  "
                f"ori max="
                f"{np.degrees(metrics['orientation_max']):.3f} deg"
            )

            if (
                min_collision_distance
                is not None
            ):

                message += (
                    f"  dmin="
                    f"{1e3 * min_collision_distance:.2f} mm"
                )

            print(
                message
            )

        # =========================================================
        # Main Gauss-Newton / SQP iterations
        # =========================================================

        for iteration in range(
            self.max_iterations
        ):

            # -----------------------------------------------------
            # Adaptive damping retries
            #
            # Static mode:
            #     this loop is executed only once unless the step
            #     succeeds.
            #
            # Adaptive mode:
            #     if the line search rejects the local model,
            #     damping is increased and the QP is rebuilt.
            # -----------------------------------------------------

            retry = 0

            while True:

                # =================================================
                # Local constrained Gauss-Newton / SQP model
                # =================================================

                (
                    c_qp,
                    osqp_result,
                    P_model,
                    g_model,
                ) = self.gauss_newton_step(
                    problem=problem,
                    current_control_points=c,
                    damping=mu,
                    verbose=False,
                )

                direction = (
                    c_qp - c
                )

                full_step_norm = float(
                    np.linalg.norm(
                        direction
                    )
                )

                # =================================================
                # QP proposes essentially no movement
                # =================================================
                #
                # Without collisions this is exactly the historical
                # stopping criterion.
                #
                # With collisions we may only declare convergence if
                # the true nonlinear collision constraints are also
                # satisfied.
                # =================================================

                if (
                    full_step_norm
                    <= (
                        self.step_tolerance
                        * (
                            1.0
                            + np.linalg.norm(
                                c
                            )
                        )
                    )
                    and self._collision_feasible(
                        problem,
                        c,
                    )
                ):

                    return (
                        CartesianOptimizationResult(
                            control_points=(
                                c.reshape(
                                    problem.spline.n_control_points,
                                    problem.spline.n_dof,
                                )
                            ),
                            success=True,
                            status="step_tolerance",
                            iterations=iteration,
                            history=history,
                        )
                    )

                # =================================================
                # Backtracking line search
                # =================================================
                #
                # Without collisions:
                #
                #     merit(c) = objective(c)
                #
                # and this reduces exactly to the historical line
                # search.
                #
                # With collisions:
                #
                #     merit(c)
                #       = objective(c)
                #       + lambda_merit * collision_violation(c)
                #
                # so an objective increase may temporarily be
                # accepted if collision feasibility improves enough.
                # =================================================

                alpha = 1.0
                accepted = False

                current_merit = (
                    self._merit_value(
                        problem,
                        c,
                    )
                )

                while (
                    alpha
                    >= self.min_alpha
                ):

                    candidate = (
                        c
                        + alpha
                        * direction
                    )

                    candidate_cost = float(
                        problem.objective(
                            candidate
                        )
                    )

                    candidate_merit = (
                        self._merit_value(
                            problem,
                            candidate,
                        )
                    )

                    if (
                        candidate_merit
                        < current_merit
                    ):

                        accepted = True
                        break

                    alpha *= (
                        self.line_search_shrink
                    )

                # =================================================
                # Successful line search
                # =================================================

                if accepted:

                    # -------------------------------------------------
                    # rho remains based on the original Gauss-Newton
                    # objective model.
                    #
                    # The merit function is only used to globalize the
                    # collision-aware SQP iterations.
                    # -------------------------------------------------

                    rho = self._gain_ratio(
                        problem=problem,
                        current_control_points=c,
                        candidate_control_points=(
                            candidate
                        ),
                        P=P_model,
                        g=g_model,
                    )

                    break

                # =================================================
                # Line search failed
                # =================================================

                # -------------------------------------------------
                # Static mode:
                #
                # preserve the historical behaviour.
                # -------------------------------------------------

                if (
                    self.damping_mode
                    == "static"
                ):

                    return (
                        CartesianOptimizationResult(
                            control_points=(
                                c.reshape(
                                    problem.spline.n_control_points,
                                    problem.spline.n_dof,
                                )
                            ),
                            success=False,
                            status="line_search_failed",
                            iterations=iteration,
                            history=history,
                        )
                    )

                # -------------------------------------------------
                # Adaptive mode:
                #
                # increase damping and rebuild the local QP.
                # -------------------------------------------------

                if (
                    retry
                    >= self.max_damping_retries
                ):

                    return (
                        CartesianOptimizationResult(
                            control_points=(
                                c.reshape(
                                    problem.spline.n_control_points,
                                    problem.spline.n_dof,
                                )
                            ),
                            success=False,
                            status="damping_limit",
                            iterations=iteration,
                            history=history,
                        )
                    )

                new_mu = float(
                    min(
                        mu
                        * self.damping_increase,
                        self.max_damping,
                    )
                )

                # Cannot increase damping any further.

                if (
                    new_mu
                    <= mu
                ):

                    return (
                        CartesianOptimizationResult(
                            control_points=(
                                c.reshape(
                                    problem.spline.n_control_points,
                                    problem.spline.n_dof,
                                )
                            ),
                            success=False,
                            status="damping_limit",
                            iterations=iteration,
                            history=history,
                        )
                    )

                retry += 1
                mu = new_mu

                if verbose:

                    print(
                        f"     line search failed -> "
                        f"increasing damping to "
                        f"mu={mu:.3e} "
                        f"(retry {retry}/"
                        f"{self.max_damping_retries})"
                    )

            # =====================================================
            # A step has been accepted
            # =====================================================

            previous_cost = (
                current_cost
            )

            previous_merit = (
                current_merit
            )

            step_norm = float(
                alpha
                * full_step_norm
            )

            # =====================================================
            # Adaptive damping update
            # =====================================================
            #
            # rho still measures agreement between the true tracking
            # objective and the undamped Gauss-Newton model.
            #
            # Collision restoration is handled separately through
            # the merit function.
            # =====================================================

            if (
                self.damping_mode
                == "adaptive"
            ):

                if (
                    rho
                    < self.rho_low
                ):

                    mu *= (
                        self.damping_increase
                    )

                elif (
                    rho
                    > self.rho_high
                    and alpha == 1.0
                ):

                    mu /= (
                        self.damping_decrease
                    )

                mu = float(
                    np.clip(
                        mu,
                        self.min_damping,
                        self.max_damping,
                    )
                )

            # =====================================================
            # Accept step
            # =====================================================

            c = candidate

            current_cost = (
                candidate_cost
            )

            current_merit = (
                candidate_merit
            )

            metrics = (
                problem.error_metrics(
                    c
                )
            )

            # =====================================================
            # Collision diagnostics
            # =====================================================

            collision = (
                self._collision_constraints(
                    problem
                )
            )

            if collision is None:

                min_collision_distance = (
                    None
                )

            else:

                min_collision_distance = (
                    collision.minimum_distance(
                        c
                    )
                )

            # =====================================================
            # Iteration history
            # =====================================================

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

            # =====================================================
            # Verbose output
            # =====================================================

            if verbose:

                if (
                    self.damping_mode
                    == "adaptive"
                ):

                    message = (
                        f"[{iteration:02d}] "
                        f"J={current_cost:.6e}  "
                        f"pos RMS="
                        f"{1e3 * metrics['position_rms']:.3f} mm  "
                        f"ori RMS="
                        f"{np.degrees(metrics['orientation_rms']):.3f} deg  "
                        f"|step|={step_norm:.3e}  "
                        f"alpha={alpha:.3f}  "
                        f"rho={rho:.3f}  "
                        f"mu={mu:.3e}  "
                        f"retry={retry}"
                    )

                else:

                    message = (
                        f"[{iteration:02d}] "
                        f"J={current_cost:.6e}  "
                        f"pos RMS="
                        f"{1e3 * metrics['position_rms']:.3f} mm  "
                        f"ori RMS="
                        f"{np.degrees(metrics['orientation_rms']):.3f} deg  "
                        f"|step|={step_norm:.3e}  "
                        f"alpha={alpha:.3f}"
                    )

                if (
                    min_collision_distance
                    is not None
                ):

                    message += (
                        f"  dmin="
                        f"{1e3 * min_collision_distance:.2f} mm"
                    )

                print(
                    message
                )

            # =====================================================
            # Step convergence
            # =====================================================
            #
            # With collision constraints, a small step is not enough:
            # the true nonlinear collision constraints must also be
            # satisfied.
            # =====================================================

            if (
                step_norm
                <= (
                    self.step_tolerance
                    * (
                        1.0
                        + np.linalg.norm(
                            c
                        )
                    )
                )
                and self._collision_feasible(
                    problem,
                    c,
                )
            ):

                return (
                    CartesianOptimizationResult(
                        control_points=(
                            c.reshape(
                                problem.spline.n_control_points,
                                problem.spline.n_dof,
                            )
                        ),
                        success=True,
                        status="step_tolerance",
                        iterations=(
                            iteration + 1
                        ),
                        history=history,
                    )
                )

            # =====================================================
            # Merit convergence
            # =====================================================
            #
            # Without collisions:
            #
            #     merit == objective
            #
            # so this is exactly the historical objective convergence
            # criterion.
            #
            # With collisions we measure progress in the same merit
            # function used by the line search.
            # =====================================================

            relative_decrease = (
                previous_merit
                - current_merit
            ) / max(
                abs(
                    previous_merit
                ),
                1e-12,
            )

            if (
                relative_decrease
                <= self.cost_tolerance
                and self._collision_feasible(
                    problem,
                    c,
                )
            ):

                return (
                    CartesianOptimizationResult(
                        control_points=(
                            c.reshape(
                                problem.spline.n_control_points,
                                problem.spline.n_dof,
                            )
                        ),
                        success=True,
                        status="cost_tolerance",
                        iterations=(
                            iteration + 1
                        ),
                        history=history,
                    )
                )

        # =========================================================
        # Maximum number of iterations reached
        # =========================================================

        return (
            CartesianOptimizationResult(
                control_points=(
                    c.reshape(
                        problem.spline.n_control_points,
                        problem.spline.n_dof,
                    )
                ),
                success=False,
                status="max_iterations",
                iterations=(
                    self.max_iterations
                ),
                history=history,
            )
        )

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
    # =========================================================
    # Gauss-Newton QP
    # =========================================================

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

        # ---------------------------------------------------------
        # Linear trajectory constraints
        # ---------------------------------------------------------

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

    def _qp_model_value(
        self,
        P,
        g,
        control_points,
    ):
        """Evaluate a full quadratic model."""

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        return float(
            0.5 * c @ P @ c
            + g @ c
        )

    def _gain_ratio(
        self,
        problem,
        current_control_points,
        candidate_control_points,
        P,
        g,
    ):
        """Compare actual and predicted objective decrease."""

        c = np.asarray(
            current_control_points,
            dtype=float,
        ).reshape(-1)

        candidate = np.asarray(
            candidate_control_points,
            dtype=float,
        ).reshape(-1)

        current_cost = problem.objective(c)
        candidate_cost = problem.objective(
            candidate
        )

        actual_reduction = (
            current_cost
            - candidate_cost
        )

        predicted_reduction = (
            self._qp_model_value(P, g, c)
            - self._qp_model_value(
                P,
                g,
                candidate,
            )
        )

        if predicted_reduction <= 0.0:
            return -np.inf

        return (
            actual_reduction
            / predicted_reduction
        )
    # =========================================================
    # One Gauss-Newton step
    # =========================================================

    def gauss_newton_step(
        self,
        problem,
        current_control_points,
        *,
        damping=None,
        verbose=False,
    ):
        """Solve one constrained Gauss-Newton QP.

        Returns
        -------
        c_qp :
            Full-step QP solution.

        result :
            OSQP result object.

        P_model, g_model :
            Undamped Gauss-Newton quadratic model used
            to evaluate the gain ratio rho.
        """

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
                current_control_points
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

        return (
            np.asarray(
                result.x,
                dtype=float,
            ),
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
        """Solve the nonlinear trajectory optimization problem."""

        c = np.asarray(
            initial_control_points,
            dtype=float,
        ).reshape(-1)

        history = []

        current_cost = problem.objective(c)

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

        if verbose:
            metrics = problem.error_metrics(c)

            print(
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

        # =========================================================
        # Main Gauss-Newton iterations
        # =========================================================

        for iteration in range(
            self.max_iterations
        ):

            # -----------------------------------------------------
            # Adaptive damping retries
            #
            # In static mode, this loop is executed only once.
            #
            # In adaptive mode, if the line search cannot find an
            # acceptable step, mu is increased and a new QP is
            # constructed.
            # -----------------------------------------------------

            retry = 0

            while True:

                # -------------------------------------------------
                # Local constrained Gauss-Newton model
                # -------------------------------------------------

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

                direction = c_qp - c

                full_step_norm = np.linalg.norm(
                    direction
                )

                # -------------------------------------------------
                # QP proposes essentially no movement
                # -------------------------------------------------

                if full_step_norm <= (
                    self.step_tolerance
                    * (
                        1.0
                        + np.linalg.norm(c)
                    )
                ):
                    return CartesianOptimizationResult(
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

                # -------------------------------------------------
                # Backtracking line search
                #
                # c and c_qp are feasible.
                #
                # Since the current constraints are linear, the
                # feasible set is convex. Therefore
                #
                #     c + alpha * (c_qp - c)
                #
                # remains feasible for alpha in [0, 1].
                # -------------------------------------------------

                alpha = 1.0
                accepted = False

                while alpha >= self.min_alpha:

                    candidate = (
                        c
                        + alpha * direction
                    )

                    candidate_cost = (
                        problem.objective(
                            candidate
                        )
                    )

                    if candidate_cost < current_cost:
                        accepted = True
                        break

                    alpha *= (
                        self.line_search_shrink
                    )

                # -------------------------------------------------
                # Successful line search
                # -------------------------------------------------

                if accepted:

                    rho = self._gain_ratio(
                        problem=problem,
                        current_control_points=c,
                        candidate_control_points=candidate,
                        P=P_model,
                        g=g_model,
                    )

                    break

                # -------------------------------------------------
                # Line search failed
                #
                # Static mode:
                # preserve the previous solver behaviour.
                # -------------------------------------------------

                if self.damping_mode == "static":

                    return CartesianOptimizationResult(
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

                # -------------------------------------------------
                # Adaptive mode:
                #
                # The current Gauss-Newton model was too
                # aggressive. Increase damping and rebuild the QP.
                # -------------------------------------------------

                if retry >= self.max_damping_retries:

                    return CartesianOptimizationResult(
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

                new_mu = float(
                    min(
                        mu * self.damping_increase,
                        self.max_damping,
                    )
                )

                # We cannot increase damping any further.
                if new_mu <= mu:

                    return CartesianOptimizationResult(
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

            previous_cost = current_cost

            step_norm = (
                alpha
                * full_step_norm
            )

            # -----------------------------------------------------
            # Update adaptive damping
            #
            # rho compares:
            #
            #   actual objective decrease
            #
            # with:
            #
            #   decrease predicted by the local quadratic model.
            # -----------------------------------------------------

            if self.damping_mode == "adaptive":

                if rho < self.rho_low:
                    mu *= self.damping_increase

                elif (
                    rho > self.rho_high
                    and alpha == 1.0
                ):
                    mu /= self.damping_decrease

                mu = float(
                    np.clip(
                        mu,
                        self.min_damping,
                        self.max_damping,
                    )
                )

            # -----------------------------------------------------
            # Accept step
            # -----------------------------------------------------

            c = candidate
            current_cost = candidate_cost

            metrics = problem.error_metrics(c)

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
                if self.damping_mode == "adaptive":
                    print(
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
            # Step convergence
            # -----------------------------------------------------

            if step_norm <= (
                self.step_tolerance
                * (
                    1.0
                    + np.linalg.norm(c)
                )
            ):
                return CartesianOptimizationResult(
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

            # -----------------------------------------------------
            # Objective convergence
            # -----------------------------------------------------

            relative_decrease = (
                previous_cost
                - current_cost
            ) / max(
                abs(previous_cost),
                1e-12,
            )

            if (
                relative_decrease
                <= self.cost_tolerance
            ):
                return CartesianOptimizationResult(
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

        # =========================================================
        # Maximum number of iterations reached
        # =========================================================

        return CartesianOptimizationResult(
            control_points=(
                c.reshape(
                    problem.spline.n_control_points,
                    problem.spline.n_dof,
                )
            ),
            success=False,
            status="max_iterations",
            iterations=self.max_iterations,
            history=history,
        )

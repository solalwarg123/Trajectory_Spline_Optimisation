import numpy as np
import scipy.sparse as sp
import osqp
from dataclasses import dataclass, field


@dataclass
class CartesianIteration:
    """Diagnostics recorded after an accepted Gauss-Newton step."""
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
    """Final trajectory, solver status and accepted-step history."""
    control_points: np.ndarray
    success: bool
    status: str
    iterations: int
    history: list = field(default_factory=list)


class GaussNewtonSolver:
    """
    Constrained Gauss-Newton solver for Cartesian trajectory optimization.

    The solver minimizes a nonlinear least-squares trajectory objective
    subject to joint-space constraints, using successive quadratic
    approximations solved with OSQP.

    At each iteration:
        1. Linearize the Cartesian tracking residuals.
        2. Build a regularized Gauss-Newton quadratic program.
        3. Solve the QP subject to trajectory constraints.
        4. Perform a backtracking line search.
        5. Update the damping parameter when adaptive mode is enabled.
        6. Check convergence and record iteration diagnostics.

    Two damping modes are supported:
        - static: Fixed Levenberg-Marquardt regularization.
        - adaptive: Adjust regularization using the gain ratio rho,
        with optional QP retries after rejected line searches.

    Collision handling is optional:
        - Without collisions, the QP uses control points only.
        - With collisions, non-negative slack variables are introduced
        for linearized collision constraints, and the line search
        uses a collision-aware merit function.

    The solver is independent of the robot model and trajectory
    representation. These are provided through the problem interface.
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
        self.damping = float(damping)

        self.damping_mode = str(damping_mode)

        self.damping_increase = float(damping_increase)

        self.damping_decrease = float(damping_decrease)

        self.rho_low = float(rho_low)

        self.rho_high = float(rho_high)

        self.min_damping = float(min_damping)

        self.max_damping = float(max_damping)

        self.max_damping_retries = int(max_damping_retries)

        self.max_iterations = int(max_iterations)

        self.step_tolerance = float(step_tolerance)

        self.cost_tolerance = float(cost_tolerance)

        self.line_search_shrink = float(line_search_shrink)

        self.min_alpha = float(min_alpha)
        self.collision_slack_weight = float(collision_slack_weight)
        self.collision_merit_weight = float(collision_merit_weight)

        self._validate_parameters()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_parameters(self):
        """Validate solver settings and damping configuration."""

        nonnegative = (
            "damping",
            "step_tolerance",
            "cost_tolerance",
        )
        positive = (
            "min_damping",
            "max_damping",
            "collision_slack_weight",
            "collision_merit_weight",
        )
        greater_than_one = (
            "damping_increase",
            "damping_decrease",
        )

        for name in nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        for name in greater_than_one:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 1.0:
                raise ValueError(f"{name} must be finite and > 1.")

        if self.damping_mode not in ("static", "adaptive"):
            raise ValueError("damping_mode must be 'static' or 'adaptive'.")

        if (
            not np.isfinite(self.rho_low)
            or not np.isfinite(self.rho_high)
            or not (0.0 <= self.rho_low < self.rho_high)
        ):
            raise ValueError(
                "rho thresholds must satisfy 0 <= rho_low < rho_high "
                "and must be finite."
            )

        if self.max_damping < self.min_damping:
            raise ValueError("max_damping must be >= min_damping.")

        if self.max_damping_retries < 0:
            raise ValueError("max_damping_retries must be non-negative.")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        if (
            not np.isfinite(self.line_search_shrink)
            or not 0.0 < self.line_search_shrink < 1.0
        ):
            raise ValueError("line_search_shrink must be in (0, 1).")

        if not np.isfinite(self.min_alpha) or not 0.0 < self.min_alpha <= 1.0:
            raise ValueError("min_alpha must be in (0, 1].")

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

        if collision is None or not collision.has_constraints:
            return None

        return collision

    def _merit_value(self, problem, control_points, *, objective=None):
        """
        Evaluate the merit function used by the backtracking line search.

        Without collisions, Phi(c) = F(c).

        With collisions:

            Phi(c) = F(c) + w_merit * sum(max(0, d_safe - d_i(c))**2)

        An already computed objective can be supplied to avoid
        redundant trajectory evaluations.
        """
        if objective is None:
            objective = float(problem.objective(control_points))

        collision = self._collision_constraints(problem)

        if collision is None:
            return objective

        distances = collision.distances(control_points).reshape(-1)
        violations = np.maximum(
            0.0,
            collision.safety_distance - distances,
        )

        violation_cost = float(violations @ violations)

        return float(
            objective + self.collision_merit_weight * violation_cost
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

        collision = self._collision_constraints(problem)

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
        """
        Build the constrained Gauss-Newton quadratic subproblem.

        At the current control points c_i, linearize the residual:

            r(c) ≈ r_i + J_i @ (c - c_i)
                = J_i @ c + d

        where:

            d = r_i - J_i @ c_i

        Including acceleration regularization gives the undamped model:

            H = J_i.T @ J_i + lambda_smooth * R2
            P_model = 2 * H
            g_model = 2 * J_i.T @ d

        Levenberg-Marquardt damping adds:

            damping * ||c - c_i||²

        giving the QP coefficients:

            P_qp = P_model + 2 * damping * I
            g_qp = g_model - 2 * damping * c_i

        Two constraint formulations are supported:
            - Without collisions: a QP directly in control points c.
            - With collisions: an augmented QP in z = [c, s], where
            s >= 0 are penalized collision slack variables.

        The undamped model is retained to compute the gain ratio rho,
        independently of the regularization used to generate the step.

        Args:
            problem: Trajectory optimization problem.
            current_control_points: Current flattened B-spline coefficients.
            damping: Optional damping override.

        Returns:
            P_osqp, g_qp, A, l, u:
                Quadratic program in OSQP form, possibly augmented
                with collision slack variables.
            P_model, g_model:
                Undamped Gauss-Newton model coefficients in control-point
                coordinates, used to calculate rho.
        """

        c_i = np.asarray(
            current_control_points,
            dtype=float,
        ).reshape(-1)

        if damping is None:
            damping = self.damping

        damping = float(damping)

        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be non-negative.")

        # ---------------------------------------------------------
        # Residual and Jacobian at current point
        # ---------------------------------------------------------

        r, J = problem.residual_and_jacobian(c_i)

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

        H_model = J.T @ J + problem.lambda_smooth * problem.R2

        P_model = 2.0 * H_model

        P_model = 0.5 * (P_model + P_model.T)

        g_model = 2.0 * (J.T @ d)

        # =========================================================
        # Damped QP
        #
        # Used by OSQP to compute the step.
        # =========================================================

        H_qp = H_model + damping * np.eye(c_i.size)

        P_qp = 2.0 * H_qp

        P_qp = 0.5 * (P_qp + P_qp.T)

        g_qp = 2.0 * (J.T @ d - damping * c_i)

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

        collision = self._collision_constraints(problem)

        # =========================================================
        # Historical path: NO COLLISIONS
        # =========================================================

        if collision is None:

            if hasattr(
                problem,
                "linearized_constraints",
            ):
                A, l, u = problem.linearized_constraints(c_i)

            else:
                A, l, u = problem.linear_constraints()

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

        A_static, l_static, u_static = problem.linear_constraints()

        A_static = sp.csc_matrix(A_static)

        l_static = np.asarray(
            l_static,
            dtype=float,
        ).reshape(-1)

        u_static = np.asarray(
            u_static,
            dtype=float,
        ).reshape(-1)

        H_col, h_col = collision.linearize(c_i)

        H_col = sp.csc_matrix(H_col)

        h_col = np.asarray(
            h_col,
            dtype=float,
        ).reshape(-1)

        n_control_variables = c_i.size

        n_slacks = h_col.size

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

        return float(0.5 * c @ P @ c + g @ c)

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

        current_cost = float(problem.objective(c))

        candidate_cost = float(problem.objective(candidate))

        actual_reduction = current_cost - candidate_cost

        predicted_reduction = self._qp_model_value(
            P,
            g,
            c,
        ) - self._qp_model_value(
            P,
            g,
            candidate,
        )

        if predicted_reduction <= 0.0:
            return -np.inf

        return float(actual_reduction / predicted_reduction)

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
            current_control_points=(c_current),
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

        if not (result.info.status.lower().startswith("solved")):
            raise RuntimeError(
                "Gauss-Newton QP failed: " f"{result.info.status}")

        solution = np.asarray(
            result.x,
            dtype=float,
        ).reshape(-1)

        n_control_variables = c_current.size

        if solution.size < n_control_variables:
            raise RuntimeError(
                "OSQP returned fewer variables " "than expected.")

        # -----------------------------------------------------
        # The first variables are always the B-spline control
        # points. Any remaining variables are collision slacks.
        # -----------------------------------------------------

        c_qp = solution[:n_control_variables]

        self._last_collision_slacks = solution[n_control_variables:].copy()

        return (
            c_qp,
            result,
            P_model,
            g_model,
        )

    # =========================================================
    # Result construction
    # =========================================================

    @staticmethod
    def _make_result(
        problem,
        control_points,
        *,
        success,
        status,
        iterations,
        history,
    ):
        """Build an optimization result from flattened control points."""
        return CartesianOptimizationResult(
            control_points=np.asarray(control_points).reshape(
                problem.spline.n_control_points,
                problem.spline.n_dof,
            ),
            success=success,
            status=status,
            iterations=iterations,
            history=history,
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
        Solve the nonlinear constrained trajectory optimization problem.

        Successive Gauss-Newton QPs are solved using OSQP. Each candidate
        step is accepted through backtracking line search.

        Without collisions, acceptance is based on the true nonlinear
        objective. With collisions, a merit function additionally penalizes
        violations of nonlinear collision constraints.

        In adaptive damping mode, a rejected line search may trigger
        another QP with increased damping. Accepted steps update the
        damping based on the gain ratio rho.

        Termination occurs upon:
            - A sufficiently small step.
            - A sufficiently small relative merit decrease.
            - The maximum number of iterations.

        Collision feasibility is additionally required before reporting
        successful convergence when collision constraints are enabled.

        Args:
            problem: Cartesian trajectory optimization problem.
            initial_control_points: Initial B-spline control points.
            verbose: Print iteration diagnostics when True.

        Returns:
            CartesianOptimizationResult containing the final trajectory,
            convergence status, iteration count, and optimization history.
        """

        # Initialize optimization state

        c = np.asarray(
            initial_control_points,
            dtype=float,
        ).reshape(-1)

        history = []

        current_cost = float(problem.objective(c))

        # Local damping may evolve without modifying self.damping.

        mu = float(
            np.clip(
                self.damping,
                self.min_damping,
                self.max_damping,
            )
        )

        # Report initial tracking and collision metrics

        if verbose:

            metrics = problem.error_metrics(c)

            collision = self._collision_constraints(problem)

            if collision is None:

                min_collision_distance = None

            else:

                min_collision_distance = collision.minimum_distance(c)

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

            if min_collision_distance is not None:

                message += f"  dmin=" f"{1e3 * min_collision_distance:.2f} mm"

            print(message)

        # Main Gauss-Newton / SQP loop

        for iteration in range(self.max_iterations):
            # Retry rejected steps with increased damping in adaptive mode.
            retry = 0

            while True:

                # Solve the local constrained QP

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

                full_step_norm = float(np.linalg.norm(direction))

                # A negligible QP step implies convergence only if collision-feasible.

                if full_step_norm <= self.step_tolerance * (
                    1.0 + np.linalg.norm(c)
                ) and self._collision_feasible(problem, c):

                    return self._make_result(
                        problem,
                        c,
                        success=True,
                        status="step_tolerance",
                        iterations=iteration,
                        history=history,
                    )

                # Backtracking on objective (no collisions) or collision-aware merit.

                alpha = 1.0
                accepted = False

                current_merit = self._merit_value(
                    problem,
                    c,
                )

                while alpha >= self.min_alpha:

                    candidate = c + alpha * direction

                    candidate_cost = float(problem.objective(candidate))

                    candidate_merit = self._merit_value(
                        problem,
                        candidate,
                        objective=candidate_cost,
                    )

                    if candidate_merit < current_merit:

                        accepted = True
                        break

                    alpha *= self.line_search_shrink

                # Evaluate model agreement using the undamped objective model.

                if accepted:

                    # rho measures the quality of the undamped tracking model.
                    # Collision penalties affect line-search acceptance, not rho.

                    rho = self._gain_ratio(
                        problem=problem,
                        current_control_points=c,
                        candidate_control_points=(candidate),
                        P=P_model,
                        g=g_model,
                    )

                    break

                # Reject the step; static mode stops, adaptive mode retries.

                if self.damping_mode == "static":

                    return self._make_result(
                        problem,
                        c,
                        success=False,
                        status="line_search_failed",
                        iterations=iteration,
                        history=history,
                    )

                # Increase damping and retry after a rejected line search.

                if retry >= self.max_damping_retries:

                    return self._make_result(
                        problem,
                        c,
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

                # Cannot increase damping any further.

                if new_mu <= mu:

                    return self._make_result(
                        problem,
                        c,
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

            # The candidate has been accepted by the line search.

            previous_merit = current_merit

            step_norm = float(alpha * full_step_norm)

            # Adjust damping from rho while respecting configured bounds.

            if self.damping_mode == "adaptive":

                if rho < self.rho_low:

                    mu *= self.damping_increase

                elif rho > self.rho_high and alpha == 1.0:

                    mu /= self.damping_decrease

                mu = float(
                    np.clip(
                        mu,
                        self.min_damping,
                        self.max_damping,
                    )
                )

            # Commit the accepted trajectory and its objective values

            c = candidate

            current_cost = candidate_cost

            current_merit = candidate_merit

            metrics = problem.error_metrics(c)

            # Collect minimum collision distance for logging

            collision = self._collision_constraints(problem)

            if collision is None:

                min_collision_distance = None

            else:

                min_collision_distance = collision.minimum_distance(c)

            # Record diagnostics for the accepted step

            history.append(
                CartesianIteration(
                    iteration=iteration,
                    objective=current_cost,
                    position_rms=metrics["position_rms"],
                    position_max=metrics["position_max"],
                    orientation_rms=metrics["orientation_rms"],
                    orientation_max=metrics["orientation_max"],
                    step_norm=step_norm,
                    alpha=alpha,
                    osqp_status=(osqp_result.info.status),
                    osqp_solve_time=float(osqp_result.info.solve_time),
                )
            )

            if verbose:

                if self.damping_mode == "adaptive":

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

                if min_collision_distance is not None:

                    message += f"  dmin=" f"{1e3 * min_collision_distance:.2f} mm"

                print(message)

            # Check step convergence and true collision feasibility

            if step_norm <= (
                self.step_tolerance * (1.0 + np.linalg.norm(c))
            ) and self._collision_feasible(
                problem,
                c,
            ):

                return self._make_result(
                    problem,
                    c,
                    success=True,
                    status="step_tolerance",
                    iterations=iteration + 1,
                    history=history,
                )

            # Check relative decrease of the acceptance merit function

            relative_decrease = (previous_merit - current_merit) / max(
                abs(previous_merit),
                1e-12,
            )

            if relative_decrease <= self.cost_tolerance and self._collision_feasible(
                problem,
                c,
            ):

                return self._make_result(
                    problem,
                    c,
                    success=True,
                    status="cost_tolerance",
                    iterations=iteration + 1,
                    history=history,
                )

        # Maximum iterations reached without convergence

        return self._make_result(
            problem,
            c,
            success=False,
            status="max_iterations",
            iterations=self.max_iterations,
            history=history,
        )

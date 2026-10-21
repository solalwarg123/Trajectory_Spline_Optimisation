
"""Receding-horizon Cartesian MPC with feasible warm starts."""

from time import perf_counter

import numpy as np
import osqp
import scipy.sparse as sp


class CartesianMPC:
    """
    Receding-horizon Cartesian trajectory optimizer.

    Features:
        - Current-state update
        - Optional feasible warm-start projection
        - Cached OSQP projection workspace
        - Optional per-obstacle collision activation
        - Final collision feasibility check

    Collision filtering is a sampled broad-phase.
    It does not guarantee continuous collision avoidance.
    """

    def __init__(
        self,
        problem,
        solver,
        initial_control_points,
        collision_activation_distance=None,
        collision_check_samples=11,
        collision_feasibility_tolerance=1e-4,
        project_warm_start=True,
        projection_tolerance=1e-8,
        linear_feasibility_tolerance: float = 1e-5,
    ):
        self.problem = problem
        self.solver = solver

        if collision_activation_distance is not None:
            if (
                not np.isfinite(collision_activation_distance)
                or collision_activation_distance < 0
            ):
                raise ValueError(
                    "Invalid collision activation distance."
                )

        if collision_check_samples < 2:
            raise ValueError(
                "collision_check_samples must be >= 2."
            )

        if (
            not np.isfinite(collision_feasibility_tolerance)
            or collision_feasibility_tolerance < 0
        ):
            raise ValueError(
                "collision_feasibility_tolerance "
                "must be non-negative."
            )

        if (
            not np.isfinite(projection_tolerance)
            or projection_tolerance <= 0
        ):
            raise ValueError(
                "projection_tolerance must be positive."
            )

        self.collision_activation_distance = (
            collision_activation_distance
        )
        self.collision_check_samples = int(
            collision_check_samples
        )
        self.collision_feasibility_tolerance = float(
            collision_feasibility_tolerance
        )

        self.project_warm_start = bool(
            project_warm_start
        )
        self.projection_tolerance = float(
            projection_tolerance
        )

        self.linear_feasibility_tolerance = float(
            linear_feasibility_tolerance
        )

        if not np.isfinite(self.linear_feasibility_tolerance) or self.linear_feasibility_tolerance < 0:
            raise ValueError(
                "linear_feasibility_tolerance must be finite and non-negative"
            )

        # Original full collision builder.
        self._full_collision = getattr(
            problem,
            "collision_constraints",
            None,
        )

        # Cache subset builders by obstacle indices.
        self._collision_cache = {}

        self.last_collision_info = None
        self.last_projection_info = None

        # Initial-state equality matrices.
        #
        # Initialized lazily: lightweight test problems
        # do not necessarily expose a complete B-spline.
        self._initial_E = None
        self._initial_G_inv = None

        # Cached OSQP projection workspace.
        self._projection_solver = None
        self._projection_A = None

        self.previous_control_points = None
        self.reset(initial_control_points)

    # ======================================================
    # Warm-start management
    # ======================================================

    def reset(self, initial_control_points):
        """Reset the MPC warm start."""

        shape = (
            self.problem.spline.n_control_points,
            self.problem.spline.n_dof,
        )

        c = np.asarray(
            initial_control_points,
            dtype=float,
        )

        if c.size != shape[0] * shape[1]:
            raise ValueError(
                f"Expected {shape[0] * shape[1]} "
                "control-point values."
            )

        if not np.all(np.isfinite(c)):
            raise ValueError(
                "Control points must be finite."
            )

        self.previous_control_points = (
            c.reshape(shape).copy()
        )

        self.last_projection_info = None

    def _initialize_initial_state_matrices(self):
        """
        Lazily initialize the mapping:

            E @ c = [q(0), qdot(0)]

        The matrix depends only on the spline.
        """

        if self._initial_E is not None:
            return

        spline = self.problem.spline
        t0 = spline.t_start

        E = np.vstack([
            spline.evaluation_matrix(
                t0,
                derivative_order=0,
            ),
            spline.evaluation_matrix(
                t0,
                derivative_order=1,
            ),
        ])

        E = np.asarray(E, dtype=float)

        expected_shape = (
            2 * spline.n_dof,
            spline.n_control_points * spline.n_dof,
        )

        if E.shape != expected_shape:
            raise ValueError(
                "Invalid initial-state evaluation matrix: "
                f"expected {expected_shape}, got {E.shape}."
            )

        G = E @ E.T

        if np.linalg.matrix_rank(G) != G.shape[0]:
            raise ValueError(
                "Initial position/velocity constraints "
                "are not independent."
            )

        self._initial_E = E
        self._initial_G_inv = np.linalg.inv(G)

    def _rebase_initial_state(self, c, q0, v0):
        """Rebase the first two control points to match q0 and v0."""

        spline = self.problem.spline

        c = np.asarray(c, dtype=float).copy()
        q0 = np.asarray(q0, dtype=float).reshape(-1)
        v0 = np.asarray(v0, dtype=float).reshape(-1)

        c = c.reshape(spline.n_control_points, spline.n_dof)

        p = spline.degree
        h = spline.knots[p + 1] - spline.knots[p]

        if p < 1 or h <= 0:
            raise ValueError("Invalid spline for initial-state rebasing.")

        if q0.size != spline.n_dof or v0.size != spline.n_dof:
            raise ValueError("Invalid initial state dimension.")

        c[0] = q0
        c[1] = q0 + (h / p) * v0

        return c

    @staticmethod
    def _linear_violation(c, A, lower, upper):
        """Maximum violation of linear constraints."""

        x = np.asarray(
            c, dtype=float
        ).reshape(-1)

        values = np.asarray(
            A @ x
        ).reshape(-1)

        return float(max(
            0.0,
            np.max(lower - values),
            np.max(values - upper),
        ))

    @staticmethod
    def _same_sparse_matrix(A, B):
        """Compare sparse structure and values."""

        if A is None or B is None:
            return False

        return (
            A.shape == B.shape
            and np.array_equal(A.indptr, B.indptr)
            and np.array_equal(A.indices, B.indices)
            and np.array_equal(A.data, B.data)
        )

    def _project_linear_feasible(
        self,
        c,
        A,
        lower,
        upper,
    ):
        """
        Euclidean projection onto linear constraints.

            minimize 0.5 * ||x - x_guess||²
            subject to lower <= A @ x <= upper

        OSQP is reused when the constraint matrix
        is unchanged.
        """

        c = np.asarray(c, dtype=float)
        shape = c.shape
        x_guess = c.reshape(-1)

        A = sp.csc_matrix(A)

        lower = np.asarray(
            lower, dtype=float
        ).reshape(-1)

        upper = np.asarray(
            upper, dtype=float
        ).reshape(-1)

        if A.shape[1] != x_guess.size:
            raise ValueError(
                "Invalid constraint matrix dimension."
            )

        if (
            A.shape[0] != lower.size
            or A.shape[0] != upper.size
        ):
            raise ValueError(
                "Invalid constraint bounds dimension."
            )

        before = self._linear_violation(
            x_guess,
            A,
            lower,
            upper,
        )

        if before <= self.projection_tolerance:
            return c.copy(), {
                "projected": False,
                "before": before,
                "after": before,
                "iterations": 0,
                "status": "already_feasible",
            }

        if not self._same_sparse_matrix(
            A,
            self._projection_A,
        ):
            self._projection_solver = None

        if self._projection_solver is None:
            qp = osqp.OSQP()

            qp.setup(
                P=sp.eye(
                    x_guess.size,
                    format="csc",
                ),
                q=-x_guess,
                A=A,
                l=lower,
                u=upper,
                verbose=False,
                polishing=True,
                eps_abs=1e-8,
                eps_rel=1e-8,
                max_iter=20000,
            )

            self._projection_solver = qp
            self._projection_A = A.copy()

        else:
            self._projection_solver.update(
                q=-x_guess,
                l=lower,
                u=upper,
            )

        result = self._projection_solver.solve()

        if (
            result.info.status_val not in (1, 2)
            or result.x is None
        ):
            raise RuntimeError(
                "Warm-start projection failed: "
                f"{result.info.status}"
            )

        projected = np.asarray(
            result.x, dtype=float
        )

        if not np.all(np.isfinite(projected)):
            raise RuntimeError(
                "Warm-start projection returned "
                "non-finite control points."
            )

        after = self._linear_violation(
            projected,
            A,
            lower,
            upper,
        )

        if after > self.projection_tolerance:
            raise RuntimeError(
                "Warm-start projection remains "
                f"infeasible: {after:.3e}"
            )

        return projected.reshape(shape), {
            "projected": True,
            "before": before,
            "after": after,
            "iterations": int(result.info.iter),
            "status": str(result.info.status),
        }

    def _prepare_warm_start(self, q0, v0):
        """
        Prepare warm start for the current MPC cycle.

        Must be called AFTER problem.update().
        """

        c_old = self.previous_control_points.copy()

        if not self.project_warm_start:
            self.last_projection_info = {
                "enabled": False,
                "projected": False,
                "before": None,
                "after": None,
                "time_ms": 0.0,
                "iterations": 0,
                "status": "disabled",
            }
            return c_old

        start = perf_counter()

        # Constraints already updated for q0 and v0.
        A, lower, upper = (
            self.problem.linear_constraints()
        )

        # First satisfy initial-state equalities.
        c_rebased = self._rebase_initial_state(
            c_old, q0, v0
        )

        # Then restore all linear constraints.
        c_projected, info = (
            self._project_linear_feasible(
                c_rebased,
                A,
                lower,
                upper,
            )
        )

        info["enabled"] = True
        info["time_ms"] = (
            perf_counter() - start
        ) * 1000.0

        self.last_projection_info = info

        return c_projected

    # ======================================================
    # Collision management
    # ======================================================

    def _collision_builder(self, indices):
        """
        Return a cached builder for selected obstacles.

        Empty selection means no collision constraints.
        """

        indices = tuple(sorted(set(indices)))

        if not indices:
            return None

        full = self._full_collision

        all_indices = tuple(
            range(full.environment.n_obstacles)
        )

        if indices == all_indices:
            return full

        if indices not in self._collision_cache:
            self._collision_cache[indices] = (
                full.subset_for_obstacles(indices)
            )

        return self._collision_cache[indices]

    def _near_obstacles(self, control_points):
        return self._full_collision.near_obstacle_indices(
            control_points,
            activation_distance=(
                self.collision_activation_distance
            ),
            n_samples=self.collision_check_samples,
        )

    def _solve_with(
        self,
        constraints,
        initial_guess,
        verbose,
    ):
        """
        Temporarily replace collision constraints.

        Restore original constraints afterwards.
        """

        original = (
            self.problem.collision_constraints
        )

        try:
            self.problem.collision_constraints = (
                constraints
            )

            return self.solver.solve(
                problem=self.problem,
                initial_control_points=initial_guess,
                verbose=verbose,
            )

        finally:
            self.problem.collision_constraints = (
                original
            )

    def _control_points(self, result):
        c = np.asarray(
            result.control_points,
            dtype=float,
        )

        shape = (
            self.problem.spline.n_control_points,
            self.problem.spline.n_dof,
        )

        if c.size != shape[0] * shape[1]:
            raise RuntimeError(
                "Solver returned invalid control point size."
            )

        if not np.all(np.isfinite(c)):
            raise RuntimeError(
                "Solver returned non-finite control points."
            )

        return c.reshape(shape)

    def _full_collision_feasible(self, c):
        """
        Check all original collision constraints
        on their collocation grid.
        """

        return self._full_collision.is_feasible(
            c,
            tolerance=self.collision_feasibility_tolerance,
        )

    def _validate_result(self, result, collision_feasible=None):
        """Accept feasible MPC iterates, including iteration-limited GN."""

        c = self._control_points(result)
        finite = np.all(np.isfinite(c))

        if hasattr(self.problem, "linear_constraints"):
            A, lower, upper = self.problem.linear_constraints()

            violation = (
                self._linear_violation(c, A, lower, upper)
                if finite else np.inf
            )

            linear_feasible = (
                np.isfinite(violation)
                and violation <= self.linear_feasibility_tolerance
            )
        else:
            violation = None
            linear_feasible = bool(finite)

        status = str(getattr(result, "status", ""))

        # A maximum-iteration exit is expected with a fixed
        # MPC computational budget.
        solver_usable = (
            bool(result.success)
            or status == "max_iterations"
        )

        accepted = (
            solver_usable
            and finite
            and linear_feasible
            and collision_feasible is not False
        )

        self.last_solution_info = {
            "solver_success": bool(result.success),
            "solver_status": status,
            "solver_usable": bool(solver_usable),
            "coefficients_finite": bool(finite),
            "linear_violation": violation,
            "linear_feasible": bool(linear_feasible),
            "collision_feasible": collision_feasible,
            "accepted": bool(accepted),
        }

        return accepted

    def _commit(self, result, collision_feasible=None):
        """Store the solution only if it is valid."""

        if self._validate_result(
            result,
            collision_feasible=collision_feasible,
        ):
            self.previous_control_points = (
                self._control_points(result).copy()
            )

    # ======================================================
    # MPC step
    # ======================================================

    def step(
        self,
        q0,
        v0,
        target_poses,
        *,
        verbose=False,
    ):
        """
        Update and solve the current MPC horizon.

        1. Update current state and target poses.
        2. Rebase and project warm start.
        3. Select nearby obstacles if enabled.
        4. Solve the trajectory optimization.
        5. Detect newly approached obstacles.
        6. Re-solve if necessary.
        7. Verify final feasibility before committing.
        """

        # Update current problem.
        self.problem.update(
            q0=q0,
            v0=v0,
            target_poses=target_poses,
        )

        # Prepare feasible initial guess.
        c_init = self._prepare_warm_start(q0, v0)

        full = self._full_collision

        has_collisions = (
            full is not None
            and full.has_constraints
        )

        # --------------------------------------------------
        # Original MPC path (without obstacle filtering)
        # --------------------------------------------------

        if (
            not has_collisions
            or self.collision_activation_distance is None
        ):
            result = self.solver.solve(
                problem=self.problem,
                initial_control_points=c_init,
                verbose=verbose,
            )

            collision_feasible = (
                self._full_collision_feasible(
                    self._control_points(result)
                )
                if has_collisions else None
            )

            self.last_collision_info = {
                "filtered": False,
                "active_indices": (
                    tuple(range(full.environment.n_obstacles))
                    if has_collisions else ()
                ),
                "solve_count": 1,
                "feasible": collision_feasible,
            }

            self._commit(
                result,
                collision_feasible=collision_feasible,
            )
            return result

        # --------------------------------------------------
        # Select obstacles near initial guess
        # --------------------------------------------------

        active = set(self._near_obstacles(c_init))

        result = self._solve_with(
            constraints=self._collision_builder(active),
            initial_guess=c_init,
            verbose=verbose,
        )

        solves = 1
        c = self._control_points(result)

        # --------------------------------------------------
        # Detect newly approached obstacles
        # --------------------------------------------------

        newly_near = set(self._near_obstacles(c))

        if not newly_near.issubset(active):
            active |= newly_near

            result = self._solve_with(
                constraints=self._collision_builder(active),
                initial_guess=c,
                verbose=verbose,
            )

            solves += 1
            c = self._control_points(result)

        # --------------------------------------------------
        # Final verification against all obstacles
        # --------------------------------------------------

        feasible = self._full_collision_feasible(c)

        if not feasible:
            all_indices = set(
                range(full.environment.n_obstacles)
            )

            if active != all_indices:
                active = all_indices

                result = self._solve_with(
                    constraints=full,
                    initial_guess=c,
                    verbose=verbose,
                )

                solves += 1
                c = self._control_points(result)

                feasible = self._full_collision_feasible(c)

        self.last_collision_info = {
            "filtered": True,
            "active_indices": tuple(sorted(active)),
            "active_count": len(active),
            "total_obstacles": full.environment.n_obstacles,
            "solve_count": solves,
            "feasible": bool(feasible),
        }

        # --------------------------------------------------
        # Validate solution and update warm start
        # --------------------------------------------------

        self._commit(
            result,
            collision_feasible=bool(feasible),
        )

        return result

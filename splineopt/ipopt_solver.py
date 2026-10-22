"""IPOPT solver interface for trajectory optimization problems.

This module provides a lightweight adapter between the internal
trajectory optimization problem interface and CyIpopt.

The mathematical problem remains independent from IPOPT.

The problem is expected to provide:

    objective(c)
    gradient(c)
    linear_constraints()
    error_metrics(c)
    spline

It may additionally provide nonlinear constraints through:

    nonlinear_constraints(c)
    nonlinear_constraints_jacobian(c)
    nonlinear_constraint_bounds()

The complete constraint vector is then

    g(c) = [A c, g_nonlinear(c)]

with corresponding lower and upper bounds.

IPOPT uses a limited-memory BFGS approximation of the Hessian,
so no exact second derivatives are required.
"""

from dataclasses import dataclass, field

from time import perf_counter

import numpy as np
import scipy.sparse as sp

try:
    import cyipopt
except ImportError:
    cyipopt = None


@dataclass
class IpoptOptimizationResult:
    """
    Result and diagnostics of an IPOPT optimization.

    Attributes:
        control_points: Optimized control points, shape (Nc, n_dof).
        success: True if IPOPT reports successful or acceptable convergence.
        status: IPOPT termination status code.
        status_message: Human-readable termination message.
        objective: Final objective value.
        iterations: Number of IPOPT iterations reported by the callback.
        solve_time: Time spent in the IPOPT solve call, in seconds.
        raw_info: Original information dictionary returned by CyIpopt.
        objective_history: Objective values recorded during iterations.
    """

    control_points: np.ndarray

    success: bool
    status: int
    status_message: str

    objective: float
    iterations: int
    solve_time: float

    raw_info: dict
    objective_history: list = field(default_factory=list)


class _CyIpoptAdapter:
    """Translate the internal problem interface to CyIpopt callbacks.

    The problem may provide:

        - static linear constraints

              l <= A x <= u

        - optional nonlinear constraints

              g_lower <= g(x) <= g_upper

    The nonlinear constraint interface is optional so that
    old/simple optimization problems remain supported.
    """

    def __init__(self, problem,):
        """
        Initialize the IPOPT callback adapter.

        The adapter retrieves linear and optional nonlinear constraints,
        validates their bounds and dimensions, and constructs a fixed
        Jacobian sparsity structure.

        Linear constraints use the sparsity of A. Nonlinear constraints
        currently use a dense Jacobian structure.

        Args:
            problem: Optimization problem implementing the required
                objective, gradient and constraint interfaces.
        """

        self.problem = problem
        self.objective_history = []

        # =====================================================
        # Linear constraints
        # =====================================================

        A, l, u = problem.linear_constraints()

        A = sp.coo_matrix(A)
        A.sum_duplicates()

        self.A = A

        self.linear_lower = np.asarray(
            l,
            dtype=float,
        ).reshape(-1)

        self.linear_upper = np.asarray(
            u,
            dtype=float,
        ).reshape(-1)

        if self.A.shape[0] != (self.linear_lower.size):
            raise ValueError(
                "Constraint matrix and lower bounds have incompatible dimensions."
            )

        if self.A.shape[0] != (self.linear_upper.size):
            raise ValueError(
                "Constraint matrix and upper bounds have incompatible dimensions."
            )

        self.n_variables = self.A.shape[1]

        self.n_linear_constraints = self.A.shape[0]

        # =====================================================
        # Optional nonlinear constraints
        # =====================================================

        has_values = hasattr(
            problem,
            "nonlinear_constraints",
        )

        has_jacobian = hasattr(
            problem,
            "nonlinear_constraints_jacobian",
        )

        has_bounds = hasattr(
            problem,
            "nonlinear_constraint_bounds",
        )

        nonlinear_interface_count = sum(
            [
                has_values,
                has_jacobian,
                has_bounds,
            ]
        )

        if nonlinear_interface_count not in {
            0,
            3,
        }:
            raise AttributeError(
                "A problem with nonlinear constraints must "
                "provide all three methods: "
                "nonlinear_constraints(c), "
                "nonlinear_constraints_jacobian(c), and "
                "nonlinear_constraint_bounds()."
            )

        self.has_nonlinear_constraints = nonlinear_interface_count == 3

        if self.has_nonlinear_constraints:

            nonlinear_lower, nonlinear_upper = problem.nonlinear_constraint_bounds()

            self.nonlinear_lower = np.asarray(
                nonlinear_lower,
                dtype=float,
            ).reshape(-1)

            self.nonlinear_upper = np.asarray(
                nonlinear_upper,
                dtype=float,
            ).reshape(-1)

        else:

            self.nonlinear_lower = np.empty(
                0,
                dtype=float,
            )

            self.nonlinear_upper = np.empty(
                0,
                dtype=float,
            )

        if self.nonlinear_lower.size != self.nonlinear_upper.size:
            raise ValueError(
                "Nonlinear lower and upper bounds " "have incompatible dimensions."
            )

        self.n_nonlinear_constraints = self.nonlinear_lower.size

        # =====================================================
        # Complete constraint bounds
        # =====================================================

        self.constraint_lower = np.concatenate(
            [
                self.linear_lower,
                self.nonlinear_lower,
            ]
        )

        self.constraint_upper = np.concatenate(
            [
                self.linear_upper,
                self.nonlinear_upper,
            ]
        )

        if np.any(self.constraint_lower > self.constraint_upper):
            raise ValueError("Constraint lower bounds exceed upper bounds.")

        # =====================================================
        # Linear Jacobian structure
        # =====================================================

        self._linear_jacobian_rows = self.A.row.astype(
            np.int32,
            copy=False,
        )

        self._linear_jacobian_cols = self.A.col.astype(
            np.int32,
            copy=False,
        )

        self._linear_jacobian_values = self.A.data.astype(
            float,
            copy=False,
        )

        # =====================================================
        # Nonlinear Jacobian structure
        # =====================================================
        #
        # CyIpopt requires a FIXED sparsity structure.
        #
        # For now we deliberately expose a dense structure for
        # the nonlinear block:
        #
        #     dg_nonlinear / dx
        #
        # This is robust and solver-independent. Later, if needed
        # for MPC/offline performance, it can be replaced by the
        # exact local B-spline sparsity pattern.
        # =====================================================

        if self.n_nonlinear_constraints > 0:

            nonlinear_rows = np.repeat(
                np.arange(
                    self.n_nonlinear_constraints,
                    dtype=np.int32,
                ),
                self.n_variables,
            )

            nonlinear_rows += self.n_linear_constraints

            nonlinear_cols = np.tile(
                np.arange(
                    self.n_variables,
                    dtype=np.int32,
                ),
                self.n_nonlinear_constraints,
            )

            self._jacobian_rows = np.concatenate(
                [
                    self._linear_jacobian_rows,
                    nonlinear_rows,
                ]
            )

            self._jacobian_cols = np.concatenate(
                [
                    self._linear_jacobian_cols,
                    nonlinear_cols,
                ]
            )

        else:

            self._jacobian_rows = self._linear_jacobian_rows

            self._jacobian_cols = self._linear_jacobian_cols

        # Updated through IPOPT's intermediate callback.
        self.iteration_count = 0

    # =========================================================
    # Objective
    # =========================================================

    def objective(self, x,):
        """Return the scalar objective value for IPOPT."""
        return float(self.problem.objective(x))

    def gradient(self, x,):
        """Return the objective gradient as a flat vector."""
        gradient = np.asarray(
            self.problem.gradient(x),
            dtype=float,
        ).reshape(-1)

        if gradient.size != len(x):
            raise ValueError(
                "problem.gradient() returned " "an invalid shape.")

        return gradient

    # =========================================================
    # Constraints
    # =========================================================

    def constraints(self, x,):
        """
        Evaluate the complete constraint vector.

        Linear constraints are returned first, followed by nonlinear
        constraints when present:

            g(x) = [A @ x, g_nonlinear(x)]
        """

        linear_values = np.asarray(
            self.A @ x,
            dtype=float,
        ).reshape(-1)

        if self.n_nonlinear_constraints == 0:
            return linear_values

        nonlinear_values = np.asarray(
            self.problem.nonlinear_constraints(x),
            dtype=float,
        ).reshape(-1)

        if nonlinear_values.size != (self.n_nonlinear_constraints):
            raise ValueError(
                "problem.nonlinear_constraints() " "returned an invalid shape."
            )

        return np.concatenate(
            [
                linear_values,
                nonlinear_values,
            ]
        )

    def jacobianstructure(self,):
        """
        Return the fixed Jacobian sparsity pattern.

        Returns:
            (rows, cols): Arrays of row and column indices for all
            Jacobian entries supplied to IPOPT.
        """

        return (
            self._jacobian_rows,
            self._jacobian_cols,
        )

    def jacobian(self, x,):
        """
        Return Jacobian values in the fixed sparsity ordering.

        Linear Jacobian values are followed by the flattened nonlinear
        Jacobian in row-major order.

        The returned values correspond exactly to the indices from
        jacobianstructure().
        """

        if self.n_nonlinear_constraints == 0:
            return self._linear_jacobian_values

        # -----------------------------------------------------
        # Nonlinear Jacobian
        # -----------------------------------------------------

        J_nonlinear = self.problem.nonlinear_constraints_jacobian(x)

        if sp.issparse(J_nonlinear):
            J_nonlinear = J_nonlinear.toarray()
        else:
            J_nonlinear = np.asarray(
                J_nonlinear,
                dtype=float,
            )

        expected_shape = (
            self.n_nonlinear_constraints,
            self.n_variables,
        )

        if J_nonlinear.shape != (expected_shape):
            raise ValueError(
                "problem."
                "nonlinear_constraints_jacobian() "
                "returned shape "
                f"{J_nonlinear.shape}, "
                f"expected {expected_shape}."
            )

        # Must use the exact same ordering as the dense
        # row/column structure created in __init__:
        #
        # row 0, all columns
        # row 1, all columns
        # ...
        nonlinear_values = J_nonlinear.reshape(-1)

        return np.concatenate(
            [
                self._linear_jacobian_values,
                nonlinear_values,
            ]
        )

    # =========================================================
    # Iteration callback
    # =========================================================

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """
        Record iteration count and objective history during optimization.

        Returns:
            True to allow IPOPT to continue.
        """

        self.iteration_count = int(iter_count)

        self.objective_history.append(float(obj_value))

        return True


class IpoptSolver:
    """Solve trajectory optimization problems with IPOPT.

    IPOPT uses its primal-dual interior-point algorithm.

    The Hessian of the Lagrangian is approximated using
    limited-memory BFGS.
    """

    def __init__(
        self,
        *,
        tolerance=1e-6,
        max_iterations=200,
        print_level=0,
        options=None,
    ):
        """
        Configure the IPOPT solver.

        Args:
            tolerance: Convergence tolerance passed to IPOPT.
            max_iterations: Maximum number of IPOPT iterations.
            print_level: IPOPT output verbosity level.
            options: Optional dictionary of additional IPOPT options.
                These options are applied last and may override defaults.
        """

        self.tolerance = float(tolerance)

        self.max_iterations = int(max_iterations)

        self.print_level = int(print_level)

        self.options = {} if options is None else dict(options)

        self._validate_parameters()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_parameters(self,):
        """Validate the solver tolerance, iteration limit and verbosity."""
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        if self.print_level < 0:
            raise ValueError("print_level must be non-negative.")

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
        Solve a trajectory optimization problem using IPOPT.

        The optimization variables are the flattened B-spline control
        points. Joint limits remain expressed as general linear
        constraints, matching the Gauss-Newton formulation.

        Args:
            problem: Optimization problem providing a spline, objective,
                gradient, linear constraints, and optional nonlinear
                constraints.
            initial_control_points: Initial control points, shaped
                (Nc, n_dof) or flattened.
            verbose: Increase IPOPT output verbosity.

        Returns:
            IpoptOptimizationResult containing optimized control points,
            termination status, timing and convergence diagnostics.

        Raises:
            ImportError: If CyIpopt is unavailable.
            ValueError: If the initial points or constraint dimensions
                are inconsistent.
        """

        if cyipopt is None:
            raise ImportError(
                "CyIpopt is not installed. "
                "Install it with:\n\n"
                "    conda install -c conda-forge cyipopt"
            )

        x0 = np.asarray(
            initial_control_points,
            dtype=float,
        ).reshape(-1)

        if not np.all(np.isfinite(x0)):
            raise ValueError("Initial control points must be finite.")

        n_variables = x0.size

        expected_variables = problem.spline.n_control_points * problem.spline.n_dof

        if n_variables != expected_variables:
            raise ValueError(
                "initial_control_points contains "
                f"{n_variables} variables, "
                f"expected {expected_variables}."
            )

        if not hasattr(
            problem,
            "gradient",
        ):
            raise AttributeError(
                "The problem must provide " "a gradient(c) method.")

        # -----------------------------------------------------
        # CyIpopt adapter
        # -----------------------------------------------------

        adapter = _CyIpoptAdapter(problem)

        if adapter.n_variables != n_variables:
            raise ValueError(
                "Linear constraint matrix has "
                f"{adapter.n_variables} columns, "
                f"expected {n_variables}."
            )

        n_constraints = adapter.constraint_lower.size

        # -----------------------------------------------------
        # Variable bounds
        #
        # For now all trajectory limits remain represented as
        # general linear constraints A c.
        #
        # This guarantees that IPOPT and Gauss-Newton solve
        # exactly the same mathematical formulation.
        # -----------------------------------------------------

        variable_lower = np.full(
            n_variables,
            -np.inf,
            dtype=float,
        )

        variable_upper = np.full(
            n_variables,
            np.inf,
            dtype=float,
        )

        # -----------------------------------------------------
        # IPOPT problem
        # -----------------------------------------------------

        nlp = cyipopt.Problem(
            n=n_variables,
            m=n_constraints,
            problem_obj=adapter,
            lb=variable_lower,
            ub=variable_upper,
            cl=adapter.constraint_lower,
            cu=adapter.constraint_upper,
        )

        # -----------------------------------------------------
        # IPOPT options
        # -----------------------------------------------------

        nlp.add_option(
            "tol",
            self.tolerance,
        )

        nlp.add_option(
            "max_iter",
            self.max_iterations,
        )

        nlp.add_option(
            "hessian_approximation",
            "limited-memory",
        )

        if verbose:
            nlp.add_option(
                "print_level",
                max(
                    self.print_level,
                    5,
                ),
            )
        else:
            nlp.add_option(
                "print_level",
                self.print_level,
            )

        # User options are applied last, allowing explicit
        # overrides when experimenting with IPOPT.
        for name, value in self.options.items():
            nlp.add_option(
                name,
                value,
            )

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------

        start_time = perf_counter()

        x_opt, info = nlp.solve(x0)

        solve_time = perf_counter() - start_time

        x_opt = np.asarray(
            x_opt,
            dtype=float,
        ).reshape(-1)

        # -----------------------------------------------------
        # IPOPT result
        #
        # 0 = Solve_Succeeded
        # 1 = Solved_To_Acceptable_Level
        # -----------------------------------------------------

        status = int(
            info.get(
                "status",
                -999,
            )
        )

        success = status in {
            0,
            1,
        }

        status_message = info.get(
            "status_msg",
            "",
        )

        if isinstance(
            status_message,
            bytes,
        ):
            status_message = status_message.decode(errors="replace")

        objective = info.get("obj_val")

        if objective is None:
            objective = problem.objective(x_opt)

        objective = float(objective)

        control_points = x_opt.reshape(
            problem.spline.n_control_points,
            problem.spline.n_dof,
        )

        return IpoptOptimizationResult(
            control_points=control_points,
            success=success,
            status=status,
            status_message=str(status_message),
            objective=objective,
            iterations=(adapter.iteration_count),
            solve_time=float(solve_time),
            raw_info=info,
            objective_history=(adapter.objective_history.copy()),
        )

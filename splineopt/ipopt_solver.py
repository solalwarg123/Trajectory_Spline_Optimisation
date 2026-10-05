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

The first implementation supports linear constraints only:

    l <= A c <= u

IPOPT uses a limited-memory BFGS approximation of the Hessian,
so no exact second derivatives are required.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp


try:
    import cyipopt
except ImportError:
    cyipopt = None


@dataclass
class IpoptOptimizationResult:
    """Result returned by IpoptSolver."""

    control_points: np.ndarray

    success: bool
    status: int
    status_message: str

    objective: float
    iterations: int
    solve_time: float

    raw_info: dict


class _CyIpoptAdapter:
    """Translate the internal problem interface to CyIpopt callbacks."""

    def __init__(
        self,
        problem,
    ):
        self.problem = problem

        # -----------------------------------------------------
        # Linear constraints
        #
        #     l <= A c <= u
        # -----------------------------------------------------

        A, l, u = (
            problem.linear_constraints()
        )

        A = sp.coo_matrix(A)
        A.sum_duplicates()

        self.A = A

        self.constraint_lower = np.asarray(
            l,
            dtype=float,
        ).reshape(-1)

        self.constraint_upper = np.asarray(
            u,
            dtype=float,
        ).reshape(-1)

        if self.A.shape[0] != (
            self.constraint_lower.size
        ):
            raise ValueError(
                "Constraint matrix and lower bounds "
                "have incompatible dimensions."
            )

        if self.A.shape[0] != (
            self.constraint_upper.size
        ):
            raise ValueError(
                "Constraint matrix and upper bounds "
                "have incompatible dimensions."
            )

        # Sparse Jacobian structure is constant because
        # current constraints are linear.
        self._jacobian_rows = (
            self.A.row.astype(
                np.int32,
                copy=False,
            )
        )

        self._jacobian_cols = (
            self.A.col.astype(
                np.int32,
                copy=False,
            )
        )

        self._jacobian_values = (
            self.A.data.astype(
                float,
                copy=False,
            )
        )

        # Updated through IPOPT's intermediate callback.
        self.iteration_count = 0

    # =========================================================
    # Objective
    # =========================================================

    def objective(
        self,
        x,
    ):
        return float(
            self.problem.objective(x)
        )

    def gradient(
        self,
        x,
    ):
        gradient = np.asarray(
            self.problem.gradient(x),
            dtype=float,
        ).reshape(-1)

        if gradient.size != len(x):
            raise ValueError(
                "problem.gradient() returned "
                "an invalid shape."
            )

        return gradient

    # =========================================================
    # Constraints
    # =========================================================

    def constraints(
        self,
        x,
    ):
        return np.asarray(
            self.A @ x,
            dtype=float,
        ).reshape(-1)

    def jacobianstructure(
        self,
    ):
        return (
            self._jacobian_rows,
            self._jacobian_cols,
        )

    def jacobian(
        self,
        x,
    ):
        # Linear constraints:
        #
        #     g(x) = A x
        #
        # therefore:
        #
        #     dg/dx = A
        #
        # The Jacobian does not depend on x.
        return self._jacobian_values

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
        self.iteration_count = int(
            iter_count
        )

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
        self.tolerance = float(
            tolerance
        )

        self.max_iterations = int(
            max_iterations
        )

        self.print_level = int(
            print_level
        )

        self.options = (
            {}
            if options is None
            else dict(options)
        )

        self._validate_parameters()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_parameters(
        self,
    ):
        if self.tolerance <= 0.0:
            raise ValueError(
                "tolerance must be positive."
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be positive."
            )

        if self.print_level < 0:
            raise ValueError(
                "print_level must be non-negative."
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
        """Solve a trajectory optimization problem using IPOPT."""

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

        n_variables = x0.size

        expected_variables = (
            problem.spline.n_control_points
            * problem.spline.n_dof
        )

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
                "The problem must provide "
                "a gradient(c) method."
            )

        # -----------------------------------------------------
        # CyIpopt adapter
        # -----------------------------------------------------

        adapter = _CyIpoptAdapter(
            problem
        )

        n_constraints = (
            adapter.constraint_lower.size
        )

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
        for name, value in (
            self.options.items()
        ):
            nlp.add_option(
                name,
                value,
            )

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------

        start_time = perf_counter()

        x_opt, info = nlp.solve(
            x0
        )

        solve_time = (
            perf_counter()
            - start_time
        )

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
            status_message = (
                status_message.decode(
                    errors="replace"
                )
            )

        objective = float(
            info.get(
                "obj_val",
                problem.objective(
                    x_opt
                ),
            )
        )

        control_points = (
            x_opt.reshape(
                problem.spline.n_control_points,
                problem.spline.n_dof,
            )
        )

        return IpoptOptimizationResult(
            control_points=control_points,

            success=success,
            status=status,
            status_message=str(
                status_message
            ),

            objective=objective,
            iterations=(
                adapter.iteration_count
            ),
            solve_time=float(
                solve_time
            ),

            raw_info=info,
        )

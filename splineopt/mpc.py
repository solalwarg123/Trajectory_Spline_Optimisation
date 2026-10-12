
import numpy as np

from splineopt.optimizer import CartesianOptimizationResult


class CartesianMPC:
    """
    Receding-horizon Cartesian trajectory optimizer.

    Reuses the same optimization problem and Gauss-Newton
    solver across MPC cycles.

    Warm start: previous optimized control points (no shift).
    """

    def __init__(self, problem, solver, initial_control_points):
        self.problem = problem
        self.solver = solver

        self.previous_control_points = None
        self.reset(initial_control_points)

    def reset(self, initial_control_points):
        """Reset the MPC warm start."""
        shape = (
            self.problem.spline.n_control_points,
            self.problem.spline.n_dof,
        )

        c = np.asarray(initial_control_points, dtype=float)

        if c.size != shape[0] * shape[1]:
            raise ValueError(
                f"Expected {shape[0] * shape[1]} "
                "control-point values."
            )

        if not np.all(np.isfinite(c)):
            raise ValueError("Control points must be finite.")

        self.previous_control_points = c.reshape(shape).copy()

    def step(self, q0, v0, target_poses, *, verbose=False):
        """
        Update and solve the current MPC horizon.

        The caller supplies the reference poses on the
        problem's existing relative tracking grid.
        """

        # 1. Update time-varying problem data.
        self.problem.update(
            q0=q0,
            v0=v0,
            target_poses=target_poses,
        )

        # 2. Warm start from previous control points.
        c_init = self.previous_control_points.copy()

        # 3. Optimize with the existing GN/SQP solver.
        result = self.solver.solve(
            problem=self.problem,
            initial_control_points=c_init,
            verbose=verbose,
        )

        # 4. Keep the last result as the next warm start.
        if result.success:
            self.previous_control_points = (
                result.control_points.copy()
            )

        return result


from types import SimpleNamespace

import numpy as np

from splineopt.mpc import CartesianMPC


class FakeProblem:
    def __init__(self):
        self.spline = SimpleNamespace(
            n_control_points=4,
            n_dof=2,
        )
        self.updates = []

    def update(self, q0, v0, target_poses):
        self.updates.append(
            (
                np.array(q0).copy(),
                np.array(v0).copy(),
                target_poses,
            )
        )


class FakeSolver:
    def __init__(self):
        self.initial_guesses = []
        self.succeed = True

    def solve(
        self,
        problem,
        initial_control_points,
        verbose=False,
    ):
        self.initial_guesses.append(
            np.array(initial_control_points).copy()
        )

        return SimpleNamespace(
            control_points=initial_control_points + 1.0,
            success=self.succeed,
            status=(
                "converged"
                if self.succeed
                else "failed"
            ),
        )


def test_mpc_reuses_previous_solution():
    problem = FakeProblem()
    solver = FakeSolver()

    c0 = np.zeros((4, 2))

    mpc = CartesianMPC(
        problem,
        solver,
        c0,
        project_warm_start=False,
    )

    result1 = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    np.testing.assert_allclose(
        solver.initial_guesses[0],
        c0,
    )

    result2 = mpc.step(
        np.ones(2),
        np.zeros(2),
        [],
    )

    np.testing.assert_allclose(
        solver.initial_guesses[1],
        result1.control_points,
    )

    assert result2.success
    assert len(problem.updates) == 2


def test_mpc_keeps_last_success_on_failure():
    problem = FakeProblem()
    solver = FakeSolver()

    mpc = CartesianMPC(
        problem,
        solver,
        np.zeros((4, 2)),
        project_warm_start=False,
    )

    first = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    solver.succeed = False

    mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    np.testing.assert_allclose(
        mpc.previous_control_points,
        first.control_points,
    )


def test_mpc_reset():
    mpc = CartesianMPC(
        FakeProblem(),
        FakeSolver(),
        np.zeros((4, 2)),
        project_warm_start=False,
    )

    mpc.reset(
        np.full((4, 2), 3.0)
    )

    np.testing.assert_allclose(
        mpc.previous_control_points,
        3.0,
    )


class ConstraintProblem(FakeProblem):
    """Fake problem with linear bounds on control points."""

    def linear_constraints(self):
        A = np.eye(8)
        lower = np.full(8, -0.5)
        upper = np.full(8, 0.5)
        return A, lower, upper


class FixedSolver(FakeSolver):
    def __init__(self, control_points):
        super().__init__()
        self.returned = np.asarray(control_points, dtype=float)

    def solve(self, problem, initial_control_points, verbose=False):
        return SimpleNamespace(
            control_points=self.returned.copy(),
            success=True,
            status="converged",
        )


def test_mpc_accepts_feasible_solution():
    expected = np.full((4, 2), 0.25)

    mpc = CartesianMPC(
        ConstraintProblem(),
        FixedSolver(expected),
        np.zeros((4, 2)),
        project_warm_start=False,
    )

    mpc.step(np.zeros(2), np.zeros(2), [])

    assert mpc.last_solution_info["accepted"]
    np.testing.assert_allclose(
        mpc.previous_control_points,
        expected,
    )


def test_mpc_rejects_infeasible_solution():
    initial = np.zeros((4, 2))
    infeasible = np.ones((4, 2))

    mpc = CartesianMPC(
        ConstraintProblem(),
        FixedSolver(infeasible),
        initial,
        project_warm_start=False,
    )

    result = mpc.step(np.zeros(2), np.zeros(2), [])

    assert result.success
    assert not mpc.last_solution_info["accepted"]
    assert not mpc.last_solution_info["linear_feasible"]

    np.testing.assert_allclose(
        mpc.previous_control_points,
        initial,
    )


def test_mpc_accepts_feasible_max_iterations():
    initial = np.zeros((4, 2))

    mpc = CartesianMPC(
        ConstraintProblem(),
        FixedSolver(initial),
        initial,
        project_warm_start=False,
    )

    result = SimpleNamespace(
        control_points=np.full((4, 2), 0.25),
        success=False,
        status="max_iterations",
    )

    mpc._commit(result)

    assert mpc.last_solution_info["accepted"]
    assert not mpc.last_solution_info["solver_success"]
    assert mpc.last_solution_info["solver_usable"]
    assert mpc.last_solution_info["linear_feasible"]

    np.testing.assert_allclose(
        mpc.previous_control_points,
        result.control_points,
    )


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
            (np.array(q0).copy(),
             np.array(v0).copy(),
             target_poses)
        )


class FakeSolver:
    def __init__(self):
        self.initial_guesses = []
        self.succeed = True

    def solve(self, problem, initial_control_points, verbose=False):
        self.initial_guesses.append(
            np.array(initial_control_points).copy()
        )
        return SimpleNamespace(
            control_points=initial_control_points + 1.0,
            success=self.succeed,
            status="converged" if self.succeed else "failed",
        )


def test_mpc_reuses_previous_solution():
    problem = FakeProblem()
    solver = FakeSolver()

    c0 = np.zeros((4, 2))

    mpc = CartesianMPC(problem, solver, c0)

    result1 = mpc.step(
        np.zeros(2), np.zeros(2), []
    )

    np.testing.assert_allclose(
        solver.initial_guesses[0], c0
    )

    result2 = mpc.step(
        np.ones(2), np.zeros(2), []
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
        problem, solver, np.zeros((4, 2))
    )

    first = mpc.step(
        np.zeros(2), np.zeros(2), []
    )

    solver.succeed = False

    mpc.step(
        np.zeros(2), np.zeros(2), []
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
    )

    mpc.reset(np.full((4, 2), 3.0))

    np.testing.assert_allclose(
        mpc.previous_control_points,
        3.0,
    )


from types import SimpleNamespace
import pytest
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
    def __init__(self, control_points, success=True):
        super().__init__()

        self.returned = np.asarray(
            control_points,
            dtype=float,
        )

        self.succeed = bool(success)

    def solve(
        self,
        problem,
        initial_control_points,
        verbose=False,
    ):
        return SimpleNamespace(
            control_points=self.returned.copy(),
            success=self.succeed,
            status=(
                "converged"
                if self.succeed
                else "max_iterations"
            ),
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
    assert mpc.last_solution_info["linear_feasible"]

    np.testing.assert_allclose(
        mpc.previous_control_points,
        result.control_points,
    )


def test_mpc_accepts_feasible_nonconverged_solution():
    expected = np.full((4, 2), 0.25)

    solver = FixedSolver(
        expected,
        success=False,
    )

    mpc = CartesianMPC(
        ConstraintProblem(),
        solver,
        np.zeros((4, 2)),
        project_warm_start=False,
    )

    result = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    assert not result.success
    assert mpc.last_solution_info["accepted"]
    assert mpc.last_solution_info["solution_source"] == "gn"

    np.testing.assert_allclose(
        mpc.previous_control_points,
        expected,
    )


def test_mpc_falls_back_to_feasible_projected_guess():
    solver = FixedSolver(
        np.ones((4, 2)),
        success=False,
    )

    mpc = CartesianMPC(
        ConstraintProblem(),
        solver,
        np.zeros((4, 2)),
        project_warm_start=True,
    )

    # Simulate a previously projected feasible warm start.
    # No real OSQP call is needed in this unit test.
    c_projected = np.full((4, 2), 0.15)

    mpc._prepare_warm_start = (
        lambda q0, v0: c_projected.copy()
    )

    result = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    assert not result.success
    assert mpc.last_solution_info["accepted"]
    assert mpc.last_solution_info["fallback_used"]

    assert (
        mpc.last_solution_info["solution_source"]
        == "projected_warm_start"
    )

    np.testing.assert_allclose(
        result.control_points,
        c_projected,
    )

    np.testing.assert_allclose(
        mpc.previous_control_points,
        c_projected,
    )


def test_mpc_does_not_accept_infeasible_fallback():
    solver = FixedSolver(
        np.ones((4, 2)),
        success=False,
    )

    initial = np.zeros((4, 2))

    mpc = CartesianMPC(
        ConstraintProblem(),
        solver,
        initial,
        project_warm_start=True,
    )

    # Deliberately infeasible fallback.
    mpc._prepare_warm_start = (
        lambda q0, v0: np.full((4, 2), 0.7)
    )

    result = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    assert not mpc.last_solution_info["accepted"]
    assert not mpc.last_solution_info["fallback_used"]

    np.testing.assert_allclose(
        result.control_points,
        np.ones((4, 2)),
    )

    np.testing.assert_allclose(
        mpc.previous_control_points,
        initial,
    )


@pytest.mark.parametrize("invalid_result", ["nan", "wrong_shape"])
@pytest.mark.parametrize("fallback_collision_feasible", [True, False])
def test_mpc_invalid_gn_result_with_collision_filtering(
    invalid_result, fallback_collision_feasible
):
    """Invalid GN results must trigger a safe, feasible fallback."""

    class FakeCollision:
        has_constraints = True
        environment = SimpleNamespace(n_obstacles=1)

        def near_obstacle_indices(self, control_points, **kwargs):
            return []

        def is_feasible(self, control_points, tolerance=1e-4):
            return fallback_collision_feasible

    class CollisionProblem(ConstraintProblem):
        def __init__(self):
            super().__init__()
            self.collision_constraints = FakeCollision()

    # Simulate an invalid GN result.
    if invalid_result == "nan":
        invalid_c = np.full((4, 2), np.nan)
    else:
        invalid_c = np.zeros((3, 2))

    initial = np.zeros((4, 2))
    c_projected = np.full((4, 2), 0.15)

    mpc = CartesianMPC(
        CollisionProblem(),
        FixedSolver(invalid_c, success=False),
        initial,
        collision_activation_distance=0.1,
        project_warm_start=True,
    )

    # Isolate the fallback logic from the OSQP projection.
    mpc._prepare_warm_start = (
        lambda q0, v0: c_projected.copy()
    )

    result = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    assert mpc.last_solution_info["accepted"] == fallback_collision_feasible

    if fallback_collision_feasible:
        assert mpc.last_solution_info["fallback_used"]
        assert mpc.last_solution_info["solution_source"] == "projected_warm_start"

        np.testing.assert_allclose(
            result.control_points,
            c_projected,
        )
        np.testing.assert_allclose(
            mpc.previous_control_points,
            c_projected,
        )
    else:
        assert not mpc.last_solution_info["fallback_used"]
        assert mpc.last_solution_info["solution_source"] == "none"

        np.testing.assert_allclose(
            mpc.previous_control_points,
            initial,
        )


@pytest.mark.parametrize(
    "invalid_control_points",
    [
        np.zeros((3, 2)),
        np.full((4, 2), np.nan),
        np.full((4, 2), np.inf),
    ],
)
def test_mpc_reset_rejects_invalid_control_points(
    invalid_control_points,
):
    initial = np.zeros((4, 2))

    mpc = CartesianMPC(
        FakeProblem(),
        FakeSolver(),
        initial,
        project_warm_start=False,
    )

    with pytest.raises(ValueError):
        mpc.reset(invalid_control_points)

    np.testing.assert_array_equal(
        mpc.previous_control_points,
        initial,
    )


@pytest.mark.parametrize(
    "tolerance, expected_accepted",
    [
        (1e-5, False),
        (1e-3, True),
    ],
)
def test_mpc_linear_feasibility_tolerance(
    tolerance,
    expected_accepted,
):
    initial = np.zeros((4, 2))

    # ConstraintProblem imposes bounds [-0.5, 0.5].
    # This candidate exceeds the upper bound by 1e-4.
    candidate = np.full((4, 2), 0.5001)

    mpc = CartesianMPC(
        ConstraintProblem(),
        FixedSolver(candidate),
        initial,
        project_warm_start=False,
        linear_feasibility_tolerance=tolerance,
    )

    mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    info = mpc.last_solution_info

    assert info["accepted"] == expected_accepted
    assert info["linear_feasible"] == expected_accepted

    np.testing.assert_allclose(
        info["linear_violation"],
        1e-4,
        atol=1e-12,
    )

    expected = candidate if expected_accepted else initial

    np.testing.assert_allclose(
        mpc.previous_control_points,
        expected,
    )


def test_mpc_accepted_solution_is_copied():
    initial = np.zeros((4, 2))
    candidate = np.full((4, 2), 0.25)

    mpc = CartesianMPC(
        ConstraintProblem(),
        FixedSolver(candidate),
        initial,
        project_warm_start=False,
    )

    result = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    assert mpc.last_solution_info["accepted"]

    saved = mpc.previous_control_points.copy()

    # Modifying the externally returned result must not
    # affect the internal MPC warm start.
    result.control_points[:] = 0.4

    np.testing.assert_array_equal(
        mpc.previous_control_points,
        saved,
    )


@pytest.mark.parametrize(
    "scenario, expected_active, expected_solves",
    [
        ("stable", (0,), 1),
        ("new_obstacle", (0, 1), 2),
        ("full_retry", (0, 1, 2), 2),
    ],
)
def test_mpc_dynamic_collision_activation(
    scenario, expected_active, expected_solves
):
    class FakeCollision:
        has_constraints = True

        def __init__(self, indices=None):
            self.environment = SimpleNamespace(n_obstacles=3)
            self.indices = indices

        def subset_for_obstacles(self, indices):
            return FakeCollision(tuple(indices))

        def near_obstacle_indices(self, c, **kwargs):
            marker = float(c[0, 0])

            if marker == 0.0:
                return [0]

            if scenario == "new_obstacle":
                return [0, 1]

            return [0]

        def is_feasible(self, c, tolerance=1e-4):
            if scenario == "full_retry":
                return float(c[0, 0]) >= 2.0
            return True

    class CollisionProblem(FakeProblem):
        def __init__(self):
            super().__init__()
            self.collision_constraints = FakeCollision()

    class RecordingSolver:
        def __init__(self):
            self.active_sets = []

        def solve(
            self, problem, initial_control_points,
            verbose=False
        ):
            collision = problem.collision_constraints
            active = (
                collision.indices
                if collision is not None
                else ()
            )
            self.active_sets.append(active)

            marker = float(len(self.active_sets))

            return SimpleNamespace(
                control_points=np.full((4, 2), marker),
                success=True,
                status="converged",
            )

    problem = CollisionProblem()
    solver = RecordingSolver()
    full_collision = problem.collision_constraints

    mpc = CartesianMPC(
        problem,
        solver,
        np.zeros((4, 2)),
        collision_activation_distance=0.1,
        project_warm_start=False,
    )

    result = mpc.step(
        np.zeros(2),
        np.zeros(2),
        [],
    )

    info = mpc.last_collision_info

    assert info["filtered"]
    assert info["solve_count"] == expected_solves
    assert info["active_indices"] == expected_active
    assert info["feasible"]
    assert mpc.last_solution_info["accepted"]

    # Collision constraints must be restored after solving.
    assert problem.collision_constraints is full_collision

    if scenario == "stable":
        assert solver.active_sets == [(0,)]

    elif scenario == "new_obstacle":
        assert solver.active_sets == [(0,), (0, 1)]

    else:
        assert solver.active_sets == [(0,), None]

    np.testing.assert_allclose(
        mpc.previous_control_points,
        result.control_points,
    )


def test_mpc_osqp_projection_and_reuse():
    mpc = CartesianMPC(
        ConstraintProblem(),
        FakeSolver(),
        np.zeros((4, 2)),
        project_warm_start=True,
    )

    A, lower, upper = mpc.problem.linear_constraints()

    # First projection: upper bounds are violated.
    c1 = np.ones((4, 2))

    projected1, info1 = mpc._project_linear_feasible(
        c1, A, lower, upper
    )

    assert info1["projected"]
    assert info1["before"] > 0
    assert info1["after"] <= mpc.projection_tolerance

    np.testing.assert_allclose(
        projected1,
        0.5,
        atol=1e-6,
    )

    first_solver = mpc._projection_solver
    assert first_solver is not None

    # Second projection: same constraint matrix,
    # different input and different active bounds.
    c2 = -np.ones((4, 2))

    projected2, info2 = mpc._project_linear_feasible(
        c2, A, lower, upper
    )

    assert info2["projected"]
    assert info2["after"] <= mpc.projection_tolerance

    np.testing.assert_allclose(
        projected2,
        -0.5,
        atol=1e-6,
    )

    # OSQP workspace should be reused.
    assert mpc._projection_solver is first_solver

    # Third call: already feasible trajectory.
    c3 = np.zeros((4, 2))

    projected3, info3 = mpc._project_linear_feasible(
        c3, A, lower, upper
    )

    assert not info3["projected"]
    assert info3["status"] == "already_feasible"
    assert info3["iterations"] == 0

    np.testing.assert_array_equal(projected3, c3)
    assert mpc._projection_solver is first_solver

import numpy as np
import scipy.sparse as sp
import pytest

from splineopt.optimizer import GaussNewtonSolver


class DummySpline:
    """Minimal spline interface required by GaussNewtonSolver."""

    n_control_points = 1
    n_dof = 1


class SimpleQuadraticProblem:
    """
    Simple 1D least-squares problem:

        minimize (c - 1)^2

    The exact optimum is c = 1.
    """

    def __init__(self):
        self.spline = DummySpline()

        self.lambda_smooth = 0.0
        self.R2 = np.zeros((1, 1))

    def residual_and_jacobian(self, c):
        c = np.asarray(c, dtype=float).reshape(-1)

        residual = np.array([
            c[0] - 1.0
        ])

        jacobian = np.array([
            [1.0]
        ])

        return residual, jacobian

    def objective(self, c):
        c = np.asarray(c, dtype=float).reshape(-1)

        return float(
            (c[0] - 1.0) ** 2
        )

    def linear_constraints(self):
        # Wide bounds so constraints do not affect
        # the solution.
        A = sp.eye(
            1,
            format="csc",
        )

        l = np.array([-10.0])
        u = np.array([10.0])

        return A, l, u

    def error_metrics(self, c):
        c = np.asarray(c, dtype=float).reshape(-1)

        error = abs(
            c[0] - 1.0
        )

        return {
            "position_rms": error,
            "position_max": error,
            "orientation_rms": 0.0,
            "orientation_max": 0.0,
        }


# ============================================================
# Parameter validation
# ============================================================


def test_adaptive_damping_mode_is_accepted():

    solver = GaussNewtonSolver(
        damping_mode="adaptive",
    )

    assert solver.damping_mode == "adaptive"


def test_invalid_damping_mode_is_rejected():

    with pytest.raises(
        ValueError,
        match="damping_mode",
    ):
        GaussNewtonSolver(
            damping_mode="invalid",
        )


def test_invalid_adaptive_parameters_are_rejected():

    with pytest.raises(ValueError):
        GaussNewtonSolver(
            damping_increase=1.0,
        )

    with pytest.raises(ValueError):
        GaussNewtonSolver(
            damping_decrease=1.0,
        )

    with pytest.raises(ValueError):
        GaussNewtonSolver(
            rho_low=0.8,
            rho_high=0.2,
        )

    with pytest.raises(ValueError):
        GaussNewtonSolver(
            max_damping_retries=-1,
        )


# ============================================================
# Real adaptive solve on a simple quadratic problem
# ============================================================


def test_adaptive_solver_reduces_objective():

    problem = SimpleQuadraticProblem()

    solver = GaussNewtonSolver(
        damping=1e-3,
        damping_mode="adaptive",
        max_iterations=10,
    )

    initial = np.array([
        [0.0]
    ])

    initial_cost = (
        problem.objective(
            initial
        )
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=initial,
    )

    final_cost = (
        problem.objective(
            result.control_points
        )
    )

    assert final_cost < initial_cost
    assert len(result.history) >= 1
    assert len(result.history) == result.iterations

    assert all(
        np.isfinite(step.objective)
        and np.isfinite(step.step_norm)
        and 0.0 < step.alpha <= 1.0
        for step in result.history
    )

    assert all(
        step.osqp_status.lower().startswith("solved")
        for step in result.history
    )


def test_adaptive_solver_converges_near_quadratic_optimum():

    problem = SimpleQuadraticProblem()

    solver = GaussNewtonSolver(
        damping=1e-3,
        damping_mode="adaptive",
        max_iterations=20,
        step_tolerance=1e-8,
        cost_tolerance=1e-10,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=np.array([
            [0.0]
        ]),
    )

    final_c = float(
        result.control_points[0, 0]
    )

    assert final_c == pytest.approx(
        1.0,
        abs=1e-5,
    )


# ============================================================
# Retry behaviour
# ============================================================


def test_adaptive_damping_retries_with_larger_mu(
    monkeypatch,
):

    problem = SimpleQuadraticProblem()

    solver = GaussNewtonSolver(
        damping=1e-3,
        damping_mode="adaptive",
        damping_increase=2.0,
        max_damping_retries=2,
        max_iterations=1,
    )

    damping_values = []

    def fake_gauss_newton_step(
        problem,
        current_control_points,
        *,
        damping=None,
        verbose=False,
    ):
        damping_values.append(
            damping
        )

        # -----------------------------------------------------
        # First QP:
        #
        # Deliberately return a bad direction.
        #
        # Starting from c = 0, moving towards -1 always
        # increases (c - 1)^2, so the line search must fail.
        # -----------------------------------------------------

        if damping < 2e-3:
            c_qp = np.array([
                -1.0
            ])

        # -----------------------------------------------------
        # Retry:
        #
        # Once damping has been increased, return a useful
        # direction towards the optimum.
        # -----------------------------------------------------

        else:
            c_qp = np.array([
                0.5
            ])

        # Undamped quadratic model:
        #
        # (c - 1)^2
        # = c^2 - 2c + 1
        #
        # Constant term is irrelevant.
        P_model = np.array([
            [2.0]
        ])

        g_model = np.array([
            -2.0
        ])

        class FakeInfo:
            status = "solved"
            solve_time = 0.0

        class FakeResult:
            info = FakeInfo()

        return (
            c_qp,
            FakeResult(),
            P_model,
            g_model,
        )

    monkeypatch.setattr(
        solver,
        "gauss_newton_step",
        fake_gauss_newton_step,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=np.array([
            [0.0]
        ]),
    )

    # First QP uses the configured damping.
    assert damping_values[0] == pytest.approx(
        1e-3
    )

    # Line search fails, therefore a second QP must be
    # constructed with increased damping.
    assert damping_values[1] == pytest.approx(
        2e-3
    )

    # A step was eventually accepted.
    assert len(result.history) == 1

    assert (
        result.control_points[0, 0]
        > 0.0
    )


# ============================================================
# Static mode must preserve old behaviour
# ============================================================


def test_static_damping_does_not_retry(
    monkeypatch,
):

    problem = SimpleQuadraticProblem()

    solver = GaussNewtonSolver(
        damping=1e-3,
        damping_mode="static",
        max_damping_retries=2,
        max_iterations=5,
    )

    damping_values = []

    def fake_gauss_newton_step(
        problem,
        current_control_points,
        *,
        damping=None,
        verbose=False,
    ):
        damping_values.append(
            damping
        )

        # Deliberately bad direction.
        c_qp = np.array([
            -1.0
        ])

        P_model = np.array([
            [2.0]
        ])

        g_model = np.array([
            -2.0
        ])

        class FakeInfo:
            status = "solved"
            solve_time = 0.0

        class FakeResult:
            info = FakeInfo()

        return (
            c_qp,
            FakeResult(),
            P_model,
            g_model,
        )

    monkeypatch.setattr(
        solver,
        "gauss_newton_step",
        fake_gauss_newton_step,
    )

    result = solver.solve(
        problem=problem,
        initial_control_points=np.array([
            [0.0]
        ]),
    )

    assert (
        result.status
        == "line_search_failed"
    )

    # Static mode must not rebuild the QP with another damping.
    assert len(damping_values) == 1

    assert damping_values[0] == pytest.approx(
        1e-3
    )


def test_adaptive_damping_retry_limit(monkeypatch):
    problem = SimpleQuadraticProblem()

    solver = GaussNewtonSolver(
        damping=1e-3,
        damping_mode="adaptive",
        damping_increase=2.0,
        max_damping_retries=2,
        max_iterations=5,
    )

    damping_values = []

    def fake_gauss_newton_step(
        problem,
        current_control_points,
        *,
        damping=None,
        verbose=False,
    ):
        damping_values.append(damping)

        # Always propose a direction increasing the objective.
        c_qp = np.array([-1.0])
        P_model = np.array([[2.0]])
        g_model = np.array([-2.0])

        class FakeInfo:
            status = "solved"
            solve_time = 0.0

        class FakeResult:
            info = FakeInfo()

        return c_qp, FakeResult(), P_model, g_model

    monkeypatch.setattr(
        solver,
        "gauss_newton_step",
        fake_gauss_newton_step,
    )

    initial = np.array([[0.0]])

    result = solver.solve(
        problem=problem,
        initial_control_points=initial,
    )

    assert not result.success
    assert result.status == "damping_limit"
    assert result.iterations == 0
    assert len(result.history) == 0

    np.testing.assert_allclose(
        result.control_points,
        initial,
    )

    np.testing.assert_allclose(
        damping_values,
        [1e-3, 2e-3, 4e-3],
    )


@pytest.mark.parametrize(
    "status, solution, expected_error",
    [
        ("primal infeasible", None, "Gauss-Newton QP failed"),
        ("maximum iterations reached", None, "Gauss-Newton QP failed"),
        ("solved", [], "OSQP returned fewer variables"),
    ],
)
def test_gauss_newton_rejects_invalid_qp_result(
    monkeypatch, status, solution, expected_error
):
    from types import SimpleNamespace
    import splineopt.optimizer as optimizer_module

    class FakeOSQP:
        def setup(self, **kwargs):
            pass

        def solve(self):
            return SimpleNamespace(
                info=SimpleNamespace(status=status),
                x=solution,
            )

    monkeypatch.setattr(
        optimizer_module.osqp,
        "OSQP",
        FakeOSQP,
    )

    problem = SimpleQuadraticProblem()
    solver = GaussNewtonSolver()

    with pytest.raises(RuntimeError, match=expected_error):
        solver.gauss_newton_step(
            problem,
            np.array([[0.0]]),
        )

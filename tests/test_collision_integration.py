import numpy as np
import scipy.sparse as sp

from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
)
from splineopt.optimizer import (
    GaussNewtonSolver,
)


# ============================================================
# Small helpers / dummy objects
# ============================================================


class DummyCollisionConstraints:
    """
    Minimal collision-constraint object used to test the
    CartesianTrajectoryProblem integration independently from
    robot geometry.
    """

    def __init__(
        self,
        H,
        h,
    ):
        self.H = sp.csc_matrix(
            H
        )

        self.h = np.asarray(
            h,
            dtype=float,
        ).reshape(-1)

        self.calls = 0

    @property
    def has_constraints(
        self,
    ):
        return (
            self.H.shape[0] > 0
        )

    def linearize(
        self,
        control_points,
    ):
        self.calls += 1

        return (
            self.H.copy(),
            self.h.copy(),
        )


class MinimalCartesianProblem(
    CartesianTrajectoryProblem
):
    """
    Lightweight subclass used only to test constraint assembly.

    We deliberately bypass CartesianTrajectoryProblem.__init__()
    because the methods under test only require:

        - linear_constraints()
        - collision_constraints
    """

    def __init__(
        self,
        A,
        l,
        u,
        collision_constraints=None,
    ):
        self._A_static = (
            sp.csc_matrix(A)
        )

        self._l_static = np.asarray(
            l,
            dtype=float,
        ).reshape(-1)

        self._u_static = np.asarray(
            u,
            dtype=float,
        ).reshape(-1)

        self.collision_constraints = (
            collision_constraints
        )

    def linear_constraints(
        self,
    ):
        return (
            self._A_static.copy(),
            self._l_static.copy(),
            self._u_static.copy(),
        )


# ============================================================
# CartesianTrajectoryProblem integration
# ============================================================


def test_linearized_constraints_without_collisions_are_unchanged():
    """
    If collision_constraints is None, the new interface must
    reproduce exactly the previous static constraint system.
    """

    A_static = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    l_static = np.array([
        -1.0,
        -2.0,
    ])

    u_static = np.array([
        1.0,
        2.0,
    ])

    problem = MinimalCartesianProblem(
        A=A_static,
        l=l_static,
        u=u_static,
        collision_constraints=None,
    )

    c = np.array([
        0.2,
        -0.1,
    ])

    A, l, u = (
        problem.linearized_constraints(
            c
        )
    )

    np.testing.assert_allclose(
        A.toarray(),
        A_static,
    )

    np.testing.assert_allclose(
        l,
        l_static,
    )

    np.testing.assert_allclose(
        u,
        u_static,
    )


def test_empty_collision_constraints_are_ignored():
    """
    A collision builder with zero rows must follow the fast path
    and leave the static problem unchanged.
    """

    A_static = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    l_static = np.array([
        -1.0,
        -1.0,
    ])

    u_static = np.array([
        1.0,
        1.0,
    ])

    collision = DummyCollisionConstraints(
        H=np.empty(
            (
                0,
                2,
            )
        ),
        h=np.empty(
            0
        ),
    )

    problem = MinimalCartesianProblem(
        A=A_static,
        l=l_static,
        u=u_static,
        collision_constraints=collision,
    )

    c = np.zeros(
        2
    )

    A, l, u = (
        problem.linearized_constraints(
            c
        )
    )

    np.testing.assert_allclose(
        A.toarray(),
        A_static,
    )

    np.testing.assert_allclose(
        l,
        l_static,
    )

    np.testing.assert_allclose(
        u,
        u_static,
    )

    # Important:
    # the fast path should avoid even calling linearize().
    assert collision.calls == 0


def test_collision_constraints_are_appended_correctly():
    """
    Check

        A =
            [ A_static ]
            [ H_col    ]

    with

        l =
            [ l_static ]
            [ h_col    ]

        u =
            [ u_static ]
            [ +inf     ].
    """

    A_static = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    l_static = np.array([
        -1.0,
        -2.0,
    ])

    u_static = np.array([
        1.0,
        2.0,
    ])

    H_col = np.array([
        [1.0, 1.0],
        [-2.0, 3.0],
    ])

    h_col = np.array([
        0.4,
        -0.2,
    ])

    collision = DummyCollisionConstraints(
        H=H_col,
        h=h_col,
    )

    problem = MinimalCartesianProblem(
        A=A_static,
        l=l_static,
        u=u_static,
        collision_constraints=collision,
    )

    c = np.array([
        0.1,
        0.2,
    ])

    A, l, u = (
        problem.linearized_constraints(
            c
        )
    )

    expected_A = np.vstack([
        A_static,
        H_col,
    ])

    expected_l = np.concatenate([
        l_static,
        h_col,
    ])

    expected_u = np.concatenate([
        u_static,
        np.full(
            h_col.size,
            np.inf,
        ),
    ])

    np.testing.assert_allclose(
        A.toarray(),
        expected_A,
    )

    np.testing.assert_allclose(
        l,
        expected_l,
    )

    np.testing.assert_allclose(
        u,
        expected_u,
    )

    assert collision.calls == 1


# ============================================================
# Linearization identity
# ============================================================


class DummyCollisionBuilder:
    """
    Small analytical collision builder used to verify the
    linearization identity at the expansion point.

    Distance:

        d(c) = a^T c + b

    which is already linear.
    """

    def __init__(
        self,
        a,
        b,
        safety_distance,
    ):
        self.a = np.asarray(
            a,
            dtype=float,
        ).reshape(-1)

        self.b = float(
            b
        )

        self.safety_distance = float(
            safety_distance
        )

        self.n_variables = (
            self.a.size
        )

    @property
    def has_constraints(
        self,
    ):
        return True

    def distance(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        return float(
            self.a @ c
            + self.b
        )

    def linearize(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        d = self.distance(
            c
        )

        H = self.a.reshape(
            1,
            -1,
        )

        h = np.array([
            self.safety_distance
            - d
            + H @ c
        ]).reshape(-1)

        return (
            sp.csc_matrix(H),
            h,
        )


def test_collision_linearization_identity_at_current_point():
    """
    At the expansion point c_s,

        H c_s - h
        =
        d(c_s) - d_safe.

    This identity is important: the linearized constraint has
    exactly the same violation as the true constraint at the
    point where it is constructed.
    """

    builder = DummyCollisionBuilder(
        a=np.array([
            1.2,
            -0.7,
            0.4,
        ]),
        b=0.15,
        safety_distance=0.20,
    )

    c = np.array([
        0.3,
        -0.1,
        0.25,
    ])

    H, h = (
        builder.linearize(
            c
        )
    )

    linearized_margin = float(
        (H @ c)[0]
        - h[0]
    )

    true_margin = (
        builder.distance(c)
        - builder.safety_distance
    )

    np.testing.assert_allclose(
        linearized_margin,
        true_margin,
        atol=1e-12,
    )


# ============================================================
# GaussNewtonSolver integration
# ============================================================


class DynamicConstraintProblem:
    """
    Minimal Gauss-Newton problem used to verify that the solver
    requests dynamic constraints around the current iterate.
    """

    def __init__(
        self,
    ):
        self.lambda_smooth = 0.0

        self.R2 = np.zeros(
            (
                2,
                2,
            )
        )

        self.last_linearization_point = (
            None
        )

    def residual_and_jacobian(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        # residual:
        #
        #     r(c) = c - target
        #
        target = np.array([
            1.0,
            2.0,
        ])

        residual = (
            c - target
        )

        jacobian = np.eye(
            2
        )

        return (
            residual,
            jacobian,
        )

    def linear_constraints(
        self,
    ):
        """
        Static fallback constraints.

        If this method were used instead of
        linearized_constraints(), A would have zero rows.
        """

        return (
            sp.csc_matrix(
                (
                    0,
                    2,
                )
            ),
            np.empty(
                0
            ),
            np.empty(
                0
            ),
        )

    def linearized_constraints(
        self,
        c,
    ):
        self.last_linearization_point = (
            np.asarray(
                c,
                dtype=float,
            ).copy()
        )

        # Dynamic constraint:
        #
        #     x >= 0.25
        #
        A = sp.csc_matrix(
            np.array([
                [1.0, 0.0],
            ])
        )

        l = np.array([
            0.25
        ])

        u = np.array([
            np.inf
        ])

        return A, l, u


def test_gauss_newton_qp_uses_linearized_constraints():
    """
    Verify that GaussNewtonSolver.build_gauss_newton_qp()
    uses problem.linearized_constraints(c) when available.
    """

    problem = (
        DynamicConstraintProblem()
    )

    solver = GaussNewtonSolver(
        damping=1e-4,
    )

    c = np.array([
        0.4,
        -0.2,
    ])

    (
        _,
        _,
        A,
        l,
        u,
        _,
        _,
    ) = solver.build_gauss_newton_qp(
        problem=problem,
        current_control_points=c,
    )

    assert (
        problem.last_linearization_point
        is not None
    )

    np.testing.assert_allclose(
        problem.last_linearization_point,
        c,
    )

    assert A.shape == (
        1,
        2,
    )

    np.testing.assert_allclose(
        A.toarray(),
        np.array([
            [1.0, 0.0]
        ]),
    )

    np.testing.assert_allclose(
        l,
        np.array([
            0.25
        ]),
    )

    assert np.isposinf(
        u[0]
    )


# ============================================================
# Backward compatibility
# ============================================================


class StaticConstraintProblem:
    """
    Old-style problem exposing only linear_constraints().
    """

    def __init__(
        self,
    ):
        self.lambda_smooth = 0.0

        self.R2 = np.zeros(
            (
                2,
                2,
            )
        )

    def residual_and_jacobian(
        self,
        c,
    ):
        c = np.asarray(
            c,
            dtype=float,
        ).reshape(-1)

        residual = c.copy()

        jacobian = np.eye(
            2
        )

        return (
            residual,
            jacobian,
        )

    def linear_constraints(
        self,
    ):
        A = sp.csc_matrix(
            np.eye(
                2
            )
        )

        l = np.array([
            -1.0,
            -2.0,
        ])

        u = np.array([
            1.0,
            2.0,
        ])

        return A, l, u


def test_gauss_newton_qp_keeps_static_constraint_compatibility():
    """
    Problems written before collision support must continue to
    work without defining linearized_constraints().
    """

    problem = (
        StaticConstraintProblem()
    )

    solver = GaussNewtonSolver(
        damping=1e-4,
    )

    c = np.array([
        0.1,
        -0.2,
    ])

    (
        _,
        _,
        A,
        l,
        u,
        _,
        _,
    ) = solver.build_gauss_newton_qp(
        problem=problem,
        current_control_points=c,
    )

    np.testing.assert_allclose(
        A.toarray(),
        np.eye(2),
    )

    np.testing.assert_allclose(
        l,
        np.array([
            -1.0,
            -2.0,
        ]),
    )

    np.testing.assert_allclose(
        u,
        np.array([
            1.0,
            2.0,
        ]),
    )


def test_nonlinear_collision_constraint_bounds():
    """
    Check that CartesianTrajectoryProblem exposes collision
    bounds in the form

        safety_distance <= d(c) <= +inf.
    """

    class DummySpline:
        n_control_points = 1
        n_dof = 2

    class DummyNonlinearCollisionConstraints:
        def __init__(self):
            self.safety_distance = 0.02
            self.n_constraints = 3

        @property
        def has_constraints(self):
            return True

    A_static = np.eye(
        2
    )

    collision = (
        DummyNonlinearCollisionConstraints()
    )

    problem = MinimalCartesianProblem(
        A=A_static,
        l=np.array([
            -1.0,
            -1.0,
        ]),
        u=np.array([
            1.0,
            1.0,
        ]),
        collision_constraints=collision,
    )

    problem.spline = (
        DummySpline()
    )

    lower, upper = (
        problem
        .nonlinear_constraint_bounds()
    )

    assert lower.shape == (
        collision.n_constraints,
    )

    assert upper.shape == (
        collision.n_constraints,
    )

    np.testing.assert_allclose(
        lower,
        collision.safety_distance,
    )

    assert np.all(
        np.isposinf(
            upper
        )
    )


def test_nonlinear_constraints_are_empty_without_collisions():
    A_static = np.eye(2)

    problem = MinimalCartesianProblem(
        A=A_static,
        l=np.array([
            -1.0,
            -1.0,
        ]),
        u=np.array([
            1.0,
            1.0,
        ]),
        collision_constraints=None,
    )

    # MinimalCartesianProblem does not inherit the full __init__,
    # but it does inherit these methods.
    problem.spline = type(
        "DummySpline",
        (),
        {
            "n_control_points": 1,
            "n_dof": 2,
        },
    )()

    c = np.zeros(
        2
    )

    values = (
        problem
        .nonlinear_constraints(
            c
        )
    )

    jacobian = (
        problem
        .nonlinear_constraints_jacobian(
            c
        )
    )

    lower, upper = (
        problem
        .nonlinear_constraint_bounds()
    )

    assert values.shape == (
        0,
    )

    assert jacobian.shape == (
        0,
        2,
    )

    assert lower.shape == (
        0,
    )

    assert upper.shape == (
        0,
    )


def test_nonlinear_collision_values_and_jacobian():
    """
    Verify that the Cartesian problem correctly forwards
    nonlinear collision distances and Jacobians.
    """

    class DummySpline:
        n_control_points = 2
        n_dof = 2

    class DummyCollisionConstraints:
        safety_distance = 0.02
        n_constraints = 2
        has_constraints = True

        def distances(self, c):
            c = np.asarray(c).reshape(-1)
            return np.array([
                [c[0] + 2.0 * c[1], c[2] - c[3]]
            ])

        def values_and_jacobian(self, c):
            values = self.distances(c).reshape(-1)

            J = sp.csc_matrix([
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ])

            return values, J

    problem = MinimalCartesianProblem(
        A=np.eye(4),
        l=np.full(4, -10.0),
        u=np.full(4, 10.0),
        collision_constraints=DummyCollisionConstraints(),
    )

    problem.spline = DummySpline()

    c = np.array([0.1, 0.2, 0.3, 0.4])

    values = problem.nonlinear_constraints(c)
    J = problem.nonlinear_constraints_jacobian(c)
    lower, upper = problem.nonlinear_constraint_bounds()

    np.testing.assert_allclose(
        values,
        [0.5, -0.1],
    )

    np.testing.assert_allclose(
        J.toarray(),
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ],
    )

    np.testing.assert_allclose(
        lower,
        [0.02, 0.02],
    )

    assert np.all(np.isposinf(upper))
    assert J.shape == (2, 4)


def test_collision_diagnostics_without_obstacles():
    problem = MinimalCartesianProblem(
        A=np.eye(2),
        l=np.array([-1.0, -1.0]),
        u=np.array([1.0, 1.0]),
        collision_constraints=None,
    )

    c = np.array([0.2, -0.1])

    distances = problem.collision_distances(c)
    min_distance = problem.minimum_collision_distance(c)
    feasible = problem.collision_feasible(c)

    assert distances.shape == (0, 0)
    assert np.isposinf(min_distance)
    assert feasible

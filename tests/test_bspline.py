import numpy as np
from scipy.interpolate import BSpline as SciPyBSpline
import pytest
from splineopt.bspline import (
    BSpline,
    gamma,
    derivative_gram_matrix,
)


# ============================================================
# Common test data
# ============================================================

DEGREE = 3
N_DOF = 2

KNOTS = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 3.0,
    4.0, 4.0, 4.0, 4.0,
])

N_CONTROL_POINTS = len(KNOTS) - DEGREE - 1

CONTROL_POINTS = np.array([
    [0.0, 1.0],
    [0.3, 1.1],
    [1.0, 0.8],
    [1.5, 0.4],
    [1.2, 0.2],
    [0.5, 0.7],
    [0.0, 1.0],
])


# ============================================================
# Gamma
# ============================================================

def test_gamma():

    s = 0.4

    np.testing.assert_allclose(
        gamma(s, degree=3, derivative_order=0),
        np.array([
            1.0,
            s,
            s**2,
            s**3,
        ]),
    )

    np.testing.assert_allclose(
        gamma(s, degree=3, derivative_order=1),
        np.array([
            0.0,
            1.0,
            2.0 * s,
            3.0 * s**2,
        ]),
    )

    np.testing.assert_allclose(
        gamma(s, degree=3, derivative_order=2),
        np.array([
            0.0,
            0.0,
            2.0,
            6.0 * s,
        ]),
    )


# ============================================================
# Spline structure
# ============================================================

def test_spline_dimensions():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    assert spline.n_control_points == 7
    assert spline.t_start == 0.0
    assert spline.t_end == 4.0


def test_segment_detection():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    assert spline.segment(0.2) == 0
    assert spline.segment(1.5) == 1
    assert spline.segment(2.5) == 2
    assert spline.segment(3.5) == 3

    # Final time must belong to final segment
    assert spline.segment(4.0) == 3


def test_normalized_time():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    s, delta = spline.normalized_time(
        t=1.5,
        segment=1,
    )

    assert np.isclose(delta, 1.0)
    assert np.isclose(s, 0.5)


# ============================================================
# Basis matrix M_k
# ============================================================

def test_basis_matrix_against_scipy():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    # Giving the identity as coefficients makes SciPy
    # return the values of all basis functions.
    scipy_basis = SciPyBSpline(
        KNOTS,
        np.eye(N_CONTROL_POINTS),
        DEGREE,
    )

    for segment in spline.segments:

        t0 = KNOTS[segment + DEGREE]
        t1 = KNOTS[segment + DEGREE + 1]

        # Test several positions inside each segment
        for s in [0.1, 0.3, 0.5, 0.8]:

            t = t0 + s * (t1 - t0)

            M = spline.basis_matrix(segment)

            local_basis = (
                gamma(s, DEGREE)
                @ M
            )

            scipy_global_basis = scipy_basis(t)

            scipy_local_basis = scipy_global_basis[
                segment:segment + DEGREE + 1
            ]

            np.testing.assert_allclose(
                local_basis,
                scipy_local_basis,
                atol=1e-10,
            )


# ============================================================
# Selection matrix S_k
# ============================================================

def test_selection_matrix():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    segment = 1

    S = spline.selection_matrix(segment)

    c = CONTROL_POINTS.reshape(-1)

    selected = S @ c

    expected = CONTROL_POINTS[
        segment:segment + DEGREE + 1
    ].reshape(-1)

    np.testing.assert_allclose(
        selected,
        expected,
        atol=1e-12,
    )


def test_selection_matrix_shape():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    S = spline.selection_matrix(0)

    assert S.shape == (
        (DEGREE + 1) * N_DOF,
        N_CONTROL_POINTS * N_DOF,
    )


# ============================================================
# Evaluation matrix A^(r)(t)
# ============================================================

def test_position_against_scipy():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    scipy_spline = SciPyBSpline(
        KNOTS,
        CONTROL_POINTS,
        DEGREE,
        axis=0,
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    for t in times:

        ours = spline.evaluate(
            CONTROL_POINTS,
            t,
            derivative_order=0,
        )

        expected = scipy_spline(t)

        np.testing.assert_allclose(
            ours,
            expected,
            atol=1e-10,
        )


def test_velocity_against_scipy():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    scipy_spline = SciPyBSpline(
        KNOTS,
        CONTROL_POINTS,
        DEGREE,
        axis=0,
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    for t in times:

        ours = spline.evaluate(
            CONTROL_POINTS,
            t,
            derivative_order=1,
        )

        expected = scipy_spline.derivative(1)(t)

        np.testing.assert_allclose(
            ours,
            expected,
            atol=1e-10,
        )


def test_acceleration_against_scipy():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    scipy_spline = SciPyBSpline(
        KNOTS,
        CONTROL_POINTS,
        DEGREE,
        axis=0,
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    for t in times:

        ours = spline.evaluate(
            CONTROL_POINTS,
            t,
            derivative_order=2,
        )

        expected = scipy_spline.derivative(2)(t)

        np.testing.assert_allclose(
            ours,
            expected,
            atol=1e-10,
        )


# ============================================================
# Evaluation matrix dimensions
# ============================================================

def test_evaluation_matrix_shape():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    A = spline.evaluation_matrix(
        t=1.5,
        derivative_order=0,
    )

    assert A.shape == (
        N_DOF,
        N_CONTROL_POINTS * N_DOF,
    )


# ============================================================
# Derivative control-point matrices
# ============================================================

def test_first_derivative_control_matrix():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    D1 = spline.derivative_control_matrix(1)

    c = CONTROL_POINTS.reshape(-1)

    derivative_control_points = (
        D1 @ c
    ).reshape(
        N_CONTROL_POINTS - 1,
        N_DOF,
    )

    # The derivative spline loses one knot
    # at each end and one degree.
    derivative_knots = KNOTS[1:-1]

    derivative_spline = SciPyBSpline(
        derivative_knots,
        derivative_control_points,
        DEGREE - 1,
        axis=0,
    )

    original_spline = SciPyBSpline(
        KNOTS,
        CONTROL_POINTS,
        DEGREE,
        axis=0,
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    np.testing.assert_allclose(
        derivative_spline(times),
        original_spline.derivative(1)(times),
        atol=1e-10,
    )


def test_second_derivative_control_matrix():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    D2 = spline.derivative_control_matrix(2)

    c = CONTROL_POINTS.reshape(-1)

    derivative_control_points = (
        D2 @ c
    ).reshape(
        N_CONTROL_POINTS - 2,
        N_DOF,
    )

    derivative_knots = KNOTS[2:-2]

    derivative_spline = SciPyBSpline(
        derivative_knots,
        derivative_control_points,
        DEGREE - 2,
        axis=0,
    )

    original_spline = SciPyBSpline(
        KNOTS,
        CONTROL_POINTS,
        DEGREE,
        axis=0,
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    np.testing.assert_allclose(
        derivative_spline(times),
        original_spline.derivative(2)(times),
        atol=1e-10,
    )


# ============================================================
# Cache
# ============================================================

def test_basis_matrix_cache():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    M1 = spline.basis_matrix(1)
    M2 = spline.basis_matrix(1)

    # Same object => returned from cache
    assert M1 is M2


def test_selection_matrix_cache():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    S1 = spline.selection_matrix(1)
    S2 = spline.selection_matrix(1)

    assert S1 is S2


def test_derivative_matrix_cache():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    D1 = spline.derivative_control_matrix(2)
    D2 = spline.derivative_control_matrix(2)

    assert D1 is D2


def test_derivative_gram_matrix_cubic_acceleration():

    Q2 = derivative_gram_matrix(
        degree=3,
        derivative_order=2,
    )

    expected = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 6.0],
        [0.0, 0.0, 6.0, 12.0],
    ])

    np.testing.assert_allclose(
        Q2,
        expected,
    )


def test_smoothness_matrix_against_numerical_integration():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    R2 = spline.smoothness_matrix(
        derivative_order=2
    )

    c = CONTROL_POINTS.reshape(-1)

    # Exact quadratic formulation
    matrix_cost = c.T @ R2 @ c

    # Numerical reference
    times = np.linspace(
        spline.t_start,
        spline.t_end,
        10000,
    )

    accelerations = np.array([
        spline.acceleration(
            CONTROL_POINTS,
            t,
        )
        for t in times
    ])

    squared_acceleration = np.sum(
        accelerations**2,
        axis=1,
    )

    numerical_cost = np.trapezoid(
        squared_acceleration,
        times,
    )

    np.testing.assert_allclose(
        matrix_cost,
        numerical_cost,
        rtol=1e-5,
        atol=1e-7,
    )


def test_smoothness_matrix_properties():

    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    R2 = spline.smoothness_matrix(2)

    expected_size = (
        N_CONTROL_POINTS * N_DOF
    )

    assert R2.shape == (
        expected_size,
        expected_size,
    )

    # Symmetric
    np.testing.assert_allclose(
        R2,
        R2.T,
        atol=1e-12,
    )

    # Positive semi-definite
    eigenvalues = np.linalg.eigvalsh(R2)

    assert np.min(eigenvalues) > -1e-10


def test_segment_near_final_time():
    """Times close to t_end must keep their correct segment."""
    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.5, 0.999999,
        1.0, 1.0, 1.0, 1.0,
    ])

    spline = BSpline(
        knots=knots,
        degree=3,
        n_dof=2,
    )

    assert spline.segment(0.999995) == 1
    assert spline.segment(0.9999995) == 2
    assert spline.segment(1.0) == 2


def test_reject_zero_duration_domain():
    """A spline must have at least one non-empty knot span."""
    knots = np.ones(10)

    with pytest.raises(ValueError, match="positive duration"):
        BSpline(
            knots=knots,
            degree=3,
            n_dof=2,
        )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_continuity_at_internal_knots(order):
    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    eps = 1e-7

    for t in [1.0, 2.0, 3.0]:
        left = spline.evaluate(
            CONTROL_POINTS, t - eps, derivative_order=order
        )
        right = spline.evaluate(
            CONTROL_POINTS, t + eps, derivative_order=order
        )

        np.testing.assert_allclose(
            left,
            right,
            rtol=0.0,
            atol=1e-5,
        )


def test_derivative_above_degree_is_zero():
    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    for order in [4, 5]:
        result = spline.evaluate(
            CONTROL_POINTS,
            t=1.5,
            derivative_order=order,
        )

        np.testing.assert_array_equal(
            result,
            np.zeros(N_DOF),
        )


@pytest.mark.parametrize("t", [-0.1, 4.1])
def test_evaluate_rejects_time_outside_domain(t):
    spline = BSpline(
        knots=KNOTS,
        degree=DEGREE,
        n_dof=N_DOF,
    )

    with pytest.raises(ValueError, match="outside spline domain"):
        spline.evaluate(CONTROL_POINTS, t)

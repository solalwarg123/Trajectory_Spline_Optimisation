import numpy as np
from math import factorial


def gamma(s, degree, derivative_order=0):
    """
    Compute Gamma_p^(r)(s).

    Gamma_p(s) = [1, s, s^2, ..., s^p]
    """

    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative.")

    if derivative_order > degree:
        return np.zeros(degree + 1)

    Gamma = np.zeros(degree + 1)

    for j in range(derivative_order, degree + 1):
        Gamma[j] = (
            factorial(j)
            / factorial(j - derivative_order)
            * s ** (j - derivative_order)
        )

    return Gamma


def derivative_gram_matrix(degree, derivative_order=0):
    """
    Compute analytically

        Q_r = integral_0^1 Gamma_p^(r)(s).T Gamma_p^(r)(s) ds
    """

    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative.")

    if derivative_order > degree:
        return np.zeros((degree + 1, degree + 1))

    Q = np.zeros((degree + 1, degree + 1))

    r = derivative_order

    for i in range(r, degree + 1):
        coeff_i = factorial(i) / factorial(i - r)

        for j in range(r, degree + 1):
            coeff_j = factorial(j) / factorial(j - r)

            Q[i, j] = (
                coeff_i
                * coeff_j
                / (i + j - 2 * r + 1)
            )

    return Q


class BSpline:
    """
    B-spline trajectory representation.

    The control points are stacked as

        c = [c_0^T, ..., c_{Nc-1}^T]^T

    with c_i in R^(n_dof).

    The trajectory and its derivatives satisfy

        q^(r)(t) = A^(r)(t) c.
    """

    def __init__(self, knots, degree, n_dof):

        self.knots = np.asarray(knots, dtype=float)
        self.degree = degree
        self.n_dof = n_dof

        self._validate()

        self.n_control_points = (
            len(self.knots) - self.degree - 1
        )

        # Non-empty spline segments
        self.segments = [
            k
            for k in range(
                self.n_control_points - self.degree
            )
            if (
                self.knots[k + self.degree]
                < self.knots[k + self.degree + 1]
            )
        ]

        # ----------------------------------------------------
        # Caches
        # ----------------------------------------------------

        # M_k : local polynomial basis matrix
        self._basis_matrix_cache = {}

        # S_k : local control-point selection matrix
        self._selection_matrix_cache = {}

        # D_r : derivative control-point matrix
        self._derivative_matrix_cache = {}

        # R_r : smoothness cost matrix
        self._smoothness_matrix_cache = {}

    # ========================================================
    # Validation
    # ========================================================

    def _validate(self):

        if self.knots.ndim != 1:
            raise ValueError("knots must be a 1D array.")

        if np.any(np.diff(self.knots) < 0):
            raise ValueError(
                "knots must be non-decreasing."
            )

        if self.degree < 0:
            raise ValueError(
                "degree must be non-negative."
            )

        if self.n_dof <= 0:
            raise ValueError(
                "n_dof must be strictly positive."
            )

        n_control_points = (
            len(self.knots) - self.degree - 1
        )

        if n_control_points <= self.degree:
            raise ValueError(
                "Not enough control points for this degree."
            )

    # ========================================================
    # Domain
    # ========================================================

    @property
    def t_start(self):
        return self.knots[self.degree]

    @property
    def t_end(self):
        return self.knots[self.n_control_points]

    def segment(self, t):
        """
        Return segment k such that

            t in [tau_{k+p}, tau_{k+p+1}].
        """

        if t < self.t_start or t > self.t_end:
            raise ValueError(
                f"t={t} outside spline domain "
                f"[{self.t_start}, {self.t_end}]."
            )

        # Final point belongs to final segment
        if np.isclose(t, self.t_end):
            return self.segments[-1]

        for k in self.segments:

            t0 = self.knots[k + self.degree]
            t1 = self.knots[k + self.degree + 1]

            if t0 <= t < t1:
                return k

        raise RuntimeError(
            f"Could not find segment for t={t}."
        )

    def normalized_time(self, t, segment):
        """
        Compute

            Delta_k = tau_{k+p+1} - tau_{k+p}

        and

            s = (t - tau_{k+p}) / Delta_k.
        """

        t0 = self.knots[segment + self.degree]
        t1 = self.knots[segment + self.degree + 1]

        delta = t1 - t0
        s = (t - t0) / delta

        return s, delta

    # ========================================================
    # Local basis matrix M_k
    # ========================================================

    def basis_matrix(self, segment):
        """
        Return M_k.

        The matrix is computed only once for each segment.
        """

        if segment not in self._basis_matrix_cache:
            self._basis_matrix_cache[segment] = (
                self._compute_basis_matrix(segment)
            )

        return self._basis_matrix_cache[segment]

    def _compute_basis_matrix(self, segment):
        """
        Compute the local basis matrix M_k such that

            [B_k,p(t), ..., B_k+p,p(t)]
            =
            Gamma_p(s) M_k.

        The Cox-de Boor recursion is performed directly in
        the local polynomial variable s.
        """

        p = self.degree
        i = segment + p

        t0 = self.knots[i]
        t1 = self.knots[i + 1]
        delta = t1 - t0

        # Temporary cache used only during construction of M_k
        polynomial_cache = {}

        def basis_polynomial(j, degree):
            """
            Polynomial coefficients of B_{j,degree}
            on the current knot interval.

            Coefficients are ordered as

                [a0, a1, ..., ap]

            for

                a0 + a1*s + ... + ap*s^p.
            """

            key = (j, degree)

            if key in polynomial_cache:
                return polynomial_cache[key]

            if degree == 0:

                result = np.array(
                    [1.0 if j == i else 0.0]
                )

                polynomial_cache[key] = result
                return result

            result = np.zeros(degree + 1)

            # ----------------------------------------------
            # First Cox-de Boor term
            # ----------------------------------------------

            denominator = (
                self.knots[j + degree]
                - self.knots[j]
            )

            if denominator != 0:

                previous = basis_polynomial(
                    j,
                    degree - 1,
                )

                factor = np.array([
                    (t0 - self.knots[j])
                    / denominator,

                    delta / denominator,
                ])

                result += np.convolve(
                    previous,
                    factor,
                )

            # ----------------------------------------------
            # Second Cox-de Boor term
            # ----------------------------------------------

            denominator = (
                self.knots[j + degree + 1]
                - self.knots[j + 1]
            )

            if denominator != 0:

                previous = basis_polynomial(
                    j + 1,
                    degree - 1,
                )

                factor = np.array([
                    (
                        self.knots[j + degree + 1]
                        - t0
                    )
                    / denominator,

                    -delta / denominator,
                ])

                result += np.convolve(
                    previous,
                    factor,
                )

            polynomial_cache[key] = result

            return result

        M = np.zeros((p + 1, p + 1))

        for column, j in enumerate(
            range(segment, segment + p + 1)
        ):
            M[:, column] = basis_polynomial(j, p)

        return M

    # ========================================================
    # Selection matrix S_k
    # ========================================================

    def selection_matrix(self, segment):
        """
        Return S_k such that

            c_{k:k+p} = S_k c.

        S_k is computed only once per segment.
        """

        if segment not in self._selection_matrix_cache:
            self._selection_matrix_cache[segment] = (
                self._compute_selection_matrix(segment)
            )

        return self._selection_matrix_cache[segment]

    def _compute_selection_matrix(self, segment):

        selector = np.zeros(
            (
                self.degree + 1,
                self.n_control_points,
            )
        )

        selector[
            :,
            segment:segment + self.degree + 1
        ] = np.eye(self.degree + 1)

        return np.kron(
            selector,
            np.eye(self.n_dof),
        )

    def smoothness_matrix(self, derivative_order=2):
        """
        Return R_r such that

            integral ||q^(r)(t)||^2 dt = c.T @ R_r @ c
        """

        if derivative_order < 0:
            raise ValueError("derivative_order must be non-negative.")

        if derivative_order > self.degree:
            raise ValueError(
                "derivative_order cannot exceed spline degree."
            )

        if derivative_order not in self._smoothness_matrix_cache:
            self._smoothness_matrix_cache[derivative_order] = (
                self._compute_smoothness_matrix(
                    derivative_order
                )
            )

        return self._smoothness_matrix_cache[
            derivative_order
        ]

    def _compute_smoothness_matrix(self, derivative_order):

        r = derivative_order

        n_variables = (
            self.n_control_points
            * self.n_dof
        )

        R = np.zeros((
            n_variables,
            n_variables,
        ))

        Q = derivative_gram_matrix(
            self.degree,
            r,
        )

        for k in self.segments:

            t0 = self.knots[
                k + self.degree
            ]

            t1 = self.knots[
                k + self.degree + 1
            ]

            delta = t1 - t0

            M = self.basis_matrix(k)
            S = self.selection_matrix(k)

            local_R_scalar = (
                M.T
                @ Q
                @ M
            )

            local_R = np.kron(
                local_R_scalar,
                np.eye(self.n_dof),
            )

            local_R *= delta ** (1 - 2 * r)

            R += (
                S.T
                @ local_R
                @ S
            )

        return R

    # ========================================================
    # Evaluation matrix A_k^(r)(t)
    # ========================================================

    def evaluation_matrix(
        self,
        t,
        derivative_order=0,
    ):
        """
        Compute

            A_k^(r)(t)

        such that

            q^(r)(t) = A_k^(r)(t) c.

        Using

                         1
        A_k^(r)(t) = ---------
                       Delta^r

            (Gamma_p^(r)(s) M_k ⊗ I) S_k.
        """

        if derivative_order < 0:
            raise ValueError(
                "derivative_order must be non-negative."
            )

        if derivative_order > self.degree:
            return np.zeros((
                self.n_dof,
                self.n_control_points * self.n_dof,
            ))

        k = self.segment(t)

        s, delta = self.normalized_time(
            t,
            k,
        )

        Gamma = gamma(
            s,
            self.degree,
            derivative_order,
        )

        M = self.basis_matrix(k)
        S = self.selection_matrix(k)

        local_basis = Gamma @ M

        A = (
            np.kron(
                local_basis,
                np.eye(self.n_dof),
            )
            @ S
        )

        return A / delta ** derivative_order

    # ========================================================
    # Trajectory evaluation
    # ========================================================

    def evaluate(
        self,
        control_points,
        t,
        derivative_order=0,
    ):
        """
        Evaluate q^(r)(t).
        """

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        expected_size = (
            self.n_control_points
            * self.n_dof
        )

        if c.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} control-point "
                f"values, got {c.size}."
            )

        A = self.evaluation_matrix(
            t,
            derivative_order,
        )

        return A @ c

    def position(self, control_points, t):
        return self.evaluate(
            control_points,
            t,
            derivative_order=0,
        )

    def velocity(self, control_points, t):
        return self.evaluate(
            control_points,
            t,
            derivative_order=1,
        )

    def acceleration(self, control_points, t):
        return self.evaluate(
            control_points,
            t,
            derivative_order=2,
        )

    # ========================================================
    # Derivative control-point matrices D_r
    # ========================================================

    def derivative_control_matrix(self, order=1):
        """
        Return D_r such that the derivative control points satisfy

            c^(r) = D_r c.

        The matrix is computed once for each derivative order.
        """

        if order < 0:
            raise ValueError(
                "Derivative order must be non-negative."
            )

        if order > self.degree:
            raise ValueError(
                "Derivative order cannot exceed spline degree."
            )

        if order not in self._derivative_matrix_cache:
            self._derivative_matrix_cache[order] = (
                self._compute_derivative_control_matrix(
                    order
                )
            )

        return self._derivative_matrix_cache[order]

    def _compute_derivative_control_matrix(
        self,
        order,
    ):
        """
        Construct D_r recursively.
        """

        if order == 0:
            return np.eye(
                self.n_control_points
                * self.n_dof
            )

        current_knots = self.knots.copy()
        current_degree = self.degree
        current_n_control_points = (
            self.n_control_points
        )

        D_total = np.eye(
            self.n_control_points
        )

        for _ in range(order):

            D = np.zeros((
                current_n_control_points - 1,
                current_n_control_points,
            ))

            for i in range(
                current_n_control_points - 1
            ):

                denominator = (
                    current_knots[
                        i + current_degree + 1
                    ]
                    - current_knots[i + 1]
                )

                if denominator != 0:

                    coefficient = (
                        current_degree
                        / denominator
                    )

                    D[i, i] = -coefficient
                    D[i, i + 1] = coefficient

            D_total = D @ D_total

            # Derivative spline:
            # remove first and last knot,
            # decrease degree by one.
            current_knots = current_knots[1:-1]
            current_degree -= 1
            current_n_control_points -= 1

        return np.kron(
            D_total,
            np.eye(self.n_dof),
        )

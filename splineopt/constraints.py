
"""
Build linear joint-space constraints for a B-spline trajectory.

The constraints are expressed in the form:

    l <= A @ c <= u

where c is the flattened vector of B-spline control points.

The constraint matrix combines:

    1. Boundary conditions:
       - Initial joint position q(t0) = q0.
       - Initial joint velocity q_dot(t0) = v0.
       - Optional initial acceleration q_ddot(t0) = a0.
       - Optional final velocity q_dot(tf) = vf.

    2. Joint position bounds:
       q_min <= c_i <= q_max.

    3. Joint velocity bounds:
       v_min <= D1 @ c <= v_max.

    4. Joint acceleration bounds:
       a_min <= D2 @ c <= a_max.

Position, velocity and acceleration bounds are imposed on
their respective B-spline control points. By the convex hull
property, these constraints guarantee continuous-time bounds
over the spline domain, although they may be conservative.

Args:
    spline: BSpline trajectory representation.
    q0: Initial joint position, shape (n_dof,).
    v0: Initial joint velocity, shape (n_dof,).
    q_min, q_max: Joint position limits, shape (n_dof,).
    v_min, v_max: Joint velocity limits, shape (n_dof,).
    a_min, a_max: Joint acceleration limits, shape (n_dof,).
    a0: Optional initial acceleration, shape (n_dof,).
    vf: Optional final velocity, shape (n_dof,).

Returns:
    A: Sparse CSC constraint matrix of shape
       (n_constraints, n_control_points * n_dof).
    l: Lower constraint bounds, shape (n_constraints,).
    u: Upper constraint bounds, shape (n_constraints,).

    Equality constraints are represented by identical lower
    and upper bounds.

Raises:
    ValueError: If bound dimensions are inconsistent or
        minimum bounds exceed maximum bounds.
"""

import numpy as np
import scipy.sparse as sp


def build_joint_constraints(
    spline,
    q0,
    v0,
    q_min,
    q_max,
    v_min,
    v_max,
    a_min,
    a_max,
    a0=None,
    vf=None,
):
    """Build linear B-spline constraints

        l <= A c <= u.

    The constraints include:
        - initial position,
        - initial velocity,
        - optional initial acceleration,
        - optional final velocity,
        - joint position limits,
        - joint velocity limits,
        - joint acceleration limits.
    """
    if spline.degree < 2:
        raise ValueError(
            "Joint acceleration constraints require "
            "a spline degree of at least 2."
        )

    n_dof = spline.n_dof
    n_control_points = spline.n_control_points
    n_variables = n_control_points * n_dof

    def dof_vector(value, name):
        value = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if value.size != n_dof:
            raise ValueError(
                f"{name} must contain {n_dof} values, "
                f"got {value.size}."
            )

        return value

    # ---------------------------------------------------------
    # Boundary conditions
    # ---------------------------------------------------------

    t0 = spline.t_start

    equality_matrices = [
        spline.evaluation_matrix(
            t0,
            derivative_order=0,
        ),
        spline.evaluation_matrix(
            t0,
            derivative_order=1,
        ),
    ]

    equality_values = [
        dof_vector(q0, "q0"),
        dof_vector(v0, "v0"),
    ]

    if a0 is not None:
        equality_matrices.append(
            spline.evaluation_matrix(
                t0,
                derivative_order=2,
            )
        )

        equality_values.append(
            dof_vector(a0, "a0")
        )

    if vf is not None:
        tf = spline.t_end

        equality_matrices.append(
            spline.evaluation_matrix(
                tf,
                derivative_order=1,
            )
        )

        equality_values.append(
            dof_vector(vf, "vf")
        )

    A_eq = sp.csc_matrix(
        np.vstack(equality_matrices)
    )

    b_eq = np.concatenate(
        equality_values
    )

    # ---------------------------------------------------------
    # Position limits
    # ---------------------------------------------------------

    q_min = dof_vector(q_min, "q_min")
    q_max = dof_vector(q_max, "q_max")

    if np.any(q_min > q_max):
        raise ValueError(
            "q_min must be <= q_max."
        )

    I = sp.eye(
        n_variables,
        format="csc",
    )

    l_position = np.tile(
        q_min,
        n_control_points,
    )

    u_position = np.tile(
        q_max,
        n_control_points,
    )

    # ---------------------------------------------------------
    # Velocity limits
    # ---------------------------------------------------------

    v_min = dof_vector(v_min, "v_min")
    v_max = dof_vector(v_max, "v_max")

    if np.any(v_min > v_max):
        raise ValueError(
            "v_min must be <= v_max."
        )

    D1 = sp.csc_matrix(
        spline.derivative_control_matrix(
            order=1
        )
    )

    l_velocity = np.tile(
        v_min,
        n_control_points - 1,
    )

    u_velocity = np.tile(
        v_max,
        n_control_points - 1,
    )

    # ---------------------------------------------------------
    # Acceleration limits
    # ---------------------------------------------------------

    a_min = dof_vector(a_min, "a_min")
    a_max = dof_vector(a_max, "a_max")

    if np.any(a_min > a_max):
        raise ValueError(
            "a_min must be <= a_max."
        )

    D2 = sp.csc_matrix(
        spline.derivative_control_matrix(
            order=2
        )
    )

    l_acceleration = np.tile(
        a_min,
        n_control_points - 2,
    )

    u_acceleration = np.tile(
        a_max,
        n_control_points - 2,
    )

    # ---------------------------------------------------------
    # Complete constraint matrix
    # ---------------------------------------------------------

    A = sp.vstack(
        [
            A_eq,
            I,
            D1,
            D2,
        ],
        format="csc",
    )

    l = np.concatenate(
        [
            b_eq,
            l_position,
            l_velocity,
            l_acceleration,
        ]
    )

    u = np.concatenate(
        [
            b_eq,
            u_position,
            u_velocity,
            u_acceleration,
        ]
    )

    return A, l, u

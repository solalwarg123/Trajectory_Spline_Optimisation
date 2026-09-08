import numpy as np
import scipy.sparse as sp
import osqp


class JointTrajectoryOptimizer:
    """
    Joint-space B-spline trajectory optimizer.

    Solve the quadratic program

        minimize
            ||A_track c - b_track||^2
            + lambda_smooth * c.T @ R2 @ c

        subject to
            q(t0)    = q0
            qdot(t0) = v0

            q_min <= c    <= q_max
            v_min <= D1 c <= v_max
            a_min <= D2 c <= a_max

    Optionally:

            qddot(t0) = a0

    The QP is written in the OSQP form

        minimize
            1/2 c.T @ P @ c + g.T @ c

        subject to
            l <= A @ c <= u
    """

    def __init__(
        self,
        spline,
        lambda_smooth=1.0,
    ):
        self.spline = spline
        self.lambda_smooth = float(lambda_smooth)

        if self.lambda_smooth < 0:
            raise ValueError(
                "lambda_smooth must be non-negative."
            )

    # ========================================================
    # QP construction
    # ========================================================

    def build_qp(
        self,
        tracking_times,
        q_ref,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        a0=None,
    ):
        """
        Build the QP matrices

            P, g, A, l, u

        for OSQP.

        Parameters
        ----------
        tracking_times :
            Times at which the optimized trajectory is compared
            with the nominal trajectory.

        q_ref :
            Reference joint positions at tracking_times.
            Shape:

                (n_tracking_times, n_dof)

        q0 :
            Initial joint position.

        v0 :
            Initial joint velocity.

        q_min, q_max :
            Joint position limits.

        v_min, v_max :
            Joint velocity limits.

        a_min, a_max :
            Joint acceleration limits.

        a0 :
            Optional initial joint acceleration.
        """

        spline = self.spline

        n_dof = spline.n_dof
        n_control_points = spline.n_control_points

        n_variables = (
            n_control_points
            * n_dof
        )

        # ====================================================
        # Inputs
        # ====================================================

        tracking_times = np.asarray(
            tracking_times,
            dtype=float,
        ).reshape(-1)

        q_ref = np.asarray(
            q_ref,
            dtype=float,
        )

        n_tracking_times = len(
            tracking_times
        )

        if q_ref.shape != (
            n_tracking_times,
            n_dof,
        ):
            raise ValueError(
                "q_ref must have shape "
                f"({n_tracking_times}, {n_dof}), "
                f"got {q_ref.shape}."
            )

        if n_tracking_times == 0:
            raise ValueError(
                "tracking_times cannot be empty."
            )

        if np.any(
            tracking_times < spline.t_start
        ) or np.any(
            tracking_times > spline.t_end
        ):
            raise ValueError(
                "All tracking_times must lie inside "
                "the spline domain."
            )

        # ====================================================
        # Tracking matrix
        #
        # J_track =
        #
        #     ||A_track c - b_track||^2
        #
        # ====================================================

        A_track = np.vstack([
            spline.evaluation_matrix(
                t,
                derivative_order=0,
            )
            for t in tracking_times
        ])

        b_track = q_ref.reshape(-1)

        # ====================================================
        # Smoothness matrix
        #
        # J_smooth = c.T R2 c
        #
        # ====================================================

        R2 = spline.smoothness_matrix(
            derivative_order=2
        )

        # ====================================================
        # Quadratic objective
        #
        # J =
        #   ||A_track c - b_track||^2
        #   + lambda c.T R2 c
        #
        # OSQP expects:
        #
        #   1/2 c.T P c + g.T c
        #
        # Therefore:
        #
        #   P =
        #       2 (
        #           A_track.T A_track
        #           + lambda R2
        #         )
        #
        #   g =
        #       -2 A_track.T b_track
        #
        # ====================================================

        P_dense = 2.0 * (
            A_track.T @ A_track
            + self.lambda_smooth * R2
        )

        # Avoid small numerical asymmetries
        P_dense = 0.5 * (
            P_dense
            + P_dense.T
        )

        g = (
            -2.0
            * A_track.T
            @ b_track
        )

        # OSQP only needs the upper triangular part of P
        P = sp.triu(
            sp.csc_matrix(P_dense),
            format="csc",
        )

        # ====================================================
        # Initial equality constraints
        #
        # A0 c = q0
        # A1 c = v0
        #
        # optionally:
        #
        # A2 c = a0
        #
        # ====================================================

        t0 = spline.t_start

        A0 = spline.evaluation_matrix(
            t0,
            derivative_order=0,
        )

        A1 = spline.evaluation_matrix(
            t0,
            derivative_order=1,
        )

        q0 = self._dof_vector(
            q0,
            "q0",
        )

        v0 = self._dof_vector(
            v0,
            "v0",
        )

        equality_matrices = [
            A0,
            A1,
        ]

        equality_values = [
            q0,
            v0,
        ]

        if a0 is not None:

            A2 = spline.evaluation_matrix(
                t0,
                derivative_order=2,
            )

            a0 = self._dof_vector(
                a0,
                "a0",
            )

            equality_matrices.append(
                A2
            )

            equality_values.append(
                a0
            )

        A_eq = sp.csc_matrix(
            np.vstack(
                equality_matrices
            )
        )

        b_eq = np.concatenate(
            equality_values
        )

        # ====================================================
        # Joint position constraints
        #
        # q_min <= c <= q_max
        #
        # Because the B-spline lies in the convex hull
        # of its control points.
        #
        # ====================================================

        q_min = self._dof_vector(
            q_min,
            "q_min",
        )

        q_max = self._dof_vector(
            q_max,
            "q_max",
        )

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

        # ====================================================
        # Joint velocity constraints
        #
        # v_min <= D1 c <= v_max
        #
        # D1 c contains the control points of qdot(t).
        #
        # ====================================================

        v_min = self._dof_vector(
            v_min,
            "v_min",
        )

        v_max = self._dof_vector(
            v_max,
            "v_max",
        )

        if np.any(v_min > v_max):
            raise ValueError(
                "v_min must be <= v_max."
            )

        D1 = sp.csc_matrix(
            spline.derivative_control_matrix(
                order=1
            )
        )

        n_velocity_control_points = (
            n_control_points - 1
        )

        l_velocity = np.tile(
            v_min,
            n_velocity_control_points,
        )

        u_velocity = np.tile(
            v_max,
            n_velocity_control_points,
        )

        # ====================================================
        # Joint acceleration constraints
        #
        # a_min <= D2 c <= a_max
        #
        # D2 c contains the control points of qddot(t).
        #
        # ====================================================

        a_min = self._dof_vector(
            a_min,
            "a_min",
        )

        a_max = self._dof_vector(
            a_max,
            "a_max",
        )

        if np.any(a_min > a_max):
            raise ValueError(
                "a_min must be <= a_max."
            )

        D2 = sp.csc_matrix(
            spline.derivative_control_matrix(
                order=2
            )
        )

        n_acceleration_control_points = (
            n_control_points - 2
        )

        l_acceleration = np.tile(
            a_min,
            n_acceleration_control_points,
        )

        u_acceleration = np.tile(
            a_max,
            n_acceleration_control_points,
        )

        # ====================================================
        # Complete constraint matrix
        #
        #           [ A0 ]
        #           [ A1 ]
        #     A  =  [ I  ]
        #           [ D1 ]
        #           [ D2 ]
        #
        # Equality constraints are represented by
        #
        #     l_i = u_i.
        #
        # ====================================================

        A = sp.vstack(
            [
                A_eq,
                I,
                D1,
                D2,
            ],
            format="csc",
        )

        l = np.concatenate([
            b_eq,
            l_position,
            l_velocity,
            l_acceleration,
        ])

        u = np.concatenate([
            b_eq,
            u_position,
            u_velocity,
            u_acceleration,
        ])

        return P, g, A, l, u

    # ========================================================
    # Solve
    # ========================================================

    def solve(
        self,
        tracking_times,
        q_ref,
        q0,
        v0,
        q_min,
        q_max,
        v_min,
        v_max,
        a_min,
        a_max,
        a0=None,
        verbose=False,
    ):
        """
        Build and solve the joint-space trajectory QP.

        Returns
        -------
        control_points :
            Optimal control points with shape

                (n_control_points, n_dof)

        result :
            Complete OSQP result object.
        """

        P, g, A, l, u = self.build_qp(
            tracking_times=tracking_times,
            q_ref=q_ref,
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
        )

        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=g,
            A=A,
            l=l,
            u=u,
            verbose=verbose,
            polishing=True,
        )

        result = solver.solve()

        if not result.info.status.lower().startswith(
            "solved"
        ):
            raise RuntimeError(
                "OSQP failed with status: "
                f"{result.info.status}"
            )

        control_points = result.x.reshape(
            self.spline.n_control_points,
            self.spline.n_dof,
        )

        return control_points, result

    # ========================================================
    # Utilities
    # ========================================================

    def _dof_vector(
        self,
        value,
        name,
    ):
        """
        Convert an input to a vector of shape (n_dof,).
        """

        value = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if value.size != self.spline.n_dof:
            raise ValueError(
                f"{name} must contain "
                f"{self.spline.n_dof} values, "
                f"got {value.size}."
            )

        return value
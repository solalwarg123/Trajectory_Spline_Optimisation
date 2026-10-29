"""Joint-space B-spline trajectory optimization using quadratic programming."""

import numpy as np
import scipy.sparse as sp
import osqp

from splineopt.constraints import build_joint_constraints


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

        if not np.isfinite(self.lambda_smooth) or self.lambda_smooth < 0.0:
            raise ValueError(
                "lambda_smooth must be finite and non-negative."
            )

    # ========================================================
    # QP construction
    # ========================================================

    def build_constraints(
        self,
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
        """
        Build linear joint-space trajectory constraints.

        Constraints include initial position and velocity, joint position,
        velocity and acceleration limits, and optional initial acceleration
        and terminal velocity conditions.

        Returns:
            A, l, u: Sparse linear constraint system satisfying
                l <= A @ c <= u.
        """

        return build_joint_constraints(
            spline=self.spline,
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
            vf=vf,
        )

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
        vf=None,
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
        vf :
            Optional terminal joint velocity.

        Returns
        -------
        P, g, A, l, u :
            QP matrices in OSQP form:

                minimize 0.5 * c.T @ P @ c + g.T @ c
                subject to l <= A @ c <= u
        """
        spline = self.spline

        n_dof = spline.n_dof

        # Validate tracking data

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

        if not np.all(np.isfinite(tracking_times)):
            raise ValueError("tracking_times must be finite.")

        if not np.all(np.isfinite(q_ref)):
            raise ValueError("q_ref must contain finite values.")

        # Assemble the joint-space tracking matrix and reference vector

        A_track = np.vstack([
            spline.evaluation_matrix(
                t,
                derivative_order=0,
            )
            for t in tracking_times
        ])

        b_track = q_ref.reshape(-1)

        # Acceleration regularization matrix: integral ||q_ddot(t)||² dt

        R2 = spline.smoothness_matrix(
            derivative_order=2
        )

        # Convert tracking and smoothness costs to OSQP quadratic form:
        # P = 2 * (A_track.T @ A_track + lambda_smooth * R2)
        # g = -2 * A_track.T @ b_track

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

        # Build joint limits and boundary constraints
        A, l, u = self.build_constraints(
            q0=q0,
            v0=v0,
            q_min=q_min,
            q_max=q_max,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            a0=a0,
            vf=vf
        )

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
        vf=None,
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
            vf=vf
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
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=20000,
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


"""
Kinova Gen3 — Cartesian MPC on a controlled feasible trajectory.

Reference:
    Smooth quintic joint trajectory with large q/v/a margins.
    Cartesian targets are obtained by forward kinematics.

The reference is analytically constructed, without offline QP.

Disturbances:
    none     : ideal execution
    kick     : one-cycle acceleration perturbation
    gaussian : random acceleration perturbations
    bias     : constant acceleration disturbance during motion

An optional HiGHS linear feasibility test checks that an
admissible MPC trajectory exists after each execution step.

Run from project root:

python -m examples.kinova_mpc_safe_tracking --mode none --show-plots

python -m examples.kinova_mpc_safe_tracking --mode kick --show-plots

python -m examples.kinova_mpc_safe_tracking --mode gaussian --show-plots

python -m examples.kinova_mpc_safe_tracking --mode bias --show-plots
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.optimize import linprog

from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.mpc import CartesianMPC
from splineopt.robotics import Robot

from examples.kinova_mpc import (
    make_spline,
    pose_from_theta,
    pose_errors,
    enforce_initial_state_on_guess,
    max_linear_violation,
)

from examples.kinova_trajectory_projection import (
    URDF_PATH,
    Q_MIN, Q_MAX,
    V_MIN, V_MAX,
    A_MIN, A_MAX,
)


# ============================================================
# Configuration
# ============================================================

DURATION = 10.0
MOVEMENT_DURATION = 6.0

# Both configurations are comfortably inside joint limits.
# Small displacement => large velocity/acceleration margins.

# Initial configuration, safely inside joint limits
START = np.array([
    0.00, 0.45, 0.15, -1.35, -0.15, 0.75, 0.10,
])

# Fundamental harmonic amplitudes
A_REF = np.array([
    0.65, 0.60, -0.55, 0.70, 0.80, -0.45, 0.75,
])

# Second harmonic amplitudes
B_REF = np.array([
    -0.25, 0.25, 0.30, -0.22, -0.35, 0.28, -0.30,
])

FEASIBILITY_TOL = 1e-4

COLOR_REF = "C1"
COLOR_MPC = "C3"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# Analytically feasible reference
# ============================================================

def reference_state(t):
    """
    Non-trivial, analytically bounded joint trajectory.

    Combines two harmonics with different amplitudes
    across the seven joints.

    Zero velocity and acceleration at both endpoints.
    Cartesian reference is obtained through FK.
    """

    t = float(np.clip(t, 0.0, DURATION))
    omega = 2.0 * np.pi / DURATION

    w1 = omega * t
    w2 = 2.0 * omega * t

    q = (
        START
        + 0.5 * A_REF * (1.0 - np.cos(w1))
        + 0.5 * B_REF * (1.0 - np.cos(w2))
    )

    v = (
        0.5 * omega * A_REF * np.sin(w1)
        + omega * B_REF * np.sin(w2)
    )

    a = (
        0.5 * omega**2 * A_REF * np.cos(w1)
        + 2.0 * omega**2 * B_REF * np.cos(w2)
    )

    return q, v, a


def reference_pose(robot, t):
    q, _, _ = reference_state(t)
    return pose_from_theta(robot, q)


def bound_violation(values, lower, upper):
    values = np.asarray(values)

    return float(max(
        0.0,
        np.max(lower - values),
        np.max(values - upper),
    ))


def validate_reference():
    """
    Analytically certify q/v/a bounds for the
    two-harmonic reference and verify them densely.
    """

    omega = 2.0 * np.pi / DURATION

    # Conservative exact position interval bounds:
    # each (1-cos(...))/2 belongs to [0, 1].
    q_lower = (
        START
        + np.minimum(A_REF, 0.0)
        + np.minimum(B_REF, 0.0)
    )

    q_upper = (
        START
        + np.maximum(A_REF, 0.0)
        + np.maximum(B_REF, 0.0)
    )

    # Conservative analytical derivative bounds
    v_bound = omega * (
        0.5 * np.abs(A_REF)
        + np.abs(B_REF)
    )

    a_bound = omega**2 * (
        0.5 * np.abs(A_REF)
        + 2.0 * np.abs(B_REF)
    )

    position_ok = np.all(
        (q_lower >= Q_MIN)
        & (q_upper <= Q_MAX)
    )

    velocity_ok = np.all(v_bound <= V_MAX)

    acceleration_ok = np.all(a_bound <= A_MAX)

    # Dense verification
    times = np.linspace(0.0, DURATION, 2001)

    states = [reference_state(t) for t in times]

    q = np.asarray([x[0] for x in states])
    v = np.asarray([x[1] for x in states])
    a = np.asarray([x[2] for x in states])

    q_violation = bound_violation(q, Q_MIN, Q_MAX)
    v_violation = bound_violation(v, V_MIN, V_MAX)
    a_violation = bound_violation(a, A_MIN, A_MAX)

    print("\n" + "=" * 70)
    print("REFERENCE FEASIBILITY")
    print("=" * 70)

    print(f"Duration: {DURATION:.2f} s")
    print(
        "Velocity usage (analytical bound): "
        f"{np.max(v_bound / V_MAX):.3f}"
    )
    print(
        "Acceleration usage (analytical bound): "
        f"{np.max(a_bound / A_MAX):.3f}"
    )
    print(
        f"Violations: q={q_violation:.3e}, "
        f"v={v_violation:.3e}, "
        f"a={a_violation:.3e}"
    )

    if not (
        position_ok
        and velocity_ok
        and acceleration_ok
    ):
        raise RuntimeError(
            "Analytical reference is not feasible."
        )

    if max(q_violation, v_violation, a_violation) > 1e-10:
        raise RuntimeError(
            "Reference feasibility verification failed."
        )

    print("Reference is analytically feasible.")
    print("=" * 70)

# ============================================================
# Execution perturbations
# ============================================================


def execute_step(
    mode,
    commanded_q,
    commanded_v,
    t_new,
    dt,
    args,
    rng,
    kick_done,
):
    """
    Execute one MPC interval.

    A small additional acceleration a_dist is applied over dt.

    Relative to the ideal commanded endpoint:

        delta_q = 0.5 * a_dist * dt²
        delta_v = a_dist * dt

    This is a simplified acceleration-disturbance model,
    not a full robot dynamics simulator.
    """

    disturbance_acceleration = np.zeros(7)

    if mode == "kick":

        if (
            not kick_done
            and t_new >= args.kick_time
        ):
            disturbance_acceleration[
                args.disturbance_joint
            ] = args.kick_accel

            kick_done = True

    elif mode == "gaussian":

        disturbance_acceleration = rng.normal(
            0.0,
            args.noise_accel_std,
            size=7,
        )

    elif mode == "bias":

        if (
            args.bias_start
            <= t_new
            <= args.bias_end
        ):
            disturbance_acceleration[
                args.disturbance_joint
            ] = args.bias_accel

    # Keep applied disturbance accelerations bounded.
    # This does not replace verifying total executed dynamics.
    disturbance_acceleration = np.clip(
        disturbance_acceleration,
        -args.max_disturbance_accel,
        args.max_disturbance_accel,
    )

    q_new = (
        commanded_q
        + 0.5 * disturbance_acceleration * dt**2
    )

    v_new = (
        commanded_v
        + disturbance_acceleration * dt
    )

    return (
        q_new,
        v_new,
        disturbance_acceleration,
        kick_done,
    )


# ============================================================
# Linear feasibility certificate
# ============================================================

def check_mpc_feasibility(problem):
    """
    Verify existence of B-spline control points satisfying
    the MPC linear constraints.

    Feasibility LP solved by HiGHS.

    No Cartesian tracking objective is included.

    This is a validation tool only, not part of MPC timing.
    """

    A, lower, upper = problem.linear_constraints()

    A = sp.csr_matrix(A)

    equality = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isclose(
            lower,
            upper,
            atol=1e-12,
            rtol=0,
        )
    )

    upper_mask = np.isfinite(upper) & ~equality
    lower_mask = np.isfinite(lower) & ~equality

    A_ub = sp.vstack([
        A[upper_mask],
        -A[lower_mask],
    ], format="csr")

    b_ub = np.r_[
        upper[upper_mask],
        -lower[lower_mask],
    ]

    result = linprog(
        c=np.zeros(A.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A[equality],
        b_eq=lower[equality],
        bounds=[(None, None)] * A.shape[1],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8,
        },
    )

    return result


# ============================================================
# Plot helpers
# ============================================================

def plot_joint_grid(
    folder,
    filename,
    title,
    times,
    reference,
    executed,
    lower,
    upper,
    unit,
):

    fig, axes = plt.subplots(
        4, 2,
        figsize=(13, 11),
        sharex=True,
    )

    axes = axes.ravel()

    for j in range(7):

        ax = axes[j]

        values = np.r_[
            reference[:, j],
            executed[:, j],
        ]

        finite = np.array([
            lower[j],
            upper[j],
        ])
        finite = finite[np.isfinite(finite)]

        if finite.size:
            values = np.r_[values, finite]

        span = max(np.ptp(values), 0.1)
        lo = np.min(values) - 0.12 * span
        hi = np.max(values) + 0.12 * span

        ax.set_ylim(lo, hi)

        # Green admissible area
        feasible_lo = max(
            lo,
            lower[j] if np.isfinite(lower[j]) else lo,
        )

        feasible_hi = min(
            hi,
            upper[j] if np.isfinite(upper[j]) else hi,
        )

        if feasible_hi > feasible_lo:
            ax.axhspan(
                feasible_lo,
                feasible_hi,
                color=COLOR_GREEN,
                alpha=0.07,
            )

        if np.isfinite(lower[j]):
            ax.axhspan(
                lo,
                lower[j],
                color=COLOR_RED,
                alpha=0.08,
            )
            ax.axhline(
                lower[j],
                color="0.4",
                linestyle=":",
            )

        if np.isfinite(upper[j]):
            ax.axhspan(
                upper[j],
                hi,
                color=COLOR_RED,
                alpha=0.08,
            )
            ax.axhline(
                upper[j],
                color="0.4",
                linestyle=":",
            )

        ax.plot(
            times,
            reference[:, j],
            "--",
            color=COLOR_REF,
            linewidth=1.8,
            label="Reference",
        )

        ax.plot(
            times,
            executed[:, j],
            color=COLOR_MPC,
            linewidth=2.2,
            label="Executed MPC",
        )

        ax.set_title(f"Joint {j+1}")
        ax.set_ylabel(unit)

        if j >= 6:
            ax.set_xlabel("Time [s]")

    axes[0].legend()
    axes[-1].axis("off")

    fig.suptitle(
        title + "\nGreen = admissible, red = forbidden"
    )
    fig.tight_layout()
    fig.savefig(
        folder / filename,
        dpi=170,
        bbox_inches="tight",
    )


def save_plots(folder, robot, data, args):

    t = np.asarray(data["times"])
    full_t = np.r_[0.0, t]

    q_hist = np.asarray(data["q"])
    v_hist = np.asarray(data["v"])

    xyz = np.asarray(data["xyz"])
    xyz_ref = np.asarray(data["xyz_ref"])

    ep = np.asarray(data["ep"])
    eo = np.asarray(data["eo"])

    compute = np.asarray(data["compute"])
    iterations = np.asarray(data["iterations"])
    violation = np.asarray(data["violation"])
    disturbance = np.asarray(data["disturbance"])

    q_ref, v_ref, a_ref = [], [], []

    for ti in full_t:
        qi, vi, ai = reference_state(ti)
        q_ref.append(qi)
        v_ref.append(vi)
        a_ref.append(ai)

    q_ref = np.asarray(q_ref)
    v_ref = np.asarray(v_ref)
    a_ref = np.asarray(a_ref)

    # --------------------------------------------------------
    # 1. Cartesian path
    # --------------------------------------------------------

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        *xyz_ref.T,
        "--",
        color=COLOR_REF,
        linewidth=2.2,
        label="Feasible Cartesian reference",
    )

    ax.plot(
        *xyz.T,
        color=COLOR_MPC,
        linewidth=2.5,
        label="Executed MPC",
    )

    ax.scatter(
        *xyz[0],
        s=60,
        color=COLOR_GREEN,
        label="Start",
    )

    ax.scatter(
        *xyz[-1],
        s=65,
        marker="X",
        color=COLOR_MPC,
        label="End",
    )

    extent = np.maximum(
        np.ptp(np.vstack([xyz, xyz_ref]), axis=0),
        1e-3,
    )

    ax.set_box_aspect(extent)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Cartesian trajectory")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        folder / "cartesian_path.png",
        dpi=170,
    )

    # --------------------------------------------------------
    # 2. Cartesian tracking
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True,
    )

    axes[0].plot(
        t,
        1000.0 * ep,
        color=COLOR_MPC,
    )
    axes[0].set_ylabel("Position error [mm]")

    axes[1].plot(
        t,
        np.rad2deg(eo),
        color=COLOR_MPC,
    )
    axes[1].set_ylabel("Orientation error [deg]")
    axes[1].set_xlabel("Time [s]")

    fig.suptitle("Cartesian tracking errors")
    fig.tight_layout()
    fig.savefig(
        folder / "tracking_errors.png",
        dpi=170,
    )

    # --------------------------------------------------------
    # 3. Joint positions and velocities
    # --------------------------------------------------------

    plot_joint_grid(
        folder,
        "joint_positions.png",
        "Joint positions",
        full_t,
        q_ref,
        q_hist,
        Q_MIN,
        Q_MAX,
        "Position [rad]",
    )

    plot_joint_grid(
        folder,
        "joint_velocities.png",
        "Joint velocities",
        full_t,
        v_ref,
        v_hist,
        V_MIN,
        V_MAX,
        "Velocity [rad/s]",
    )

    # Cycle-average acceleration estimate for plotting.
    # Not an exact continuous-time acceleration measurement.
    a_executed = np.zeros_like(v_hist)
    a_executed[1:] = np.diff(
        v_hist,
        axis=0,
    ) / args.dt

    plot_joint_grid(
        folder,
        "joint_accelerations.png",
        "Joint accelerations (cycle-average estimate)",
        full_t,
        a_ref,
        a_executed,
        A_MIN,
        A_MAX,
        "Acceleration [rad/s²]",
    )

    # --------------------------------------------------------
    # 4. Disturbances and tracking
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True,
    )

    axes[0].plot(
        t,
        np.linalg.norm(disturbance, axis=1),
        color="C0",
    )
    axes[0].set_ylabel(
        "Disturbance acceleration [rad/s²]"
    )

    axes[1].plot(
        t,
        1000.0 * ep,
        color=COLOR_MPC,
    )
    axes[1].set_ylabel("Position error [mm]")
    axes[1].set_xlabel("Time [s]")

    fig.suptitle("Perturbation and recovery")
    fig.tight_layout()
    fig.savefig(
        folder / "perturbation_recovery.png",
        dpi=170,
    )

    # --------------------------------------------------------
    # 5. MPC performance
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3, 1,
        figsize=(11, 9),
        sharex=True,
    )

    axes[0].plot(
        t,
        compute,
        ".-",
        color="C0",
    )

    axes[0].axhline(
        1000 * args.dt,
        linestyle="--",
        color=COLOR_RED,
        label="MPC deadline",
    )

    axes[0].axhline(
        np.percentile(compute, 95),
        linestyle=":",
        color=COLOR_GREEN,
        label="P95",
    )

    axes[0].set_ylabel("Compute [ms]")
    axes[0].legend()

    axes[1].step(
        t,
        iterations,
        where="mid",
        color="C0",
    )
    axes[1].set_ylabel("GN iterations")

    axes[2].semilogy(
        t,
        np.maximum(violation, 1e-16),
        color=COLOR_MPC,
    )
    axes[2].axhline(
        FEASIBILITY_TOL,
        color=COLOR_RED,
        linestyle="--",
    )
    axes[2].set_ylabel("Linear violation")
    axes[2].set_xlabel("Time [s]")

    fig.suptitle("MPC computation and feasibility")
    fig.tight_layout()
    fig.savefig(
        folder / "mpc_performance.png",
        dpi=170,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["none", "kick", "gaussian", "bias"],
        default="none",
    )

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--control-points", type=int, default=12)
    parser.add_argument("--tracking-points", type=int, default=15)
    parser.add_argument("--orientation-scale", type=float, default=0.1)

    # Disturbance parameters
    parser.add_argument("--disturbance-joint", type=int, default=0)
    parser.add_argument("--kick-time", type=float, default=3.0)
    parser.add_argument("--kick-accel", type=float, default=0.2)
    parser.add_argument("--noise-accel-std", type=float, default=0.05)
    parser.add_argument("--bias-accel", type=float, default=0.05)
    parser.add_argument("--bias-start", type=float, default=2.0)
    parser.add_argument("--bias-end", type=float, default=4.0)
    parser.add_argument("--max-disturbance-accel", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    # Validation / visualization
    parser.add_argument("--no-lp-check", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kinova_mpc_safe_tracking"),
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error("Require 0 < dt < horizon")

    if args.disturbance_joint not in range(7):
        parser.error("Joint index must be 0..6")

    if args.max_disturbance_accel <= 0:
        parser.error("max-disturbance-accel must be positive")

    if args.noise_accel_std < 0:
        parser.error("noise-accel-std must be non-negative")

    if args.steps is None:
        args.steps = int(round(DURATION / args.dt))

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(args.seed)

    # --------------------------------------------------------
    # Validate reference
    # --------------------------------------------------------

    validate_reference()

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    # --------------------------------------------------------
    # B-spline MPC
    # --------------------------------------------------------

    spline = make_spline(
        args.horizon,
        robot.nv,
        args.control_points,
    )

    grid = np.linspace(
        0.0,
        args.horizon,
        args.tracking_points,
    )

    q, v, _ = reference_state(0.0)

    c0 = np.tile(
        q,
        (spline.n_control_points, 1),
    )

    initial_targets = [
        reference_pose(robot, tau)
        for tau in grid
    ]

    problem = CartesianTrajectoryProblem(
        robot=robot,
        spline=spline,
        tracking_times=grid,
        target_poses=initial_targets,
        q0=q,
        v0=v,
        q_min=Q_MIN,
        q_max=Q_MAX,
        v_min=V_MIN,
        v_max=V_MAX,
        a_min=A_MIN,
        a_max=A_MAX,
        orientation_scale=args.orientation_scale,
        lambda_smooth=1e-5,
        lambda_terminal=1.0,
    )

    solver = GaussNewtonSolver(
        damping_mode="adaptive",
        damping=1e-3,
        max_iterations=args.max_iterations,
    )

    mpc = CartesianMPC(
        problem,
        solver,
        c0,
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    data = {
        "times": [],
        "compute": [],
        "iterations": [],
        "statuses": [],
        "violation": [],
        "ep": [],
        "eo": [],
        "disturbance": [],
        "q": [q.copy()],
        "v": [v.copy()],
        "xyz": [
            pose_from_theta(
                robot, q
            ).translation.copy()
        ],
        "xyz_ref": [
            reference_pose(
                robot, 0.0
            ).translation.copy()
        ],
    }

    kick_done = False

    print("\n" + "=" * 70)
    print("KINOVA — SAFE CARTESIAN MPC TRACKING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"dt: {args.dt:.3f} s")
    print(f"Horizon: {args.horizon:.2f} s")
    print(f"GN iterations: <= {args.max_iterations}")
    print(f"Steps: {args.steps}")
    print(f"LP feasibility check: {not args.no_lp_check}")

    # --------------------------------------------------------
    # MPC loop
    # --------------------------------------------------------

    for k in range(args.steps):

        t = k * args.dt
        t_new = (k + 1) * args.dt

        targets = [
            reference_pose(robot, t + tau)
            for tau in grid
        ]

        # Timing excludes reference construction and
        # the independent LP feasibility test.

        tic = time.perf_counter()

        mpc.previous_control_points = (
            enforce_initial_state_on_guess(
                spline,
                mpc.previous_control_points,
                q,
                v,
            )
        )

        try:
            result = mpc.step(
                q0=q,
                v0=v,
                target_poses=targets,
            )
        except RuntimeError as exc:
            print(
                f"\nGN failed at cycle {k+1}, "
                f"t={t:.3f} s: {exc}"
            )

            lp = check_mpc_feasibility(problem)
            print("Independent HiGHS check:", lp.message)

            raise

        elapsed_ms = 1000.0 * (
            time.perf_counter() - tic
        )

        c = np.asarray(
            result.control_points,
            dtype=float,
        ).reshape(
            spline.n_control_points,
            robot.nv,
        )

        if not np.all(np.isfinite(c)):
            raise RuntimeError(
                f"Cycle {k+1}: non-finite control points."
            )

        violation = max_linear_violation(problem, c)

        if violation > FEASIBILITY_TOL:
            raise RuntimeError(
                f"Cycle {k+1}: planned trajectory "
                f"infeasible ({violation:.3e})."
            )

        # A feasible result may still have status
        # max_iterations or damping_limit.
        # Keep it but report the lack of convergence.

        if not result.success:
            if result.status not in (
                "max_iterations",
                "damping_limit",
            ):
                raise RuntimeError(
                    f"Cycle {k+1}: GN failed "
                    f"({result.status})."
                )

        mpc.previous_control_points = c.copy()

        # ----------------------------------------------------
        # Execute first interval
        # ----------------------------------------------------

        commanded_q = spline.position(
            c,
            args.dt,
        )

        commanded_v = spline.velocity(
            c,
            args.dt,
        )

        (
            q_new,
            v_new,
            disturbance,
            kick_done,
        ) = execute_step(
            args.mode,
            commanded_q,
            commanded_v,
            t_new,
            args.dt,
            args,
            rng,
            kick_done,
        )

        # ----------------------------------------------------
        # Check physical state limits
        # ----------------------------------------------------

        q_violation = bound_violation(
            q_new,
            Q_MIN,
            Q_MAX,
        )

        v_violation = bound_violation(
            v_new,
            V_MIN,
            V_MAX,
        )

        if max(q_violation, v_violation) > 1e-7:
            raise RuntimeError(
                f"Cycle {k+1}: executed state violates "
                f"q/v limits. q={q_violation:.3e}, "
                f"v={v_violation:.3e}"
            )

        # ----------------------------------------------------
        # Independent feasibility test for next state
        # ----------------------------------------------------

        if not args.no_lp_check:

            problem.update(
                q0=q_new,
                v0=v_new,
                target_poses=[
                    reference_pose(
                        robot, t_new + tau
                    )
                    for tau in grid
                ],
            )

            lp = check_mpc_feasibility(problem)

            if not lp.success:
                raise RuntimeError(
                    f"Cycle {k+1}: next MPC state is "
                    f"not certified feasible by HiGHS: "
                    f"{lp.message}"
                )

        # ----------------------------------------------------
        # Actual state / Cartesian error
        # ----------------------------------------------------

        q = q_new
        v = v_new

        actual_pose = pose_from_theta(robot, q)
        desired_pose = reference_pose(robot, t_new)

        ep, eo = pose_errors(
            actual_pose,
            desired_pose,
        )

        data["times"].append(t_new)
        data["compute"].append(elapsed_ms)
        data["iterations"].append(result.iterations)
        data["statuses"].append(result.status)
        data["violation"].append(violation)
        data["ep"].append(ep)
        data["eo"].append(eo)
        data["disturbance"].append(
            disturbance.copy()
        )
        data["q"].append(q.copy())
        data["v"].append(v.copy())

        data["xyz"].append(
            actual_pose.translation.copy()
        )

        data["xyz_ref"].append(
            desired_pose.translation.copy()
        )

        print(
            f"[{k+1:03d}/{args.steps:03d}] "
            f"t={t_new:5.2f}s "
            f"{elapsed_ms:7.1f}ms "
            f"GN={result.iterations:2d} "
            f"pos={1000*ep:7.3f}mm "
            f"ori={np.rad2deg(eo):6.3f}deg "
            f"status={result.status}"
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    compute = np.asarray(data["compute"])
    ep = np.asarray(data["ep"])
    eo = np.asarray(data["eo"])
    violations = np.asarray(data["violation"])

    print("\n" + "=" * 70)
    print("MPC STATISTICS")
    print("=" * 70)

    print(f"Mode: {args.mode}")

    print(
        f"Compute [ms]: mean={compute.mean():.2f}, "
        f"median={np.median(compute):.2f}, "
        f"p95={np.percentile(compute, 95):.2f}, "
        f"max={compute.max():.2f}"
    )

    print(
        f"Deadline misses: "
        f"{np.count_nonzero(compute > 1000*args.dt)}"
        f"/{len(compute)}"
    )

    print(
        f"Position RMS: "
        f"{1000*np.sqrt(np.mean(ep**2)):.3f} mm"
    )

    print(
        f"Position max: {1000*ep.max():.3f} mm"
    )

    print(
        f"Orientation RMS: "
        f"{np.rad2deg(np.sqrt(np.mean(eo**2))):.4f} deg"
    )

    print(
        f"Max planned linear violation: "
        f"{violations.max():.3e}"
    )

    print(
        "GN statuses:",
        {
            status: data["statuses"].count(status)
            for status in sorted(set(data["statuses"]))
        },
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = args.output / "cycles.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "compute_ms",
            "iterations",
            "status",
            "position_error_m",
            "orientation_error_rad",
            "linear_violation",
            "disturbance_accel_norm",
        ])

        for i in range(len(data["times"])):
            writer.writerow([
                data["times"][i],
                data["compute"][i],
                data["iterations"][i],
                data["statuses"][i],
                data["ep"][i],
                data["eo"][i],
                data["violation"][i],
                np.linalg.norm(
                    data["disturbance"][i]
                ),
            ])

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    save_plots(
        args.output,
        robot,
        data,
        args,
    )

    print(
        "\nSaved CSV and figures to:",
        args.output.resolve(),
    )

    # --------------------------------------------------------
    # Meshcat
    # --------------------------------------------------------

    if not args.no_animation:

        print("\nOpening Meshcat...")

        robot.init_viewer(
            open_browser=True,
        )

        robot.display_target(
            reference_pose(robot, DURATION),
            name="final_reference",
        )

        configurations = (
            robot.joint_trajectory_to_configurations(
                data["q"]
            )
        )

        robot.animate(
            configurations,
            dt=args.dt,
        )

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

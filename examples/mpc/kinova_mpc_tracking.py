
"""Kinova Gen3 — Cartesian B-spline MPC tracking.

Track a feasible two-harmonic Cartesian reference using:
    --solver gn
    --solver ipopt
    --solver both

Both solvers use identical constraints, reference trajectories
and pre-generated disturbances in independent closed-loop runs.

Examples:
    python -m examples.mpc.kinova_mpc_tracking
    python -m examples.mpc.kinova_mpc_tracking --solver both --show-plots
    python -m examples.mpc.kinova_mpc_tracking --solver gn --mode kick
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.ipopt_solver import IpoptSolver
from splineopt.mpc import CartesianMPC
from splineopt.robotics import Robot

from examples.utils.kinova_helpers import (
    URDF_PATH,
    Q_MIN, Q_MAX,
    V_MIN, V_MAX,
    A_MIN, A_MAX,
)
from examples.utils.mpc_helpers import (
    DURATION,
    make_spline,
    pose_from_theta,
    pose_errors,
    enforce_initial_state_on_guess,
    max_linear_violation,
    reference_state,
    reference_pose,
    validate_reference,
    bound_violation,
    check_mpc_feasibility,
)

from examples.utils.kinova_visualization import (
    animate_kinova_trajectories,
)

FEASIBILITY_TOL = 1e-4

COLORS = {
    "Reference": "C1",
    "GN": "C0",
    "IPOPT": "C3",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# Disturbances and solver configuration
# ============================================================

def generate_disturbances(args):
    """Generate the same disturbance sequence for every solver."""
    rng = np.random.default_rng(args.seed)
    disturbances = np.zeros((args.steps, 7))

    if args.mode == "gaussian":
        disturbances = rng.normal(
            0.0,
            args.noise_accel_std,
            size=(args.steps, 7),
        )

    elif args.mode == "kick":
        index = max(
            0,
            int(np.ceil(args.kick_time / args.dt)) - 1,
        )
        if index < args.steps:
            disturbances[index, args.disturbance_joint] = (
                args.kick_accel
            )

    elif args.mode == "bias":
        for k in range(args.steps):
            t = (k + 1) * args.dt
            if args.bias_start <= t <= args.bias_end:
                disturbances[k, args.disturbance_joint] = (
                    args.bias_accel
                )

    return np.clip(
        disturbances,
        -args.max_disturbance_accel,
        args.max_disturbance_accel,
    )


def make_solver(name, args):
    if name == "GN":
        return GaussNewtonSolver(
            damping_mode="adaptive",
            damping=1e-3,
            max_iterations=args.gn_iterations,
        )

    if name == "IPOPT":
        return IpoptSolver(
            tolerance=args.ipopt_tolerance,
            max_iterations=args.ipopt_iterations,
            print_level=0,
        )

    raise ValueError(f"Unknown solver: {name}")


def acceptable_result(name, result):
    """Accept feasible nonconverged iterates at iteration limits."""
    if result.success:
        return True

    if name == "GN":
        return result.status in (
            "max_iterations",
            "damping_limit",
        )

    if name == "IPOPT":
        return result.status == -1

    return False


# ============================================================
# Closed-loop simulation
# ============================================================

def run_mpc(name, args, disturbances):
    """Run an independent MPC experiment with one solver."""
    print("\n" + "=" * 70)
    print(f"KINOVA CARTESIAN MPC — {name}")
    print("=" * 70)

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

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
    c0 = np.tile(q, (spline.n_control_points, 1))

    problem = CartesianTrajectoryProblem(
        robot=robot,
        spline=spline,
        tracking_times=grid,
        target_poses=[
            reference_pose(robot, tau)
            for tau in grid
        ],
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

    mpc = CartesianMPC(
        problem,
        make_solver(name, args),
        c0,
    )

    initial_pose = pose_from_theta(robot, q)

    data = {
        "time": [],
        "compute_ms": [],
        "iterations": [],
        "status": [],
        "pos_error": [],
        "ori_error": [],
        "violation": [],
        "disturbance": [],
        "q": [q.copy()],
        "v": [v.copy()],
        "xyz": [initial_pose.translation.copy()],
        "xyz_ref": [
            reference_pose(robot, 0.0).translation.copy()
        ],
    }

    for k in range(args.steps):
        t = k * args.dt
        t_new = (k + 1) * args.dt

        targets = [
            reference_pose(robot, t + tau)
            for tau in grid
        ]

        # Timing includes initial-guess correction and MPC solve.
        # Reference construction and LP validation are excluded.
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
                f"{name} failed at cycle {k + 1}: {exc}"
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
                f"{name}, cycle {k + 1}: non-finite solution."
            )

        violation = max_linear_violation(problem, c)

        if violation > FEASIBILITY_TOL:
            raise RuntimeError(
                f"{name}, cycle {k + 1}: linear constraint "
                f"violation = {violation:.3e}"
            )

        if not acceptable_result(name, result):
            raise RuntimeError(
                f"{name}, cycle {k + 1}: solver failed "
                f"with status {result.status}. "
                f"{getattr(result, 'status_message', '')}"
            )

        # Feasibility is required independently of solver status.
        mpc.previous_control_points = c.copy()

        commanded_q = spline.position(c, args.dt)
        commanded_v = spline.velocity(c, args.dt)

        a_dist = disturbances[k]

        q_new = (
            commanded_q
            + 0.5 * a_dist * args.dt**2
        )
        v_new = commanded_v + a_dist * args.dt

        q_violation = bound_violation(
            q_new, Q_MIN, Q_MAX,
        )
        v_violation = bound_violation(
            v_new, V_MIN, V_MAX,
        )

        if max(q_violation, v_violation) > 1e-7:
            raise RuntimeError(
                f"{name}, cycle {k + 1}: executed q/v "
                f"limits violated."
            )

        if not args.no_lp_check:
            problem.update(
                q0=q_new,
                v0=v_new,
                target_poses=[
                    reference_pose(robot, t_new + tau)
                    for tau in grid
                ],
            )

            lp = check_mpc_feasibility(problem)

            if not lp.success:
                raise RuntimeError(
                    f"{name}, cycle {k + 1}: next state "
                    f"not feasible according to HiGHS: "
                    f"{lp.message}"
                )

        q, v = q_new, v_new

        actual_pose = pose_from_theta(robot, q)
        desired_pose = reference_pose(robot, t_new)
        ep, eo = pose_errors(actual_pose, desired_pose)

        data["time"].append(t_new)
        data["compute_ms"].append(elapsed_ms)
        data["iterations"].append(result.iterations)
        data["status"].append(result.status)
        data["pos_error"].append(ep)
        data["ori_error"].append(eo)
        data["violation"].append(violation)
        data["disturbance"].append(a_dist.copy())
        data["q"].append(q.copy())
        data["v"].append(v.copy())
        data["xyz"].append(
            actual_pose.translation.copy()
        )
        data["xyz_ref"].append(
            desired_pose.translation.copy()
        )

        print(
            f"[{name:5s} {k + 1:03d}/{args.steps:03d}] "
            f"{elapsed_ms:7.1f} ms  "
            f"iter={result.iterations:3d}  "
            f"pos={1000 * ep:7.3f} mm  "
            f"ori={np.degrees(eo):6.3f} deg  "
            f"status={result.status}"
        )

    return {
        key: np.asarray(values)
        for key, values in data.items()
    }


# ============================================================
# Statistics and CSV
# ============================================================

def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def print_summary(results, args):
    names = list(results)

    print("\n" + "=" * 85)
    print("MPC TRACKING RESULTS")
    print("=" * 85)

    print(
        f"{'Metric':<34}"
        + "".join(f"{name:>20}" for name in names)
    )
    print("-" * (34 + 20 * len(names)))

    metrics = {
        "Mean solve time [ms]": lambda d: np.mean(
            d["compute_ms"]
        ),
        "Median solve time [ms]": lambda d: np.median(
            d["compute_ms"]
        ),
        "P95 solve time [ms]": lambda d: np.percentile(
            d["compute_ms"], 95
        ),
        "Maximum solve time [ms]": lambda d: np.max(
            d["compute_ms"]
        ),
        "Mean iterations": lambda d: np.mean(
            d["iterations"]
        ),
        "Position RMS [mm]": lambda d: 1000 * rms(
            d["pos_error"]
        ),
        "Position max [mm]": lambda d: 1000 * np.max(
            d["pos_error"]
        ),
        "Orientation RMS [deg]": lambda d: np.degrees(
            rms(d["ori_error"])
        ),
        "Orientation max [deg]": lambda d: np.degrees(
            np.max(d["ori_error"])
        ),
        "Max linear violation": lambda d: np.max(
            d["violation"]
        ),
    }

    for label, metric in metrics.items():
        print(
            f"{label:<34}"
            + "".join(
                f"{metric(results[name]):>20.5f}"
                for name in names
            )
        )

    print("-" * (34 + 20 * len(names)))

    for name, data in results.items():
        within_deadline = np.mean(
            data["compute_ms"] <= 1000 * args.dt
        )

        statuses, counts = np.unique(
            data["status"].astype(str),
            return_counts=True,
        )

        print(
            f"{name}: measured deadline hit rate = "
            f"{100 * within_deadline:.1f}%"
        )
        print(
            f"{name}: statuses = "
            f"{dict(zip(statuses, counts))}"
        )

    print(
        "\nTiming covers warm-start correction and MPC solve, "
        "not the complete control loop."
    )


def save_csv(results, args):
    path = args.output / "mpc_tracking.csv"

    with path.open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "solver",
            "cycle",
            "time_s",
            "compute_ms",
            "iterations",
            "status",
            "position_error_mm",
            "orientation_error_deg",
            "linear_violation",
            "disturbance_accel_norm",
        ])

        for name, data in results.items():
            for k in range(len(data["time"])):
                writer.writerow([
                    name,
                    k + 1,
                    data["time"][k],
                    data["compute_ms"][k],
                    data["iterations"][k],
                    data["status"][k],
                    1000 * data["pos_error"][k],
                    np.degrees(data["ori_error"][k]),
                    data["violation"][k],
                    np.linalg.norm(data["disturbance"][k]),
                ])

    print(f"\nSaved CSV: {path}")


# ============================================================
# Plot helpers
# ============================================================

def save_figure(fig, folder, filename):
    fig.tight_layout()
    fig.savefig(
        folder / filename,
        dpi=170,
        bbox_inches="tight",
    )


def plot_joint_grid(
    folder, filename, title, times,
    reference, trajectories, lower, upper, unit,
):
    """Compare q/v/a for all solvers on the same seven joint axes."""
    fig, axes = plt.subplots(
        4, 2,
        figsize=(13, 11),
        sharex=True,
    )
    axes = axes.ravel()

    for j, ax in enumerate(axes[:7]):
        values = [reference[:, j]]

        for actual in trajectories.values():
            values.append(actual[:, j])

        finite_bounds = np.array([
            lower[j], upper[j],
        ])
        finite_bounds = finite_bounds[
            np.isfinite(finite_bounds)
        ]

        if finite_bounds.size:
            values.append(finite_bounds)

        values = np.concatenate(values)

        span = max(np.ptp(values), 0.1)
        lo = np.min(values) - 0.12 * span
        hi = np.max(values) + 0.12 * span

        ax.set_ylim(lo, hi)

        low = lower[j] if np.isfinite(lower[j]) else lo
        high = upper[j] if np.isfinite(upper[j]) else hi

        feasible_lo = max(lo, low)
        feasible_hi = min(hi, high)

        if feasible_hi > feasible_lo:
            ax.axhspan(
                feasible_lo,
                feasible_hi,
                color="#2ca02c",
                alpha=0.07,
            )

        if np.isfinite(lower[j]):
            ax.axhspan(
                lo, lower[j],
                color="#d62728",
                alpha=0.08,
            )
            ax.axhline(
                lower[j],
                color="0.4",
                linestyle=":",
            )

        if np.isfinite(upper[j]):
            ax.axhspan(
                upper[j], hi,
                color="#d62728",
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
            color=COLORS["Reference"],
            linewidth=1.8,
            label="Reference",
        )

        for name, actual in trajectories.items():
            ax.plot(
                times,
                actual[:, j],
                color=COLORS[name],
                linewidth=2.0,
                label=name,
            )

        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel(unit)

        if j >= 6:
            ax.set_xlabel("Time [s]")

    axes[0].legend()
    axes[7].axis("off")

    fig.suptitle(
        title + "\nGreen = admissible, red = forbidden"
    )

    save_figure(fig, folder, filename)


def plot_joint_signals(results, args):
    """Joint positions, velocities and accelerations for all solvers."""
    first = next(iter(results.values()))
    full_t = np.r_[0.0, first["time"]]

    states = [
        reference_state(t)
        for t in full_t
    ]

    q_ref = np.asarray([state[0] for state in states])
    v_ref = np.asarray([state[1] for state in states])
    a_ref = np.asarray([state[2] for state in states])

    positions = {
        name: data["q"]
        for name, data in results.items()
    }

    velocities = {
        name: data["v"]
        for name, data in results.items()
    }

    # Cycle-average acceleration estimates.
    # These are not exact continuous-time spline accelerations.
    accelerations = {}

    for name, data in results.items():
        a = np.zeros_like(data["v"])

        a[1:] = np.diff(
            data["v"],
            axis=0,
        ) / args.dt

        accelerations[name] = a

    quantities = (
        (
            "joint_positions.png",
            "Joint positions",
            q_ref,
            positions,
            Q_MIN,
            Q_MAX,
            "Position [rad]",
        ),
        (
            "joint_velocities.png",
            "Joint velocities",
            v_ref,
            velocities,
            V_MIN,
            V_MAX,
            "Velocity [rad/s]",
        ),
        (
            "joint_accelerations.png",
            "Joint accelerations (cycle-average estimates)",
            a_ref,
            accelerations,
            A_MIN,
            A_MAX,
            "Acceleration [rad/s²]",
        ),
    )

    for (
        filename, title, reference,
        trajectories, lower, upper, unit,
    ) in quantities:
        plot_joint_grid(
            args.output,
            filename,
            title,
            full_t,
            reference,
            trajectories,
            lower,
            upper,
            unit,
        )

    # Disturbance and Cartesian tracking recovery.
    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True,
    )

    # Disturbances were generated once and are shared.
    disturbance = first["disturbance"]

    axes[0].plot(
        first["time"],
        np.linalg.norm(disturbance, axis=1),
        color="C2",
    )

    for name, data in results.items():
        axes[1].plot(
            data["time"],
            1000 * data["pos_error"],
            color=COLORS[name],
            label=name,
        )

    axes[0].set_ylabel(
        "Disturbance acceleration [rad/s²]"
    )
    axes[1].set_ylabel("Position error [mm]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()

    fig.suptitle("Disturbance and tracking recovery")
    save_figure(
        fig,
        args.output,
        "perturbation_recovery.png",
    )


def plot_cartesian_paths(results, args):
    """3D reference, GN and/or IPOPT executed trajectories."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    reference = next(iter(results.values()))["xyz_ref"]

    ax.plot(
        *reference.T,
        "--",
        color=COLORS["Reference"],
        linewidth=2.3,
        label="Feasible reference",
    )

    all_points = [reference]

    for name, data in results.items():
        xyz = data["xyz"]
        all_points.append(xyz)

        ax.plot(
            *xyz.T,
            color=COLORS[name],
            linewidth=2.2,
            label=f"{name} executed",
        )

        ax.scatter(
            *xyz[-1],
            color=COLORS[name],
            marker="X",
            s=60,
        )

    ax.scatter(
        *reference[0],
        color="#2ca02c",
        s=65,
        label="Start",
    )

    extent = np.maximum(
        np.ptp(np.vstack(all_points), axis=0),
        1e-3,
    )

    ax.set_box_aspect(extent)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        f"Cartesian MPC tracking — {args.mode}"
    )
    ax.legend()

    save_figure(
        fig,
        args.output,
        "cartesian_trajectories.png",
    )


def plot_comparison(results, args):
    """Solver timing, Cartesian errors and convergence diagnostics."""
    fig, axes = plt.subplots(
        3, 1,
        figsize=(11, 10),
        sharex=True,
    )

    for name, data in results.items():
        t = data["time"]

        axes[0].plot(
            t,
            data["compute_ms"],
            color=COLORS[name],
            label=name,
        )
        axes[1].plot(
            t,
            1000 * data["pos_error"],
            color=COLORS[name],
            label=name,
        )
        axes[2].plot(
            t,
            np.degrees(data["ori_error"]),
            color=COLORS[name],
            label=name,
        )

    axes[0].axhline(
        1000 * args.dt,
        color="black",
        linestyle="--",
        label="Control period",
    )

    axes[0].set_ylabel("Computation [ms]")
    axes[1].set_ylabel("Position error [mm]")
    axes[2].set_ylabel("Orientation error [deg]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.legend()

    fig.suptitle(f"MPC tracking — {args.mode}")

    save_figure(
        fig,
        args.output,
        "mpc_tracking_comparison.png",
    )

    # Solver iterations and linear feasibility.
    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True,
    )

    for name, data in results.items():
        t = data["time"]

        axes[0].step(
            t,
            data["iterations"],
            where="mid",
            color=COLORS[name],
            label=name,
        )

        axes[1].semilogy(
            t,
            np.maximum(data["violation"], 1e-16),
            color=COLORS[name],
            label=name,
        )

    axes[0].set_ylabel("Solver iterations")
    axes[1].set_ylabel("Linear violation")
    axes[1].set_xlabel("Time [s]")

    axes[1].axhline(
        FEASIBILITY_TOL,
        color="black",
        linestyle="--",
        label="Tolerance",
    )

    axes[0].legend()
    axes[1].legend()

    fig.suptitle("MPC solver diagnostics")

    save_figure(
        fig,
        args.output,
        "solver_diagnostics.png",
    )


def plot_solver_boxplot(results, args):
    """Computation time distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results)

    ax.boxplot(
        [results[name]["compute_ms"] for name in names],
        tick_labels=names,
        showfliers=True,
    )

    ax.axhline(
        1000 * args.dt,
        color="black",
        linestyle="--",
        label="Control period",
    )

    ax.set_ylabel("Computation time [ms]")
    ax.set_title("MPC solver runtime distribution")
    ax.legend()

    save_figure(
        fig,
        args.output,
        "solver_timing_boxplot.png",
    )


def save_plots(results, args):
    """Save all relevant figures for one or both solvers."""
    plot_cartesian_paths(results, args)
    plot_comparison(results, args)

    # Joint signals now appear in both single- and dual-solver modes.
    plot_joint_signals(results, args)

    if len(results) > 1:
        plot_solver_boxplot(results, args)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--solver",
        choices=("gn", "ipopt", "both"),
        default="gn",
    )

    parser.add_argument(
        "--mode",
        choices=("none", "kick", "gaussian", "bias"),
        default="none",
    )

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=None)

    parser.add_argument(
        "--gn-iterations", type=int, default=5,
    )
    parser.add_argument(
        "--ipopt-iterations", type=int, default=100,
    )
    parser.add_argument(
        "--ipopt-tolerance", type=float, default=1e-5,
    )

    parser.add_argument(
        "--control-points", type=int, default=12,
    )
    parser.add_argument(
        "--tracking-points", type=int, default=15,
    )
    parser.add_argument(
        "--orientation-scale", type=float, default=0.1,
    )

    parser.add_argument(
        "--noise-accel-std", type=float, default=0.35,
    )
    parser.add_argument(
        "--kick-time", type=float, default=3.0,
    )
    parser.add_argument(
        "--kick-accel", type=float, default=4.8,
    )
    parser.add_argument(
        "--bias-accel", type=float, default=1.0,
    )
    parser.add_argument(
        "--bias-start", type=float, default=2.0,
    )
    parser.add_argument(
        "--bias-end", type=float, default=4.0,
    )
    parser.add_argument(
        "--disturbance-joint", type=int, default=0,
    )
    parser.add_argument(
        "--max-disturbance-accel",
        type=float,
        default=4.9,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lp-check", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument(
        "--animation-speed",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kinova_mpc_tracking"),
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error("Require 0 < dt < horizon")

    if args.steps is None:
        args.steps = int(round(DURATION / args.dt))

    if args.steps < 1:
        parser.error("steps must be positive")

    if args.gn_iterations < 1 or args.ipopt_iterations < 1:
        parser.error("iteration limits must be positive")

    if args.ipopt_tolerance <= 0:
        parser.error("ipopt-tolerance must be positive")

    if args.disturbance_joint not in range(7):
        parser.error("disturbance-joint must be 0..6")

    if args.noise_accel_std < 0:
        parser.error("noise-accel-std must be non-negative")

    if args.max_disturbance_accel <= 0:
        parser.error("max-disturbance-accel must be positive")

    if args.animation_speed <= 0:
        parser.error("animation-speed must be positive")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_reference()
    disturbances = generate_disturbances(args)

    if args.solver == "gn":
        names = ["GN"]
    elif args.solver == "ipopt":
        names = ["IPOPT"]
    else:
        names = ["GN", "IPOPT"]

    print("\nKINOVA CARTESIAN MPC TRACKING")
    print(f"Solvers: {', '.join(names)}")
    print(f"Mode: {args.mode}")
    print(f"Steps: {args.steps}")
    print(f"dt: {args.dt:.3f} s")
    print(f"Horizon: {args.horizon:.2f} s")
    print(f"Seed: {args.seed}")

    results = {
        name: run_mpc(name, args, disturbances)
        for name in names
    }

    print_summary(results, args)
    save_csv(results, args)
    save_plots(results, args)

    print(
        f"\nResults saved to: {args.output.resolve()}"
    )

    if args.show_plots:
        plt.show(block=False)

    if not args.no_animation:
        robot = Robot(
            urdf_path=URDF_PATH,
            end_effector_frame="end_effector_link",
        )

        first = next(iter(results.values()))

        animate_kinova_trajectories(
            robot=robot,
            trajectories={
                name: {
                    "q": data["q"],
                    "label": name,
                }
                for name, data in results.items()
            },
            dt=args.dt,
            reference_xyz=first["xyz_ref"],
            target_pose=reference_pose(
                robot, float(first["time"][-1]),
            ),
            speed=args.animation_speed,
            title="KINOVA MPC TRACKING",
        )

    elif args.show_plots:
        plt.show()

    else:
        plt.close("all")


if __name__ == "__main__":
    main()

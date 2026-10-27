
"""Kinova Gen3 — Cartesian MPC: Gauss-Newton vs IPOPT.

Both solvers:
- Track the same feasible Cartesian reference.
- Use identical B-spline and physical constraints.
- Receive the same pre-generated disturbances.
- Run in independent closed-loop simulations.
- Use the previous solution as a warm start.

The comparison includes:
- Computation times and deadlines
- Cartesian tracking errors
- Solver iterations and statuses
- Linear constraint violations
- Executed joint position/velocity limits
- 3D Cartesian trajectories
- Interactive Meshcat playback

Examples:
    python -m examples.kinova_mpc_solver_comparison
    python -m examples.kinova_mpc_solver_comparison --show-plots
    python -m examples.kinova_mpc_solver_comparison --mode kick
    python -m examples.kinova_mpc_solver_comparison --no-animation
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import meshcat.geometry as mg
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


FEASIBILITY_TOL = 1e-4

COLORS = {
    "Reference": "C1",
    "GN": "C0",
    "IPOPT": "C3",
}

MESHCAT_COLORS = {
    "Reference": 0xFF9900,
    "GN": 0x2277CC,
    "IPOPT": 0xCC3366,
}

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# Disturbances
# ============================================================

def generate_disturbances(args):
    """Generate the same disturbances for both closed-loop runs."""
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
            t_new = (k + 1) * args.dt

            if args.bias_start <= t_new <= args.bias_end:
                disturbances[k, args.disturbance_joint] = (
                    args.bias_accel
                )

    return np.clip(
        disturbances,
        -args.max_disturbance_accel,
        args.max_disturbance_accel,
    )


# ============================================================
# Solver factory
# ============================================================

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


# ============================================================
# Independent closed-loop simulation
# ============================================================

def run_mpc(name, args, disturbances):
    """Run one closed-loop MPC simulation using a given solver."""
    print("\n" + "=" * 70)
    print(f"Running {name}")
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

    solver = make_solver(name, args)
    mpc = CartesianMPC(problem, solver, c0)

    initial_pose = pose_from_theta(robot, q)

    data = {
        "time": [],
        "compute_ms": [],
        "iterations": [],
        "status": [],
        "pos_error": [],
        "ori_error": [],
        "violation": [],
        "q": [q.copy()],
        "v": [v.copy()],
        "xyz": [initial_pose.translation.copy()],
        "xyz_ref": [reference_pose(
            robot, 0.0
        ).translation.copy()],
    }

    for k in range(args.steps):
        t = k * args.dt
        t_new = (k + 1) * args.dt

        targets = [
            reference_pose(robot, t + tau)
            for tau in grid
        ]

        # Timing includes warm-start correction and solver,
        # but excludes target construction and LP validation.
        tic = time.perf_counter()

        mpc.previous_control_points = (
            enforce_initial_state_on_guess(
                spline,
                mpc.previous_control_points,
                q,
                v,
            )
        )

        result = mpc.step(
            q0=q,
            v0=v,
            target_poses=targets,
        )

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
                f"{name}, cycle {k+1}: non-finite solution."
            )

        violation = max_linear_violation(problem, c)

        if violation > FEASIBILITY_TOL:
            raise RuntimeError(
                f"{name}, cycle {k+1}: linear constraint "
                f"violation = {violation:.3e}"
            )

        if not result.success:
            if name == "GN":
                acceptable = result.status in (
                    "max_iterations",
                    "damping_limit",
                )
            else:
                # IPOPT -1 means maximum iterations exceeded.
                acceptable = result.status == -1

            if not acceptable:
                raise RuntimeError(
                    f"{name}, cycle {k+1}: solver failed "
                    f"with status {result.status}. "
                    f"{getattr(result, 'status_message', '')}"
                )

        # Keep feasible iterates even when convergence
        # was not reached within the iteration budget.
        mpc.previous_control_points = c.copy()

        # Execute the first MPC interval.
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
                f"{name}, cycle {k+1}: executed q/v "
                f"limits violated."
            )

        # Independent next-state feasibility check.
        # This is excluded from solve-time measurements.
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
                    f"{name}, cycle {k+1}: next state "
                    f"not feasible according to HiGHS: "
                    f"{lp.message}"
                )

        q, v = q_new, v_new

        actual_pose = pose_from_theta(robot, q)
        desired_pose = reference_pose(robot, t_new)

        ep, eo = pose_errors(
            actual_pose,
            desired_pose,
        )

        data["time"].append(t_new)
        data["compute_ms"].append(elapsed_ms)
        data["iterations"].append(result.iterations)
        data["status"].append(result.status)
        data["pos_error"].append(ep)
        data["ori_error"].append(eo)
        data["violation"].append(violation)
        data["q"].append(q.copy())
        data["v"].append(v.copy())
        data["xyz"].append(
            actual_pose.translation.copy()
        )
        data["xyz_ref"].append(
            desired_pose.translation.copy()
        )

        print(
            f"[{name:5s} {k+1:03d}/{args.steps:03d}] "
            f"time={elapsed_ms:7.1f} ms  "
            f"iter={result.iterations:3d}  "
            f"pos={1000*ep:7.3f} mm  "
            f"ori={np.degrees(eo):6.3f} deg  "
            f"status={result.status}"
        )

    return {
        key: np.asarray(values)
        for key, values in data.items()
    }


# ============================================================
# Statistics
# ============================================================

def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def print_summary(results, args):
    print("\n" + "=" * 88)
    print("MPC SOLVER COMPARISON")
    print("=" * 88)

    print(
        f"{'Metric':<31}"
        f"{'GN':>20}"
        f"{'IPOPT':>20}"
    )
    print("-" * 71)

    metrics = {
        "Mean time [ms]": lambda d: np.mean(
            d["compute_ms"]
        ),
        "Median time [ms]": lambda d: np.median(
            d["compute_ms"]
        ),
        "P95 time [ms]": lambda d: np.percentile(
            d["compute_ms"], 95
        ),
        "Maximum time [ms]": lambda d: np.max(
            d["compute_ms"]
        ),
        "Mean iterations": lambda d: np.mean(
            d["iterations"]
        ),
        "Position RMS [mm]": lambda d: (
            1000 * rms(d["pos_error"])
        ),
        "Position max [mm]": lambda d: (
            1000 * np.max(d["pos_error"])
        ),
        "Orientation RMS [deg]": lambda d: (
            np.degrees(rms(d["ori_error"]))
        ),
        "Orientation max [deg]": lambda d: (
            np.degrees(np.max(d["ori_error"]))
        ),
        "Max linear violation": lambda d: np.max(
            d["violation"]
        ),
    }

    for label, metric in metrics.items():
        gn = metric(results["GN"])
        ipopt = metric(results["IPOPT"])

        print(
            f"{label:<31}"
            f"{gn:>20.5f}"
            f"{ipopt:>20.5f}"
        )

    print("-" * 71)

    for name, data in results.items():
        within_budget = np.mean(
            data["compute_ms"] <= 1000 * args.dt
        )
        statuses, counts = np.unique(
            data["status"].astype(str),
            return_counts=True,
        )

        print(
            f"{name} deadline hit rate: "
            f"{100 * within_budget:.1f}%"
        )
        print(
            f"{name} statuses: "
            f"{dict(zip(statuses, counts))}"
        )

    print(
        "\nNote: the deadline rate concerns the measured "
        "optimizer call, not the full control loop."
    )


# ============================================================
# Plots
# ============================================================

def plot_comparison(results, args):
    """Generate and save all solver comparison plots."""
    fig, axes = plt.subplots(
        3, 1,
        figsize=(11, 10),
        sharex=True,
    )

    for name, data in results.items():
        t = data["time"]
        color = COLORS[name]

        axes[0].plot(
            t, data["compute_ms"],
            label=name, color=color,
        )
        axes[1].plot(
            t, 1000 * data["pos_error"],
            label=name, color=color,
        )
        axes[2].plot(
            t, np.degrees(data["ori_error"]),
            label=name, color=color,
        )

    axes[0].axhline(
        1000 * args.dt,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Control period",
    )

    axes[0].set_ylabel("Computation [ms]")
    axes[1].set_ylabel("Position error [mm]")
    axes[2].set_ylabel("Orientation error [deg]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Cartesian MPC — GN vs IPOPT ({args.mode})"
    )
    fig.tight_layout()
    fig.savefig(
        args.output / "solver_comparison.png",
        dpi=180,
    )

    # Computation-time distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        [
            results["GN"]["compute_ms"],
            results["IPOPT"]["compute_ms"],
        ],
        tick_labels=["GN", "IPOPT"],
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
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        args.output / "solver_timing_boxplot.png",
        dpi=180,
    )

    # Cartesian 3D trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    reference_xyz = results["GN"]["xyz_ref"]

    ax.plot(
        *reference_xyz.T,
        "--",
        color=COLORS["Reference"],
        linewidth=2.5,
        label="Feasible reference",
    )

    for name in ("GN", "IPOPT"):
        xyz = results[name]["xyz"]

        ax.plot(
            *xyz.T,
            color=COLORS[name],
            linewidth=2.1,
            label=f"{name} executed",
        )

        ax.scatter(
            *xyz[-1],
            color=COLORS[name],
            marker="X",
            s=55,
        )

    ax.scatter(
        *reference_xyz[0],
        color="#2ca02c",
        s=65,
        label="Start",
    )

    all_points = np.vstack([
        reference_xyz,
        results["GN"]["xyz"],
        results["IPOPT"]["xyz"],
    ])
    extent = np.maximum(
        np.ptp(all_points, axis=0),
        1e-3,
    )
    ax.set_box_aspect(extent)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        f"Executed Cartesian trajectories — {args.mode}"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        args.output / "cartesian_trajectories_3d.png",
        dpi=180,
    )


# ============================================================
# Meshcat animation
# ============================================================

def animate_comparison(results, args):
    """Replay GN or IPOPT execution until the user enters 0."""
    print("\nOpening Meshcat...", flush=True)

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )
    robot.init_viewer(open_browser=True)

    viewer = robot.viewer.viewer

    paths = {
        "Reference": results["GN"]["xyz_ref"],
        "GN": results["GN"]["xyz"],
        "IPOPT": results["IPOPT"]["xyz"],
    }

    for name, points in paths.items():
        viewer[f"trajectories/{name.lower()}"].set_object(
            mg.Line(
                mg.PointsGeometry(
                    np.asarray(points, dtype=np.float32).T
                ),
                mg.LineBasicMaterial(
                    color=MESHCAT_COLORS[name],
                    linewidth=3,
                ),
            )
        )

    final_time = float(results["GN"]["time"][-1])
    robot.display_target(
        reference_pose(robot, final_time),
        name="final_reference",
    )

    configurations = {
        name: robot.joint_trajectory_to_configurations(
            results[name]["q"]
        )
        for name in ("GN", "IPOPT")
    }

    robot.display(configurations["GN"][0])

    try:
        while True:
            print("\n" + "=" * 45)
            print("MPC SOLVER COMPARISON — ANIMATION")
            print("=" * 45)
            print("  1. Gauss-Newton trajectory")
            print("  2. IPOPT trajectory")
            print("  0. Close plots and exit")

            try:
                choice = input(
                    "\nSelect trajectory: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "0":
                break

            if choice not in ("1", "2"):
                print("Invalid selection.")
                continue

            name = "GN" if choice == "1" else "IPOPT"
            trajectory = configurations[name]

            print(f"\nPlaying {name} trajectory...")

            robot.display(trajectory[0])
            time.sleep(0.2)

            for i, configuration in enumerate(trajectory):
                robot.display(configuration)

                if i + 1 < len(trajectory):
                    time.sleep(
                        args.dt / args.animation_speed
                    )

            print("Animation finished.")

    finally:
        plt.close("all")
        print("Animation session closed.")


# ============================================================
# CSV
# ============================================================

def save_csv(results, args):
    path = args.output / "solver_comparison.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "solver",
            "cycle",
            "time_s",
            "compute_ms",
            "iterations",
            "status",
            "pos_error_mm",
            "ori_error_deg",
            "linear_violation",
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
                ])

    print(f"\nSaved CSV: {path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        choices=("none", "kick", "gaussian", "bias"),
        default="gaussian",
    )

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=None)

    parser.add_argument(
        "--gn-iterations", type=int, default=2,
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
        "--noise-accel-std", type=float, default=0.05,
    )
    parser.add_argument(
        "--kick-time", type=float, default=3.0,
    )
    parser.add_argument(
        "--kick-accel", type=float, default=0.2,
    )
    parser.add_argument(
        "--bias-accel", type=float, default=0.05,
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
        default=0.3,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lp-check", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--show-plots", action="store_true")

    parser.add_argument(
        "--animation-speed",
        type=float,
        default=2.0,
        help="Meshcat playback speed multiplier.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kinova_mpc_solver_comparison"),
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

    print("\nKINOVA CARTESIAN MPC — SOLVER COMPARISON")
    print(f"Mode: {args.mode}")
    print(f"Steps: {args.steps}")
    print(f"dt: {args.dt} s")
    print(f"Horizon: {args.horizon} s")
    print(f"GN max iterations: {args.gn_iterations}")
    print(f"IPOPT max iterations: {args.ipopt_iterations}")
    print(f"Seed: {args.seed}")

    results = {}

    for name in ("GN", "IPOPT"):
        results[name] = run_mpc(
            name, args, disturbances,
        )

    print_summary(results, args)
    save_csv(results, args)
    plot_comparison(results, args)

    print(f"\nResults saved to: {args.output}")

    # Display plots before the interactive Meshcat menu.
    if args.show_plots:
        plt.show(block=False)

    if not args.no_animation:
        animate_comparison(results, args)

    elif args.show_plots:
        plt.show()

    else:
        plt.close("all")


if __name__ == "__main__":
    main()

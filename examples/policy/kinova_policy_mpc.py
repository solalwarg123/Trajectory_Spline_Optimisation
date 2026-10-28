
"""Kinova Gen3: noisy Cartesian policy vs constrained B-spline MPC.

- Policy emits 16-pose chunks every 1.5 simulated seconds.
- New chunks become available after a simulated inference latency.
- RAW holds its last command during inference.
- MPC continues executing its last B-spline.
- Predictions beyond the available chunk are hold-padded.
- MPC optimization period: 100 ms by default.
- Execution sampling: 10 ms by default.

Inference latency is simulated, not measured.
RAW inverse kinematics is precomputed offline.

Examples
--------
python -m examples.policy.kinova_policy_mpc
python -m examples.policy.kinova_policy_mpc --gn-iters 2 --show-plots
python -m examples.policy.kinova_policy_mpc --no-animation
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.mpc import CartesianMPC
from splineopt.robotics import Robot
from splineopt.robotics.ik import solve_pose_ik

from examples.utils.kinova_helpers import (
    URDF_PATH,
    Q_MIN,
    Q_MAX,
    V_MIN,
    V_MAX,
    A_MIN,
    A_MAX,
)
from examples.utils.mpc_helpers import make_spline, pose_from_theta, max_linear_violation
from examples.utils.kinova_visualization import (
    animate_kinova_trajectories,
)
from examples.policy.mock_policy import (
    DURATION,
    nominal_state,
    nominal_diagnostics,
    MockChunkPolicy,
    pregenerate_chunks,
    available_chunk_index,
)


C_RAW = "#e68625"
C_MPC = "#c6384c"
C_NOM = "#222222"
C_WAIT = "#e6b75b"


# ============================================================
# Utilities
# ============================================================

def rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (
        float(np.sqrt(np.mean(values**2)))
        if values.size
        else float("nan")
    )


def max_violation(x, lower, upper):
    x = np.asarray(x, dtype=float)
    return float(np.max(np.maximum(
        np.maximum(lower - x, x - upper), 0.0
    )))


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def derivatives(q, dt):
    """Finite differences of commanded joint positions.

    These are not motor-dynamics derivatives. In particular,
    RAW command discontinuities can create large acceleration
    and jerk estimates.
    """
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    j = np.zeros_like(q)

    v[1:] = np.diff(q, axis=0) / dt
    a[2:] = np.diff(v[1:], axis=0) / dt
    j[3:] = np.diff(a[2:], axis=0) / dt

    return v, a, j


# ============================================================
# RAW inverse kinematics
# ============================================================

def precompute_ik(robot, chunks, args):
    """Compute RAW IK offline, outside online timing."""
    print(
        "\nPRECOMPUTING RAW IK "
        "(excluded from online timings)"
    )

    seed = nominal_state(0.0)[0].copy()
    all_q = []
    failures = 0

    for k, chunk in enumerate(chunks):
        start = perf_counter()
        q_chunk = []

        for target in chunk.poses:
            result = solve_pose_ik(
                robot=robot,
                target_pose=target,
                q0=robot.joint_positions_to_configuration(seed),
                tolerance=args.ik_tolerance,
                max_iterations=args.ik_max_iterations,
                step_size=0.2,
                damping=1e-5,
            )

            if result.success:
                q = robot.configuration_to_joint_positions(
                    result.q
                )
                q = np.asarray(q, dtype=float).reshape(-1)

                # Use the angular branch closest to the previous pose.
                q = seed + (q - seed + np.pi) % (2*np.pi) - np.pi
                seed = q.copy()
            else:
                failures += 1
                q = seed.copy()

            q_chunk.append(q)

        all_q.append(np.asarray(q_chunk))

        print(
            f"  Chunk {k+1:02d}/{len(chunks)}: "
            f"{1000*(perf_counter()-start):.1f} ms | "
            f"IK failures: {failures}"
        )

    if failures:
        raise RuntimeError(
            f"RAW has {failures} IK failures: benchmark invalid"
        )

    return all_q


# ============================================================
# RAW trajectory
# ============================================================

def raw_motion(chunks, ik_chunks, times, latency, q_initial):
    """Build the directly commanded RAW trajectory.

    At a chunk boundary, RAW holds the preceding position
    until the new chunk becomes available. The first motion
    segment after availability interpolates from the held
    position to the second waypoint of the new chunk.
    """
    q_values = np.empty((len(times), len(q_initial)))
    waiting = np.zeros(len(times), dtype=bool)

    dt = chunks[0].dt
    boundaries = np.asarray([
        chunk.start_time for chunk in chunks
    ])

    for n, t in enumerate(times):
        index = int(np.clip(
            np.searchsorted(
                boundaries, t + 1e-9, side="right"
            ) - 1,
            0,
            len(chunks) - 1,
        ))

        chunk = chunks[index]
        q_chunk = ik_chunks[index]
        tau = t - chunk.start_time

        if index == 0:
            if tau <= dt:
                alpha = np.clip(tau / dt, 0.0, 1.0)
                q_values[n] = (
                    q_initial + alpha * (q_chunk[1] - q_initial)
                )
                continue
        else:
            previous_last = ik_chunks[index - 1][-1]

            if tau < latency - 1e-9:
                q_values[n] = previous_last
                waiting[n] = True
                continue

            if tau <= dt + 1e-9:
                alpha = np.clip(
                    (tau - latency) / (dt - latency),
                    0.0,
                    1.0,
                )
                q_values[n] = (
                    previous_last
                    + alpha * (q_chunk[1] - previous_last)
                )
                continue

        x = np.clip(tau / dt, 0.0, len(q_chunk) - 1)
        j = min(int(np.floor(x)), len(q_chunk) - 2)
        alpha = x - j

        q_values[n] = (
            (1.0 - alpha) * q_chunk[j]
            + alpha * q_chunk[j + 1]
        )

    return q_values, waiting


# ============================================================
# MPC initialization
# ============================================================

def create_mpc(robot, chunks, args):
    spline = make_spline(
        args.horizon,
        robot.nv,
        args.control_points,
    )

    grid = np.linspace(
        0.0, args.horizon, args.tracking_points
    )

    q0 = nominal_state(0.0)[0]

    problem = CartesianTrajectoryProblem(
        robot=robot,
        spline=spline,
        tracking_times=grid,
        target_poses=chunks[0].horizon(0.0, grid),
        q0=q0,
        v0=np.zeros(robot.nv),
        q_min=Q_MIN,
        q_max=Q_MAX,
        v_min=V_MIN,
        v_max=V_MAX,
        a_min=A_MIN,
        a_max=A_MAX,
        orientation_scale=args.orientation_scale,
        lambda_smooth=args.lambda_smooth,
        lambda_terminal=args.lambda_terminal,
    )

    solver = GaussNewtonSolver(
        damping=1e-3,
        damping_mode="adaptive",
        max_iterations=args.gn_iters,
    )

    mpc = CartesianMPC(
        problem,
        solver,
        np.tile(q0, (spline.n_control_points, 1)),
        project_warm_start=True,
        projection_tolerance=args.projection_tol,
        linear_feasibility_tolerance=(
            args.linear_feasibility_tol
        ),
    )

    return spline, grid, problem, mpc


# ============================================================
# MPC execution
# ============================================================

def simulate_mpc(robot, chunks, times, args):
    """Simulate continuous B-spline execution and chunk updates."""
    spline, grid, problem, mpc = create_mpc(
        robot, chunks, args
    )

    n_cycles = int(round(DURATION / args.dt))
    regular_times = np.arange(n_cycles) * args.dt

    boundaries = np.asarray([
        chunk.start_time for chunk in chunks[1:]
    ])

    if args.inference_latency > 0 and boundaries.size:
        at_boundary = np.any(
            np.isclose(
                regular_times[:, None],
                boundaries[None, :],
                atol=1e-9,
                rtol=0.0,
            ),
            axis=1,
        )

        solve_times = np.sort(np.concatenate([
            regular_times[~at_boundary],
            boundaries + args.inference_latency,
        ]))
    else:
        solve_times = regular_times

    q_dense = np.full(
        (len(times), robot.nv), np.nan
    )

    q_initial = nominal_state(0.0)[0].copy()
    q_dense[0] = q_initial

    previous_c = None
    previous_time = None
    rows = []
    fallback_count = 0

    for k, t in enumerate(solve_times):
        t = float(t)

        if previous_c is None:
            q = q_initial.copy()
            v = np.zeros(robot.nv)
        else:
            elapsed = t - previous_time
            q = np.asarray(
                spline.position(previous_c, elapsed)
            ).reshape(-1)
            v = np.asarray(
                spline.velocity(previous_c, elapsed)
            ).reshape(-1)

        index = available_chunk_index(
            chunks, t, args.inference_latency
        )
        targets = chunks[index].horizon(t, grid)

        start = perf_counter()
        result = mpc.step(
            q0=q,
            v0=v,
            target_poses=targets,
        )
        solve_ms = 1000 * (perf_counter() - start)

        info = mpc.last_solution_info or {}
        projection = mpc.last_projection_info or {}

        if not info.get("accepted", False):
            raise RuntimeError(
                f"MPC rejected solve {k+1} at t={t:.3f}s: "
                f"status={getattr(result, 'status', '?')}, "
                f"linear={info.get('linear_violation')}, "
                f"projection={projection}"
            )

        c = np.asarray(
            result.control_points, dtype=float
        ).reshape(
            spline.n_control_points, robot.nv
        )

        violation = max_linear_violation(problem, c)

        if (
            not np.isfinite(violation)
            or violation > args.linear_feasibility_tol
        ):
            raise RuntimeError(
                f"MPC infeasible at t={t:.3f}s: "
                f"violation={violation:.3e}"
            )

        next_t = (
            float(solve_times[k + 1])
            if k + 1 < len(solve_times)
            else DURATION
        )

        i_start = int(round(t / args.sample_dt))
        i_stop = int(round(next_t / args.sample_dt))

        for i in range(i_start, i_stop + 1):
            tau = np.clip(
                times[i] - t, 0.0, args.horizon
            )
            q_dense[i] = np.asarray(
                spline.position(c, float(tau))
            ).reshape(-1)

        previous_c = c.copy()
        previous_time = t

        source = info.get("solution_source", "gn")
        fallback_count += (
            source == "projected_warm_start"
        )

        rows.append({
            "cycle": k + 1,
            "time_s": round(t, 5),
            "chunk_used": index + 1,
            "chunk_ready": int(
                index > 0 and t >= (
                    chunks[index].start_time
                    + args.inference_latency - 1e-9
                )
            ),
            "solve_ms": round(solve_ms, 5),
            "solver_status": str(
                getattr(result, "status", "unknown")
            ),
            "solver_converged": bool(
                getattr(result, "success", False)
            ),
            "accepted": True,
            "solution_source": source,
            "fallback_used": int(
                source == "projected_warm_start"
            ),
            "projected": int(bool(
                projection.get("projected", False)
            )),
            "linear_violation": violation,
        })

        if args.print_every > 0 and (
            (k + 1) % args.print_every == 0 or k == 0
        ):
            print(
                f"[MPC {k+1:03d}/{len(solve_times)}] "
                f"t={t:5.2f}s | "
                f"chunk={index+1:02d} | "
                f"{solve_ms:6.2f} ms | "
                f"source={source} | "
                f"linear={violation:.1e}"
            )

    if not np.all(np.isfinite(q_dense)):
        raise RuntimeError(
            "MPC execution timeline contains missing samples"
        )

    print(
        f"MPC complete: {len(solve_times)} solves; "
        f"projected warm-start fallbacks={fallback_count}"
    )

    return q_dense, rows


# ============================================================
# Metrics
# ============================================================

def metric_summary(
    q, dt, robot, nominal_xyz, nominal_rot, method
):
    """Evaluate RAW and MPC over the same sampling timeline."""
    v, a, j = derivatives(q, dt)

    ep = np.empty(len(q))
    eo = np.empty(len(q))

    for i, qi in enumerate(q):
        pose = pose_from_theta(robot, qi)

        ep[i] = (
            1000 * np.linalg.norm(
                pose.translation - nominal_xyz[i]
            )
        )
        eo[i] = np.linalg.norm(
            pin.log3(pose.rotation.T @ nominal_rot[i])
        )

    a_violation = np.maximum(
        np.maximum(A_MIN - a[2:], a[2:] - A_MAX),
        0.0,
    )

    summary = {
        "method": method,
        "position_rms_mm": rms(ep),
        "orientation_rms_deg": float(
            np.degrees(rms(eo))
        ),
        "velocity_rms_rad_s": rms(v[1:]),
        "acceleration_rms_rad_s2": rms(a[2:]),
        "jerk_rms_rad_s3": rms(j[3:]),
        "max_v_violation": max_violation(
            v[1:], V_MIN, V_MAX
        ),
        "max_a_violation": max_violation(
            a[2:], A_MIN, A_MAX
        ),
        "acceleration_violation_pct": (
            100 * float(np.mean(
                np.any(a_violation > 1e-6, axis=1)
            ))
        ),
    }

    return summary, (v, a, j, ep, eo)


# ============================================================
# Matplotlib
# ============================================================

def inference_bands(ax, chunks, latency, alpha=0.2):
    """Highlight simulated policy-inference intervals."""
    for i, chunk in enumerate(chunks[1:]):
        ax.axvspan(
            chunk.start_time,
            chunk.start_time + latency,
            facecolor=C_WAIT,
            alpha=alpha,
            label=(
                f"Policy inference ({1000*latency:.0f} ms)"
                if i == 0 else None
            ),
        )


def save_figure(fig, args, filename, dpi=170):
    fig.tight_layout()
    fig.savefig(
        args.output / filename,
        dpi=dpi,
    )


def plot_results(
    times, chunks, q_raw, q_mpc,
    raw_d, mpc_d, metrics, cycles, args
):
    """Generate the six benchmark figures."""
    vr, ar, jr, ep_raw, _ = raw_d
    vm, am, jm, ep_mpc, _ = mpc_d

    # --------------------------------------------------------
    # Figure 1 — Global smoothness
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10), sharex=True
    )

    panels = (
        (
            np.linalg.norm(vr, axis=1),
            np.linalg.norm(vm, axis=1),
            "Joint speed norm [rad/s]",
        ),
        (
            np.linalg.norm(ar, axis=1),
            np.linalg.norm(am, axis=1),
            "Joint acceleration norm [rad/s²]",
        ),
        (
            np.linalg.norm(jr, axis=1),
            np.linalg.norm(jm, axis=1),
            "Joint jerk norm [rad/s³]",
        ),
    )

    for ax, (raw, mpc, ylabel) in zip(axes, panels):
        ax.plot(
            times, raw,
            color=C_RAW,
            lw=1.0,
            alpha=0.75,
            label="RAW",
        )
        ax.plot(
            times, mpc,
            color=C_MPC,
            lw=1.5,
            label="B-spline MPC",
        )

        inference_bands(
            ax, chunks, args.inference_latency
        )

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    axes[1].set_yscale("symlog", linthresh=2.0)
    axes[2].set_yscale("symlog", linthresh=10.0)
    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        "Motion smoothness — execution samples"
    )
    save_figure(
        fig, args, "smoothness_raw_vs_mpc.png"
    )

    # --------------------------------------------------------
    # Figure 2 — Chunk-boundary zooms
    # --------------------------------------------------------
    boundaries = [
        c.start_time for c in chunks[1:4]
    ]

    if boundaries:
        fig, axes = plt.subplots(
            len(boundaries),
            3,
            figsize=(18, 3.5 * len(boundaries)),
            squeeze=False,
        )

        joint = args.zoom_joint - 1

        for i, boundary in enumerate(boundaries):
            mask = (
                (times >= boundary - 0.12)
                & (times <= boundary + 0.20)
            )

            datasets = (
                (
                    q_raw[:, joint],
                    q_mpc[:, joint],
                    "q [rad]",
                ),
                (
                    vr[:, joint],
                    vm[:, joint],
                    "v [rad/s]",
                ),
                (
                    ar[:, joint],
                    am[:, joint],
                    "a [rad/s²]",
                ),
            )

            for j, (raw, mpc, ylabel) in enumerate(datasets):
                ax = axes[i, j]

                ax.plot(
                    times[mask], raw[mask],
                    color=C_RAW,
                    lw=1.8,
                    marker=".",
                    ms=2,
                    label="RAW" if i == 0 else None,
                )
                ax.plot(
                    times[mask], mpc[mask],
                    color=C_MPC,
                    lw=2,
                    label="MPC" if i == 0 else None,
                )

                ax.axvspan(
                    boundary,
                    boundary + args.inference_latency,
                    facecolor=C_WAIT,
                    alpha=0.35,
                )
                ax.axvline(
                    boundary,
                    color="gray",
                    linestyle=":",
                )
                ax.axvline(
                    boundary + args.inference_latency,
                    color="#ae7f28",
                    linestyle="--",
                )

                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"Boundary {boundary:.2f} s — "
                    f"joint {args.zoom_joint}"
                )
                ax.grid(alpha=0.2)

                if i == 0:
                    ax.legend()

                if i == len(boundaries) - 1:
                    ax.set_xlabel("Time [s]")

        fig.suptitle(
            "Policy inference: RAW holds position, "
            "MPC keeps moving"
        )
        save_figure(
            fig, args, "inference_hold_zooms.png",
            dpi=190,
        )

    # --------------------------------------------------------
    # Figure 3 — Cartesian errors
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4.2))

    ax.plot(
        times, ep_raw,
        color=C_RAW,
        lw=1.0,
        label="RAW",
    )
    ax.plot(
        times, ep_mpc,
        color=C_MPC,
        lw=1.5,
        label="MPC",
    )

    inference_bands(
        ax, chunks, args.inference_latency
    )

    ax.set(
        xlabel="Time [s]",
        ylabel="Error [mm]",
        title=(
            "Cartesian position error "
            "vs noiseless reference"
        ),
    )
    ax.grid(alpha=0.2)
    ax.legend()

    save_figure(
        fig, args, "cartesian_errors.png"
    )

    # --------------------------------------------------------
    # Figure 4 — Joint positions and velocities
    # --------------------------------------------------------
    joint = args.zoom_joint - 1

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True
    )

    datasets = (
        (
            q_raw[:, joint],
            q_mpc[:, joint],
            "Joint position [rad]",
        ),
        (
            vr[:, joint],
            vm[:, joint],
            "Joint velocity [rad/s]",
        ),
    )

    for ax, (raw, mpc, ylabel) in zip(axes, datasets):
        ax.plot(
            times, raw,
            color=C_RAW,
            alpha=0.7,
            lw=1,
            label="RAW",
        )
        ax.plot(
            times, mpc,
            color=C_MPC,
            lw=1.7,
            label="MPC",
        )

        inference_bands(
            ax, chunks, args.inference_latency
        )

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        f"Joint {joint+1}: noisy commands "
        "versus B-spline MPC"
    )

    save_figure(
        fig, args, "joint_comparison.png"
    )

    # --------------------------------------------------------
    # Figure 5 — Computation time vs policy latency
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4.5))

    cycle_times = np.asarray([
        row["time_s"] for row in cycles
    ])
    compute_times = np.asarray([
        row["solve_ms"] for row in cycles
    ])

    ax.plot(
        cycle_times,
        compute_times,
        color=C_MPC,
        marker=".",
        ms=3,
        lw=1,
        label="MPC solve time (measured)",
    )

    ax.axhline(
        1000 * args.inference_latency,
        color="#a77d26",
        linestyle="--",
        label=(
            "Simulated policy latency: "
            f"{1000*args.inference_latency:.0f} ms"
        ),
    )
    ax.axhline(
        1000 * args.dt,
        color="black",
        linestyle=":",
        label=f"MPC period: {1000*args.dt:.0f} ms",
    )

    inference_bands(
        ax, chunks, args.inference_latency
    )

    ax.set(
        xlabel="Simulation time [s]",
        ylabel="Time [ms]",
        title=(
            "Measured MPC computation "
            "vs simulated policy latency"
        ),
    )
    ax.grid(alpha=0.2)
    ax.legend()

    save_figure(
        fig, args, "timing_and_inference.png"
    )

    # --------------------------------------------------------
    # Figure 6 — Quantitative comparison
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4)
    )

    names = [row["method"] for row in metrics]

    quantities = (
        ("position_rms_mm", "Position RMS [mm]"),
        (
            "acceleration_rms_rad_s2",
            "Acceleration RMS [rad/s²]",
        ),
        ("jerk_rms_rad_s3", "Jerk RMS [rad/s³]"),
    )

    for ax, (key, title) in zip(axes, quantities):
        values = [row[key] for row in metrics]

        bars = ax.bar(
            names,
            values,
            color=[C_RAW, C_MPC],
            width=0.55,
        )
        ax.bar_label(
            bars, fmt="%.2f", padding=3
        )

        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Same {DURATION:.1f} s execution window "
        "for both methods"
    )

    save_figure(
        fig, args, "benchmark_summary.png"
    )


# ============================================================
# Shared Meshcat animation
# ============================================================

def animate_results(
    robot, times, q_raw, q_mpc,
    nominal_xyz, nominal_poses, args
):
    """Replay RAW and MPC with the shared Kinova viewer."""
    animate_kinova_trajectories(
        robot=robot,
        trajectories={
            "RAW": {
                "q": q_raw,
                "times": times,
                "label": "RAW noisy policy",
                "color": 0xE68625,
            },
            "MPC": {
                "q": q_mpc,
                "times": times,
                "label": "B-spline MPC",
                "color": 0xC6384C,
            },
        },
        reference_xyz=nominal_xyz,
        target_pose=nominal_poses[-1],
        speed=args.animation_speed,
        title="NOISY POLICY VS B-SPLINE MPC",
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Simulation and MPC.
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--sample-dt", type=float, default=0.01
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16
    )
    parser.add_argument(
        "--horizon", type=float, default=1.5
    )
    parser.add_argument(
        "--inference-latency",
        type=float,
        default=0.02,
        help="Simulated inference latency [s]",
    )
    parser.add_argument(
        "--control-points", type=int, default=12
    )
    parser.add_argument(
        "--tracking-points", type=int, default=16
    )
    parser.add_argument(
        "--gn-iters", type=int, default=2
    )

    # Mock policy.
    parser.add_argument(
        "--noise-std", type=float, default=0.0015
    )
    parser.add_argument(
        "--bias-std", type=float, default=0.0005
    )
    parser.add_argument(
        "--tremor-amplitude",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--tremor-frequency",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    # Optimization weights.
    parser.add_argument(
        "--orientation-scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lambda-terminal",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--projection-tol",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--linear-feasibility-tol",
        type=float,
        default=1e-5,
    )

    # Inverse kinematics.
    parser.add_argument(
        "--ik-tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--ik-max-iterations",
        type=int,
        default=300,
    )

    # Output and visualization.
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--zoom-joint",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
    )
    parser.add_argument(
        "--animation-speed",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kinova_policy_mpc"),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------
    if args.dt <= 0 or args.sample_dt <= 0:
        parser.error("dt and sample-dt must be positive")

    ratio = args.dt / args.sample_dt
    if abs(ratio - round(ratio)) > 1e-8:
        parser.error("dt must be a multiple of sample-dt")

    if not 0 <= args.inference_latency < args.dt:
        parser.error(
            "inference-latency must be between 0 and dt"
        )

    latency_ratio = args.inference_latency / args.sample_dt
    if abs(latency_ratio - round(latency_ratio)) > 1e-8:
        parser.error(
            "inference-latency must be a multiple of sample-dt"
        )

    if args.chunk_size < 2:
        parser.error("chunk-size must be at least 2")

    if not 1 <= args.zoom_joint <= 7:
        parser.error("zoom-joint must be between 1 and 7")

    if args.gn_iters < 1:
        parser.error("gn-iters must be positive")

    if args.animation_speed <= 0:
        parser.error("animation-speed must be positive")

    if abs(
        args.horizon - (args.chunk_size - 1) * args.dt
    ) > 1e-8:
        parser.error(
            "horizon must equal (chunk-size-1)*dt"
        )

    duration_ratio = DURATION / args.dt
    if abs(duration_ratio - round(duration_ratio)) > 1e-8:
        parser.error(
            "simulation duration must be a multiple of dt"
        )

    args.output.mkdir(
        parents=True, exist_ok=True
    )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------
    robot = Robot(
        URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    if robot.nv != 7:
        raise ValueError(
            f"Expected 7 joints, got {robot.nv}"
        )

    print("=" * 85)
    print("KINOVA GEN3 — NOISY POLICY VS B-SPLINE MPC")
    print("SIMULATED POLICY INFERENCE LATENCY")
    print("=" * 85)

    print(
        f"Duration={DURATION:.2f}s | "
        f"MPC dt={1000*args.dt:.0f}ms | "
        f"sample dt={1000*args.sample_dt:.0f}ms | "
        f"inference={1000*args.inference_latency:.0f}ms"
    )
    print(
        f"Noise={1000*args.noise_std:.1f}mm | "
        f"bias={1000*args.bias_std:.1f}mm | "
        f"tremor={1000*args.tremor_amplitude:.1f}mm"
    )

    diagnostics = nominal_diagnostics()
    print(
        "Nominal feasibility: "
        f"q={diagnostics['max_q_violation']:.2e} | "
        f"v={diagnostics['max_v_violation']:.2e} | "
        f"a={diagnostics['max_a_violation']:.2e} | "
        f"accel violating time="
        f"{diagnostics['accel_violation_pct']:.2f}%"
    )

    # --------------------------------------------------------
    # Generate identical policy predictions
    # --------------------------------------------------------
    policy = MockChunkPolicy(
        robot,
        chunk_size=args.chunk_size,
        dt=args.dt,
        noise_std=args.noise_std,
        bias_std=args.bias_std,
        tremor_amplitude=args.tremor_amplitude,
        tremor_frequency=args.tremor_frequency,
        seed=args.seed,
    )

    chunks = pregenerate_chunks(
        policy, duration=DURATION
    )

    print(
        f"Generated {len(chunks)} identical chunks "
        "for RAW and MPC"
    )

    for i, chunk in enumerate(chunks):
        diff = np.asarray([
            pose.translation - nominal.translation
            for pose, nominal in zip(
                chunk.poses, chunk.nominal_poses
            )
        ])

        availability = chunk.start_time + (
            0.0 if i == 0 else args.inference_latency
        )

        print(
            f"  Chunk {i+1:02d} | "
            f"{chunk.start_time:.2f}-"
            f"{chunk.end_time:.2f}s | "
            f"available at {availability:.2f}s | "
            f"noise RMS="
            f"{1000*rms(np.linalg.norm(diff, axis=1)):.2f}mm"
        )

    # --------------------------------------------------------
    # Precompute inverse kinematics
    # --------------------------------------------------------
    ik_chunks = precompute_ik(
        robot, chunks, args
    )

    # --------------------------------------------------------
    # Common execution timeline
    # --------------------------------------------------------
    n_samples = int(round(
        DURATION / args.sample_dt
    )) + 1

    times = np.linspace(
        0.0, DURATION, n_samples
    )

    q_initial = nominal_state(0.0)[0].copy()

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------
    print(
        "\nSIMULATION: RAW "
        "(command hold during inference)"
    )

    q_raw, waiting = raw_motion(
        chunks,
        ik_chunks,
        times,
        args.inference_latency,
        q_initial,
    )

    print(
        f"RAW hold samples: {int(np.sum(waiting))} "
        f"({1000*args.sample_dt:.0f} ms each)"
    )

    # --------------------------------------------------------
    # MPC
    # --------------------------------------------------------
    print(
        "\nSIMULATION: MPC "
        "(hold-padded horizon, fixed GN budget)"
    )

    q_mpc, cycles = simulate_mpc(
        robot, chunks, times, args
    )

    # --------------------------------------------------------
    # Nominal reference
    # --------------------------------------------------------
    nominal_q = nominal_state(times)[0]

    nominal_poses = [
        pose_from_theta(robot, q)
        for q in nominal_q
    ]

    nominal_xyz = np.asarray([
        pose.translation.copy()
        for pose in nominal_poses
    ])

    nominal_rot = np.asarray([
        pose.rotation.copy()
        for pose in nominal_poses
    ])

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    raw_metrics, raw_d = metric_summary(
        q_raw,
        args.sample_dt,
        robot,
        nominal_xyz,
        nominal_rot,
        "RAW",
    )

    mpc_metrics, mpc_d = metric_summary(
        q_mpc,
        args.sample_dt,
        robot,
        nominal_xyz,
        nominal_rot,
        "MPC",
    )

    metrics = [raw_metrics, mpc_metrics]

    print(
        "\nFINAL BENCHMARK "
        "(same execution window)"
    )

    for row in metrics:
        print(
            f"{row['method']:4s} | "
            f"position={row['position_rms_mm']:7.2f} mm | "
            f"accel RMS="
            f"{row['acceleration_rms_rad_s2']:8.2f} rad/s² | "
            f"jerk RMS="
            f"{row['jerk_rms_rad_s3']:9.2f} rad/s³ | "
            f"max a violation="
            f"{row['max_a_violation']:.2e}"
        )

    durations = np.asarray([
        row["solve_ms"] for row in cycles
    ])

    print(
        f"MPC solve time: "
        f"mean={np.mean(durations):.2f} ms | "
        f"p95={np.percentile(durations, 95):.2f} ms | "
        f"deadline misses="
        f"{int(np.sum(durations > 1000*args.dt))}"
    )

    print(
        "\nNOTE: Policy inference latency is simulated, "
        "not measured."
    )
    print(
        "NOTE: RAW online timings exclude precomputed IK."
    )
    print(
        "NOTE: Acceleration and jerk are finite differences "
        "of commanded positions."
    )

    # --------------------------------------------------------
    # Exports
    # --------------------------------------------------------
    save_csv(
        args.output / "summary.csv",
        metrics,
    )
    save_csv(
        args.output / "mpc_cycles.csv",
        cycles,
    )

    np.savez(
        args.output / "trajectories.npz",
        t=times,
        raw_q=q_raw,
        mpc_q=q_mpc,
        raw_v=raw_d[0],
        mpc_v=mpc_d[0],
        raw_a=raw_d[1],
        mpc_a=mpc_d[1],
        raw_jerk=raw_d[2],
        mpc_jerk=mpc_d[2],
        raw_hold=waiting,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plot_results(
        times,
        chunks,
        q_raw,
        q_mpc,
        raw_d,
        mpc_d,
        metrics,
        cycles,
        args,
    )

    print(
        "\nSaved results and six figures to: "
        f"{args.output.resolve()}"
    )

    # --------------------------------------------------------
    # Interactive Meshcat
    # --------------------------------------------------------
    if args.show_plots:
        plt.show(block=False)

        # Force Matplotlib to process pending drawing events
        # before entering the interactive Meshcat menu.
        for number in plt.get_fignums():
            fig = plt.figure(number)
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.pause(0.2)

    if not args.no_animation:
        animate_results(
            robot,
            times,
            q_raw,
            q_mpc,
            nominal_xyz,
            nominal_poses,
            args,
        )

    elif args.show_plots:
        plt.show()

    else:
        plt.close("all")


if __name__ == "__main__":
    main()

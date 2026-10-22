
"""Kinova Gen3: noisy Cartesian policy vs constrained B-spline MPC.

- Policy emits 16-pose chunks every 1.5 simulated seconds.
- After each chunk (except the preloaded first), the next
  chunk is available after --inference-latency (default 20 ms).
- RAW holds its final commanded joint position during this delay.
- MPC keeps executing its latest B-spline.
- MPC targets beyond the available chunk are hold-padded.
- MPC optimization period: 100 ms.
- Motion visualization sampling: 10 ms.

Inference delay is SIMULATED, not a measured model latency.
RAW IK is precomputed and excluded from online timings.

Run:
    python -m examples.kinova_policy_mpc --gn-iters 2 --show-plots
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

from examples.kinova_mpc import (
    make_spline,
    pose_from_theta,
)

from examples.kinova_trajectory_projection import (
    URDF_PATH,
    Q_MIN,
    Q_MAX,
    V_MIN,
    V_MAX,
    A_MIN,
    A_MAX,
)

from examples.mock_policy import (
    DURATION,
    WAYPOINT_TIMES,
    nominal_state,
    nominal_diagnostics,
    normalized_usage,
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
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return float("nan")

    return float(np.sqrt(np.mean(a*a)))


def pose_error(a, b):
    position = float(np.linalg.norm(
        a.translation - b.translation
    ))

    orientation = float(np.linalg.norm(
        pin.log3(a.rotation.T @ b.rotation)
    ))

    return position, orientation


def max_violation(x, lo, hi):
    return float(np.max(
        np.maximum(
            np.maximum(lo - x, x - hi),
            0.0,
        )
    ))


def linear_violation(problem, c):
    A, lo, hi = problem.linear_constraints()

    values = np.asarray(
        A @ np.asarray(c).reshape(-1)
    ).reshape(-1)

    return float(max(
        0.0,
        np.max(lo - values),
        np.max(values - hi),
    ))


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def derivatives(q, h):
    """Discrete derivatives of sampled commanded positions.

    The same convention is used for RAW and MPC.

    These describe the commanded motion, not motor dynamics.
    RAW hold/release boundaries can cause very large peaks.
    """
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    j = np.zeros_like(q)

    v[1:] = np.diff(q, axis=0) / h

    a[2:] = (
        np.diff(v[1:], axis=0) / h
    )

    j[3:] = (
        np.diff(a[2:], axis=0) / h
    )

    return v, a, j


# ============================================================
# IK preprocessing
# ============================================================

def precompute_ik(robot, chunks, args):
    print(
        "\nPRECOMPUTING RAW IK "
        "(offline; not counted in online time)"
    )

    all_q = []

    seed = nominal_state(0.0)[0].copy()
    failures = 0

    for k, chunk in enumerate(chunks):
        t0 = perf_counter()
        q_chunk = []

        for target in chunk.poses:
            result = solve_pose_ik(
                robot=robot,
                target_pose=target,
                q0=robot.joint_positions_to_configuration(
                    seed
                ),
                tolerance=args.ik_tolerance,
                max_iterations=args.ik_max_iterations,
                step_size=0.2,
                damping=1e-5,
            )

            if result.success:
                q = (
                    robot.configuration_to_joint_positions(
                        result.q
                    )
                )

                q = np.asarray(
                    q, dtype=float
                ).reshape(-1)

                # Keep continuous joints on the closest
                # equivalent angular branch.
                q = (
                    seed
                    + (q - seed + np.pi)
                    % (2*np.pi)
                    - np.pi
                )

                seed = q.copy()

            else:
                failures += 1
                q = seed.copy()

            q_chunk.append(q)

        all_q.append(
            np.asarray(q_chunk)
        )

        elapsed_ms = (
            1000 * (perf_counter() - t0)
        )

        print(
            f"  chunk {k+1:02d}/{len(chunks)}: "
            f"{elapsed_ms:.1f} ms | "
            f"IK failures so far: {failures}"
        )

    if failures:
        raise RuntimeError(
            f"RAW has {failures} IK failures: "
            "benchmark invalid"
        )

    return all_q


# ============================================================
# RAW execution
# ============================================================

def raw_motion(
    chunks,
    ik_chunks,
    times,
    latency,
    q_initial,
):
    """Generate RAW commanded positions at high resolution.

    At a chunk boundary:
      - finish the old chunk;
      - hold its final position during inference;
      - resume following the new predictions when available.

    The new chunk's first segment is interpolated from the
    held position to its second prediction. This avoids
    introducing an artificial position teleport on release.

    After that, RAW follows successive noisy IK predictions.
    """

    q_values = np.empty(
        (len(times), len(q_initial))
    )

    waiting = np.zeros(
        len(times),
        dtype=bool,
    )

    dt = chunks[0].dt

    boundaries = np.array([
        c.start_time for c in chunks
    ])

    for n, t in enumerate(times):
        # Use the nominal boundary, even if the next
        # chunk is not yet available.
        i = int(np.clip(
            np.searchsorted(
                boundaries,
                t + 1e-9,
                side="right",
            ) - 1,
            0,
            len(chunks) - 1,
        ))

        chunk = chunks[i]
        q_chunk = ik_chunks[i]

        tau = t - chunk.start_time

        # ------------------------------------------------
        # First chunk: start from actual initial q
        # ------------------------------------------------

        if i == 0:
            if tau <= dt:
                alpha = np.clip(
                    tau / dt,
                    0.0,
                    1.0,
                )

                q_values[n] = (
                    q_initial
                    + alpha
                    * (q_chunk[1] - q_initial)
                )
                continue

        else:
            # --------------------------------------------
            # The last point of the previous chunk.
            # --------------------------------------------
            prev_last = ik_chunks[i - 1][-1]

            # --------------------------------------------
            # Policy inference: no new command.
            # Hold last position exactly.
            # --------------------------------------------
            if tau < latency - 1e-9:
                q_values[n] = prev_last
                waiting[n] = True
                continue

            # --------------------------------------------
            # The new chunk has arrived.
            #
            # Resume smoothly in POSITION from the hold
            # value to the next predicted waypoint.
            #
            # Velocity may still jump at release.
            # This is a naive raw-command baseline.
            # --------------------------------------------
            if tau <= dt + 1e-9:
                alpha = np.clip(
                    (tau - latency) / (dt - latency),
                    0.0,
                    1.0,
                )

                q_values[n] = (
                    prev_last
                    + alpha
                    * (q_chunk[1] - prev_last)
                )
                continue

        # ------------------------------------------------
        # Interior of an available chunk:
        # direct noisy waypoint tracking.
        # ------------------------------------------------

        x = np.clip(
            tau / dt,
            0.0,
            len(q_chunk) - 1,
        )

        j = min(
            int(np.floor(x)),
            len(q_chunk) - 2,
        )

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

    tracking_grid = np.linspace(
        0.0,
        args.horizon,
        args.tracking_points,
    )

    q0 = nominal_state(0.0)[0]

    problem = CartesianTrajectoryProblem(
        robot=robot,
        spline=spline,
        tracking_times=tracking_grid,
        target_poses=chunks[0].horizon(
            0.0, tracking_grid
        ),
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
        np.tile(
            q0,
            (spline.n_control_points, 1),
        ),
        project_warm_start=True,
        projection_tolerance=args.projection_tol,
        linear_feasibility_tolerance=(
            args.linear_feasibility_tol
        ),
    )

    return spline, tracking_grid, problem, mpc


# ============================================================
# MPC execution
# ============================================================

def simulate_mpc(robot, chunks, times, args):
    """
    MPC with delayed chunk availability.

    At a chunk boundary, continue executing the previous
    B-spline during inference, then replan immediately
    when the new chunk becomes available.

    Regular replanning: every args.dt.
    Execution sampling: every args.sample_dt.
    """
    spline, grid, problem, mpc = create_mpc(
        robot, chunks, args
    )

    n_cycles = int(round(DURATION / args.dt))

    regular_times = (
        np.arange(n_cycles, dtype=float) * args.dt
    )

    boundaries = np.array(
        [chunk.start_time for chunk in chunks[1:]],
        dtype=float,
    )

    # Replace each solve exactly at a chunk boundary
    # with a solve after the inference latency.
    if args.inference_latency > 0.0 and boundaries.size:
        at_boundary = np.any(
            np.isclose(
                regular_times[:, None],
                boundaries[None, :],
                atol=1e-9,
                rtol=0.0,
            ),
            axis=1,
        )

        delayed_times = (
            boundaries + args.inference_latency
        )

        solve_times = np.sort(
            np.concatenate([
                regular_times[~at_boundary],
                delayed_times,
            ])
        )
    else:
        solve_times = regular_times

    q_dense = np.full(
        (len(times), robot.nv),
        np.nan,
    )

    q_initial = nominal_state(0.0)[0].copy()
    q_dense[0] = q_initial

    previous_c = None
    previous_solve_time = None

    rows = []
    fallback_count = 0

    for k, t in enumerate(solve_times):
        t = float(t)

        # Determine the actual state from the spline
        # that was being executed immediately before t.
        if previous_c is None:
            q = q_initial.copy()
            v = np.zeros(robot.nv)
        else:
            elapsed = t - previous_solve_time

            q = np.asarray(
                spline.position(previous_c, elapsed),
                dtype=float,
            ).reshape(-1)

            v = np.asarray(
                spline.velocity(previous_c, elapsed),
                dtype=float,
            ).reshape(-1)

        # At t=1.52, chunk 2 is now available.
        idx = available_chunk_index(
            chunks,
            t,
            args.inference_latency,
        )

        targets = chunks[idx].horizon(t, grid)

        start = perf_counter()

        result = mpc.step(
            q0=q,
            v0=v,
            target_poses=targets,
        )

        solve_ms = 1000.0 * (
            perf_counter() - start
        )

        info = mpc.last_solution_info or {}
        projection = mpc.last_projection_info or {}

        if not info.get("accepted", False):
            raise RuntimeError(
                f"MPC rejected solve {k+1} "
                f"at t={t:.3f}s: "
                f"status={getattr(result, 'status', '?')}, "
                f"linear={info.get('linear_violation')}, "
                f"projection={projection}"
            )

        c = np.asarray(
            result.control_points,
            dtype=float,
        ).reshape(
            spline.n_control_points,
            robot.nv,
        )

        violation = linear_violation(
            problem, c
        )

        if (
            not np.isfinite(violation)
            or violation > args.linear_feasibility_tol
        ):
            raise RuntimeError(
                "Selected MPC trajectory infeasible "
                f"at t={t:.3f}s: "
                f"{violation:.3e}"
            )

        # Execute this spline until the next solve.
        #
        # Across a chunk boundary, the interval can be
        # 120 ms instead of 100 ms. No hold or artificial
        # position discontinuity is introduced.
        next_t = (
            float(solve_times[k + 1])
            if k + 1 < len(solve_times)
            else DURATION
        )

        i_start = int(round(
            t / args.sample_dt
        ))

        i_stop = int(round(
            next_t / args.sample_dt
        ))

        for i in range(i_start, i_stop + 1):
            tau = float(times[i] - t)

            # Correct floating-point rounding near spline boundaries.
            tau = float(np.clip(tau, 0.0, args.horizon))

            q_dense[i] = np.asarray(
                spline.position(c, tau),
                dtype=float,
            ).reshape(-1)

        previous_c = c.copy()
        previous_solve_time = t

        source = info.get(
            "solution_source", "gn"
        )

        fallback_count += (
            source == "projected_warm_start"
        )

        rows.append({
            "cycle": k + 1,
            "time_s": round(t, 5),
            "chunk_used": idx + 1,
            "chunk_ready": int(
                idx > 0
                and t >= (
                    chunks[idx].start_time
                    + args.inference_latency
                    - 1e-9
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

        if (
            args.print_every > 0
            and (
                (k + 1) % args.print_every == 0
                or k == 0
            )
        ):
            print(
                f"[MPC {k+1:03d}/{len(solve_times)}] "
                f"t={t:5.2f}s "
                f"chunk={idx+1:02d} | "
                f"{solve_ms:6.2f} ms | "
                f"source={source} | "
                f"linear={violation:.1e}"
            )

    if not np.all(np.isfinite(q_dense)):
        raise RuntimeError(
            "MPC execution timeline contains "
            "missing samples"
        )

    print(
        f"MPC complete: {len(solve_times)} solves; "
        "projected warm-start fallback="
        f"{fallback_count}"
    )

    return q_dense, rows

# ============================================================
# Evaluation metrics
# ============================================================


def metric_summary(
    q,
    h,
    robot,
    nominal_xyz,
    nominal_rot,
    mode,
):
    """Compute metrics over the SAME execution timeline."""

    v, a, j = derivatives(q, h)

    ep = np.empty(len(q))
    eo = np.empty(len(q))

    for i, qi in enumerate(q):
        pose = pose_from_theta(
            robot, qi
        )

        ep[i] = (
            np.linalg.norm(
                pose.translation
                - nominal_xyz[i]
            )
            * 1000
        )

        eo[i] = np.linalg.norm(
            pin.log3(
                pose.rotation.T
                @ nominal_rot[i]
            )
        )

    accel_violation = np.maximum(
        np.maximum(
            A_MIN - a[2:],
            a[2:] - A_MAX,
        ),
        0.0,
    )

    summary = {
        "method": mode,
        "position_rms_mm": rms(ep),
        "orientation_rms_deg": float(
            np.rad2deg(rms(eo))
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
                np.any(
                    accel_violation > 1e-6,
                    axis=1,
                )
            ))
        ),
    }

    histories = (
        v, a, j, ep, eo
    )

    return summary, histories


# ============================================================
# Plotting
# ============================================================

def inference_bands(
    ax,
    chunks,
    latency,
    label_first=True,
    alpha=0.2,
):
    """Shade all simulated policy-inference periods."""

    for i, chunk in enumerate(chunks[1:]):
        ax.axvspan(
            chunk.start_time,
            chunk.start_time + latency,
            facecolor=C_WAIT,
            alpha=alpha,
            label=(
                f"Policy inference ({1000*latency:.0f} ms)"
                if label_first and i == 0
                else None
            ),
        )


def plot_results(
    times,
    chunks,
    q_raw,
    q_mpc,
    raw_d,
    mpc_d,
    metrics,
    cycles,
    args,
):
    out = args.output

    vr, ar, jr, ep_raw, _ = raw_d
    vm, am, jm, ep_mpc, _ = mpc_d

    # ========================================================
    # Figure 1: global smoothness
    # ========================================================

    fig, axes = plt.subplots(
        3, 1,
        figsize=(15, 10),
        sharex=True,
    )

    panels = [
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
    ]

    for ax, (raw, mpc, label) in zip(
        axes, panels
    ):
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
            ax, chunks,
            args.inference_latency,
        )

        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    # Symmetric-log scale keeps smaller MPC variations
    # visible despite large RAW peaks.
    axes[1].set_yscale(
        "symlog",
        linthresh=2.0,
    )

    axes[2].set_yscale(
        "symlog",
        linthresh=10.0,
    )

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        "Motion smoothness - 10 ms execution samples"
    )

    fig.tight_layout()
    fig.savefig(
        out / "smoothness_raw_vs_mpc.png",
        dpi=170,
    )

    # ========================================================
    # Figure 2: inference hold zooms
    # ========================================================

    boundaries = [
        c.start_time
        for c in chunks[1:4]
    ]

    fig, axes = plt.subplots(
        len(boundaries),
        3,
        figsize=(18, 10),
    )

    joint = args.zoom_joint - 1

    for i, b in enumerate(boundaries):
        mask = (
            (times >= b - 0.12)
            & (times <= b + 0.20)
        )

        datasets = [
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
        ]

        for j, (raw, mpc, unit) in enumerate(datasets):
            ax = axes[i, j]

            ax.plot(
                times[mask],
                raw[mask],
                color=C_RAW,
                lw=1.8,
                marker=".",
                ms=2,
                label="RAW" if i == 0 else None,
            )

            ax.plot(
                times[mask],
                mpc[mask],
                color=C_MPC,
                lw=2,
                label="MPC" if i == 0 else None,
            )

            # Inference period.
            ax.axvspan(
                b,
                b + args.inference_latency,
                facecolor=C_WAIT,
                alpha=0.35,
            )

            # Chunk boundary.
            ax.axvline(
                b,
                color="gray",
                ls=":",
                lw=1,
            )

            # New chunk ready.
            ax.axvline(
                b + args.inference_latency,
                color="#ae7f28",
                ls="--",
                lw=1,
            )

            ax.set_ylabel(unit)

            if i == 0:
                ax.set_title(
                    f"Boundary {b:.2f}s "
                    f"- joint {args.zoom_joint}"
                )
            else:
                ax.set_title(
                    f"Boundary {b:.2f}s"
                )

            ax.grid(alpha=0.2)

            if i == 0:
                ax.legend(loc="best")

            if i == len(boundaries) - 1:
                ax.set_xlabel("Time [s]")

    fig.suptitle(
        "Policy inference: RAW holds position; "
        "MPC keeps moving"
    )

    fig.tight_layout()
    fig.savefig(
        out / "inference_hold_zooms.png",
        dpi=190,
    )

    # ========================================================
    # Figure 3: Cartesian tracking
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 4.2)
    )

    ax.plot(
        times,
        ep_raw,
        color=C_RAW,
        lw=1.0,
        label="RAW",
    )

    ax.plot(
        times,
        ep_mpc,
        color=C_MPC,
        lw=1.5,
        label="MPC",
    )

    inference_bands(
        ax, chunks,
        args.inference_latency,
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

    fig.tight_layout()
    fig.savefig(
        out / "cartesian_errors.png",
        dpi=170,
    )

    # ========================================================
    # Figure 4: joint comparison
    # ========================================================

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 7),
        sharex=True,
    )

    datasets = [
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
    ]

    for ax, (raw, mpc, unit) in zip(
        axes, datasets
    ):
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
            ax, chunks,
            args.inference_latency,
        )

        ax.set_ylabel(unit)
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        f"Joint {joint+1}: noisy commands "
        "versus B-spline MPC"
    )

    fig.tight_layout()
    fig.savefig(
        out / "joint_comparison.png",
        dpi=170,
    )

    # ========================================================
    # Figure 5: measured MPC time / simulated latency
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 4.5)
    )

    ct = np.array([
        r["time_s"] for r in cycles
    ])

    ms = np.array([
        r["solve_ms"] for r in cycles
    ])

    ax.plot(
        ct,
        ms,
        color=C_MPC,
        marker=".",
        ms=3,
        lw=1,
        label="MPC solve time (measured)",
    )

    ax.axhline(
        args.inference_latency * 1000,
        color="#a77d26",
        ls="--",
        label=(
            "Policy latency (simulated): "
            f"{args.inference_latency*1000:.0f} ms"
        ),
    )

    ax.axhline(
        args.dt * 1000,
        color="black",
        ls=":",
        label=(
            f"MPC period: {args.dt*1000:.0f} ms"
        ),
    )

    inference_bands(
        ax, chunks,
        args.inference_latency,
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

    fig.tight_layout()
    fig.savefig(
        out / "timing_and_inference.png",
        dpi=170,
    )

    # ========================================================
    # Figure 6: quantitative summary
    # ========================================================

    names = [
        row["method"] for row in metrics
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),
    )

    keys = [
        "position_rms_mm",
        "acceleration_rms_rad_s2",
        "jerk_rms_rad_s3",
    ]

    titles = [
        "Position RMS [mm]",
        "Acceleration RMS [rad/s²]",
        "Jerk RMS [rad/s³]",
    ]

    for ax, key, title in zip(
        axes, keys, titles
    ):
        vals = [
            row[key] for row in metrics
        ]

        bars = ax.bar(
            names,
            vals,
            color=[C_RAW, C_MPC],
            width=0.55,
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
        )

        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Same 20.7 s execution window "
        "for both methods"
    )

    fig.tight_layout()
    fig.savefig(
        out / "benchmark_summary.png",
        dpi=170,
    )

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # MPC and execution sampling.
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--inference-latency",
        type=float,
        default=0.02,
        help="Simulated waiting time after each chunk [s]",
    )

    parser.add_argument(
        "--control-points",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--tracking-points",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gn-iters",
        type=int,
        default=2,
    )

    # Mock policy noise.
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0025,
        help="Independent Cartesian noise [m]",
    )

    parser.add_argument(
        "--bias-std",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--tremor-amplitude",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--tremor-frequency",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # MPC weights.
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

    # IK.
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

    # Display and output.
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
        "--output",
        type=Path,
        default=Path(
            "outputs/kinova_policy_mpc"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if args.dt <= 0 or args.sample_dt <= 0:
        parser.error(
            "dt and sample-dt must be positive"
        )

    ratio = args.dt / args.sample_dt

    if abs(ratio - round(ratio)) > 1e-8:
        parser.error(
            "dt must be a multiple of sample-dt"
        )

    if (
        args.inference_latency < 0
        or args.inference_latency >= args.dt
    ):
        parser.error(
            "inference latency must lie "
            "between 0 and dt"
        )

    latency_ratio = (
        args.inference_latency / args.sample_dt
    )

    if abs(
        latency_ratio - round(latency_ratio)
    ) > 1e-8:
        parser.error(
            "inference latency must be "
            "a multiple of sample-dt"
        )

    if (
        args.chunk_size < 2
        or not (1 <= args.zoom_joint <= 7)
    ):
        parser.error(
            "invalid chunk-size or zoom-joint"
        )

    chunk_duration = (
        args.chunk_size - 1
    ) * args.dt

    if abs(
        args.horizon - chunk_duration
    ) > 1e-8:
        parser.error(
            "horizon must equal "
            "(chunk-size-1)*dt"
        )

    duration_ratio = DURATION / args.dt

    if abs(
        duration_ratio - round(duration_ratio)
    ) > 1e-8:
        parser.error(
            "simulation duration must "
            "be a multiple of dt"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
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
            f"Expected nv=7, got {robot.nv}"
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("=" * 85)
    print(
        "KINOVA GEN3 - NOISY POLICY vs B-SPLINE MPC"
    )
    print(
        "SIMULATED POLICY INFERENCE LATENCY"
    )
    print("=" * 85)

    print(
        f"Duration={DURATION:.2f}s | "
        f"MPC dt={args.dt*1000:.0f}ms | "
        f"sample dt={args.sample_dt*1000:.0f}ms | "
        f"simulated inference="
        f"{args.inference_latency*1000:.0f}ms"
    )

    print(
        f"Noise: independent="
        f"{1000*args.noise_std:.1f}mm | "
        f"bias={1000*args.bias_std:.1f}mm | "
        f"tremor amplitude="
        f"{1000*args.tremor_amplitude:.1f}mm"
    )

    d = nominal_diagnostics()

    print(
        f"Nominal feasibility: "
        f"q={d['max_q_violation']:.2e} | "
        f"v={d['max_v_violation']:.2e} | "
        f"a={d['max_a_violation']:.2e}; "
        f"accel violating time="
        f"{d['accel_violation_pct']:.2f}%"
    )

    # --------------------------------------------------------
    # Pre-generate identical policy chunks
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
        policy,
        duration=DURATION,
    )

    print(
        f"Generated {len(chunks)} identical "
        "chunks for both methods"
    )

    for i, chunk in enumerate(chunks):
        diff = np.array([
            p.translation - n.translation
            for p, n in zip(
                chunk.poses,
                chunk.nominal_poses,
            )
        ])

        available_time = (
            chunk.start_time
            + (
                0.0 if i == 0
                else args.inference_latency
            )
        )

        print(
            f"  chunk {i+1:02d} | "
            f"time {chunk.start_time:5.2f}-"
            f"{chunk.end_time:5.2f}s | "
            f"available at "
            f"{available_time:5.2f}s | "
            f"prediction noise RMS="
            f"{1000*rms(np.linalg.norm(diff, axis=1)):.2f}mm"
        )

    # --------------------------------------------------------
    # RAW preprocessing
    # --------------------------------------------------------

    ik_chunks = precompute_ik(
        robot, chunks, args
    )

    # --------------------------------------------------------
    # Common high-resolution execution timeline
    # --------------------------------------------------------

    n_samples = int(round(
        DURATION / args.sample_dt
    )) + 1

    times = np.linspace(
        0.0,
        DURATION,
        n_samples,
    )

    q_initial = nominal_state(
        0.0
    )[0].copy()

    # --------------------------------------------------------
    # RAW simulation
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
        f"RAW hold samples: "
        f"{int(np.sum(waiting))} "
        f"({1000*args.sample_dt:.0f}ms each)"
    )

    # --------------------------------------------------------
    # MPC simulation
    # --------------------------------------------------------

    print(
        "\nSIMULATION: MPC "
        "(hold-padded horizon; fixed GN budget)"
    )

    q_mpc, cycles = simulate_mpc(
        robot,
        chunks,
        times,
        args,
    )

    # --------------------------------------------------------
    # Ideal reference FK, evaluated at identical times
    # --------------------------------------------------------

    nominal_q = nominal_state(
        times
    )[0]

    nominal_poses = [
        pose_from_theta(robot, q)
        for q in nominal_q
    ]

    nom_xyz = np.asarray([
        pose.translation.copy()
        for pose in nominal_poses
    ])

    nom_rot = np.asarray([
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
        nom_xyz,
        nom_rot,
        "RAW",
    )

    mpc_metrics, mpc_d = metric_summary(
        q_mpc,
        args.sample_dt,
        robot,
        nom_xyz,
        nom_rot,
        "MPC",
    )

    metrics = [
        raw_metrics,
        mpc_metrics,
    ]

    print(
        "\nFINAL BENCHMARK "
        "(same execution window, "
        "10-ms derivatives)"
    )

    for s in metrics:
        print(
            f"{s['method']:4s} | "
            f"position="
            f"{s['position_rms_mm']:7.2f}mm | "
            f"accel RMS="
            f"{s['acceleration_rms_rad_s2']:8.2f}rad/s² | "
            f"jerk RMS="
            f"{s['jerk_rms_rad_s3']:9.2f}rad/s³ | "
            f"a viol="
            f"{s['max_a_violation']:.2e}"
        )

    durations = np.array([
        row["solve_ms"]
        for row in cycles
    ])

    print(
        f"MPC solve time "
        f"mean={np.mean(durations):.2f}ms | "
        f"p95={np.percentile(durations, 95):.2f}ms | "
        f"deadline misses="
        f"{int(np.sum(durations > 1000*args.dt))}"
    )

    print(
        "NOTE: policy inference latency is "
        "SIMULATED (not measured)."
    )

    print(
        "NOTE: RAW online timings exclude "
        "precomputed IK."
    )

    print(
        "NOTE: acceleration/jerk are finite "
        "differences of commanded positions."
    )

    print(
        "RAW hold/release transitions "
        "can produce high peaks."
    )

    # --------------------------------------------------------
    # Save CSV and trajectories
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
    # Plot
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


if __name__ == "__main__":
    main()

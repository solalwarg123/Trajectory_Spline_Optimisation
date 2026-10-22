
"""
Kinova Gen3 — Static-obstacle Cartesian MPC benchmark.

Features
--------
- Automatic static obstacle placement near the nominal motion.
- Per-obstacle activation handled inside CartesianMPC.
- GN / IPOPT, multiple iteration budgets.
- Optional Gaussian acceleration disturbances.
- Independent closed-loop runs with identical disturbances.
- Collision validation on a dense grid.
- Computation-time and feasibility plots.
- 3D end-effector and capsule trajectories.
- CSV exports.

Examples
--------
python -m examples.kinova_mpc_obstacle_benchmark \
    --solver gn --limits 5 10 --steps 100 --show-plots

python -m examples.kinova_mpc_obstacle_benchmark \
    --solver gn --limits 5 10 --noise-std 0.05

python -m examples.kinova_mpc_obstacle_benchmark \
    --solver both --limits 20 50

The activation filtering is implemented in splineopt.mpc,
not in this example.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from splineopt.robotics import Robot
from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.ipopt_solver import IpoptSolver
from splineopt.mpc import CartesianMPC

from splineopt.collision import (
    SphereObstacle,
    CollisionEnvironment,
    CollisionConstraintBuilder,
)

from examples.kinova_mpc import (
    make_spline,
    pose_from_theta,
    pose_errors,
    enforce_initial_state_on_guess,
    max_linear_violation,
)

from examples.kinova_mpc_safe_tracking import (
    DURATION,
    reference_state,
    reference_pose,
    validate_reference,
    bound_violation,
)

from examples.kinova_collision_optimization import (
    KINOVA_CAPSULES,
    capsule_endpoints,
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


# ============================================================
# General utilities
# ============================================================

def rms(values):
    x = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def unit_vector(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise ValueError("Zero-length direction")
    return x / norm


def make_environment(robot, obstacle):
    return CollisionEnvironment(
        robot=robot,
        capsules=KINOVA_CAPSULES,
        obstacles=[obstacle],
    )


def clearance(environment, q):
    distances, _ = environment.evaluate(
        q,
        compute_jacobian=False,
    )
    return float(np.min(distances))


def reference_clearances(robot, obstacle, n=301):
    env = make_environment(robot, obstacle)

    times = np.linspace(0, DURATION, n)
    distances = np.empty(n)

    for i, t in enumerate(times):
        q, _, _ = reference_state(t)
        distances[i] = clearance(env, q)

    return times, distances


def get_capsule(name):
    for capsule in KINOVA_CAPSULES:
        if capsule.name == name:
            return capsule
    raise ValueError(f"Unknown capsule: {name}")


def capsule_midpoint(robot, q, capsule):
    p0, p1 = capsule_endpoints(robot, q, capsule)
    return 0.5 * (p0 + p1)


# ============================================================
# Obstacle placement
# ============================================================

def create_obstacle(robot, args):
    """
    Place one static sphere close to the nominal robot motion.

    We search along several fixed spatial directions and choose
    a position achieving the requested nominal clearance.

    This is a deterministic scenario generator, not a planner.
    """

    target = args.target_nominal_clearance

    if args.obstacle_time > DURATION:
        raise ValueError(
            "Obstacle time exceeds reference duration"
        )

    capsule = get_capsule(args.obstacle_capsule)

    q_anchor, _, _ = reference_state(args.obstacle_time)
    q_initial, _, _ = reference_state(0.0)

    anchor = capsule_midpoint(
        robot, q_anchor, capsule
    )

    directions = [
        unit_vector([1, 0, 0]),
        unit_vector([-1, 0, 0]),
        unit_vector([0, 1, 0]),
        unit_vector([0, -1, 0]),
        unit_vector([0, 0, 1]),
        unit_vector([0, 0, -1]),
    ]

    best = None

    # Precompute nominal robot configurations for the
    # scenario selection, then evaluate obstacle geometry.
    reference_q = [
        reference_state(t)[0]
        for t in np.linspace(
            0, DURATION, args.placement_samples
        )
    ]

    def evaluate_candidate(center):
        obstacle = SphereObstacle(
            name="static_obstacle",
            center=np.asarray(center, dtype=float),
            radius=args.obstacle_radius,
        )

        env = make_environment(robot, obstacle)

        min_nominal = min(
            clearance(env, q) for q in reference_q
        )

        initial = clearance(env, q_initial)

        return obstacle, min_nominal, initial

    # Scan first; do not assume clearance is globally
    # monotonic with obstacle displacement.
    shifts = np.linspace(
        0.0, args.max_obstacle_shift, 36
    )

    for direction in directions:

        samples = []

        for shift in shifts:
            center = anchor + shift * direction

            obstacle, d_nom, d_initial = (
                evaluate_candidate(center)
            )

            samples.append(
                (shift, d_nom, d_initial, obstacle)
            )

        # Find a local bracket crossing the target clearance.
        for left, right in zip(
            samples[:-1], samples[1:]
        ):
            f_left = left[1] - target
            f_right = right[1] - target

            if f_left * f_right > 0:
                continue

            lo, hi = left[0], right[0]
            f_lo = f_left

            selected = None

            for _ in range(22):
                mid = 0.5 * (lo + hi)

                obs, d_nom, d_initial = (
                    evaluate_candidate(
                        anchor + mid * direction
                    )
                )

                selected = (
                    obs, d_nom, d_initial, mid
                )

                f_mid = d_nom - target

                if f_mid * f_lo <= 0:
                    hi = mid
                else:
                    lo = mid
                    f_lo = f_mid

            obs, d_nom, d_initial, shift = selected

            if d_initial < (
                args.safety_distance
                + args.initial_clearance_buffer
            ):
                continue

            error = abs(d_nom - target)

            # Prefer an accurate target and a generous
            # clearance at the initial configuration.
            score = (
                error
                - 0.001 * d_initial
            )

            if best is None or score < best["score"]:
                best = {
                    "obstacle": obs,
                    "nominal": d_nom,
                    "initial": d_initial,
                    "shift": shift,
                    "direction": direction.copy(),
                    "score": score,
                }

    if best is None:
        raise RuntimeError(
            "Could not position an obstacle with the "
            "requested clearance and a safe initial state. "
            "Try another obstacle time, capsule, radius "
            "or larger --max-obstacle-shift."
        )

    return best["obstacle"], best


# ============================================================
# Solver construction
# ============================================================

def make_solver(name, limit):

    if name == "GN":
        return GaussNewtonSolver(
            damping_mode="adaptive",
            damping=1e-3,
            max_iterations=limit,
            collision_slack_weight=1e4,
            collision_merit_weight=1e4,
        )

    if name == "IPOPT":
        return IpoptSolver(
            tolerance=1e-5,
            max_iterations=limit,
            print_level=0,
        )

    raise ValueError(name)


def acceptable_status(result, solver_name):

    if result.success:
        return True

    if solver_name == "GN":
        return result.status in (
            "max_iterations",
            "damping_limit",
        )

    return result.status == -1


# ============================================================
# Disturbances
# ============================================================

def generate_noise(args, nv):
    rng = np.random.default_rng(args.seed)

    return rng.normal(
        0.0,
        args.noise_std,
        size=(args.steps, nv),
    )


# ============================================================
# One MPC simulation
# ============================================================

def run_experiment(
    robot,
    obstacle,
    args,
    solver_name,
    limit,
    disturbances,
):

    name = f"{solver_name}-{limit}"

    print("\n" + "=" * 76)
    print(f"Experiment: {name}")
    print("=" * 76)

    spline = make_spline(
        args.horizon,
        robot.nv,
        args.control_points,
    )

    tracking_times = np.linspace(
        0.0,
        args.horizon,
        args.tracking_points,
    )

    collision_times = np.linspace(
        0.0,
        args.horizon,
        args.collision_points,
    )

    validation_times = np.linspace(
        0.0,
        args.horizon,
        args.validation_points,
    )

    environment = make_environment(
        robot, obstacle
    )

    collision = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=collision_times,
        safety_distance=args.safety_distance,
    )

    dense_collision = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=validation_times,
        safety_distance=args.safety_distance,
    )

    q, v, _ = reference_state(0.0)

    c0 = np.tile(
        q,
        (spline.n_control_points, 1),
    )

    problem = CartesianTrajectoryProblem(
        robot=robot,
        spline=spline,
        tracking_times=tracking_times,
        target_poses=[
            reference_pose(robot, t)
            for t in tracking_times
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
        collision_constraints=collision,
    )

    mpc = CartesianMPC(
        problem=problem,
        solver=make_solver(solver_name, limit),
        initial_control_points=c0,
        collision_activation_distance=(
            args.activation_distance
        ),
        collision_check_samples=(
            args.activation_samples
        ),
        collision_feasibility_tolerance=(
            args.feasibility_tol
        ),
    )

    capsule = get_capsule(args.obstacle_capsule)

    records = []

    ee_path = [
        pose_from_theta(
            robot, q
        ).translation.copy()
    ]

    capsule_path = [
        capsule_midpoint(
            robot, q, capsule
        )
    ]

    q_path = [q.copy()]

    failure = None

    for k in range(args.steps):

        t = k * args.dt
        t_next = (k + 1) * args.dt

        targets = [
            reference_pose(robot, t + tau)
            for tau in tracking_times
        ]

        # Same warm-start correction as existing MPC
        # examples. We do not change this policy here.
        mpc.previous_control_points = (
            enforce_initial_state_on_guess(
                spline,
                mpc.previous_control_points,
                q,
                v,
            )
        )

        row = {
            "cycle": k + 1,
            "time_s": t_next,
            "compute_ms": np.nan,
            "iterations": np.nan,
            "status": "not_started",
            "active_obstacles": np.nan,
            "solve_count": np.nan,
            "linear_violation": np.nan,
            "collision_violation": np.nan,
            "min_planned_clearance_m": np.nan,
            "instant_clearance_m": np.nan,
            "position_error_mm": np.nan,
            "orientation_error_deg": np.nan,
            "feasible": False,
        }

        try:

            # Time only MPC solving and its internal
            # obstacle-selection logic.
            tic = time.perf_counter()

            result = mpc.step(
                q0=q,
                v0=v,
                target_poses=targets,
            )

            row["compute_ms"] = 1000.0 * (
                time.perf_counter() - tic
            )

            info = mpc.last_collision_info

            row["iterations"] = result.iterations
            row["status"] = str(result.status)
            row["active_obstacles"] = info.get(
                "active_count",
                len(info.get("active_indices", ())),
            )
            row["solve_count"] = info.get(
                "solve_count", 1
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
                    "Non-finite control points"
                )

            # Independent feasibility validation
            # against ALL obstacles.
            linear_violation = max_linear_violation(
                problem, c
            )

            min_planned = (
                collision.minimum_distance(c)
            )

            collision_violation = max(
                0.0,
                args.safety_distance - min_planned,
            )

            row["linear_violation"] = linear_violation
            row["collision_violation"] = (
                collision_violation
            )
            row["min_planned_clearance_m"] = (
                min_planned
            )

            if not acceptable_status(
                result, solver_name
            ):
                raise RuntimeError(
                    f"Solver status {result.status}"
                )

            if linear_violation > args.feasibility_tol:
                raise RuntimeError(
                    "Linear constraint violation: "
                    f"{linear_violation:.3e}"
                )

            if collision_violation > args.feasibility_tol:
                raise RuntimeError(
                    "Collision margin violation: "
                    f"{collision_violation:.3e}"
                )

            # Verify the first planned segment on
            # a finer grid before execution.
            segment_times = np.linspace(
                0,
                args.dt,
                args.execution_points,
            )

            segment_clearances = []

            for tau in segment_times:
                q_tau = spline.position(c, tau)
                segment_clearances.append(
                    clearance(environment, q_tau)
                )

            if min(segment_clearances) < (
                args.safety_distance
                - args.feasibility_tol
            ):
                raise RuntimeError(
                    "Planned executed segment violates "
                    "the collision margin."
                )

            # ------------------------------------------------
            # Execution with optional acceleration disturbance
            # ------------------------------------------------

            q_command = spline.position(
                c, args.dt
            )

            v_command = spline.velocity(
                c, args.dt
            )

            a_noise = disturbances[k]

            q_next = (
                q_command
                + 0.5 * a_noise * args.dt**2
            )

            v_next = (
                v_command
                + a_noise * args.dt
            )

            q_violation = bound_violation(
                q_next, Q_MIN, Q_MAX
            )

            v_violation = bound_violation(
                v_next, V_MIN, V_MAX
            )

            instant_clearance = clearance(
                environment, q_next
            )

            if max(
                q_violation, v_violation
            ) > args.feasibility_tol:
                raise RuntimeError(
                    "Executed joint limits violated"
                )

            if instant_clearance < (
                args.safety_distance
                - args.feasibility_tol
            ):
                raise RuntimeError(
                    "Executed state violates collision margin"
                )

            # Accept executed state.
            q = q_next
            v = v_next

            actual_pose = pose_from_theta(robot, q)
            desired_pose = reference_pose(
                robot, t_next
            )

            ep, eo = pose_errors(
                actual_pose,
                desired_pose,
            )

            # Preserve feasible non-converged iterate
            # as warm start, as in safe_tracking.
            mpc.previous_control_points = c.copy()

            ee_path.append(
                actual_pose.translation.copy()
            )

            capsule_path.append(
                capsule_midpoint(
                    robot, q, capsule
                )
            )

            q_path.append(q.copy())

            row["feasible"] = True
            row["instant_clearance_m"] = (
                instant_clearance
            )
            row["position_error_mm"] = 1000 * ep
            row["orientation_error_deg"] = (
                np.degrees(eo)
            )

            records.append(row)

            active = row["active_obstacles"]

            print(
                f"[{name:9s} {k+1:03d}] "
                f"{row['compute_ms']:7.1f} ms | "
                f"iter={result.iterations:3d} | "
                f"obs={active}/1 | "
                f"solves={row['solve_count']} | "
                f"clear={min_planned:.4f} m | "
                f"pos={1000*ep:.3f} mm"
            )

        except Exception as exc:
            records.append(row)
            failure = f"Cycle {k+1}: {exc}"
            print(f"FAILED: {failure}")
            break

    complete = failure is None

    return {
        "name": name,
        "solver": solver_name,
        "limit": limit,
        "complete": complete,
        "failure": failure,
        "records": records,
        "ee_path": np.asarray(ee_path),
        "capsule_path": np.asarray(capsule_path),
        "q_path": np.asarray(q_path),
    }


# ============================================================
# Benchmark statistics
# ============================================================

def summarize(run, args):

    valid = [
        row for row in run["records"]
        if row["feasible"]
    ]

    timed = [
        row for row in run["records"]
        if np.isfinite(row["compute_ms"])
    ]

    times = np.array(
        [r["compute_ms"] for r in timed],
        dtype=float,
    )

    if len(times):
        mean_ms = float(np.mean(times))
        p95_ms = float(np.percentile(times, 95))
        max_ms = float(np.max(times))
        deadline = float(
            100 * np.mean(times <= 1000 * args.dt)
        )
    else:
        mean_ms = p95_ms = max_ms = deadline = np.nan

    return {
        "experiment": run["name"],
        "complete": run["complete"],
        "successful_cycles": len(valid),
        "attempted_cycles": len(run["records"]),
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "max_ms": max_ms,
        "deadline_pct": deadline,
        "position_rms_mm": (
            rms([r["position_error_mm"] for r in valid])
            if valid else np.nan
        ),
        "orientation_rms_deg": (
            rms([r["orientation_error_deg"] for r in valid])
            if valid else np.nan
        ),
        "min_executed_clearance_m": (
            float(min(
                r["instant_clearance_m"] for r in valid
            ))
            if valid else np.nan
        ),
        "failure": run["failure"] or "",
    }


# ============================================================
# CSV export
# ============================================================

def save_results(runs, summaries, output):

    cycle_fields = [
        "experiment",
        "cycle",
        "time_s",
        "compute_ms",
        "iterations",
        "status",
        "active_obstacles",
        "solve_count",
        "linear_violation",
        "collision_violation",
        "min_planned_clearance_m",
        "instant_clearance_m",
        "position_error_mm",
        "orientation_error_deg",
        "feasible",
    ]

    with (output / "cycles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=cycle_fields
        )
        writer.writeheader()

        for run in runs:
            for row in run["records"]:
                writer.writerow({
                    "experiment": run["name"],
                    **row,
                })

    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0]),
        )

        writer.writeheader()
        writer.writerows(summaries)


# ============================================================
# 3D visualization utilities
# ============================================================

def plot_sphere(ax, center, radius, color, alpha):

    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 24)

    x = center[0] + radius * np.outer(
        np.cos(u), np.sin(v)
    )
    y = center[1] + radius * np.outer(
        np.sin(u), np.sin(v)
    )
    z = center[2] + radius * np.outer(
        np.ones_like(u), np.cos(v)
    )

    ax.plot_surface(
        x, y, z,
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=True,
    )


def equal_3d_axes(ax, points):

    p = np.asarray(points)
    minimum = np.min(p, axis=0)
    maximum = np.max(p, axis=0)

    center = 0.5 * (minimum + maximum)
    half_range = max(
        0.5 * np.max(maximum - minimum),
        0.05,
    ) * 1.15

    ax.set_xlim(center[0]-half_range, center[0]+half_range)
    ax.set_ylim(center[1]-half_range, center[1]+half_range)
    ax.set_zlim(center[2]-half_range, center[2]+half_range)

    ax.set_box_aspect((1, 1, 1))


# ============================================================
# Plotting
# ============================================================

def plot_results(
    robot,
    obstacle,
    runs,
    summaries,
    nominal_times,
    nominal_clearances,
    args,
):

    colors = plt.cm.tab10(
        np.linspace(0, 1, max(len(runs), 1))
    )

    # --------------------------------------------------------
    # Figure 1: computation, clearance, tracking
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True
    )

    for color, run in zip(colors, runs):

        rows = run["records"]
        if not rows:
            continue

        t = [r["time_s"] for r in rows]

        axes[0].plot(
            t,
            [r["compute_ms"] for r in rows],
            label=run["name"],
            color=color,
        )

        axes[1].plot(
            t,
            [r["min_planned_clearance_m"] for r in rows],
            label=run["name"],
            color=color,
        )

        axes[2].plot(
            t,
            [r["position_error_mm"] for r in rows],
            label=run["name"],
            color=color,
        )

    axes[0].axhline(
        1000 * args.dt,
        color="black",
        linestyle="--",
        label="Control period",
    )

    axes[1].axhline(
        args.safety_distance,
        color="red",
        linestyle="--",
        label="Safety margin",
    )

    axes[0].set_ylabel("MPC solve time [ms]")
    axes[1].set_ylabel("Min horizon clearance [m]")
    axes[2].set_ylabel("Position error [mm]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        args.output / "mpc_cycles.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # Figure 2: instantaneous collision feasibility
    # --------------------------------------------------------

    fig2, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        nominal_times,
        nominal_clearances,
        "k--",
        linewidth=2,
        label="Nominal reference",
    )

    for color, run in zip(colors, runs):

        rows = [
            r for r in run["records"]
            if r["feasible"]
        ]

        if not rows:
            continue

        ax.plot(
            [r["time_s"] for r in rows],
            [r["instant_clearance_m"] for r in rows],
            color=color,
            label=run["name"],
        )

    ax.axhline(
        args.safety_distance,
        color="red",
        linestyle="--",
        label="Safety margin",
    )

    ax.axhline(
        0.0,
        color="gray",
        linestyle=":",
        label="Geometric contact",
    )

    ax.axhspan(
        -1,
        args.safety_distance,
        color="red",
        alpha=0.07,
    )

    ax.set_ylim(
        bottom=min(
            -0.04,
            float(np.min(nominal_clearances)) - 0.02,
        )
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Robot-obstacle clearance [m]")
    ax.set_title("Collision feasibility")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig2.tight_layout()
    fig2.savefig(
        args.output / "collision_feasibility.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # Figure 3: timing vs iteration budget
    # --------------------------------------------------------

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    for solver in sorted(
        set(run["solver"] for run in runs)
    ):

        selection = [
            (run, summary)
            for run, summary in zip(runs, summaries)
            if run["solver"] == solver
        ]

        ax3.plot(
            [run["limit"] for run, _ in selection],
            [s["mean_ms"] for _, s in selection],
            "o-",
            label=solver,
        )

    ax3.axhline(
        1000 * args.dt,
        color="black",
        linestyle="--",
        label="Control period",
    )

    ax3.set_xlabel("Max iterations")
    ax3.set_ylabel("Mean MPC solve time [ms]")
    ax3.set_title("Iteration budget vs computation time")
    ax3.grid(alpha=0.3)
    ax3.legend()

    fig3.tight_layout()
    fig3.savefig(
        args.output / "iteration_budget.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # Figure 4: active obstacle count
    # --------------------------------------------------------

    fig4, ax4 = plt.subplots(figsize=(11, 4))

    for color, run in zip(colors, runs):
        rows = run["records"]
        if not rows:
            continue

        ax4.step(
            [r["time_s"] for r in rows],
            [r["active_obstacles"] for r in rows],
            where="post",
            color=color,
            label=run["name"],
        )

    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Number of active obstacles")
    ax4.set_yticks([0, 1])
    ax4.set_ylim(-0.15, 1.15)
    ax4.set_title("Per-obstacle activation")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=8)

    fig4.tight_layout()
    fig4.savefig(
        args.output / "obstacle_activation.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # Figure 5: 3D trajectories
    # --------------------------------------------------------

    capsule = get_capsule(args.obstacle_capsule)

    ref_times = np.linspace(
        0,
        DURATION,
        301,
    )

    nominal_ee = []
    nominal_capsule = []

    for t in ref_times:
        q_ref, _, _ = reference_state(t)

        nominal_ee.append(
            pose_from_theta(
                robot, q_ref
            ).translation.copy()
        )

        nominal_capsule.append(
            capsule_midpoint(
                robot, q_ref, capsule
            )
        )

    nominal_ee = np.asarray(nominal_ee)
    nominal_capsule = np.asarray(nominal_capsule)

    fig5 = plt.figure(figsize=(12, 9))
    ax5 = fig5.add_subplot(
        111, projection="3d"
    )

    ax5.plot(
        nominal_ee[:, 0],
        nominal_ee[:, 1],
        nominal_ee[:, 2],
        "k--",
        linewidth=2,
        label="Nominal end-effector",
    )

    ax5.plot(
        nominal_capsule[:, 0],
        nominal_capsule[:, 1],
        nominal_capsule[:, 2],
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Nominal selected capsule",
    )

    all_points = [
        nominal_ee,
        nominal_capsule,
    ]

    for color, run in zip(colors, runs):

        ee = run["ee_path"]
        cap = run["capsule_path"]

        ax5.plot(
            ee[:, 0],
            ee[:, 1],
            ee[:, 2],
            color=color,
            linewidth=2.5,
            label=f"{run['name']} end-effector",
        )

        ax5.plot(
            cap[:, 0],
            cap[:, 1],
            cap[:, 2],
            color=color,
            linewidth=1.6,
            linestyle=":",
            label=f"{run['name']} capsule",
        )

        all_points.extend([ee, cap])

    plot_sphere(
        ax5,
        obstacle.center,
        obstacle.radius,
        color="red",
        alpha=0.3,
    )

    # The activation sphere would be visually misleading
    # because proximity is measured from all robot capsules,
    # not only from the end-effector.
    ax5.scatter(
        *obstacle.center,
        color="red",
        marker="x",
        s=70,
        label="Static obstacle",
    )

    ax5.scatter(
        *nominal_ee[0],
        marker="o",
        color="black",
        s=40,
    )

    ax5.scatter(
        *nominal_ee[-1],
        marker="^",
        color="black",
        s=40,
    )

    sphere_bounds = np.array([
        obstacle.center
        - args.obstacle_radius,
        obstacle.center
        + args.obstacle_radius,
    ])

    all_points.append(sphere_bounds)

    equal_3d_axes(
        ax5,
        np.vstack(all_points),
    )

    ax5.set_xlabel("X [m]")
    ax5.set_ylabel("Y [m]")
    ax5.set_zlabel("Z [m]")

    ax5.set_title(
        "Kinova MPC — end-effector, capsule and obstacle"
    )

    ax5.view_init(
        elev=23,
        azim=-58,
    )

    ax5.legend(fontsize=8)
    fig5.tight_layout()

    fig5.savefig(
        args.output / "trajectories_3d.png",
        dpi=180,
    )

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--solver",
        choices=["gn", "ipopt", "both"],
        default="gn",
    )

    parser.add_argument(
        "--limits",
        nargs="+",
        type=int,
        default=[5, 10, 20],
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--control-points",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--tracking-points",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--collision-points",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--validation-points",
        type=int,
        default=81,
    )

    parser.add_argument(
        "--execution-points",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--orientation-scale",
        type=float,
        default=0.1,
    )

    # --------------------------------------------------------
    # Collision geometry
    # --------------------------------------------------------

    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--safety-distance",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--target-nominal-clearance",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--obstacle-time",
        type=float,
        default=4.5,
    )

    parser.add_argument(
        "--obstacle-capsule",
        type=str,
        default="bracelet",
    )

    parser.add_argument(
        "--max-obstacle-shift",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--placement-samples",
        type=int,
        default=101,
    )

    parser.add_argument(
        "--initial-clearance-buffer",
        type=float,
        default=0.02,
    )

    # --------------------------------------------------------
    # MPC per-obstacle activation
    # --------------------------------------------------------

    parser.add_argument(
        "--activation-distance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--activation-samples",
        type=int,
        default=11,
    )

    # --------------------------------------------------------
    # Disturbances and validation
    # --------------------------------------------------------

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--feasibility-tol",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/kinova_mpc_obstacle_benchmark"
        ),
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error("Require 0 < dt < horizon")

    if args.steps <= 0:
        parser.error("steps must be positive")

    if any(n <= 0 for n in args.limits):
        parser.error("Iteration limits must be positive")

    if args.noise_std < 0:
        parser.error("noise-std must be non-negative")

    if args.activation_distance < 0:
        parser.error(
            "activation-distance must be non-negative"
        )

    if min(
        args.collision_points,
        args.validation_points,
        args.execution_points,
        args.activation_samples,
    ) < 2:
        parser.error(
            "Sampling counts must be at least 2"
        )

    if args.steps * args.dt > DURATION + 1e-9:
        parser.error(
            "Simulation exceeds reference duration"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_reference()

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    obstacle, placement = create_obstacle(
        robot, args
    )

    nominal_times, nominal_clearances = (
        reference_clearances(
            robot,
            obstacle,
        )
    )

    print("\n" + "=" * 76)
    print("STATIC OBSTACLE MPC BENCHMARK")
    print("=" * 76)

    print(f"Obstacle center: {obstacle.center}")
    print(f"Obstacle radius: {obstacle.radius:.4f} m")
    print(f"Selected capsule: {args.obstacle_capsule}")
    print(f"Safety margin: {args.safety_distance:.4f} m")
    print(
        "Target nominal clearance: "
        f"{args.target_nominal_clearance:.4f} m"
    )
    print(
        "Measured nominal clearance: "
        f"{np.min(nominal_clearances):.4f} m"
    )
    print(
        "Initial clearance: "
        f"{placement['initial']:.4f} m"
    )
    print(
        "Activation distance: "
        f"{args.activation_distance:.4f} m"
    )
    print(f"Noise std: {args.noise_std:.4f}")

    disturbances = generate_noise(
        args, robot.nv
    )

    solvers = (
        ["GN", "IPOPT"]
        if args.solver == "both"
        else [args.solver.upper()]
    )

    runs = []

    for solver in solvers:
        for limit in args.limits:

            run = run_experiment(
                robot=robot,
                obstacle=obstacle,
                args=args,
                solver_name=solver,
                limit=limit,
                disturbances=disturbances,
            )

            runs.append(run)

    summaries = [
        summarize(run, args)
        for run in runs
    ]

    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"{'Experiment':<14}"
        f"{'Complete':>10}"
        f"{'Cycles':>9}"
        f"{'Mean ms':>12}"
        f"{'P95 ms':>12}"
        f"{'Pos RMS':>12}"
        f"{'Min clear':>13}"
    )

    for s in summaries:
        print(
            f"{s['experiment']:<14}"
            f"{str(s['complete']):>10}"
            f"{s['successful_cycles']:>9}"
            f"{s['mean_ms']:>12.2f}"
            f"{s['p95_ms']:>12.2f}"
            f"{s['position_rms_mm']:>12.3f}"
            f"{s['min_executed_clearance_m']:>13.4f}"
        )

        if s["failure"]:
            print(f"    {s['failure']}")

    save_results(
        runs,
        summaries,
        args.output,
    )

    plot_results(
        robot=robot,
        obstacle=obstacle,
        runs=runs,
        summaries=summaries,
        nominal_times=nominal_times,
        nominal_clearances=nominal_clearances,
        args=args,
    )

    print(
        f"\nResults saved to: {args.output}"
    )


if __name__ == "__main__":
    main()

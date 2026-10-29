
"""Kinova Gen3 — Static-obstacle Cartesian MPC benchmark.

Compare GN and IPOPT with several iteration budgets in independent
closed-loop simulations, using identical disturbances.

Features:
- Deterministic automatic obstacle placement
- Per-obstacle activation inside CartesianMPC
- Dense first-segment collision validation
- Computation, tracking and clearance statistics
- CSV exports, Matplotlib plots and replayable Meshcat animation

Examples:
    python -m examples.mpc.kinova_mpc_obstacle_benchmark --solver gn --limits 5
    python -m examples.mpc.kinova_mpc_obstacle_benchmark --solver both --limits 20 50
    python -m examples.mpc.kinova_mpc_obstacle_benchmark --solver gn --limits 5 --show-plots
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
)
from examples.utils.collision_visualization import (
    CAPSULES as KINOVA_CAPSULES,
    capsule_endpoints,
)
from examples.utils.kinova_visualization import (
    animate_kinova_trajectories,
)

# ============================================================
# Geometry and reference
# ============================================================


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def print_linear_diagnostics(problem, control_points, q0, v0):
    """Report the rows responsible for linear constraint violations."""
    A, lower, upper = problem.linear_constraints()

    x = np.asarray(control_points, dtype=float).reshape(-1)
    values = np.asarray(A @ x).reshape(-1)

    violations = np.maximum.reduce([
        np.zeros_like(values),
        lower - values,
        values - upper,
    ])

    n = problem.spline.n_dof

    # The first 2*n rows are initial position and velocity
    # equalities. Remaining rows are static joint constraints.
    groups = {
        "Initial position": slice(0, n),
        "Initial velocity": slice(n, 2 * n),
        "Other joint constraints": slice(2 * n, len(values)),
    }

    print("\n  LINEAR CONSTRAINT DIAGNOSTICS")
    print(f"  q0: {np.array2string(q0, precision=5)}")
    print(f"  v0: {np.array2string(v0, precision=5)}")

    for label, indices in groups.items():
        group = violations[indices]
        if group.size:
            print(f"  {label}: max violation = {np.max(group):.3e}")

    worst = np.argsort(violations)[-5:][::-1]

    print("  Worst constraint rows:")
    for i in worst:
        if violations[i] <= 0:
            continue

        print(
            f"    row={i:4d} "
            f"value={values[i]: .6e} "
            f"lower={lower[i]: .6e} "
            f"upper={upper[i]: .6e} "
            f"violation={violations[i]:.3e}"
        )


def unit_vector(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    if norm < 1e-12:
        raise ValueError("Zero-length direction")
    return values / norm


def make_environment(robot, obstacle):
    return CollisionEnvironment(
        robot=robot,
        capsules=KINOVA_CAPSULES,
        obstacles=[obstacle],
    )


def clearance(environment, q):
    distances, _ = environment.evaluate(
        q, compute_jacobian=False,
    )
    return float(np.min(distances))


def get_capsule(name):
    for capsule in KINOVA_CAPSULES:
        if capsule.name == name:
            return capsule
    raise ValueError(f"Unknown capsule: {name}")


def capsule_midpoint(robot, q, capsule):
    p0, p1 = capsule_endpoints(robot, q, capsule)
    return 0.5 * (p0 + p1)


def reference_clearances(robot, obstacle, n=301):
    environment = make_environment(robot, obstacle)
    times = np.linspace(0.0, DURATION, n)
    distances = np.array([
        clearance(environment, reference_state(t)[0])
        for t in times
    ])
    return times, distances


def create_obstacle(robot, args):
    """Find a static sphere with the requested nominal clearance.

    Preserve the original six-direction scan and bisection search.
    """
    if args.obstacle_time > DURATION:
        raise ValueError(
            "Obstacle time exceeds reference duration"
        )

    capsule = get_capsule(args.obstacle_capsule)
    q_anchor = reference_state(args.obstacle_time)[0]
    q_initial = reference_state(0.0)[0]

    anchor = capsule_midpoint(
        robot, q_anchor, capsule,
    )

    directions = [
        unit_vector(v)
        for v in (
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        )
    ]

    reference_q = [
        reference_state(t)[0]
        for t in np.linspace(
            0.0, DURATION, args.placement_samples,
        )
    ]

    def evaluate(center):
        obstacle = SphereObstacle(
            name="static_obstacle",
            center=np.asarray(center, dtype=float),
            radius=args.obstacle_radius,
        )
        environment = make_environment(robot, obstacle)

        nominal = min(
            clearance(environment, q)
            for q in reference_q
        )
        initial = clearance(environment, q_initial)

        return obstacle, nominal, initial

    best = None
    shifts = np.linspace(
        0.0, args.max_obstacle_shift, 36,
    )

    for direction in directions:
        samples = []

        for shift in shifts:
            obstacle, nominal, initial = evaluate(
                anchor + shift * direction
            )
            samples.append((
                shift, nominal, initial, obstacle,
            ))

        for left, right in zip(
            samples[:-1], samples[1:],
        ):
            target = args.target_nominal_clearance
            f_left = left[1] - target
            f_right = right[1] - target

            if f_left * f_right > 0:
                continue

            lo, hi = left[0], right[0]
            f_lo = f_left
            selected = None

            for _ in range(22):
                mid = 0.5 * (lo + hi)
                obstacle, nominal, initial = evaluate(
                    anchor + mid * direction
                )
                selected = (
                    obstacle, nominal, initial, mid,
                )

                if (nominal - target) * f_lo <= 0:
                    hi = mid
                else:
                    lo = mid
                    f_lo = nominal - target

            obstacle, nominal, initial, shift = selected

            if initial < (
                args.safety_distance
                + args.initial_clearance_buffer
            ):
                continue

            score = (
                abs(nominal - target)
                - 0.001 * initial
            )

            if best is None or score < best["score"]:
                best = {
                    "obstacle": obstacle,
                    "nominal": nominal,
                    "initial": initial,
                    "shift": shift,
                    "direction": direction.copy(),
                    "score": score,
                }

    if best is None:
        raise RuntimeError(
            "Could not place a suitable obstacle. "
            "Try changing its time, capsule, radius "
            "or maximum shift."
        )

    return best["obstacle"], best


# ============================================================
# Solvers and disturbances
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

    raise ValueError(f"Unknown solver: {name}")


def acceptable_status(result, solver_name):
    if result.success:
        return True

    if solver_name == "GN":
        return result.status in (
            "max_iterations",
            "damping_limit",
        )

    return result.status == -1


def generate_noise(args, nv):
    rng = np.random.default_rng(args.seed)
    return rng.normal(
        0.0,
        args.noise_std,
        size=(args.steps, nv),
    )


# ============================================================
# Closed-loop MPC
# ============================================================

def run_experiment(
    robot, obstacle, args,
    solver_name, limit, disturbances,
):
    """Run one independent closed-loop collision-aware MPC."""
    name = f"{solver_name}-{limit}"

    print("\n" + "=" * 76)
    print(f"Experiment: {name}")
    print("=" * 76)

    spline = make_spline(
        args.horizon, robot.nv, args.control_points,
    )

    tracking_times = np.linspace(
        0, args.horizon, args.tracking_points,
    )
    collision_times = np.linspace(
        0, args.horizon, args.collision_points,
    )

    environment = make_environment(robot, obstacle)

    collision = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=collision_times,
        safety_distance=args.safety_distance,
    )

    q, v, _ = reference_state(0.0)
    c0 = np.tile(q, (spline.n_control_points, 1))

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
    q_path = [q.copy()]
    ee_path = [
        pose_from_theta(robot, q).translation.copy()
    ]
    capsule_path = [
        capsule_midpoint(robot, q, capsule)
    ]

    failure = None

    for k in range(args.steps):
        t = k * args.dt
        t_next = (k + 1) * args.dt

        targets = [
            reference_pose(robot, t + tau)
            for tau in tracking_times
        ]

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
            tic = time.perf_counter()

            result = mpc.step(
                q0=q,
                v0=v,
                target_poses=targets,
            )

            row["compute_ms"] = (
                1000 * (time.perf_counter() - tic)
            )

            info = mpc.last_collision_info
            row["iterations"] = result.iterations
            row["status"] = str(result.status)
            row["active_obstacles"] = info.get(
                "active_count",
                len(info.get("active_indices", ())),
            )
            row["solve_count"] = info.get(
                "solve_count", 1,
            )

            c = np.asarray(
                result.control_points, dtype=float,
            ).reshape(
                spline.n_control_points, robot.nv,
            )

            if not np.all(np.isfinite(c)):
                raise RuntimeError(
                    "Non-finite control points"
                )

            linear_violation = max_linear_violation(
                problem, c,
            )
            min_planned = collision.minimum_distance(c)
            collision_violation = max(
                0.0,
                args.safety_distance - min_planned,
            )

            row["linear_violation"] = linear_violation
            row["collision_violation"] = (
                collision_violation
            )
            row["min_planned_clearance_m"] = min_planned

            if not acceptable_status(
                result, solver_name,
            ):
                raise RuntimeError(
                    f"Solver status {result.status}"
                )

            if linear_violation > args.feasibility_tol:
                if args.verbose:
                    print_linear_diagnostics(problem, c, q, v)

                raise RuntimeError(
                    "Linear constraint violation: "
                    f"{linear_violation:.3e}"
                )

            if collision_violation > args.feasibility_tol:
                raise RuntimeError(
                    "Collision margin violation: "
                    f"{collision_violation:.3e}"
                )

            # Validate the entire first planned interval,
            # not only the endpoint.
            segment_times = np.linspace(
                0.0, args.dt, args.execution_points,
            )

            segment_clearances = [
                clearance(
                    environment,
                    spline.position(c, tau),
                )
                for tau in segment_times
            ]

            if min(segment_clearances) < (
                args.safety_distance
                - args.feasibility_tol
            ):
                raise RuntimeError(
                    "Planned executed segment violates "
                    "the collision margin."
                )

            # Simplified acceleration-disturbance model.
            q_command = spline.position(c, args.dt)
            v_command = spline.velocity(c, args.dt)
            a_noise = disturbances[k]

            q_next = (
                q_command
                + 0.5 * a_noise * args.dt**2
            )
            v_next = v_command + a_noise * args.dt

            q_violation = bound_violation(
                q_next, Q_MIN, Q_MAX,
            )
            v_violation = bound_violation(
                v_next, V_MIN, V_MAX,
            )
            instant_clearance = clearance(
                environment, q_next,
            )

            if max(
                q_violation, v_violation,
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

            q, v = q_next, v_next
            actual_pose = pose_from_theta(robot, q)
            desired_pose = reference_pose(robot, t_next)
            ep, eo = pose_errors(
                actual_pose, desired_pose,
            )

            mpc.previous_control_points = c.copy()

            q_path.append(q.copy())
            ee_path.append(
                actual_pose.translation.copy()
            )
            capsule_path.append(
                capsule_midpoint(robot, q, capsule)
            )

            row["feasible"] = True
            row["instant_clearance_m"] = instant_clearance
            row["position_error_mm"] = 1000 * ep
            row["orientation_error_deg"] = np.degrees(eo)

            records.append(row)

            print(
                f"[{name:9s} {k+1:03d}] "
                f"{row['compute_ms']:7.1f} ms | "
                f"iter={result.iterations:3d} | "
                f"obs={row['active_obstacles']}/1 | "
                f"solves={row['solve_count']} | "
                f"clear={min_planned:.4f} m | "
                f"pos={1000*ep:.3f} mm"
            )

        except Exception as exc:
            records.append(row)
            failure = f"Cycle {k+1}: {exc}"
            print(f"FAILED: {failure}")
            break

    return {
        "name": name,
        "solver": solver_name,
        "limit": limit,
        "complete": failure is None,
        "failure": failure,
        "records": records,
        "ee_path": np.asarray(ee_path),
        "capsule_path": np.asarray(capsule_path),
        "q_path": np.asarray(q_path),
    }


# ============================================================
# Benchmark results
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

    times = np.asarray([
        row["compute_ms"] for row in timed
    ])

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
            min(r["instant_clearance_m"] for r in valid)
            if valid else np.nan
        ),
        "failure": run["failure"] or "",
    }


def save_results(runs, summaries, output):
    fields = [
        "experiment", "cycle", "time_s",
        "compute_ms", "iterations", "status",
        "active_obstacles", "solve_count",
        "linear_violation", "collision_violation",
        "min_planned_clearance_m", "instant_clearance_m",
        "position_error_mm", "orientation_error_deg",
        "feasible",
    ]

    with (output / "cycles.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields,
        )
        writer.writeheader()

        for run in runs:
            for row in run["records"]:
                writer.writerow({
                    "experiment": run["name"],
                    **row,
                })

    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summaries[0]),
        )
        writer.writeheader()
        writer.writerows(summaries)


def print_summary(summaries):
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


# ============================================================
# Matplotlib
# ============================================================

def save_figure(fig, output, filename):
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)


def plot_sphere(ax, center, radius):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 24)

    x = center[0] + radius * np.outer(
        np.cos(u), np.sin(v),
    )
    y = center[1] + radius * np.outer(
        np.sin(u), np.sin(v),
    )
    z = center[2] + radius * np.outer(
        np.ones_like(u), np.cos(v),
    )

    ax.plot_surface(
        x, y, z,
        color="red",
        alpha=0.3,
        linewidth=0,
    )


def equal_3d_axes(ax, points):
    points = np.asarray(points)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)

    center = (minimum + maximum) / 2
    half_range = max(
        0.5 * np.max(maximum - minimum),
        0.05,
    ) * 1.15

    ax.set_xlim(
        center[0] - half_range,
        center[0] + half_range,
    )
    ax.set_ylim(
        center[1] - half_range,
        center[1] + half_range,
    )
    ax.set_zlim(
        center[2] - half_range,
        center[2] + half_range,
    )
    ax.set_box_aspect((1, 1, 1))


def plot_results(
    robot, obstacle, runs, summaries,
    nominal_times, nominal_clearances, args,
):
    colors = plt.cm.tab10(
        np.linspace(0, 1, max(len(runs), 1))
    )

    # 1. MPC times, clearance and tracking
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
    )

    for color, run in zip(colors, runs):
        rows = run["records"]
        if not rows:
            continue

        t = [r["time_s"] for r in rows]
        quantities = (
            "compute_ms",
            "min_planned_clearance_m",
            "position_error_mm",
        )

        for ax, key in zip(axes, quantities):
            ax.plot(
                t,
                [r[key] for r in rows],
                color=color,
                label=run["name"],
            )

    axes[0].axhline(
        1000 * args.dt,
        color="black", ls="--",
        label="Control period",
    )
    axes[1].axhline(
        args.safety_distance,
        color="red", ls="--",
        label="Safety margin",
    )

    axes[0].set_ylabel("MPC solve time [ms]")
    axes[1].set_ylabel("Min horizon clearance [m]")
    axes[2].set_ylabel("Position error [mm]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.legend(fontsize=8)

    save_figure(fig, args.output, "mpc_cycles.png")

    # 2. Executed collision clearance
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        nominal_times, nominal_clearances,
        "k--", lw=2, label="Nominal reference",
    )

    for color, run in zip(colors, runs):
        rows = [
            r for r in run["records"]
            if r["feasible"]
        ]
        if rows:
            ax.plot(
                [r["time_s"] for r in rows],
                [r["instant_clearance_m"] for r in rows],
                color=color,
                label=run["name"],
            )

    ax.axhline(
        args.safety_distance,
        color="red", ls="--",
        label="Safety margin",
    )
    ax.axhline(
        0.0, color="gray", ls=":",
        label="Geometric contact",
    )
    ax.axhspan(
        -1.0, args.safety_distance,
        color="red", alpha=0.07,
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
    ax.legend(fontsize=8)

    save_figure(
        fig, args.output, "collision_feasibility.png",
    )

    # 3. Iteration budget vs computation time
    fig, ax = plt.subplots(figsize=(10, 5))

    for solver in sorted({
        run["solver"] for run in runs
    }):
        selection = [
            (run, summary)
            for run, summary in zip(runs, summaries)
            if run["solver"] == solver
        ]

        ax.plot(
            [run["limit"] for run, _ in selection],
            [s["mean_ms"] for _, s in selection],
            "o-",
            label=solver,
        )

    ax.axhline(
        1000 * args.dt,
        color="black", ls="--",
        label="Control period",
    )
    ax.set_xlabel("Max iterations")
    ax.set_ylabel("Mean MPC solve time [ms]")
    ax.set_title("Iteration budget vs computation time")
    ax.legend()

    save_figure(fig, args.output, "iteration_budget.png")

    # 4. Collision activation
    fig, ax = plt.subplots(figsize=(11, 4))

    for color, run in zip(colors, runs):
        rows = run["records"]
        if rows:
            ax.step(
                [r["time_s"] for r in rows],
                [r["active_obstacles"] for r in rows],
                where="post",
                color=color,
                label=run["name"],
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Active obstacles")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("Per-obstacle activation")
    ax.legend(fontsize=8)

    save_figure(
        fig, args.output, "obstacle_activation.png",
    )

    # 5. Cartesian and selected capsule trajectories
    capsule = get_capsule(args.obstacle_capsule)
    ref_times = np.linspace(0, DURATION, 301)

    nominal_ee = []
    nominal_capsule = []

    for t in ref_times:
        q = reference_state(t)[0]
        nominal_ee.append(
            pose_from_theta(robot, q).translation.copy()
        )
        nominal_capsule.append(
            capsule_midpoint(robot, q, capsule)
        )

    nominal_ee = np.asarray(nominal_ee)
    nominal_capsule = np.asarray(nominal_capsule)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        *nominal_ee.T,
        "k--", lw=2,
        label="Nominal end-effector",
    )
    ax.plot(
        *nominal_capsule.T,
        color="gray", ls=":", lw=1.5,
        label="Nominal selected capsule",
    )

    all_points = [nominal_ee, nominal_capsule]

    for color, run in zip(colors, runs):
        ee = run["ee_path"]
        cap = run["capsule_path"]

        ax.plot(
            *ee.T,
            color=color, lw=2.5,
            label=f"{run['name']} end-effector",
        )
        ax.plot(
            *cap.T,
            color=color, ls=":", lw=1.6,
            label=f"{run['name']} capsule",
        )
        all_points.extend([ee, cap])

    plot_sphere(
        ax, obstacle.center, obstacle.radius,
    )

    ax.scatter(
        *obstacle.center,
        color="red", marker="x", s=70,
        label="Static obstacle",
    )
    ax.scatter(
        *nominal_ee[0], color="black", s=40,
    )
    ax.scatter(
        *nominal_ee[-1],
        color="black", marker="^", s=40,
    )

    all_points.append(np.array([
        obstacle.center - obstacle.radius,
        obstacle.center + obstacle.radius,
    ]))

    equal_3d_axes(ax, np.vstack(all_points))

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        "Kinova MPC — end-effector, capsule and obstacle"
    )
    ax.view_init(elev=23, azim=-58)
    ax.legend(fontsize=8)

    save_figure(
        fig, args.output, "trajectories_3d.png",
    )


# ============================================================
# Interactive Meshcat
# ============================================================

def animate_results(robot, obstacle, runs, args):
    """Replay MPC trajectories using the shared Meshcat helper."""
    reference_times = np.arange(args.steps + 1) * args.dt

    reference_xyz = np.asarray([
        pose_from_theta(
            robot,
            reference_state(t)[0],
        ).translation.copy()
        for t in reference_times
    ])

    trajectories = {}

    for run in runs:
        status = (
            "complete" if run["complete"] else "stopped early"
        )

        trajectories[run["name"]] = {
            "q": run["q_path"],
            "label": (
                f"{run['name']} "
                f"({len(run['q_path']) - 1} cycles, {status})"
            ),
        }

    animate_kinova_trajectories(
        robot=robot,
        trajectories=trajectories,
        dt=args.dt,
        obstacles=[obstacle],
        reference_xyz=reference_xyz,
        target_pose=reference_pose(
            robot, args.steps * args.dt,
        ),
        speed=args.animation_speed,
        title="STATIC OBSTACLE MPC — ANIMATION",
    )


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
        "--limits", nargs="+", type=int,
        default=[5, 10, 20],
    )

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--control-points", type=int, default=12)
    parser.add_argument("--tracking-points", type=int, default=15)
    parser.add_argument("--collision-points", type=int, default=20)
    parser.add_argument("--validation-points", type=int, default=81)
    parser.add_argument("--execution-points", type=int, default=11)
    parser.add_argument("--orientation-scale", type=float, default=0.1)

    parser.add_argument("--obstacle-radius", type=float, default=0.07)
    parser.add_argument("--safety-distance", type=float, default=0.02)
    parser.add_argument(
        "--target-nominal-clearance",
        type=float, default=0.012,
    )
    parser.add_argument("--obstacle-time", type=float, default=4.5)
    parser.add_argument(
        "--obstacle-capsule", type=str, default="bracelet",
    )
    parser.add_argument(
        "--max-obstacle-shift", type=float, default=0.5,
    )
    parser.add_argument(
        "--placement-samples", type=int, default=101,
    )
    parser.add_argument(
        "--initial-clearance-buffer",
        type=float, default=0.02,
    )

    parser.add_argument(
        "--activation-distance", type=float, default=0.35,
    )
    parser.add_argument(
        "--activation-samples", type=int, default=11,
    )

    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feasibility-tol", type=float, default=1e-4,
    )

    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostics when linear constraints fail.",
    )
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument(
        "--animation-speed", type=float, default=2.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kinova_mpc_obstacle_benchmark"),
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error("Require 0 < dt < horizon")

    if args.steps <= 0:
        parser.error("steps must be positive")

    if any(limit <= 0 for limit in args.limits):
        parser.error("Iteration limits must be positive")

    if args.noise_std < 0:
        parser.error("noise-std must be non-negative")

    if args.activation_distance < 0:
        parser.error("activation-distance must be non-negative")

    if min(
        args.collision_points,
        args.validation_points,
        args.execution_points,
        args.activation_samples,
    ) < 2:
        parser.error("Sampling counts must be at least 2")

    if args.steps * args.dt > DURATION + 1e-9:
        parser.error("Simulation exceeds reference duration")

    if args.animation_speed <= 0:
        parser.error("animation-speed must be positive")

    args.output.mkdir(
        parents=True, exist_ok=True,
    )

    validate_reference()

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    obstacle, placement = create_obstacle(robot, args)
    nominal_times, nominal_clearances = (
        reference_clearances(robot, obstacle)
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

    disturbances = generate_noise(args, robot.nv)

    solvers = (
        ["GN", "IPOPT"]
        if args.solver == "both"
        else [args.solver.upper()]
    )

    runs = [
        run_experiment(
            robot=robot,
            obstacle=obstacle,
            args=args,
            solver_name=solver,
            limit=limit,
            disturbances=disturbances,
        )
        for solver in solvers
        for limit in args.limits
    ]

    summaries = [
        summarize(run, args)
        for run in runs
    ]

    print_summary(summaries)
    save_results(runs, summaries, args.output)

    plot_results(
        robot,
        obstacle,
        runs,
        summaries,
        nominal_times,
        nominal_clearances,
        args,
    )

    print(f"\nResults saved to: {args.output.resolve()}")

    if args.show_plots:
        plt.show(block=False)

    if not args.no_animation:
        animate_results(robot, obstacle, runs, args)
    elif args.show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

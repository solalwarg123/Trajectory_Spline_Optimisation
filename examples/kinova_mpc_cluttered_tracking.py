
"""
Kinova Gen3 - Cartesian MPC benchmark (Phase 1).

Experiments:
    OPEN       : open-loop reference execution
    FREE-GN2   : MPC without collision constraints
    FREE-GN5
    SAFE-GN2   : collision-aware MPC
    SAFE-GN5

Features:
    - Native warm-start rebasing and OSQP projection
    - Final GN solution acceptance diagnostics
    - Collision activation and feasibility checks
    - Detailed cycle-by-cycle printing
    - CSV exports and diagnostic plots

Run:
    python -m examples.kinova_mpc_cluttered_tracking

    python -m examples.kinova_mpc_cluttered_tracking \
        --gn-iters 2 5 --noise-std 0.35 --debug-projection

    python -m examples.kinova_mpc_cluttered_tracking \
        --gn-iters 2 5 --show-plots
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
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a**2))) if a.size else np.nan


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(np.linalg.norm(x), 1e-12)


def finite_values(rows, key):
    values = np.asarray(
        [r.get(key, np.nan) for r in rows],
        dtype=float,
    )
    return values[np.isfinite(values)]


def linear_violation(problem, control_points):
    A, lower, upper = problem.linear_constraints()
    x = np.asarray(control_points, dtype=float).reshape(-1)
    values = np.asarray(A @ x).reshape(-1)

    if not np.all(np.isfinite(values)):
        return np.inf

    return float(max(
        0.0,
        np.max(lower - values),
        np.max(values - upper),
    ))


def print_linear_failure(problem, control_points):
    A, lower, upper = problem.linear_constraints()
    x = np.asarray(control_points).reshape(-1)
    values = np.asarray(A @ x).reshape(-1)

    violations = np.maximum(
        np.maximum(lower - values, 0.0),
        np.maximum(values - upper, 0.0),
    )

    order = np.argsort(violations)[::-1][:8]
    n = problem.spline.n_dof
    m = problem.spline.n_control_points

    n_eq = 2 * n
    pos_end = n_eq + m * n
    vel_end = pos_end + (m - 1) * n

    print("\n" + "-" * 75)
    print("LINEAR CONSTRAINT FAILURE")
    print("-" * 75)

    for i in order:
        if violations[i] <= 1e-10:
            continue

        if i < n:
            family, joint = "q0", i
        elif i < n_eq:
            family, joint = "v0", i - n
        elif i < pos_end:
            family, joint = "position", (i - n_eq) % n
        elif i < vel_end:
            family, joint = "velocity", (i - pos_end) % n
        else:
            family, joint = "acceleration", (i - vel_end) % n

        print(
            f"Row {i:4d} | {family:14s} | "
            f"Joint {joint + 1} | "
            f"value={values[i]: .6f} | "
            f"bounds=[{lower[i]: .6f}, {upper[i]: .6f}] | "
            f"violation={violations[i]:.3e}"
        )


# ============================================================
# Collision helpers
# ============================================================

def make_environment(robot, obstacles):
    return CollisionEnvironment(
        robot=robot,
        capsules=KINOVA_CAPSULES,
        obstacles=obstacles,
    )


def obstacle_clearances(environment, q):
    distances, _ = environment.evaluate(
        q,
        compute_jacobian=False,
    )

    return np.asarray(distances).reshape(
        environment.n_capsules,
        environment.n_obstacles,
    ).min(axis=0)


def reference_clearances(environment, times):
    return np.asarray([
        obstacle_clearances(environment, reference_state(t)[0])
        for t in times
    ])


def capsule_midpoint(robot, q, capsule):
    p0, p1 = capsule_endpoints(robot, q, capsule)
    return 0.5 * (p0 + p1)


def generate_obstacles(robot, args):
    """
    Place static spheres near the nominal trajectory.

    The nominal trajectory remains collision-feasible.
    """
    anchor_times = np.linspace(
        2.5, 7.5, args.n_obstacles
    )
    targets = np.linspace(
        args.nominal_clearance_min,
        args.nominal_clearance_max,
        args.n_obstacles,
    )

    sample_times = np.linspace(
        0, DURATION, args.placement_samples
    )
    reference_q = [
        reference_state(t)[0] for t in sample_times
    ]

    capsule = next(
        (c for c in KINOVA_CAPSULES
         if c.name == args.anchor_capsule),
        None,
    )

    if capsule is None:
        raise ValueError(
            f"Unknown capsule: {args.anchor_capsule}"
        )

    directions = [
        unit(x) for x in (
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        )
    ]

    obstacles = []
    placement = []

    for index, (anchor_t, target) in enumerate(
        zip(anchor_times, targets)
    ):
        anchor = capsule_midpoint(
            robot,
            reference_state(anchor_t)[0],
            capsule,
        )

        def evaluate(shift, direction):
            sphere = SphereObstacle(
                center=anchor + shift * direction,
                radius=args.obstacle_radius,
                name=f"obstacle_{index + 1}",
            )

            env = make_environment(robot, [sphere])
            distances = np.asarray([
                obstacle_clearances(env, q)[0]
                for q in reference_q
            ])

            closest = int(np.argmin(distances))

            return {
                "sphere": sphere,
                "min_clearance": float(distances[closest]),
                "closest_time": float(sample_times[closest]),
                "initial_clearance": float(distances[0]),
            }

        best = None
        shifts = np.linspace(
            0,
            args.max_obstacle_shift,
            args.placement_shifts,
        )

        for direction in directions:
            previous = None

            for shift in shifts:
                current = evaluate(shift, direction)

                if previous is not None:
                    old_shift, old_data = previous
                    f0 = old_data["min_clearance"] - target
                    f1 = current["min_clearance"] - target

                    if f0 * f1 <= 0:
                        lo, hi = old_shift, shift
                        flo = f0
                        candidate = current

                        for _ in range(15):
                            mid = 0.5 * (lo + hi)
                            candidate = evaluate(mid, direction)
                            fm = candidate["min_clearance"] - target

                            if flo * fm <= 0:
                                hi = mid
                            else:
                                lo = mid
                                flo = fm

                        sphere = candidate["sphere"]

                        if candidate["initial_clearance"] < (
                            args.safety_distance + args.initial_buffer
                        ):
                            continue

                        separated = all(
                            np.linalg.norm(
                                sphere.center - other.center
                            ) >= (
                                sphere.radius + other.radius
                                + args.min_sphere_gap
                            )
                            for other in obstacles
                        )

                        if not separated:
                            continue

                        score = (
                            abs(candidate["min_clearance"] - target)
                            + args.time_placement_weight
                            * abs(candidate["closest_time"] - anchor_t)
                        )

                        if best is None or score < best["score"]:
                            best = {
                                **candidate,
                                "score": score,
                                "anchor_time": anchor_t,
                            }

                previous = (shift, current)

        if best is None:
            raise RuntimeError(
                f"Could not place obstacle {index + 1}"
            )

        obstacles.append(best["sphere"])
        placement.append(best)

    environment = make_environment(robot, obstacles)

    check_times = np.linspace(
        0, DURATION, args.reference_check_points
    )

    distances = reference_clearances(
        environment, check_times
    )

    minima = distances.min(axis=0)

    if np.any(minima < (
        args.safety_distance + args.nominal_feasibility_buffer
    )):
        raise RuntimeError(
            "Nominal trajectory not collision-feasible: "
            f"{minima}"
        )

    return (
        obstacles,
        environment,
        placement,
        (check_times, distances),
    )


# ============================================================
# MPC construction
# ============================================================

def create_mpc(robot, environment, args, safe, gn_iters):
    spline = make_spline(
        args.horizon,
        robot.nv,
        args.control_points,
    )

    tracking_times = np.linspace(
        0, args.horizon, args.tracking_points
    )
    collision_times = np.linspace(
        0, args.horizon, args.collision_points
    )

    q0, v0, _ = reference_state(0)

    initial_c = np.tile(
        q0, (spline.n_control_points, 1)
    )

    collision = None

    if safe:
        collision = CollisionConstraintBuilder(
            spline=spline,
            environment=environment,
            collocation_times=collision_times,
            safety_distance=args.safety_distance,
        )

    problem = CartesianTrajectoryProblem(
        robot=robot,
        spline=spline,
        tracking_times=tracking_times,
        target_poses=[
            reference_pose(robot, t)
            for t in tracking_times
        ],
        q0=q0,
        v0=v0,
        q_min=Q_MIN,
        q_max=Q_MAX,
        v_min=V_MIN,
        v_max=V_MAX,
        a_min=A_MIN,
        a_max=A_MAX,
        orientation_scale=args.orientation_scale,
        lambda_smooth=1e-5,
        lambda_terminal=args.lambda_terminal,
        collision_constraints=collision,
    )

    solver = GaussNewtonSolver(
        damping_mode="adaptive",
        damping=1e-3,
        max_iterations=gn_iters,
        collision_slack_weight=1e4,
        collision_merit_weight=1e4,
    )

    options = {}

    if safe:
        options = {
            "collision_activation_distance":
                args.activation_distance,
            "collision_check_samples":
                args.activation_samples,
            "collision_feasibility_tolerance":
                args.feasibility_tol,
        }

    mpc = CartesianMPC(
        problem=problem,
        solver=solver,
        initial_control_points=initial_c,
        project_warm_start=args.project_warm_start,
        projection_tolerance=args.projection_tol,
        **options,
    )

    return mpc, spline, problem, collision, tracking_times


# ============================================================
# Simulation
# ============================================================

def empty_record(name, mode, cycle, time_s):
    return {
        "experiment": name,
        "mode": mode,
        "cycle": cycle,
        "time_s": time_s,
        "accepted": False,
        "executed": False,
        "status": "",
        "solver_success": False,
        "linear_feasible": False,
        "collision_feasible": "",
        "compute_ms": np.nan,
        "gn_iterations": np.nan,
        "gn_linear_violation": np.nan,
        "projection_needed": False,
        "projection_status": "",
        "projection_ms": 0.0,
        "warm_violation_before": np.nan,
        "warm_violation_after": np.nan,
        "active_count": 0,
        "active_indices": "",
        "solve_count": 0,
        "min_planned_clearance": np.nan,
        "collision_violation": np.nan,
        "min_actual_clearance": np.nan,
        "position_error_mm": np.nan,
        "orientation_error_deg": np.nan,
        "joint_error_rad": np.nan,
        "joint_limit_violation": np.nan,
        "velocity_limit_violation": np.nan,
    }


def simulate(
    name,
    robot,
    environment,
    args,
    disturbances,
    mode,
    gn_iters=None,
):
    q, v, _ = reference_state(0.0)

    mpc = spline = problem = collision = tracking_times = None

    if mode != "OPEN":
        mpc, spline, problem, collision, tracking_times = (
            create_mpc(
                robot, environment, args,
                safe=(mode == "SAFE"),
                gn_iters=gn_iters,
            )
        )

    e_q = np.zeros(robot.nv)
    e_v = np.zeros(robot.nv)

    records = []
    q_path = [q.copy()]
    ee_path = [pose_from_theta(robot, q).translation.copy()]

    failure = None

    print("\n" + "=" * 115)
    print(f"EXPERIMENT: {name}")
    print("=" * 115)

    for k in range(args.steps):
        t = k * args.dt
        t_next = (k + 1) * args.dt
        disturbance = disturbances[k]

        row = empty_record(name, mode, k + 1, t_next)
        candidate_c = None

        try:
            if mode == "OPEN":
                # Open-loop disturbance accumulation.
                e_q += (
                    e_v * args.dt
                    + 0.5 * disturbance * args.dt**2
                )
                e_v += disturbance * args.dt

                q_ref, v_ref, _ = reference_state(t_next)
                q_next = q_ref + e_q
                v_next = v_ref + e_v

                row["status"] = "open_loop"
                row["solver_success"] = True
                row["linear_feasible"] = True
                row["accepted"] = True

            else:
                targets = [
                    reference_pose(robot, t + tau)
                    for tau in tracking_times
                ]

                start = time.perf_counter()

                result = mpc.step(
                    q0=q,
                    v0=v,
                    target_poses=targets,
                )

                row["compute_ms"] = (
                    1000 * (time.perf_counter() - start)
                )
                row["gn_iterations"] = getattr(
                    result, "iterations", np.nan
                )
                row["status"] = str(result.status)

                # Native projection diagnostics.
                projection = mpc.last_projection_info or {}

                row["projection_needed"] = bool(
                    projection.get("projected", False)
                )
                row["projection_status"] = str(
                    projection.get("status", "")
                )
                row["projection_ms"] = float(
                    projection.get("time_ms", 0.0)
                )

                for source, dest in (
                    ("before", "warm_violation_before"),
                    ("after", "warm_violation_after"),
                ):
                    value = projection.get(source)
                    if value is not None:
                        row[dest] = float(value)

                # Collision activation diagnostics.
                collision_info = mpc.last_collision_info or {}
                indices = collision_info.get(
                    "active_indices", ()
                )

                row["active_count"] = len(indices)
                row["active_indices"] = ",".join(
                    map(str, indices)
                )
                row["solve_count"] = collision_info.get(
                    "solve_count", 1
                )

                # Final MPC acceptance diagnostics.
                solution_info = mpc.last_solution_info or {}

                row["solver_success"] = bool(
                    solution_info.get(
                        "solver_success", result.success
                    )
                )

                row["linear_feasible"] = bool(
                    solution_info.get("linear_feasible", False)
                )

                collision_ok = solution_info.get(
                    "collision_feasible"
                )
                row["collision_feasible"] = (
                    "" if collision_ok is None
                    else bool(collision_ok)
                )

                row["accepted"] = bool(
                    solution_info.get("accepted", False)
                )

                # Independent measurement for reporting.
                candidate_c = np.asarray(
                    result.control_points,
                    dtype=float,
                ).reshape(
                    spline.n_control_points,
                    robot.nv,
                )

                if not np.all(np.isfinite(candidate_c)):
                    raise RuntimeError(
                        "GN returned non-finite coefficients"
                    )

                row["gn_linear_violation"] = (
                    linear_violation(problem, candidate_c)
                )

                if mode == "SAFE":
                    min_planned = float(
                        collision.minimum_distance(candidate_c)
                    )
                    row["min_planned_clearance"] = min_planned
                    row["collision_violation"] = max(
                        0.0,
                        args.safety_distance - min_planned,
                    )

                if args.debug_all or (
                    args.debug_projection
                    and row["projection_needed"]
                ):
                    print(
                        f"\n[DIAGNOSTICS {name} cycle {k+1}]"
                    )
                    print(
                        "  Warm violation before:",
                        row["warm_violation_before"]
                    )
                    print(
                        "  Warm violation after :",
                        row["warm_violation_after"]
                    )
                    print(
                        "  Projection status    :",
                        row["projection_status"]
                    )
                    print(
                        "  Projection time      :",
                        f"{row['projection_ms']:.3f} ms"
                    )
                    print(
                        "  GN linear violation  :",
                        f"{row['gn_linear_violation']:.3e}"
                    )
                    print(
                        "  Solver success       :",
                        row["solver_success"]
                    )
                    print(
                        "  Linear feasible      :",
                        row["linear_feasible"]
                    )
                    print(
                        "  Collision feasible   :",
                        row["collision_feasible"]
                    )
                    print(
                        "  Accepted             :",
                        row["accepted"]
                    )

                # Important: do not execute rejected results.
                if not row["accepted"]:
                    print("\nMPC SOLUTION REJECTED")
                    print("  Experiment:", name)
                    print("  Cycle:", k + 1)
                    print("  Solver status:", result.status)
                    print("  Solver success:", row["solver_success"])
                    print("  Linear feasible:", row["linear_feasible"])
                    print("  Collision feasible:", row["collision_feasible"])
                    print(
                        "  GN linear violation:",
                        row["gn_linear_violation"]
                    )

                    if np.isfinite(
                        row["gn_linear_violation"]
                    ):
                        print_linear_failure(
                            problem, candidate_c
                        )

                    raise RuntimeError(
                        "MPC rejected final GN solution"
                    )

                # Execute only the first MPC interval.
                q_command = spline.position(
                    candidate_c, args.dt
                )
                v_command = spline.velocity(
                    candidate_c, args.dt
                )

                q_next = (
                    q_command
                    + 0.5 * disturbance * args.dt**2
                )
                v_next = (
                    v_command + disturbance * args.dt
                )

            # Executed-state diagnostics.
            actual_distances = obstacle_clearances(
                environment, q_next
            )
            min_actual = float(actual_distances.min())

            q_violation = bound_violation(
                q_next, Q_MIN, Q_MAX
            )
            v_violation = bound_violation(
                v_next, V_MIN, V_MAX
            )

            row["min_actual_clearance"] = min_actual
            row["joint_limit_violation"] = q_violation
            row["velocity_limit_violation"] = v_violation

            if max(q_violation, v_violation) > args.feasibility_tol:
                raise RuntimeError(
                    "Executed joint or velocity limit violation"
                )

            q = np.asarray(q_next, dtype=float).copy()
            v = np.asarray(v_next, dtype=float).copy()

            actual_pose = pose_from_theta(robot, q)
            desired_pose = reference_pose(robot, t_next)
            ep, eo = pose_errors(actual_pose, desired_pose)

            row["position_error_mm"] = 1000.0 * ep
            row["orientation_error_deg"] = np.degrees(eo)
            row["joint_error_rad"] = np.linalg.norm(
                q - reference_state(t_next)[0]
            )

            row["executed"] = True

            q_path.append(q.copy())
            ee_path.append(actual_pose.translation.copy())
            records.append(row)

            safety = (
                "SAFE"
                if min_actual >= args.safety_distance
                else "CLEARANCE VIOLATION"
            )

            compute_text = (
                "    --"
                if mode == "OPEN"
                else f"{row['compute_ms']:7.2f}"
            )

            gn_text = (
                "--"
                if mode == "OPEN"
                else f"{row['gn_linear_violation']:.1e}"
            )

            print(
                f"[{name:10s} {k+1:03d}/{args.steps:03d}] "
                f"{compute_text} ms | "
                f"pos={row['position_error_mm']:7.3f} mm | "
                f"ori={row['orientation_error_deg']:6.3f} deg | "
                f"clear={min_actual:7.4f} m | "
                f"active={row['active_count']}/{args.n_obstacles} | "
                f"proj={int(row['projection_needed'])} | "
                f"GN={gn_text} | "
                f"accepted={int(row['accepted'])} | "
                f"{safety}"
            )

        except Exception as exc:
            row["status"] = (
                f"{row['status']} | ERROR: {exc}"
            )
            records.append(row)

            failure = f"Cycle {k+1}: {exc}"

            print("\n" + "!" * 85)
            print(f"FAILED: {name}")
            print(failure)
            print("!" * 85)

            break

    return {
        "name": name,
        "mode": mode,
        "gn_iters": gn_iters,
        "records": records,
        "complete": failure is None,
        "failure": failure,
        "q_path": np.asarray(q_path),
        "ee_path": np.asarray(ee_path),
    }


# ============================================================
# Statistics and CSV
# ============================================================

def summarize(run, args):
    rows = [
        r for r in run["records"] if r["executed"]
    ]
    attempted = [
        r for r in run["records"] if r["mode"] != "OPEN"
    ]

    times = finite_values(rows, "compute_ms")
    clearances = finite_values(rows, "min_actual_clearance")
    projection_times = finite_values(
        attempted, "projection_ms"
    )

    def mean(x):
        return float(x.mean()) if x.size else np.nan

    def percentile(x, p):
        return (
            float(np.percentile(x, p))
            if x.size else np.nan
        )

    return {
        "experiment": run["name"],
        "complete": run["complete"],
        "cycles": len(rows),
        "attempted_cycles": len(run["records"]),
        "accepted_solutions": sum(
            bool(r["accepted"]) for r in attempted
        ),
        "rejected_solutions": sum(
            not bool(r["accepted"]) for r in attempted
        ),
        "mean_ms": mean(times),
        "median_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "max_ms": (
            float(times.max()) if times.size else np.nan
        ),
        "position_rms_mm": rms(
            finite_values(rows, "position_error_mm")
        ),
        "orientation_rms_deg": rms(
            finite_values(rows, "orientation_error_deg")
        ),
        "joint_rms_rad": rms(
            finite_values(rows, "joint_error_rad")
        ),
        "min_clearance_m": (
            float(clearances.min())
            if clearances.size else np.nan
        ),
        "clearance_violation_cycles": sum(
            r["min_actual_clearance"] < args.safety_distance
            for r in rows
        ),
        "active_cycles": sum(
            r["active_count"] > 0 for r in attempted
        ),
        "projection_cycles": sum(
            bool(r["projection_needed"]) for r in attempted
        ),
        "projection_mean_ms": mean(projection_times),
        "max_warm_violation": (
            float(x.max())
            if (x := finite_values(
                attempted, "warm_violation_before"
            )).size else np.nan
        ),
        "max_gn_violation": (
            float(x.max())
            if (x := finite_values(
                attempted, "gn_linear_violation"
            )).size else np.nan
        ),
        "failure": run["failure"] or "",
    }


def export_csv(runs, summaries, output):
    rows = [
        r for run in runs for r in run["records"]
    ]

    if rows:
        with (output / "cycles.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summaries[0].keys())
        )
        writer.writeheader()
        writer.writerows(summaries)


# ============================================================
# Plot helpers
# ============================================================

def save_plot(fig, output, name):
    fig.tight_layout()
    fig.savefig(
        output / name,
        dpi=180,
        bbox_inches="tight",
    )


def draw_sphere(ax, center, radius, color, alpha):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 14)

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
    )


def equal_axes_3d(ax, arrays):
    points = np.vstack(arrays)
    center = 0.5 * (
        points.min(axis=0) + points.max(axis=0)
    )
    radius = max(
        0.5 * np.max(
            points.max(axis=0) - points.min(axis=0)
        ),
        0.05,
    ) * 1.1

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


# ============================================================
# Plotting
# ============================================================

def plot_results(robot, obstacles, runs, args, nominal_data):
    colors = {
        run["name"]: f"C{i % 10}"
        for i, run in enumerate(runs)
    }

    def executed(run):
        return [
            r for r in run["records"] if r["executed"]
        ]

    def attempted(run):
        return [
            r for r in run["records"]
            if run["mode"] != "OPEN"
        ]

    # --------------------------------------------------------
    # 1. Tracking errors
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True
    )

    for run in runs:
        rows = executed(run)
        t = [r["time_s"] for r in rows]

        axes[0].plot(
            t,
            [r["position_error_mm"] for r in rows],
            label=run["name"],
            color=colors[run["name"]],
        )
        axes[1].plot(
            t,
            [r["orientation_error_deg"] for r in rows],
            label=run["name"],
            color=colors[run["name"]],
        )

    axes[0].set_ylabel("Position error [mm]")
    axes[1].set_ylabel("Orientation error [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("Cartesian tracking errors")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    save_plot(fig, args.output, "tracking_errors.png")

    # --------------------------------------------------------
    # 2. Joint tracking
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        7, 1, figsize=(14, 16), sharex=True
    )

    reference_times = np.arange(args.steps + 1) * args.dt
    q_ref = np.asarray([
        reference_state(t)[0]
        for t in reference_times
    ])

    for j, ax in enumerate(axes):
        ax.plot(
            reference_times,
            q_ref[:, j],
            "k--",
            linewidth=1.6,
            label="Reference",
        )
        ax.axhline(
            Q_MIN[j], color="red",
            linestyle=":", alpha=0.6
        )
        ax.axhline(
            Q_MAX[j], color="red",
            linestyle=":", alpha=0.6
        )

        for run in runs:
            path = run["q_path"]
            t = np.arange(len(path)) * args.dt

            ax.plot(
                t, path[:, j],
                color=colors[run["name"]],
                label=run["name"],
                linewidth=1.2,
            )

        ax.set_ylabel(f"q{j+1} [rad]")
        ax.grid(True)

    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Joint-space tracking and position limits")

    save_plot(fig, args.output, "joint_tracking.png")

    # --------------------------------------------------------
    # 3. Obstacle clearances
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        args.n_obstacles,
        1,
        figsize=(13, 3.4 * args.n_obstacles),
        sharex=True,
        squeeze=False,
    )

    nominal_times, nominal_clearances = nominal_data

    for j in range(args.n_obstacles):
        ax = axes[j, 0]

        ax.plot(
            nominal_times,
            nominal_clearances[:, j],
            "k--",
            linewidth=1.8,
            label="Nominal reference",
        )

        for run in runs:
            path = run["q_path"]
            env = make_environment(robot, [obstacles[j]])

            distances = np.asarray([
                obstacle_clearances(env, q)[0]
                for q in path
            ])

            t = np.arange(len(path)) * args.dt

            ax.plot(
                t, distances,
                label=run["name"],
                color=colors[run["name"]],
            )

        ax.axhline(
            args.safety_distance,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label="Safety threshold",
        )
        ax.axhline(
            0.0, color="black",
            linewidth=0.8
        )

        ax.set_ylabel(f"Obstacle {j+1} [m]")
        ax.grid(True)
        ax.legend(ncol=3, fontsize=8)

    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle("Robot-obstacle clearances", y=1.01)

    save_plot(fig, args.output, "obstacle_clearances.png")

    # --------------------------------------------------------
    # 4. MPC timing and iterations
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True
    )

    for run in runs:
        if run["mode"] == "OPEN":
            continue

        rows = attempted(run)
        t = [r["time_s"] for r in rows]
        color = colors[run["name"]]

        axes[0].plot(
            t,
            [r["compute_ms"] for r in rows],
            color=color,
            label=run["name"],
        )
        axes[1].plot(
            t,
            [r["gn_iterations"] for r in rows],
            color=color,
            label=run["name"],
        )

    axes[0].axhline(
        args.dt * 1000,
        linestyle="--",
        color="black",
        label="MPC time budget",
    )
    axes[0].set_ylabel("Compute time [ms]")
    axes[1].set_ylabel("GN iterations")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("MPC computational performance")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    save_plot(fig, args.output, "mpc_timing.png")

    # --------------------------------------------------------
    # 5. Warm-start feasibility and GN
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True
    )

    keys = [
        "warm_violation_before",
        "warm_violation_after",
        "gn_linear_violation",
    ]
    titles = [
        "Before OSQP projection",
        "After OSQP projection",
        "After Gauss-Newton",
    ]

    for run in runs:
        if run["mode"] == "OPEN":
            continue

        rows = attempted(run)
        t = np.asarray([r["time_s"] for r in rows])

        for ax, key in zip(axes, keys):
            values = np.asarray(
                [r[key] for r in rows],
                dtype=float,
            )
            valid = np.isfinite(values)

            ax.semilogy(
                t[valid],
                np.maximum(values[valid], 1e-12),
                color=colors[run["name"]],
                label=run["name"],
            )

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_ylabel("Max violation")
        ax.grid(True, which="both")
        ax.legend()

    axes[1].axhline(
        args.projection_tol,
        color="red",
        linestyle=":",
        label="Projection tolerance",
    )
    axes[2].axhline(
        args.projection_tol,
        color="red",
        linestyle=":",
        label="MPC acceptance tolerance",
    )

    axes[-1].set_xlabel("Time [s]")
    save_plot(
        fig, args.output,
        "warm_start_feasibility.png",
    )

    # --------------------------------------------------------
    # 6. Obstacle activation
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True
    )

    for run in runs:
        if run["mode"] != "SAFE":
            continue

        rows = attempted(run)
        t = [r["time_s"] for r in rows]
        color = colors[run["name"]]

        axes[0].step(
            t,
            [r["active_count"] for r in rows],
            where="post",
            label=run["name"],
            color=color,
        )
        axes[1].step(
            t,
            [r["solve_count"] for r in rows],
            where="post",
            label=run["name"],
            color=color,
        )

    axes[0].set_ylabel("Active obstacles")
    axes[1].set_ylabel("GN solves / cycle")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_ylim(-0.2, args.n_obstacles + 0.5)

    for ax in axes:
        ax.grid(True)
        ax.legend()

    save_plot(
        fig, args.output,
        "obstacle_activation.png",
    )

    # --------------------------------------------------------
    # 7. Cartesian 3D trajectories
    # --------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    reference_positions = np.asarray([
        pose_from_theta(
            robot, reference_state(t)[0]
        ).translation
        for t in reference_times
    ])

    ax.plot(
        reference_positions[:, 0],
        reference_positions[:, 1],
        reference_positions[:, 2],
        "k--",
        linewidth=2.5,
        label="Reference",
    )

    paths = [reference_positions]

    for run in runs:
        path = run["ee_path"]
        paths.append(path)

        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=colors[run["name"]],
            label=run["name"],
        )

    for j, obstacle in enumerate(obstacles):
        draw_sphere(
            ax,
            obstacle.center,
            obstacle.radius,
            color="red",
            alpha=0.25,
        )
        ax.scatter(
            *obstacle.center,
            marker="x",
            color="red",
            s=50,
        )
        ax.text(
            *obstacle.center,
            f"O{j+1}",
        )

    equal_axes_3d(ax, paths)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Cartesian trajectories - Kinova Gen3")
    ax.legend()

    save_plot(fig, args.output, "trajectories_3d.png")

    # --------------------------------------------------------
    # 8. Additional Phase 1 diagnostics
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True
    )

    for run in runs:
        if run["mode"] == "OPEN":
            continue

        rows = attempted(run)
        t = [r["time_s"] for r in rows]

        axes[0].step(
            t,
            [int(r["accepted"]) for r in rows],
            where="post",
            label=run["name"],
            color=colors[run["name"]],
        )
        axes[1].plot(
            t,
            [r["projection_ms"] for r in rows],
            label=run["name"],
            color=colors[run["name"]],
        )

    axes[0].set_ylabel("MPC accepted [0/1]")
    axes[0].set_ylim(-0.15, 1.15)
    axes[1].set_ylabel("Projection time [ms]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("Phase 1 - MPC validation diagnostics")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    save_plot(fig, args.output, "mpc_diagnostics.png")

    if args.show_plots:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--gn-iters", nargs="+", type=int, default=[2, 5]
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.35
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control-points", type=int, default=12
    )
    parser.add_argument(
        "--tracking-points", type=int, default=15
    )
    parser.add_argument(
        "--collision-points", type=int, default=20
    )
    parser.add_argument(
        "--orientation-scale", type=float, default=0.1
    )
    parser.add_argument(
        "--lambda-terminal", type=float, default=1.0
    )

    parser.add_argument(
        "--activation-distance", type=float, default=0.25
    )
    parser.add_argument(
        "--activation-samples", type=int, default=11
    )
    parser.add_argument(
        "--feasibility-tol", type=float, default=1e-4
    )
    parser.add_argument(
        "--projection-tol", type=float, default=1e-8
    )
    parser.add_argument(
        "--no-project-warm-start",
        dest="project_warm_start",
        action="store_false",
    )
    parser.set_defaults(project_warm_start=True)

    parser.add_argument(
        "--debug-all", action="store_true"
    )
    parser.add_argument(
        "--debug-projection", action="store_true"
    )
    parser.add_argument(
        "--show-plots", action="store_true"
    )

    # Obstacle configuration.
    parser.add_argument(
        "--n-obstacles", type=int, default=3
    )
    parser.add_argument(
        "--obstacle-radius", type=float, default=0.06
    )
    parser.add_argument(
        "--safety-distance", type=float, default=0.02
    )
    parser.add_argument(
        "--nominal-clearance-min", type=float, default=0.03
    )
    parser.add_argument(
        "--nominal-clearance-max", type=float, default=0.05
    )
    parser.add_argument(
        "--nominal-feasibility-buffer",
        type=float, default=0.002,
    )
    parser.add_argument(
        "--anchor-capsule", type=str, default="bracelet"
    )
    parser.add_argument(
        "--placement-samples", type=int, default=101
    )
    parser.add_argument(
        "--placement-shifts", type=int, default=16
    )
    parser.add_argument(
        "--max-obstacle-shift", type=float, default=0.5
    )
    parser.add_argument(
        "--reference-check-points", type=int, default=401
    )
    parser.add_argument(
        "--initial-buffer", type=float, default=0.04
    )
    parser.add_argument(
        "--min-sphere-gap", type=float, default=0.05
    )
    parser.add_argument(
        "--time-placement-weight", type=float, default=0.005
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/kinova_mpc_cluttered_tracking"
        ),
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error("Require 0 < dt < horizon")

    if args.steps * args.dt > DURATION + 1e-9:
        parser.error("Simulation exceeds reference duration")

    if args.n_obstacles < 1:
        parser.error("At least one obstacle is required")

    if args.noise_std < 0:
        parser.error("Noise standard deviation must be >= 0")

    if any(i < 1 for i in args.gn_iters):
        parser.error("GN iterations must be >= 1")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_reference()

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    print("\n" + "=" * 115)
    print("KINOVA GEN3 - CARTESIAN MPC BENCHMARK")
    print("PHASE 1 - WARM START + FINAL FEASIBILITY VALIDATION")
    print("=" * 115)

    print("\nGenerating static obstacles...")

    (
        obstacles,
        environment,
        placement,
        nominal_data,
    ) = generate_obstacles(robot, args)

    print("\nOBSTACLE PLACEMENT")
    print("-" * 90)

    for j, (obstacle, info) in enumerate(
        zip(obstacles, placement)
    ):
        print(
            f"Obstacle {j+1} | "
            f"center={np.round(obstacle.center, 4)} | "
            f"radius={obstacle.radius:.3f} m | "
            f"closest_t={info['closest_time']:.2f} s | "
            f"nominal_clearance={info['min_clearance']:.4f} m"
        )

    print("\nMPC CONFIGURATION")
    print("-" * 90)
    print(f"Time step              : {args.dt:.3f} s")
    print(f"Horizon                : {args.horizon:.3f} s")
    print(f"Simulation steps       : {args.steps}")
    print(f"GN budgets             : {args.gn_iters}")
    print(f"Control points         : {args.control_points}")
    print(f"Tracking samples       : {args.tracking_points}")
    print(f"Collision samples      : {args.collision_points}")
    print(f"Acceleration noise std : {args.noise_std}")
    print(f"Random seed            : {args.seed}")
    print(f"Warm-start projection  : {args.project_warm_start}")
    print(f"Projection tolerance   : {args.projection_tol:.1e}")
    print(f"Safety distance        : {args.safety_distance:.3f} m")
    print(f"Activation distance    : {args.activation_distance:.3f} m")
    print(f"Output directory       : {args.output}")

    rng = np.random.default_rng(args.seed)

    # Shared disturbances ensure fair comparisons.
    disturbances = rng.normal(
        0.0,
        args.noise_std,
        size=(args.steps, robot.nv),
    )

    runs = [
        simulate(
            "OPEN", robot, environment,
            args, disturbances, "OPEN",
        )
    ]

    for iters in args.gn_iters:
        runs.append(
            simulate(
                f"FREE-GN{iters}",
                robot,
                environment,
                args,
                disturbances,
                "FREE",
                gn_iters=iters,
            )
        )

        runs.append(
            simulate(
                f"SAFE-GN{iters}",
                robot,
                environment,
                args,
                disturbances,
                "SAFE",
                gn_iters=iters,
            )
        )

    summaries = [
        summarize(run, args) for run in runs
    ]

    print("\n" + "=" * 125)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 125)

    print(
        f"{'Experiment':<15}"
        f"{'Complete':>10}"
        f"{'Cycles':>9}"
        f"{'Mean ms':>11}"
        f"{'P95 ms':>11}"
        f"{'Pos RMS':>11}"
        f"{'Min clear':>12}"
        f"{'Projected':>12}"
        f"{'Rejected':>11}"
        f"{'GN max':>13}"
    )

    for s in summaries:
        print(
            f"{s['experiment']:<15}"
            f"{str(s['complete']):>10}"
            f"{s['cycles']:>9}"
            f"{s['mean_ms']:>11.2f}"
            f"{s['p95_ms']:>11.2f}"
            f"{s['position_rms_mm']:>11.3f}"
            f"{s['min_clearance_m']:>12.4f}"
            f"{s['projection_cycles']:>12}"
            f"{s['rejected_solutions']:>11}"
            f"{s['max_gn_violation']:>13.2e}"
        )

        if s["failure"]:
            print("    FAILURE:", s["failure"])

    print("\nDETAILED MPC STATISTICS")
    print("-" * 100)

    for s in summaries:
        print(f"\n{s['experiment']}")
        print(f"  Complete              : {s['complete']}")
        print(f"  Successful cycles     : {s['cycles']}/{args.steps}")
        print(f"  Accepted GN solutions : {s['accepted_solutions']}")
        print(f"  Rejected GN solutions : {s['rejected_solutions']}")
        print(f"  Mean compute          : {s['mean_ms']:.3f} ms")
        print(f"  Median compute        : {s['median_ms']:.3f} ms")
        print(f"  P95 compute           : {s['p95_ms']:.3f} ms")
        print(f"  Max compute           : {s['max_ms']:.3f} ms")
        print(f"  Position RMS          : {s['position_rms_mm']:.3f} mm")
        print(f"  Orientation RMS       : {s['orientation_rms_deg']:.3f} deg")
        print(f"  Joint error RMS       : {s['joint_rms_rad']:.5f} rad")
        print(f"  Minimum clearance     : {s['min_clearance_m']:.5f} m")
        print(f"  Clearance violations  : {s['clearance_violation_cycles']}")
        print(f"  Active obstacle cycles: {s['active_cycles']}")
        print(f"  Projection cycles     : {s['projection_cycles']}")
        print(f"  Mean projection time  : {s['projection_mean_ms']:.3f} ms")
        print(f"  Max warm violation    : {s['max_warm_violation']:.3e}")
        print(f"  Max GN violation      : {s['max_gn_violation']:.3e}")

    export_csv(runs, summaries, args.output)

    print("\nGenerating figures...")

    plot_results(
        robot,
        obstacles,
        runs,
        args,
        nominal_data,
    )

    print("\n" + "=" * 90)
    print("BENCHMARK FINISHED")
    print("=" * 90)
    print("Results saved to:", args.output)
    print("  - cycles.csv")
    print("  - summary.csv")
    print("  - tracking_errors.png")
    print("  - joint_tracking.png")
    print("  - obstacle_clearances.png")
    print("  - mpc_timing.png")
    print("  - warm_start_feasibility.png")
    print("  - obstacle_activation.png")
    print("  - trajectories_3d.png")
    print("  - mpc_diagnostics.png")


if __name__ == "__main__":
    main()

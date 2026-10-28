
"""Kinova Gen3 — Cartesian MPC benchmark in a cluttered environment.

Experiments:
    OPEN       : open-loop reference execution
    FREE-GN2   : MPC without collision constraints
    FREE-GN5
    SAFE-GN2   : collision-aware MPC
    SAFE-GN5

Features:
    - Native warm-start rebasing and OSQP projection
    - Feasible nonconverged GN iterate acceptance
    - Collision activation and feasibility diagnostics
    - Independent closed-loop simulations
    - Shared acceleration disturbances
    - CSV exports and diagnostic figures
    - Interactive Meshcat trajectory replay

Examples:
    python -m examples.mpc.kinova_mpc_cluttered_tracking
    python -m examples.mpc.kinova_mpc_cluttered_tracking --show-plots
    python -m examples.mpc.kinova_mpc_cluttered_tracking --gn-iters 2 5
    python -m examples.mpc.kinova_mpc_cluttered_tracking --steps 20 --no-animation
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
    reference_state,
    reference_pose,
    validate_reference,
    bound_violation,
    max_linear_violation
)
from examples.utils.collision_visualization import (
    CAPSULES as KINOVA_CAPSULES,
    capsule_endpoints,
)
from examples.utils.kinova_visualization import (
    animate_kinova_trajectories,
)

# ============================================================
# General utilities
# ============================================================


def rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (
        float(np.sqrt(np.mean(values**2)))
        if values.size else np.nan
    )


def unit(values):
    values = np.asarray(values, dtype=float)
    return values / max(np.linalg.norm(values), 1e-12)


def finite_values(rows, key):
    values = np.asarray(
        [row.get(key, np.nan) for row in rows],
        dtype=float,
    )
    return values[np.isfinite(values)]


def print_linear_failure(problem, control_points):
    """Show the largest violated linear constraint rows."""
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
# Collision geometry
# ============================================================

def make_environment(robot, obstacles):
    return CollisionEnvironment(
        robot=robot,
        capsules=KINOVA_CAPSULES,
        obstacles=obstacles,
    )


def obstacle_clearances(environment, q):
    """Minimum capsule clearance for each obstacle."""
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
    """Place separated static spheres near the feasible reference.

    Preserve the original deterministic placement search.
    The nominal reference must remain collision-feasible.
    """
    anchor_times = np.linspace(
        2.5, 7.5, args.n_obstacles,
    )
    targets = np.linspace(
        args.nominal_clearance_min,
        args.nominal_clearance_max,
        args.n_obstacles,
    )

    sample_times = np.linspace(
        0.0, DURATION, args.placement_samples,
    )
    reference_q = [
        reference_state(t)[0]
        for t in sample_times
    ]

    capsule = next(
        (
            c for c in KINOVA_CAPSULES
            if c.name == args.anchor_capsule
        ),
        None,
    )

    if capsule is None:
        raise ValueError(
            f"Unknown capsule: {args.anchor_capsule}"
        )

    directions = [
        unit(direction)
        for direction in (
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

            environment = make_environment(robot, [sphere])

            distances = np.asarray([
                obstacle_clearances(environment, q)[0]
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
            0.0,
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
                            args.safety_distance
                            + args.initial_buffer
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
                            * abs(
                                candidate["closest_time"] - anchor_t
                            )
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
        0.0, DURATION, args.reference_check_points,
    )
    distances = reference_clearances(
        environment, check_times,
    )
    minima = distances.min(axis=0)

    if np.any(minima < (
        args.safety_distance
        + args.nominal_feasibility_buffer
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
# MPC setup
# ============================================================

def create_mpc(robot, environment, args, safe, gn_iters):
    spline = make_spline(
        args.horizon,
        robot.nv,
        args.control_points,
    )

    tracking_times = np.linspace(
        0.0, args.horizon, args.tracking_points,
    )
    collision_times = np.linspace(
        0.0, args.horizon, args.collision_points,
    )

    q0, v0, _ = reference_state(0.0)
    initial_c = np.tile(
        q0, (spline.n_control_points, 1),
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
# Simulation records
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


def print_cycle_diagnostics(name, cycle, row):
    print(f"\n[DIAGNOSTICS {name} cycle {cycle}]")

    fields = {
        "Warm violation before": "warm_violation_before",
        "Warm violation after": "warm_violation_after",
        "Projection status": "projection_status",
        "Projection time [ms]": "projection_ms",
        "GN linear violation": "gn_linear_violation",
        "Solver success": "solver_success",
        "Linear feasible": "linear_feasible",
        "Collision feasible": "collision_feasible",
        "Accepted": "accepted",
    }

    for label, key in fields.items():
        print(f"  {label:<24}: {row[key]}")


# ============================================================
# Closed-loop simulation
# ============================================================

def simulate(
    name, robot, environment, args,
    disturbances, mode, gn_iters=None,
):
    q, v, _ = reference_state(0.0)

    mpc = spline = problem = collision = tracking_times = None

    if mode != "OPEN":
        mpc, spline, problem, collision, tracking_times = (
            create_mpc(
                robot,
                environment,
                args,
                safe=(mode == "SAFE"),
                gn_iters=gn_iters,
            )
        )

    e_q = np.zeros(robot.nv)
    e_v = np.zeros(robot.nv)

    records = []
    q_path = [q.copy()]
    ee_path = [
        pose_from_theta(robot, q).translation.copy()
    ]

    failure = None

    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {name}")
    print("=" * 100)

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

                tic = time.perf_counter()

                result = mpc.step(
                    q0=q,
                    v0=v,
                    target_poses=targets,
                )

                row["compute_ms"] = (
                    1000 * (time.perf_counter() - tic)
                )
                row["gn_iterations"] = getattr(
                    result, "iterations", np.nan,
                )
                row["status"] = str(result.status)

                # Native warm-start projection diagnostics.
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
                    "active_indices", (),
                )

                row["active_count"] = len(indices)
                row["active_indices"] = ",".join(
                    map(str, indices)
                )
                row["solve_count"] = collision_info.get(
                    "solve_count", 1,
                )

                # Final solution acceptance diagnostics.
                solution_info = mpc.last_solution_info or {}

                row["solver_success"] = bool(
                    solution_info.get(
                        "solver_success", result.success,
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

                # Independent linear/collision measurements.
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
                    max_linear_violation(problem, candidate_c)
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
                    print_cycle_diagnostics(
                        name, k + 1, row,
                    )

                # Never execute rejected MPC solutions.
                if not row["accepted"]:
                    print("\nMPC SOLUTION REJECTED")
                    print_cycle_diagnostics(
                        name, k + 1, row,
                    )

                    if np.isfinite(
                        row["gn_linear_violation"]
                    ):
                        print_linear_failure(
                            problem, candidate_c,
                        )

                    raise RuntimeError(
                        "MPC rejected final GN solution"
                    )

                q_command = spline.position(
                    candidate_c, args.dt,
                )
                v_command = spline.velocity(
                    candidate_c, args.dt,
                )

                q_next = (
                    q_command
                    + 0.5 * disturbance * args.dt**2
                )
                v_next = v_command + disturbance * args.dt

            # Executed-state feasibility measurements.
            actual_distances = obstacle_clearances(
                environment, q_next,
            )
            min_actual = float(actual_distances.min())

            q_violation = bound_violation(
                q_next, Q_MIN, Q_MAX,
            )
            v_violation = bound_violation(
                v_next, V_MIN, V_MAX,
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
            ep, eo = pose_errors(
                actual_pose, desired_pose,
            )

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
            failure = f"Cycle {k + 1}: {exc}"

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
        row for row in run["records"]
        if row["executed"]
    ]
    attempted = (
        run["records"]
        if run["mode"] != "OPEN"
        else []
    )

    times = finite_values(rows, "compute_ms")
    clearances = finite_values(
        rows, "min_actual_clearance",
    )
    projection_times = finite_values(
        attempted, "projection_ms",
    )

    def mean(values):
        return float(values.mean()) if values.size else np.nan

    def percentile(values, p):
        return (
            float(np.percentile(values, p))
            if values.size else np.nan
        )

    warm = finite_values(
        attempted, "warm_violation_before",
    )
    gn = finite_values(
        attempted, "gn_linear_violation",
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
            float(warm.max()) if warm.size else np.nan
        ),
        "max_gn_violation": (
            float(gn.max()) if gn.size else np.nan
        ),
        "failure": run["failure"] or "",
    }


def export_csv(runs, summaries, output):
    rows = [
        row
        for run in runs
        for row in run["records"]
    ]

    if rows:
        with (output / "cycles.csv").open(
            "w", newline="", encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)


def print_summary(summaries, args):
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
        print(
            f"  Successful cycles     : "
            f"{s['cycles']}/{args.steps}"
        )
        print(
            f"  Accepted GN solutions : "
            f"{s['accepted_solutions']}"
        )
        print(
            f"  Rejected GN solutions : "
            f"{s['rejected_solutions']}"
        )
        print(f"  Mean compute          : {s['mean_ms']:.3f} ms")
        print(f"  Median compute        : {s['median_ms']:.3f} ms")
        print(f"  P95 compute           : {s['p95_ms']:.3f} ms")
        print(f"  Max compute           : {s['max_ms']:.3f} ms")
        print(
            f"  Position RMS          : "
            f"{s['position_rms_mm']:.3f} mm"
        )
        print(
            f"  Orientation RMS       : "
            f"{s['orientation_rms_deg']:.3f} deg"
        )
        print(
            f"  Joint error RMS       : "
            f"{s['joint_rms_rad']:.5f} rad"
        )
        print(
            f"  Minimum clearance     : "
            f"{s['min_clearance_m']:.5f} m"
        )
        print(
            f"  Clearance violations  : "
            f"{s['clearance_violation_cycles']}"
        )
        print(
            f"  Active obstacle cycles: "
            f"{s['active_cycles']}"
        )
        print(
            f"  Projection cycles     : "
            f"{s['projection_cycles']}"
        )
        print(
            f"  Mean projection time  : "
            f"{s['projection_mean_ms']:.3f} ms"
        )
        print(
            f"  Max warm violation    : "
            f"{s['max_warm_violation']:.3e}"
        )
        print(
            f"  Max GN violation      : "
            f"{s['max_gn_violation']:.3e}"
        )


# ============================================================
# Visualization helpers
# ============================================================

def save_plot(fig, output, name):
    fig.tight_layout()
    fig.savefig(
        output / name,
        dpi=180,
        bbox_inches="tight",
    )


def draw_sphere(ax, center, radius, color="red", alpha=0.25):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 14)

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


def executed_rows(run):
    return [
        row for row in run["records"]
        if row["executed"]
    ]


def mpc_rows(run):
    return (
        run["records"]
        if run["mode"] != "OPEN"
        else []
    )


# ============================================================
# Diagnostic plots
# ============================================================

def plot_results(robot, obstacles, runs, args, nominal_data):
    colors = {
        run["name"]: f"C{i % 10}"
        for i, run in enumerate(runs)
    }

    reference_times = np.arange(
        args.steps + 1
    ) * args.dt

    # 1. Cartesian tracking errors
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
    )

    for run in runs:
        rows = executed_rows(run)
        t = [r["time_s"] for r in rows]

        for ax, key in zip(
            axes,
            ("position_error_mm", "orientation_error_deg"),
        ):
            ax.plot(
                t,
                [r[key] for r in rows],
                color=colors[run["name"]],
                label=run["name"],
            )

    axes[0].set_ylabel("Position error [mm]")
    axes[1].set_ylabel("Orientation error [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("Cartesian tracking errors")

    for ax in axes:
        ax.legend()

    save_plot(fig, args.output, "tracking_errors.png")

    # 2. Joint-space tracking
    fig, axes = plt.subplots(
        4, 2, figsize=(14, 12), sharex=True,
    )
    axes = axes.ravel()

    q_ref = np.asarray([
        reference_state(t)[0]
        for t in reference_times
    ])

    for j, ax in enumerate(axes[:7]):
        ax.plot(
            reference_times,
            q_ref[:, j],
            "k--",
            lw=1.7,
            label="Reference",
        )

        ax.axhline(
            Q_MIN[j], color="red", ls=":", alpha=0.6,
        )
        ax.axhline(
            Q_MAX[j], color="red", ls=":", alpha=0.6,
        )

        for run in runs:
            q_path = run["q_path"]
            t = np.arange(len(q_path)) * args.dt

            ax.plot(
                t,
                q_path[:, j],
                color=colors[run["name"]],
                label=run["name"],
                lw=1.3,
            )

        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel("Position [rad]")

        if j >= 6:
            ax.set_xlabel("Time [s]")

    axes[0].legend(ncol=2, fontsize=8)
    axes[7].axis("off")
    fig.suptitle("Joint-space tracking and position limits")

    save_plot(fig, args.output, "joint_tracking.png")

    # 3. Individual obstacle clearances
    fig, axes = plt.subplots(
        args.n_obstacles,
        1,
        figsize=(13, 3.4 * args.n_obstacles),
        sharex=True,
        squeeze=False,
    )

    nominal_times, nominal_clearances = nominal_data

    for j, obstacle in enumerate(obstacles):
        ax = axes[j, 0]

        ax.plot(
            nominal_times,
            nominal_clearances[:, j],
            "k--",
            lw=1.8,
            label="Nominal reference",
        )

        environment = make_environment(robot, [obstacle])

        for run in runs:
            q_path = run["q_path"]
            clearances = np.asarray([
                obstacle_clearances(environment, q)[0]
                for q in q_path
            ])
            t = np.arange(len(q_path)) * args.dt

            ax.plot(
                t,
                clearances,
                color=colors[run["name"]],
                label=run["name"],
            )

        ax.axhline(
            args.safety_distance,
            color="red",
            ls=":",
            lw=1.5,
            label="Safety threshold",
        )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel(f"Obstacle {j + 1} [m]")
        ax.legend(ncol=3, fontsize=8)

    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle("Robot-obstacle clearances")

    save_plot(fig, args.output, "obstacle_clearances.png")

    # 4. MPC solve times and GN iterations
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
    )

    for run in runs:
        rows = mpc_rows(run)
        if not rows:
            continue

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
        1000 * args.dt,
        color="black",
        ls="--",
        label="MPC time budget",
    )

    axes[0].set_ylabel("Compute time [ms]")
    axes[1].set_ylabel("GN iterations")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("MPC computational performance")

    for ax in axes:
        ax.legend()

    save_plot(fig, args.output, "mpc_timing.png")

    # 5. Warm-start and final GN feasibility
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True,
    )

    keys = (
        "warm_violation_before",
        "warm_violation_after",
        "gn_linear_violation",
    )
    titles = (
        "Before OSQP projection",
        "After OSQP projection",
        "After Gauss-Newton",
    )

    for run in runs:
        rows = mpc_rows(run)
        if not rows:
            continue

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
        ls=":",
        label="Projection tolerance",
    )
    axes[2].axhline(
        args.feasibility_tol,
        color="red",
        ls=":",
        label="MPC feasibility tolerance",
    )
    axes[-1].set_xlabel("Time [s]")

    save_plot(
        fig, args.output, "warm_start_feasibility.png",
    )

    # 6. Collision activation and solve count
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
    )

    for run in runs:
        if run["mode"] != "SAFE":
            continue

        rows = mpc_rows(run)
        t = [r["time_s"] for r in rows]

        axes[0].step(
            t,
            [r["active_count"] for r in rows],
            where="post",
            color=colors[run["name"]],
            label=run["name"],
        )
        axes[1].step(
            t,
            [r["solve_count"] for r in rows],
            where="post",
            color=colors[run["name"]],
            label=run["name"],
        )

    axes[0].set_ylabel("Active obstacles")
    axes[1].set_ylabel("GN solves / cycle")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_ylim(-0.2, args.n_obstacles + 0.5)

    for ax in axes:
        ax.legend()

    save_plot(fig, args.output, "obstacle_activation.png")

    # 7. Cartesian trajectories and obstacles
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    reference_positions = np.asarray([
        pose_from_theta(
            robot, reference_state(t)[0]
        ).translation.copy()
        for t in reference_times
    ])

    ax.plot(
        *reference_positions.T,
        "k--",
        lw=2.5,
        label="Reference",
    )

    paths = [reference_positions]

    for run in runs:
        path = run["ee_path"]
        paths.append(path)

        ax.plot(
            *path.T,
            color=colors[run["name"]],
            label=run["name"],
        )

    for j, obstacle in enumerate(obstacles):
        draw_sphere(
            ax,
            obstacle.center,
            obstacle.radius,
        )
        ax.scatter(
            *obstacle.center,
            marker="x",
            color="red",
            s=50,
        )
        ax.text(
            *obstacle.center,
            f"O{j + 1}",
        )

    equal_axes_3d(ax, paths)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Cartesian trajectories — Kinova Gen3")
    ax.legend()

    save_plot(fig, args.output, "trajectories_3d.png")

    # 8. MPC acceptance and projection time
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
    )

    for run in runs:
        rows = mpc_rows(run)
        if not rows:
            continue

        t = [r["time_s"] for r in rows]

        axes[0].step(
            t,
            [int(r["accepted"]) for r in rows],
            where="post",
            color=colors[run["name"]],
            label=run["name"],
        )
        axes[1].plot(
            t,
            [r["projection_ms"] for r in rows],
            color=colors[run["name"]],
            label=run["name"],
        )

    axes[0].set_ylabel("MPC accepted [0/1]")
    axes[0].set_ylim(-0.15, 1.15)
    axes[1].set_ylabel("Projection time [ms]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title("MPC acceptance and projection diagnostics")

    for ax in axes:
        ax.legend()

    save_plot(fig, args.output, "mpc_diagnostics.png")


# ============================================================
# Interactive Meshcat
# ============================================================

def animate_results(robot, obstacles, runs, args):
    """Replay OPEN, FREE and SAFE trajectories using Meshcat."""
    reference_times = np.arange(args.steps + 1) * args.dt

    reference_xyz = np.asarray([
        pose_from_theta(
            robot,
            reference_state(t)[0],
        ).translation.copy()
        for t in reference_times
    ])

    colors = (
        0x888888,
        0x2277CC,
        0xCC3366,
        0x33AA77,
        0x9933AA,
        0xDDAA22,
        0x555555,
    )

    trajectories = {}

    for i, run in enumerate(runs):
        status = (
            "complete" if run["complete"] else "stopped early"
        )

        trajectories[run["name"]] = {
            "q": run["q_path"],
            "label": (
                f"{run['name']} "
                f"({len(run['q_path']) - 1} cycles, {status})"
            ),
            "color": colors[i % len(colors)],
        }

    animate_kinova_trajectories(
        robot=robot,
        trajectories=trajectories,
        dt=args.dt,
        obstacles=obstacles,
        reference_xyz=reference_xyz,
        target_pose=reference_pose(
            robot, reference_times[-1],
        ),
        speed=args.animation_speed,
        title="CLUTTERED MPC — ANIMATION",
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--gn-iters",
        nargs="+",
        type=int,
        default=[2, 5],
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.35,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control-points", type=int, default=12,
    )
    parser.add_argument(
        "--tracking-points", type=int, default=15,
    )
    parser.add_argument(
        "--collision-points", type=int, default=20,
    )
    parser.add_argument(
        "--orientation-scale", type=float, default=0.1,
    )
    parser.add_argument(
        "--lambda-terminal", type=float, default=1.0,
    )

    # MPC collision activation and warm start.
    parser.add_argument(
        "--activation-distance", type=float, default=0.25,
    )
    parser.add_argument(
        "--activation-samples", type=int, default=11,
    )
    parser.add_argument(
        "--feasibility-tol", type=float, default=1e-4,
    )
    parser.add_argument(
        "--projection-tol", type=float, default=1e-8,
    )
    parser.add_argument(
        "--no-project-warm-start",
        dest="project_warm_start",
        action="store_false",
    )
    parser.set_defaults(project_warm_start=True)

    parser.add_argument("--debug-all", action="store_true")
    parser.add_argument(
        "--debug-projection", action="store_true",
    )

    # Obstacle configuration.
    parser.add_argument(
        "--n-obstacles", type=int, default=3,
    )
    parser.add_argument(
        "--obstacle-radius", type=float, default=0.06,
    )
    parser.add_argument(
        "--safety-distance", type=float, default=0.02,
    )
    parser.add_argument(
        "--nominal-clearance-min",
        type=float, default=0.023,
    )
    parser.add_argument(
        "--nominal-clearance-max",
        type=float, default=0.028,
    )
    parser.add_argument(
        "--nominal-feasibility-buffer",
        type=float, default=0.002,
    )
    parser.add_argument(
        "--anchor-capsule", type=str, default="bracelet",
    )
    parser.add_argument(
        "--placement-samples", type=int, default=101,
    )
    parser.add_argument(
        "--placement-shifts", type=int, default=16,
    )
    parser.add_argument(
        "--max-obstacle-shift", type=float, default=0.5,
    )
    parser.add_argument(
        "--reference-check-points", type=int, default=401,
    )
    parser.add_argument(
        "--initial-buffer", type=float, default=0.04,
    )
    parser.add_argument(
        "--min-sphere-gap", type=float, default=0.05,
    )
    parser.add_argument(
        "--time-placement-weight", type=float, default=0.005,
    )

    # Visualization and export.
    parser.add_argument(
        "--show-plots", action="store_true",
    )
    parser.add_argument(
        "--no-animation", action="store_true",
    )
    parser.add_argument(
        "--animation-speed", type=float, default=2.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kinova_mpc_cluttered_tracking"),
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error("Require 0 < dt < horizon")

    if args.steps <= 0:
        parser.error("steps must be positive")

    if args.steps * args.dt > DURATION + 1e-9:
        parser.error("Simulation exceeds reference duration")

    if args.n_obstacles < 1:
        parser.error("At least one obstacle is required")

    if args.noise_std < 0:
        parser.error("Noise standard deviation must be >= 0")

    if any(i < 1 for i in args.gn_iters):
        parser.error("GN iterations must be >= 1")

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

    print("\n" + "=" * 100)
    print("KINOVA GEN3 — CARTESIAN MPC BENCHMARK")
    print("WARM START + FINAL FEASIBILITY VALIDATION")
    print("=" * 100)

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

    # Shared disturbances for fair comparisons.
    disturbances = rng.normal(
        0.0,
        args.noise_std,
        size=(args.steps, robot.nv),
    )

    runs = [
        simulate(
            "OPEN",
            robot,
            environment,
            args,
            disturbances,
            "OPEN",
        )
    ]

    for iterations in args.gn_iters:
        for mode in ("FREE", "SAFE"):
            runs.append(
                simulate(
                    f"{mode}-GN{iterations}",
                    robot,
                    environment,
                    args,
                    disturbances,
                    mode,
                    gn_iters=iterations,
                )
            )

    summaries = [
        summarize(run, args)
        for run in runs
    ]

    print_summary(summaries, args)
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
    print("Results saved to:", args.output.resolve())

    print("\nSaved files:")
    for filename in (
        "cycles.csv",
        "summary.csv",
        "tracking_errors.png",
        "joint_tracking.png",
        "obstacle_clearances.png",
        "mpc_timing.png",
        "warm_start_feasibility.png",
        "obstacle_activation.png",
        "trajectories_3d.png",
        "mpc_diagnostics.png",
    ):
        print(f"  - {filename}")

    if args.show_plots:
        plt.show(block=False)

    if not args.no_animation:
        animate_results(robot, obstacles, runs, args)
    elif args.show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

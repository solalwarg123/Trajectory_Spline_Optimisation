
"""
Kinova Gen3 — Cluttered Cartesian MPC benchmark.

Compare:
    OPEN      : no feedback
    FREE-GN2  : Cartesian MPC, two GN iterations
    FREE-GN5  : Cartesian MPC, five GN iterations
    SAFE-GN2  : collision-aware MPC, two GN iterations
    SAFE-GN5  : collision-aware MPC, five GN iterations

Warm-start feasibility restoration is handled internally
by splineopt.mpc.CartesianMPC.

Run:
    python -m examples.kinova_mpc_cluttered_tracking \
        --gn-iters 2 5 --noise-std 0.35 --show-plots

    python -m examples.kinova_mpc_cluttered_tracking \
        --gn-iters 2 5 --noise-std 0.35 --debug-projection

    python -m examples.kinova_mpc_cluttered_tracking \
        --gn-iters 2 5 --noise-std 0.35 \
        --no-project-warm-start
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
# Utilities
# ============================================================

def rms(values):
    x = np.asarray(values, dtype=float)
    return (
        float(np.sqrt(np.mean(x**2)))
        if x.size else np.nan
    )


def unit(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    return x / max(norm, 1e-12)


def linear_violation(problem, control_points):
    A, lower, upper = problem.linear_constraints()

    x = np.asarray(
        control_points, dtype=float
    ).reshape(-1)

    values = np.asarray(A @ x).reshape(-1)

    return float(max(
        0.0,
        np.max(lower - values),
        np.max(values - upper),
    ))


def print_linear_failure(problem, c):
    A, lower, upper = problem.linear_constraints()

    values = np.asarray(
        A @ np.asarray(c).reshape(-1)
    ).reshape(-1)

    violations = np.maximum(
        np.maximum(lower - values, 0.0),
        np.maximum(values - upper, 0.0),
    )

    order = np.argsort(violations)[::-1][:5]

    n = problem.spline.n_dof
    m = problem.spline.n_control_points

    n_eq = 2 * n
    pos_end = n_eq + m * n
    vel_end = pos_end + (m - 1) * n

    print("\nLINEAR CONSTRAINT DIAGNOSTICS")

    for index in order:
        violation = violations[index]

        if violation <= 1e-10:
            continue

        if index < n:
            family = "initial_position"
            joint = index
            cp = 0

        elif index < n_eq:
            family = "initial_velocity"
            joint = index - n
            cp = 0

        elif index < pos_end:
            offset = index - n_eq
            family = "position"
            joint = offset % n
            cp = offset // n

        elif index < vel_end:
            offset = index - pos_end
            family = "velocity"
            joint = offset % n
            cp = offset // n

        else:
            offset = index - vel_end
            family = "acceleration"
            joint = offset % n
            cp = offset // n

        print(
            f"  row={index} | "
            f"{family} | "
            f"joint={joint + 1} | "
            f"cp={cp} | "
            f"value={values[index]:.6f} | "
            f"lower={lower[index]:.6f} | "
            f"upper={upper[index]:.6f} | "
            f"violation={violation:.6e}"
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
    if environment.n_obstacles == 0:
        return np.empty(0)

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
        obstacle_clearances(
            environment,
            reference_state(t)[0],
        )
        for t in times
    ])


def get_capsule(name):
    for capsule in KINOVA_CAPSULES:
        if capsule.name == name:
            return capsule

    raise ValueError(
        f"Unknown capsule: {name}"
    )


def capsule_midpoint(robot, q, capsule):
    p0, p1 = capsule_endpoints(
        robot, q, capsule
    )
    return 0.5 * (p0 + p1)


# ============================================================
# Feasible static obstacle placement
# ============================================================

def generate_obstacles(robot, args):
    """
    Generate spheres close to, but not intersecting,
    the nominal joint-space reference trajectory.
    """

    anchor_times = np.linspace(
        2.5, 7.5, args.n_obstacles
    )

    desired_clearances = np.linspace(
        args.nominal_clearance_min,
        args.nominal_clearance_max,
        args.n_obstacles,
    )

    sample_times = np.linspace(
        0.0,
        DURATION,
        args.placement_samples,
    )

    reference_q = [
        reference_state(t)[0]
        for t in sample_times
    ]

    q_start = reference_q[0]

    directions = [
        unit([1, 0, 0]),
        unit([-1, 0, 0]),
        unit([0, 1, 0]),
        unit([0, -1, 0]),
        unit([0, 0, 1]),
        unit([0, 0, -1]),
    ]

    capsule = get_capsule(args.anchor_capsule)

    obstacles = []
    placement = []

    for index, (anchor_t, target) in enumerate(
        zip(anchor_times, desired_clearances)
    ):
        q_anchor = reference_state(anchor_t)[0]

        anchor = capsule_midpoint(
            robot, q_anchor, capsule
        )

        best = None

        def evaluate_candidate(shift, direction):
            center = anchor + shift * direction

            sphere = SphereObstacle(
                center=center,
                radius=args.obstacle_radius,
                name=f"obstacle_{index + 1}",
            )

            environment = make_environment(
                robot, [sphere]
            )

            distances = np.asarray([
                obstacle_clearances(
                    environment, q
                )[0]
                for q in reference_q
            ])

            closest = int(np.argmin(distances))

            initial = obstacle_clearances(
                environment, q_start
            )[0]

            return {
                "sphere": sphere,
                "min_clearance": float(
                    distances[closest]
                ),
                "closest_time": float(
                    sample_times[closest]
                ),
                "initial_clearance": float(initial),
            }

        shifts = np.linspace(
            0,
            args.max_obstacle_shift,
            args.placement_shifts,
        )

        for direction in directions:
            previous = None

            for shift in shifts:
                current = evaluate_candidate(
                    shift, direction
                )

                if previous is not None:
                    previous_shift, previous_data = previous

                    f0 = (
                        previous_data["min_clearance"]
                        - target
                    )
                    f1 = (
                        current["min_clearance"]
                        - target
                    )

                    if f0 * f1 <= 0:
                        lo = previous_shift
                        hi = shift
                        flo = f0
                        refined = current

                        for _ in range(15):
                            mid = 0.5 * (lo + hi)

                            refined = evaluate_candidate(
                                mid, direction
                            )

                            fm = (
                                refined["min_clearance"]
                                - target
                            )

                            if flo * fm <= 0:
                                hi = mid
                            else:
                                lo = mid
                                flo = fm

                        sphere = refined["sphere"]

                        if refined[
                            "initial_clearance"
                        ] < (
                            args.safety_distance
                            + args.initial_buffer
                        ):
                            continue

                        separated = all(
                            np.linalg.norm(
                                sphere.center - other.center
                            ) >= (
                                sphere.radius
                                + other.radius
                                + args.min_sphere_gap
                            )
                            for other in obstacles
                        )

                        if not separated:
                            continue

                        score = (
                            abs(
                                refined["min_clearance"]
                                - target
                            )
                            + args.time_placement_weight
                            * abs(
                                refined["closest_time"]
                                - anchor_t
                            )
                        )

                        if (
                            best is None
                            or score < best["score"]
                        ):
                            best = {
                                **refined,
                                "anchor_time": anchor_t,
                                "score": score,
                            }

                previous = (shift, current)

        if best is None:
            raise RuntimeError(
                f"Could not place obstacle {index + 1}"
            )

        obstacles.append(best["sphere"])
        placement.append(best)

    environment = make_environment(
        robot, obstacles
    )

    check_times = np.linspace(
        0, DURATION,
        args.reference_check_points,
    )

    reference_distances = reference_clearances(
        environment, check_times
    )

    minima = reference_distances.min(axis=0)

    if np.any(
        minima
        < args.safety_distance
        + args.nominal_feasibility_buffer
    ):
        raise RuntimeError(
            "Nominal reference violates obstacle "
            f"feasibility margin: {minima}"
        )

    return (
        obstacles,
        environment,
        placement,
        (check_times, reference_distances),
    )


# ============================================================
# MPC construction
# ============================================================

def create_mpc(
    robot,
    environment,
    args,
    safe,
    gn_iters,
):

    spline = make_spline(
        args.horizon,
        robot.nv,
        args.control_points,
    )

    tracking_times = np.linspace(
        0,
        args.horizon,
        args.tracking_points,
    )

    collision_times = np.linspace(
        0,
        args.horizon,
        args.collision_points,
    )

    q0, v0, _ = reference_state(0)

    initial_c = np.tile(
        q0,
        (spline.n_control_points, 1),
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
            reference_pose(robot, tau)
            for tau in tracking_times
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

    mpc_kwargs = {}

    if safe:
        mpc_kwargs.update({
            "collision_activation_distance":
                args.activation_distance,
            "collision_check_samples":
                args.activation_samples,
            "collision_feasibility_tolerance":
                args.feasibility_tol,
        })

    mpc = CartesianMPC(
        problem=problem,
        solver=solver,
        initial_control_points=initial_c,
        project_warm_start=args.project_warm_start,
        projection_tolerance=args.projection_tol,
        **mpc_kwargs,
    )

    return (
        mpc,
        spline,
        problem,
        collision,
        tracking_times,
    )


# ============================================================
# Simulation
# ============================================================

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

    mpc = None
    spline = None
    problem = None
    collision = None
    tracking_times = None

    if mode != "OPEN":
        (
            mpc,
            spline,
            problem,
            collision,
            tracking_times,
        ) = create_mpc(
            robot,
            environment,
            args,
            safe=(mode == "SAFE"),
            gn_iters=gn_iters,
        )

    e_q = np.zeros(robot.nv)
    e_v = np.zeros(robot.nv)

    records = []

    q_path = [q.copy()]

    ee_path = [
        pose_from_theta(
            robot, q
        ).translation.copy()
    ]

    failure = None

    print("\n" + "=" * 90)
    print(f"EXPERIMENT: {name}")
    print("=" * 90)

    for k in range(args.steps):
        t = k * args.dt
        t_next = (k + 1) * args.dt

        disturbance = disturbances[k]

        row = {
            "experiment": name,
            "mode": mode,
            "cycle": k + 1,
            "time_s": t_next,
            "compute_ms": np.nan,
            "gn_iterations": np.nan,
            "status": "",
            "active_count": 0,
            "active_indices": "",
            "solve_count": 0,
            "projection_needed": False,
            "projection_status": "",
            "projection_ms": 0.0,
            "warm_violation_before": np.nan,
            "warm_violation_after": np.nan,
            "gn_linear_violation": np.nan,
            "collision_violation": np.nan,
            "min_planned_clearance": np.nan,
            "min_actual_clearance": np.nan,
            "position_error_mm": np.nan,
            "orientation_error_deg": np.nan,
            "joint_error_rad": np.nan,
            "joint_limit_violation": np.nan,
            "velocity_limit_violation": np.nan,
            "accepted": False,
        }

        candidate_c = None

        try:
            if mode == "OPEN":
                # Open-loop execution: disturbances
                # accumulate without feedback.
                e_q = (
                    e_q
                    + e_v * args.dt
                    + 0.5 * disturbance * args.dt**2
                )

                e_v = (
                    e_v
                    + disturbance * args.dt
                )

                q_ref, v_ref, _ = reference_state(
                    t_next
                )

                q_next = q_ref + e_q
                v_next = v_ref + e_v

                row["status"] = "open_loop"

            else:
                # ====================================
                # Native MPC step
                #
                # IMPORTANT:
                # No external warm-start correction.
                # No external OSQP projection.
                # CartesianMPC handles both.
                # ====================================

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

                elapsed_ms = (
                    1000.0
                    * (time.perf_counter() - tic)
                )

                row["compute_ms"] = elapsed_ms
                row["gn_iterations"] = result.iterations
                row["status"] = str(result.status)

                # ====================================
                # Native warm-start diagnostics
                # ====================================

                projection_info = (
                    mpc.last_projection_info or {}
                )

                row["projection_needed"] = bool(
                    projection_info.get(
                        "projected", False
                    )
                )

                row["projection_status"] = (
                    projection_info.get(
                        "status", ""
                    )
                )

                row["projection_ms"] = float(
                    projection_info.get(
                        "time_ms", 0.0
                    )
                )

                before = projection_info.get(
                    "before", None
                )

                after = projection_info.get(
                    "after", None
                )

                if before is not None:
                    row["warm_violation_before"] = (
                        float(before)
                    )

                if after is not None:
                    row["warm_violation_after"] = (
                        float(after)
                    )

                # ====================================
                # Obstacle activation diagnostics
                # ====================================

                collision_info = (
                    mpc.last_collision_info or {}
                )

                active_indices = collision_info.get(
                    "active_indices", ()
                )

                row["active_count"] = len(
                    active_indices
                )

                row["active_indices"] = ",".join(
                    str(i) for i in active_indices
                )

                row["solve_count"] = (
                    collision_info.get(
                        "solve_count", 1
                    )
                )

                # ====================================
                # Validate solver output
                # ====================================

                candidate_c = np.asarray(
                    result.control_points,
                    dtype=float,
                ).reshape(
                    spline.n_control_points,
                    robot.nv,
                )

                if not np.all(
                    np.isfinite(candidate_c)
                ):
                    raise RuntimeError(
                        "Non-finite GN control points"
                    )

                gn_violation = linear_violation(
                    problem,
                    candidate_c,
                )

                row["gn_linear_violation"] = (
                    gn_violation
                )

                # ====================================
                # Print projection diagnostics
                # ====================================

                if (
                    args.debug_all
                    or (
                        args.debug_projection
                        and row["projection_needed"]
                    )
                ):
                    print(
                        f"\n[PROJECTION {name} "
                        f"cycle {k+1}]"
                    )

                    print(
                        "  violation before: "
                        f"{row['warm_violation_before']:.3e}"
                    )

                    print(
                        "  violation after : "
                        f"{row['warm_violation_after']:.3e}"
                    )

                    print(
                        "  projection time : "
                        f"{row['projection_ms']:.3f} ms"
                    )

                    print(
                        "  projection status: "
                        f"{row['projection_status']}"
                    )

                    print(
                        "  GN violation    : "
                        f"{gn_violation:.3e}"
                    )

                if gn_violation > args.feasibility_tol:
                    print(
                        f"\nLINEAR FAILURE | "
                        f"{name} | cycle={k+1}"
                    )

                    print(
                        "  Warm start before:",
                        row["warm_violation_before"]
                    )

                    print(
                        "  Warm start after:",
                        row["warm_violation_after"]
                    )

                    print(
                        "  GN violation:",
                        gn_violation
                    )

                    print(
                        "  Current velocity:",
                        np.round(v, 5)
                    )

                    print(
                        "  Reference velocity:",
                        np.round(
                            reference_state(t)[1],
                            5,
                        )
                    )

                    print_linear_failure(
                        problem,
                        candidate_c,
                    )

                    raise RuntimeError(
                        "GN linear violation: "
                        f"{gn_violation:.6e}"
                    )

                # ====================================
                # Validate planned collisions
                # ====================================

                if mode == "SAFE":
                    min_planned = (
                        collision.minimum_distance(
                            candidate_c
                        )
                    )

                    collision_violation = max(
                        0.0,
                        args.safety_distance
                        - min_planned,
                    )

                    row["min_planned_clearance"] = (
                        min_planned
                    )

                    row["collision_violation"] = (
                        collision_violation
                    )

                    if (
                        collision_violation
                        > args.feasibility_tol
                    ):
                        raise RuntimeError(
                            "Planned collision violation: "
                            f"{collision_violation:.6e}"
                        )

                # ====================================
                # GN status
                #
                # Keep the existing conservative
                # behavior for now.
                # Status policy will be handled in
                # the next project step.
                # ====================================

                if (
                    not result.success
                    and str(result.status)
                    not in ("max_iterations",)
                ):
                    raise RuntimeError(
                        f"GN status: {result.status}"
                    )

                # ====================================
                # Execute first MPC interval
                # ====================================

                q_command = spline.position(
                    candidate_c,
                    args.dt,
                )

                v_command = spline.velocity(
                    candidate_c,
                    args.dt,
                )

                q_next = (
                    q_command
                    + 0.5 * disturbance * args.dt**2
                )

                v_next = (
                    v_command
                    + disturbance * args.dt
                )

                # Do not overwrite:
                # mpc.previous_control_points
                #
                # Warm-start ownership now belongs
                # exclusively to CartesianMPC.

            # ========================================
            # Executed state diagnostics
            # ========================================

            actual_distances = obstacle_clearances(
                environment, q_next
            )

            min_actual = float(
                actual_distances.min()
            )

            q_violation = bound_violation(
                q_next,
                Q_MIN,
                Q_MAX,
            )

            v_violation = bound_violation(
                v_next,
                V_MIN,
                V_MAX,
            )

            row["min_actual_clearance"] = (
                min_actual
            )

            row["joint_limit_violation"] = (
                q_violation
            )

            row["velocity_limit_violation"] = (
                v_violation
            )

            if max(
                q_violation,
                v_violation,
            ) > args.feasibility_tol:
                raise RuntimeError(
                    "Executed joint/velocity "
                    "limit violation"
                )

            # Accept executed state.
            q = np.asarray(
                q_next, dtype=float
            ).copy()

            v = np.asarray(
                v_next, dtype=float
            ).copy()

            actual_pose = pose_from_theta(
                robot, q
            )

            desired_pose = reference_pose(
                robot, t_next
            )

            ep, eo = pose_errors(
                actual_pose,
                desired_pose,
            )

            q_ref = reference_state(
                t_next
            )[0]

            row["position_error_mm"] = (
                1000.0 * ep
            )

            row["orientation_error_deg"] = (
                np.degrees(eo)
            )

            row["joint_error_rad"] = (
                np.linalg.norm(q - q_ref)
            )

            row["accepted"] = True

            q_path.append(q.copy())

            ee_path.append(
                actual_pose.translation.copy()
            )

            records.append(row)

            safety = (
                "OK"
                if min_actual >= args.safety_distance
                else "VIOLATION"
            )

            compute = row["compute_ms"]

            if mode == "OPEN":
                compute_text = "    nan"
            else:
                compute_text = f"{compute:7.1f}"

            print(
                f"[{name:12s} {k+1:03d}] "
                f"{compute_text} ms | "
                f"pos={row['position_error_mm']:7.3f} mm | "
                f"ori={row['orientation_error_deg']:6.3f} deg | "
                f"clear={min_actual:.4f} m | "
                f"obs={row['active_count']}/{args.n_obstacles} | "
                f"proj={int(row['projection_needed'])} | "
                f"GN={row['gn_linear_violation']:.1e} | "
                f"{safety}"
            )

        except Exception as exc:
            row["status"] = str(exc)
            records.append(row)

            failure = f"Cycle {k+1}: {exc}"

            print(
                f"\n[{name}] FAILED: {failure}"
            )

            if (
                problem is not None
                and candidate_c is not None
            ):
                print_linear_failure(
                    problem,
                    candidate_c,
                )

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
        if row["accepted"]
    ]

    def values(key):
        x = np.asarray([
            row[key] for row in rows
        ], dtype=float)

        return x[np.isfinite(x)]

    times = values("compute_ms")
    clearances = values("min_actual_clearance")

    projection_times = values(
        "projection_ms"
    )

    gn_violations = values(
        "gn_linear_violation"
    )

    warm_violations = values(
        "warm_violation_before"
    )

    return {
        "experiment": run["name"],
        "complete": run["complete"],
        "cycles": len(rows),
        "mean_ms": (
            float(times.mean())
            if times.size else np.nan
        ),
        "median_ms": (
            float(np.median(times))
            if times.size else np.nan
        ),
        "p95_ms": (
            float(np.percentile(times, 95))
            if times.size else np.nan
        ),
        "position_rms_mm": rms(
            values("position_error_mm")
        ),
        "orientation_rms_deg": rms(
            values("orientation_error_deg")
        ),
        "min_clearance_m": (
            float(clearances.min())
            if clearances.size else np.nan
        ),
        "collision_violation_cycles": int(
            np.sum(
                clearances < args.safety_distance
            )
        ),
        "active_cycles": sum(
            row["active_count"] > 0
            for row in rows
        ),
        "projection_cycles": sum(
            row["projection_needed"]
            for row in rows
        ),
        "projection_mean_ms": (
            float(projection_times.mean())
            if projection_times.size else np.nan
        ),
        "max_warm_violation": (
            float(warm_violations.max())
            if warm_violations.size else np.nan
        ),
        "max_gn_violation": (
            float(gn_violations.max())
            if gn_violations.size else np.nan
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
        with (
            output / "cycles.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(rows)

    with (
        output / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summaries)


# ============================================================
# Plotting
# ============================================================

def draw_sphere(ax, center, radius, color, alpha):
    u = np.linspace(
        0, 2 * np.pi, 24
    )
    v = np.linspace(
        0, np.pi, 14
    )

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
        points.min(axis=0)
        + points.max(axis=0)
    )

    radius = max(
        0.5 * np.max(
            points.max(axis=0)
            - points.min(axis=0)
        ),
        0.05,
    ) * 1.1

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )
    ax.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )
    ax.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )
    ax.set_box_aspect((1, 1, 1))


def plot_results(
    robot,
    obstacles,
    runs,
    args,
    nominal_data,
):
    ref_times, ref_clearances = nominal_data

    colors = {
        run["name"]: f"C{i % 10}"
        for i, run in enumerate(runs)
    }

    def accepted(run):
        return [
            row for row in run["records"]
            if row["accepted"]
        ]

    # --------------------------------------------------------
    # 1. Cartesian tracking
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
    )

    for run in runs:
        rows = accepted(run)

        if not rows:
            continue

        t = [
            row["time_s"] for row in rows
        ]

        axes[0].plot(
            t,
            [
                row["position_error_mm"]
                for row in rows
            ],
            color=colors[run["name"]],
            label=run["name"],
        )

        axes[1].plot(
            t,
            [
                row["orientation_error_deg"]
                for row in rows
            ],
            color=colors[run["name"]],
            label=run["name"],
        )

    axes[0].set_ylabel("Position error [mm]")
    axes[1].set_ylabel("Orientation error [deg]")
    axes[1].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Cartesian tracking under disturbances"
    )
    fig.tight_layout()
    fig.savefig(
        args.output / "tracking_errors.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # 2. Obstacle clearances
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        args.n_obstacles,
        1,
        figsize=(12, 3.3 * args.n_obstacles),
        sharex=True,
        squeeze=False,
    )

    for j, obstacle in enumerate(obstacles):
        ax = axes[j, 0]

        ax.plot(
            ref_times,
            ref_clearances[:, j],
            "k--",
            label="Nominal",
        )

        env_j = make_environment(
            robot, [obstacle]
        )

        for run in runs:
            path = run["q_path"]
            t = np.arange(len(path)) * args.dt

            clearances = [
                obstacle_clearances(
                    env_j, q
                )[0]
                for q in path
            ]

            ax.plot(
                t,
                clearances,
                color=colors[run["name"]],
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
        )

        ax.set_ylabel("Clearance [m]")
        ax.set_title(obstacle.name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time [s]")

    fig.suptitle("Robot-obstacle clearances")
    fig.tight_layout()
    fig.savefig(
        args.output / "obstacle_clearances.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # 3. MPC timings
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
    )

    for run in runs:
        if run["mode"] == "OPEN":
            continue

        rows = accepted(run)
        t = [r["time_s"] for r in rows]

        axes[0].plot(
            t,
            [r["compute_ms"] for r in rows],
            color=colors[run["name"]],
            label=run["name"],
        )

        axes[1].plot(
            t,
            [r["gn_iterations"] for r in rows],
            color=colors[run["name"]],
            label=run["name"],
        )

    axes[0].axhline(
        args.dt * 1000,
        color="black",
        linestyle="--",
        label="Control period",
    )

    axes[0].set_ylabel("Total MPC time [ms]")
    axes[1].set_ylabel("GN iterations")
    axes[1].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("MPC computation")
    fig.tight_layout()
    fig.savefig(
        args.output / "mpc_timing.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # 4. Obstacle activation
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
    )

    for run in runs:
        if run["mode"] != "SAFE":
            continue

        rows = accepted(run)
        t = [r["time_s"] for r in rows]

        axes[0].step(
            t,
            [r["active_count"] for r in rows],
            where="post",
            label=run["name"],
        )

        axes[1].step(
            t,
            [r["solve_count"] for r in rows],
            where="post",
            label=run["name"],
        )

    axes[0].set_ylabel("Active obstacles")
    axes[1].set_ylabel("Solves per cycle")
    axes[1].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Obstacle activation")
    fig.tight_layout()
    fig.savefig(
        args.output / "obstacle_activation.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # 5. Native projection diagnostics
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 10),
        sharex=True,
    )

    fields = [
        "warm_violation_before",
        "warm_violation_after",
        "gn_linear_violation",
    ]

    labels = [
        "Before projection",
        "After projection",
        "After Gauss-Newton",
    ]

    for run in runs:
        if run["mode"] == "OPEN":
            continue

        rows = accepted(run)
        t = [r["time_s"] for r in rows]

        for ax, field in zip(axes, fields):
            values = np.array([
                r[field] for r in rows
            ], dtype=float)

            values = np.where(
                np.isfinite(values),
                np.maximum(values, 1e-12),
                np.nan,
            )

            ax.semilogy(
                t,
                values,
                color=colors[run["name"]],
                label=run["name"],
            )

    for ax, label in zip(axes, labels):
        ax.axhline(
            args.feasibility_tol,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Tolerance",
        )
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")

    fig.suptitle(
        "Native MPC warm-start feasibility"
    )
    fig.tight_layout()
    fig.savefig(
        args.output / "warm_start_feasibility.png",
        dpi=180,
    )

    # --------------------------------------------------------
    # 6. 3D Cartesian trajectories
    # --------------------------------------------------------

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ref_t = np.linspace(
        0, DURATION, 301
    )

    ref_ee = np.asarray([
        reference_pose(
            robot, t
        ).translation.copy()
        for t in ref_t
    ])

    ax.plot(
        ref_ee[:, 0],
        ref_ee[:, 1],
        ref_ee[:, 2],
        "k--",
        linewidth=2,
        label="Nominal EE",
    )

    all_points = [ref_ee]

    for run in runs:
        path = run["ee_path"]

        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=colors[run["name"]],
            linewidth=2,
            label=run["name"],
        )

        all_points.append(path)

    for j, sphere in enumerate(obstacles):
        color = f"C{j + 5}"

        draw_sphere(
            ax,
            sphere.center,
            sphere.radius,
            color,
            0.35,
        )

        draw_sphere(
            ax,
            sphere.center,
            sphere.radius + args.safety_distance,
            color,
            0.055,
        )

        ax.text(
            *sphere.center,
            f"O{j + 1}",
        )

        radius = (
            sphere.radius
            + args.safety_distance
        )

        all_points.append(
            np.array([
                sphere.center - radius,
                sphere.center + radius,
            ])
        )

    equal_axes_3d(ax, all_points)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        "Kinova Gen3 — cluttered tracking"
    )

    ax.view_init(
        elev=23,
        azim=-55,
    )

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
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
        "--gn-iters",
        nargs="+",
        type=int,
        default=[2, 5],
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--orientation-scale",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lambda-terminal",
        type=float,
        default=1.0,
    )

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

    parser.add_argument(
        "--feasibility-tol",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--projection-tol",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--no-project-warm-start",
        dest="project_warm_start",
        action="store_false",
    )

    parser.set_defaults(
        project_warm_start=True
    )

    parser.add_argument(
        "--debug-all",
        action="store_true",
    )

    parser.add_argument(
        "--debug-projection",
        action="store_true",
    )

    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--safety-distance",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--nominal-clearance-min",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--nominal-clearance-max",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--nominal-feasibility-buffer",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--anchor-capsule",
        type=str,
        default="bracelet",
    )

    parser.add_argument(
        "--placement-samples",
        type=int,
        default=101,
    )

    parser.add_argument(
        "--placement-shifts",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-obstacle-shift",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--reference-check-points",
        type=int,
        default=401,
    )

    parser.add_argument(
        "--initial-buffer",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--min-sphere-gap",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--time-placement-weight",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/kinova_mpc_cluttered_tracking"
        ),
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
    )

    args = parser.parse_args()

    if not 0 < args.dt < args.horizon:
        parser.error(
            "Require 0 < dt < horizon."
        )

    if args.steps * args.dt > DURATION + 1e-9:
        parser.error(
            "Simulation exceeds reference duration."
        )

    if args.n_obstacles < 1:
        parser.error(
            "n-obstacles must be >= 1."
        )

    if args.noise_std < 0:
        parser.error(
            "noise-std must be non-negative."
        )

    if any(i < 1 for i in args.gn_iters):
        parser.error(
            "GN iterations must be positive."
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

    print("\n" + "=" * 90)
    print("KINOVA MPC — NATIVE FEASIBLE WARM START")
    print("=" * 90)

    (
        obstacles,
        environment,
        placement,
        nominal_data,
    ) = generate_obstacles(
        robot, args
    )

    print("\nOBSTACLE PLACEMENT")

    for i, (obstacle, info) in enumerate(
        zip(obstacles, placement)
    ):
        print(
            f"O{i+1} | "
            f"center={np.round(obstacle.center, 4)} | "
            f"radius={obstacle.radius:.3f} m | "
            f"closest t={info['closest_time']:.2f}s | "
            f"nominal clearance="
            f"{info['min_clearance']:.4f} m"
        )

    print("\nMPC CONFIGURATION")

    print(f"dt: {args.dt:.3f} s")
    print(f"Horizon: {args.horizon:.3f} s")
    print(f"GN iteration budgets: {args.gn_iters}")
    print(f"Acceleration noise std: {args.noise_std}")
    print(f"Native projection: {args.project_warm_start}")
    print(f"Projection tolerance: {args.projection_tol:.1e}")
    print(f"Safety distance: {args.safety_distance:.3f} m")
    print(f"Activation distance: {args.activation_distance:.3f} m")

    rng = np.random.default_rng(
        args.seed
    )

    disturbances = rng.normal(
        0.0,
        args.noise_std,
        size=(args.steps, robot.nv),
    )

    runs = []

    runs.append(
        simulate(
            "OPEN",
            robot,
            environment,
            args,
            disturbances,
            "OPEN",
        )
    )

    for iters in args.gn_iters:
        runs.append(
            simulate(
                f"FREE-GN{iters}",
                robot,
                environment,
                args,
                disturbances,
                "FREE",
                iters,
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
                iters,
            )
        )

    summaries = [
        summarize(run, args)
        for run in runs
    ]

    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    print(
        f"{'Experiment':<15}"
        f"{'Complete':>10}"
        f"{'Cycles':>9}"
        f"{'Mean ms':>11}"
        f"{'P95 ms':>11}"
        f"{'Pos RMS':>11}"
        f"{'Min clear':>12}"
        f"{'Projections':>13}"
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
            f"{s['projection_cycles']:>13}"
            f"{s['max_gn_violation']:>13.2e}"
        )

        if s["failure"]:
            print(
                f"    Failure: {s['failure']}"
            )

    export_csv(
        runs,
        summaries,
        args.output,
    )

    plot_results(
        robot,
        obstacles,
        runs,
        args,
        nominal_data,
    )

    print(
        "\nOutput directory:",
        args.output,
    )


if __name__ == "__main__":
    main()

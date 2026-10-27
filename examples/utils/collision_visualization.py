
"""Collision utilities for the Kinova offline examples.

Capsules, obstacle generation, collision constraints, validation,
Matplotlib figures and interactive Meshcat playback.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from splineopt.collision import (
    Capsule, SphereObstacle, CollisionEnvironment,
    CollisionConstraintBuilder,
)

CAPSULES = [
    Capsule("upper_arm_1", "shoulder_link", "half_arm_1_link", 0.075),
    Capsule("upper_arm_2", "half_arm_1_link", "half_arm_2_link", 0.075),
    Capsule("forearm", "half_arm_2_link", "forearm_link", 0.070),
    Capsule("wrist_1", "forearm_link", "spherical_wrist_1_link", 0.060),
    Capsule("wrist_2", "spherical_wrist_1_link",
            "spherical_wrist_2_link", 0.055),
    Capsule("bracelet", "spherical_wrist_2_link", "bracelet_link", 0.055),
    Capsule("tool", "bracelet_link", "end_effector_link", 0.050),
]

COLORS = {
    "initial": "0.5",
    "free": "tab:orange",
    "gn": "tab:purple",
    "ipopt": "tab:blue",
}
MESHCAT_COLORS = {
    "initial": 0x888888,
    "free": 0xFF8800,
    "gn": 0x9933CC,
    "ipopt": 0x0077FF,
}


def capsule_endpoints(robot, theta, capsule):
    q = robot.joint_positions_to_configuration(theta)
    return (
        robot.frame_position(q, frame=capsule.start_frame),
        robot.frame_position(q, frame=capsule.end_frame),
    )


def point_clearance(robot, theta, point, capsules=CAPSULES):
    """Minimum surface distance from a point to the robot capsules."""
    distances = []
    for capsule in capsules:
        a, b = capsule_endpoints(robot, theta, capsule)
        ab = b - a
        u = np.clip(
            np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-12),
            0.0, 1.0,
        )
        nearest = a + u * ab
        distances.append(np.linalg.norm(point - nearest) - capsule.radius)
    return float(min(distances))


def perpendicular(axis, angle=0.0):
    axis = np.asarray(axis, dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-12)
    seed = np.array([0., 0., 1.])
    if abs(np.dot(seed, axis)) > 0.9:
        seed = np.array([1., 0., 0.])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return np.cos(angle) * u + np.sin(angle) * v


def make_collision_builder(
    robot, spline, obstacles, safety=0.02, samples=50,
):
    environment = CollisionEnvironment(
        robot=robot, capsules=CAPSULES, obstacles=list(obstacles),
    )
    times = np.linspace(spline.t_start, spline.t_end, samples)
    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=times,
        safety_distance=safety,
    )
    return builder


def clearance_curve(builder, controls):
    distances = np.asarray(builder.distances(controls))
    if distances.ndim != 2:
        raise ValueError(f"Unexpected distance array: {distances.shape}")
    if distances.shape[1] == 0:
        return np.full(distances.shape[0], np.inf)
    return np.min(distances, axis=1)


def place_avoidance_obstacle(
    robot, spline, free_controls, q_initial, safety=0.02,
    radius_max=0.07,
):
    """Obstacle on the FREE path, away from the initial robot."""
    best = None
    t0, tf = spline.t_start, spline.t_end
    names = {"forearm", "wrist_1", "wrist_2", "bracelet"}

    for t in np.linspace(t0 + 0.30*(tf-t0), t0 + 0.85*(tf-t0), 60):
        theta = spline.position(free_controls, t)
        for capsule in CAPSULES:
            if capsule.name not in names:
                continue
            a, b = capsule_endpoints(robot, theta, capsule)
            center = 0.5*(a+b)
            score = point_clearance(robot, q_initial, center)
            if best is None or score > best["score"]:
                best = {
                    "center": center.copy(),
                    "time": float(t),
                    "capsule": capsule.name,
                    "score": score,
                }

    if best is None:
        raise RuntimeError("No valid obstacle candidate.")

    max_radius = best["score"] - safety - 0.015
    if max_radius <= 0.02:
        raise RuntimeError("Could not place a safe initial obstacle.")

    obstacle = SphereObstacle(
        name="automatic_obstacle",
        center=best["center"],
        radius=min(radius_max, 0.8*max_radius),
    )
    return obstacle, best


def place_solver_obstacle(
    robot, spline, initial_controls, safety=0.02, radius=0.05,
):
    """Find a bracelet obstacle with safe start and shallow QP penetration.

    The search is deliberately bounded and reproducible.
    """
    capsule = next(c for c in CAPSULES if c.name == "bracelet")
    t0, tf = spline.t_start, spline.t_end
    target_time = t0 + 0.55*(tf-t0)

    theta_target = spline.position(initial_controls, target_time)
    a, b = capsule_endpoints(robot, theta_target, capsule)
    midpoint = (a+b)/2

    q_start = spline.position(initial_controls, t0)
    best = None
    search_times = np.linspace(t0, tf, 101)
    theta_samples = [
        spline.position(initial_controls, t) for t in search_times
    ]

    for angle in np.linspace(0, 2*np.pi, 36, endpoint=False):
        direction = perpendicular(b-a, angle)

        for gap in np.linspace(-0.01, 0.12, 45):
            center = midpoint + direction*(capsule.radius+radius+gap)

            initial = point_clearance(robot, q_start, center)-radius
            if initial < safety + 0.02:
                continue

            curve = np.array([
                point_clearance(robot, q, center)-radius
                for q in theta_samples
            ])
            index = int(np.argmin(curve))
            t_min = search_times[index]
            minimum = float(curve[index])

            # Require a small but meaningful penetration around mid-motion.
            if not (t0+0.35*(tf-t0) <= t_min <= t0+0.75*(tf-t0)):
                continue
            if not (-0.025 <= minimum <= safety-0.002):
                continue

            score = abs(minimum+0.005) + 0.01*abs(
                (t_min-t0)/(tf-t0)-0.55
            )
            if best is None or score < best["score"]:
                best = {
                    "center": center.copy(), "time": float(t_min),
                    "minimum": minimum, "score": score,
                    "capsule": capsule.name,
                }

    if best is None:
        raise RuntimeError(
            "No suitable solver obstacle found; "
            "try the avoidance scenario."
        )

    return SphereObstacle(
        name="solver_obstacle", center=best["center"], radius=radius,
    ), best


def evaluate_collision_results(
    robot, spline, controls_by_name, target_times,
    target_poses, obstacle, safety, dense_samples=400,
):
    times = np.linspace(spline.t_start, spline.t_end, dense_samples)
    builder = make_collision_builder(
        robot, spline, [obstacle], safety, dense_samples,
    )

    ref_xyz = np.array([p.translation for p in target_poses])
    reference = np.column_stack([
        np.interp(times, target_times, ref_xyz[:, j])
        for j in range(3)
    ])

    results = {}
    for name, controls in controls_by_name.items():
        theta = np.array([spline.position(controls, t) for t in times])
        q_pin = robot.joint_trajectory_to_configurations(theta)
        xyz = np.array([
            robot.frame_pose(q).translation for q in q_pin
        ])
        error = np.linalg.norm(xyz-reference, axis=1)
        clearances = clearance_curve(builder, controls)

        results[name] = {
            "q": theta,
            "xyz": xyz,
            "position_error": error,
            "clearance": clearances,
            "rms": float(np.sqrt(np.mean(error**2))),
            "minimum": float(np.min(clearances)),
        }

    return times, reference, results


def print_collision_metrics(results, times_by_name, safety, tol=1e-3):
    print("\n" + "="*81)
    print("COLLISION OPTIMIZATION RESULTS")
    print("="*81)
    print(
        f"{'Method':<12} {'Pos RMS [mm]':>15} "
        f"{'Clearance [mm]':>18} {'Solve [s]':>12} {'Safe':>8}"
    )
    for name, result in results.items():
        minimum = result["minimum"]
        print(
            f"{name:<12} {1000*result['rms']:>15.2f} "
            f"{1000*minimum:>18.2f} "
            f"{times_by_name.get(name, 0.):>12.3f} "
            f"{'PASS' if minimum >= safety-tol else 'FAIL':>8}"
        )


def plot_collision_results(times, reference, results, obstacle, safety):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(*reference.T, "k--", lw=1.6, label="Cartesian reference")
    for name, result in results.items():
        xyz = result["xyz"]
        ax.plot(*xyz.T, color=COLORS[name], lw=2.2, label=name.upper())

    u = np.linspace(0, 2*np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    x = obstacle.center[0] + obstacle.radius*np.outer(np.cos(u), np.sin(v))
    y = obstacle.center[1] + obstacle.radius*np.outer(np.sin(u), np.sin(v))
    z = obstacle.center[2] + obstacle.radius*np.outer(
        np.ones_like(u), np.cos(v)
    )
    ax.plot_surface(x, y, z, color="tab:red", alpha=0.25)
    ax.set(
        title="End-effector trajectories and obstacle",
        xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
    )
    ax.legend()
    fig.tight_layout()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, result in results.items():
        axes[0].plot(
            times, result["position_error"]*1000,
            color=COLORS[name], label=name.upper(),
        )
        axes[1].plot(
            times, result["clearance"]*1000,
            color=COLORS[name], label=name.upper(),
        )

    axes[1].axhspan(-200, 1000*safety, color="tab:red", alpha=0.08)
    axes[1].axhline(
        1000*safety, color="tab:red", linestyle="--",
        label="Safety threshold",
    )
    axes[0].set_ylabel("Position error [mm]")
    axes[1].set_ylabel("Minimum clearance [mm]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()


def animate_collision_trajectories(
    robot, results, times, obstacle, speed=2.0,
):
    """Replay trajectories until 0; keep Matplotlib windows visible."""
    import meshcat.geometry as mg
    import meshcat.transformations as mt

    robot.init_viewer(open_browser=True)
    viewer = robot.viewer.viewer

    # Static obstacle and collision geometry.
    node = viewer["collision/obstacle"]
    node.set_object(
        mg.Sphere(obstacle.radius),
        mg.MeshLambertMaterial(color=0xFF4444, opacity=0.5, transparent=True),
    )
    node.set_transform(mt.translation_matrix(obstacle.center))

    for name, result in results.items():
        points = np.asarray(result["xyz"], dtype=np.float32).T
        viewer[f"trajectories/{name}"].set_object(
            mg.Line(
                mg.PointsGeometry(points),
                mg.LineBasicMaterial(
                    color=MESHCAT_COLORS[name], linewidth=3
                ),
            )
        )

    # Display moving capsule approximations.
    def display_capsules(q):
        for capsule in CAPSULES:
            a = robot.frame_position(q, frame=capsule.start_frame)
            b = robot.frame_position(q, frame=capsule.end_frame)
            center = (a+b)/2
            direction = b-a
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            rotation = np.eye(3)
            y = direction / length
            x = perpendicular(y)
            z = np.cross(x, y)
            rotation[:, 0], rotation[:, 1], rotation[:, 2] = x, y, z
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = center

            path = viewer[f"collision/capsules/{capsule.name}"]
            path.set_object(
                mg.Cylinder(length, capsule.radius),
                mg.MeshLambertMaterial(
                    color=0x33AAFF, opacity=0.25, transparent=True
                ),
            )
            path.set_transform(transform)

    choices = list(results)
    try:
        while True:
            print("\n" + "="*45)
            print("COLLISION ANIMATION")
            print("="*45)
            for i, name in enumerate(choices, 1):
                print(f"  {i}. {name.upper()}")
            print("  0. Close plots and exit")

            try:
                selected = input("\nSelect trajectory: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if selected == "0":
                break
            if not selected.isdigit() or not (
                1 <= int(selected) <= len(choices)
            ):
                print("Invalid selection.")
                continue

            name = choices[int(selected)-1]
            theta = results[name]["q"]
            q_pin = robot.joint_trajectory_to_configurations(theta)

            print(f"Playing {name.upper()}...")
            for k, q in enumerate(q_pin):
                robot.display(q)
                display_capsules(q)
                if k + 1 < len(times):
                    dt = (times[k + 1] - times[k]) / speed
                    time.sleep(max(0.0, dt))
            print("Finished.")

    finally:
        plt.close("all")

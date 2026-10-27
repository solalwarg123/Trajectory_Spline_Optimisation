
"""Shared interactive Meshcat visualization for Kinova examples.

Supports:
- Multiple joint-space trajectories and replay selection
- Uniform or explicitly sampled time vectors
- Reference end-effector path
- Static spherical obstacles
- Optional moving collision capsules
- Repeated playback, with 0 to close Matplotlib and exit

No optimization or simulation is performed here.
"""

import time

import matplotlib.pyplot as plt
import meshcat.geometry as mg
import meshcat.transformations as mt
import numpy as np


COLORS = (
    0x2277CC,
    0xCC3366,
    0x33AA77,
    0x9933AA,
    0xDDAA22,
    0x888888,
    0x00AAAA,
)

REFERENCE_COLOR = 0xFF9900
OBSTACLE_COLOR = 0xDD3333
CAPSULE_COLOR = 0x33AAFF


def _draw_line(viewer, name, points, color):
    """Display a Cartesian path if at least two points exist."""
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"{name}: expected Cartesian points of shape (N, 3)"
        )

    if len(points) < 2:
        return

    viewer[f"trajectories/{name}"].set_object(
        mg.Line(
            mg.PointsGeometry(
                points.astype(np.float32).T
            ),
            mg.LineBasicMaterial(
                color=color,
                linewidth=3,
            ),
        )
    )


def _perpendicular(direction):
    """Return a stable unit vector perpendicular to direction."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    seed = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(seed, direction)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])

    normal = np.cross(direction, seed)
    return normal / np.linalg.norm(normal)


def _display_capsules(viewer, robot, configuration, capsules):
    """Update simplified collision capsules for the current pose."""
    for capsule in capsules:
        a = robot.frame_position(
            configuration, frame=capsule.start_frame,
        )
        b = robot.frame_position(
            configuration, frame=capsule.end_frame,
        )

        direction = b - a
        length = float(np.linalg.norm(direction))

        if length < 1e-9:
            continue

        y_axis = direction / length
        x_axis = _perpendicular(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        transform = np.eye(4)
        transform[:3, :3] = np.column_stack(
            (x_axis, y_axis, z_axis)
        )
        transform[:3, 3] = 0.5 * (a + b)

        node = viewer[f"collision/capsules/{capsule.name}"]

        node.set_object(
            mg.Cylinder(length, capsule.radius),
            mg.MeshLambertMaterial(
                color=CAPSULE_COLOR,
                opacity=0.25,
                transparent=True,
            ),
        )
        node.set_transform(transform)


def animate_kinova_trajectories(
    robot,
    trajectories,
    dt=None,
    *,
    obstacles=None,
    reference_xyz=None,
    capsules=None,
    target_pose=None,
    speed=2.0,
    title="KINOVA ANIMATION",
):
    """Visualize and replay Kinova joint trajectories.

    Parameters
    ----------
    robot:
        Kinova Robot wrapper.

    trajectories:
        Mapping from display name to either:

            q_array

        or:

            {
                "q": q_array,
                "times": time_array,  # optional if dt is set
                "color": 0x2277CC,    # optional
                "label": "Description"  # optional
            }

        q_array has shape (N, robot.nv).

    dt:
        Default sampling interval in seconds. Each trajectory
        may instead provide an explicit time vector.

    obstacles:
        Optional list of SphereObstacle objects.

    reference_xyz:
        Optional reference Cartesian path of shape (N, 3).

    capsules:
        Optional collision Capsule objects to display and update.

    target_pose:
        Optional final SE3 target to display in Meshcat.

    speed:
        Playback speed multiplier.
    """
    if not trajectories:
        raise ValueError("At least one trajectory is required")

    if speed <= 0:
        raise ValueError("Playback speed must be positive")

    obstacles = [] if obstacles is None else list(obstacles)
    capsules = [] if capsules is None else list(capsules)

    prepared = {}

    # Validate all data before opening the viewer.
    for index, (name, entry) in enumerate(trajectories.items()):
        if isinstance(entry, dict):
            q = entry["q"]
            times = entry.get("times")
            color = entry.get("color", COLORS[index % len(COLORS)])
            label = entry.get("label", name)
        else:
            q = entry
            times = None
            color = COLORS[index % len(COLORS)]
            label = name

        q = np.asarray(q, dtype=float)

        if q.ndim != 2 or q.shape[1] != robot.nv:
            raise ValueError(
                f"{name}: expected joint array (N, {robot.nv})"
            )

        if len(q) == 0 or not np.all(np.isfinite(q)):
            raise ValueError(
                f"{name}: trajectory must be nonempty and finite"
            )

        if times is None:
            if dt is None or dt <= 0:
                raise ValueError(
                    f"{name}: provide dt or an explicit time vector"
                )
            times = np.arange(len(q), dtype=float) * dt
        else:
            times = np.asarray(times, dtype=float)

        if (
            times.shape != (len(q),)
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                f"{name}: invalid trajectory timestamps"
            )

        prepared[name] = {
            "q": q,
            "times": times,
            "color": int(color),
            "label": str(label),
        }

    print("\nOpening Meshcat...")
    robot.init_viewer(open_browser=True)

    viewer = robot.viewer.viewer

    # Draw static obstacles.
    for index, obstacle in enumerate(obstacles, start=1):
        node = viewer[f"obstacles/obstacle_{index}"]

        node.set_object(
            mg.Sphere(float(obstacle.radius)),
            mg.MeshLambertMaterial(
                color=OBSTACLE_COLOR,
                opacity=0.5,
                transparent=True,
            ),
        )
        node.set_transform(
            mt.translation_matrix(obstacle.center)
        )

    # Draw the reference path.
    if reference_xyz is not None:
        _draw_line(
            viewer,
            "reference",
            reference_xyz,
            REFERENCE_COLOR,
        )

    if target_pose is not None:
        robot.display_target(
            target_pose,
            name="final_reference",
        )

    # Convert each trajectory only once.
    for index, (name, data) in enumerate(prepared.items()):
        configurations = (
            robot.joint_trajectory_to_configurations(data["q"])
        )

        xyz = np.asarray([
            robot.frame_pose(q).translation.copy()
            for q in configurations
        ])

        data["configurations"] = configurations

        _draw_line(
            viewer,
            f"method_{index}",
            xyz,
            data["color"],
        )

    choices = list(prepared)

    def display_configuration(configuration):
        robot.display(configuration)

        if capsules:
            _display_capsules(
                viewer, robot, configuration, capsules,
            )

    try:
        display_configuration(
            prepared[choices[0]]["configurations"][0]
        )

        while True:
            print("\n" + "=" * 50)
            print(title)
            print("=" * 50)

            for index, name in enumerate(choices, start=1):
                data = prepared[name]
                print(
                    f"  {index}. {data['label']} "
                    f"({len(data['q']) - 1} intervals)"
                )

            print("  0. Close plots and exit")

            try:
                choice = input(
                    "\nSelect trajectory: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "0":
                break

            if not choice.isdigit():
                print("Invalid selection.")
                continue

            index = int(choice) - 1

            if not 0 <= index < len(choices):
                print("Invalid selection.")
                continue

            name = choices[index]
            data = prepared[name]

            configurations = data["configurations"]
            times = data["times"]

            print(f"\nPlaying {data['label']}...")

            display_configuration(configurations[0])
            time.sleep(0.2)

            for k, configuration in enumerate(configurations):
                display_configuration(configuration)

                if k + 1 < len(configurations):
                    delay = (
                        times[k + 1] - times[k]
                    ) / speed
                    time.sleep(max(0.0, float(delay)))

            print("Animation finished.")

    finally:
        plt.close("all")
        print("Animation session closed.")

from pathlib import Path
import time

import numpy as np
import meshcat.geometry as mg
import meshcat.transformations as mt

from splineopt.robotics import Robot
from splineopt.collision import (
    Capsule,
    SphereObstacle,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


# ============================================================
# Kinova collision model
# ============================================================


KINOVA_CAPSULES = [
    Capsule(
        name="upper_arm_1",
        start_frame="shoulder_link",
        end_frame="half_arm_1_link",
        radius=0.06,
    ),
    Capsule(
        name="upper_arm_2",
        start_frame="half_arm_1_link",
        end_frame="half_arm_2_link",
        radius=0.06,
    ),
    Capsule(
        name="forearm",
        start_frame="half_arm_2_link",
        end_frame="forearm_link",
        radius=0.055,
    ),
    Capsule(
        name="wrist_1",
        start_frame="forearm_link",
        end_frame="spherical_wrist_1_link",
        radius=0.05,
    ),
    Capsule(
        name="wrist_2",
        start_frame="spherical_wrist_1_link",
        end_frame="spherical_wrist_2_link",
        radius=0.045,
    ),
    Capsule(
        name="bracelet",
        start_frame="spherical_wrist_2_link",
        end_frame="bracelet_link",
        radius=0.045,
    ),
    Capsule(
        name="tool",
        start_frame="bracelet_link",
        end_frame="end_effector_link",
        radius=0.04,
    ),
]


OBSTACLES = [
    SphereObstacle(
        name="obstacle_1",
        center=np.array([
            0.40,
            0.12,
            0.45,
        ]),
        radius=0.10,
    ),
    SphereObstacle(
        name="obstacle_2",
        center=np.array([
            0.25,
            -0.22,
            0.65,
        ]),
        radius=0.08,
    ),
]


# ============================================================
# Meshcat helpers
# ============================================================


def rotation_matrix_from_y(direction):
    """
    Return a 3x3 rotation matrix that maps the local +Y axis
    onto `direction`.

    Meshcat cylinders are aligned with their local Y axis.
    """

    direction = np.asarray(
        direction,
        dtype=float,
    ).reshape(3)

    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return np.eye(3)

    y_axis = direction / norm

    # Choose a reference vector that is not parallel
    # to the desired Y axis.
    if abs(y_axis[2]) < 0.9:
        reference = np.array([
            0.0,
            0.0,
            1.0,
        ])
    else:
        reference = np.array([
            1.0,
            0.0,
            0.0,
        ])

    x_axis = np.cross(
        y_axis,
        reference,
    )

    x_axis /= np.linalg.norm(
        x_axis
    )

    z_axis = np.cross(
        x_axis,
        y_axis,
    )

    R = np.column_stack([
        x_axis,
        y_axis,
        z_axis,
    ])

    return R


def make_transform(
    position,
    rotation=None,
):
    """
    Build a 4x4 homogeneous transform.
    """

    T = np.eye(4)

    T[:3, 3] = np.asarray(
        position,
        dtype=float,
    ).reshape(3)

    if rotation is not None:
        T[:3, :3] = rotation

    return T


def display_sphere(
    viewer,
    name,
    center,
    radius,
    opacity=0.45,
):
    """
    Display a sphere in Meshcat.
    """

    node = viewer[
        "collision"
    ][
        "obstacles"
    ][
        name
    ]

    material = mg.MeshLambertMaterial(
        color=0xD95F02,
        opacity=opacity,
        transparent=True,
    )

    node.set_object(
        mg.Sphere(radius),
        material,
    )

    node.set_transform(
        mt.translation_matrix(
            center
        )
    )


def display_capsule(
    viewer,
    name,
    start,
    end,
    radius,
    opacity=0.45,
):
    """
    Display a capsule approximately using:

        sphere + cylinder + sphere

    between `start` and `end`.

    This is only a visualization primitive. The collision
    mathematics still uses the analytical capsule model.
    """

    start = np.asarray(
        start,
        dtype=float,
    ).reshape(3)

    end = np.asarray(
        end,
        dtype=float,
    ).reshape(3)

    direction = end - start

    length = np.linalg.norm(
        direction
    )

    if length < 1e-10:
        raise ValueError(
            f"Capsule {name!r} has "
            "almost zero length."
        )

    midpoint = 0.5 * (
        start + end
    )

    rotation = (
        rotation_matrix_from_y(
            direction
        )
    )

    root = viewer[
        "collision"
    ][
        "capsules"
    ][
        name
    ]

    material = mg.MeshLambertMaterial(
        color=0x1F78B4,
        opacity=opacity,
        transparent=True,
    )

    # --------------------------------------------------------
    # Cylinder
    # --------------------------------------------------------

    cylinder_node = root[
        "cylinder"
    ]

    cylinder_node.set_object(
        mg.Cylinder(
            length,
            radius,
        ),
        material,
    )

    cylinder_node.set_transform(
        make_transform(
            midpoint,
            rotation,
        )
    )

    # --------------------------------------------------------
    # Start sphere
    # --------------------------------------------------------

    start_node = root[
        "start"
    ]

    start_node.set_object(
        mg.Sphere(radius),
        material,
    )

    start_node.set_transform(
        mt.translation_matrix(
            start
        )
    )

    # --------------------------------------------------------
    # End sphere
    # --------------------------------------------------------

    end_node = root[
        "end"
    ]

    end_node.set_object(
        mg.Sphere(radius),
        material,
    )

    end_node.set_transform(
        mt.translation_matrix(
            end
        )
    )


def update_capsules(
    robot,
    q,
    capsules,
):
    """
    Update all robot capsules from the current Pinocchio
    configuration q.
    """

    if robot.viewer is None:
        raise RuntimeError(
            "Robot viewer has not been initialized."
        )

    meshcat_viewer = (
        robot.viewer.viewer
    )

    for capsule in capsules:

        start = robot.frame_position(
            q,
            frame=capsule.start_frame,
        )

        end = robot.frame_position(
            q,
            frame=capsule.end_frame,
        )

        display_capsule(
            viewer=meshcat_viewer,
            name=capsule.name,
            start=start,
            end=end,
            radius=capsule.radius,
        )


def display_obstacles(
    robot,
    obstacles,
):
    """
    Display all spherical obstacles.
    """

    if robot.viewer is None:
        raise RuntimeError(
            "Robot viewer has not been initialized."
        )

    meshcat_viewer = (
        robot.viewer.viewer
    )

    for i, obstacle in enumerate(
        obstacles
    ):
        name = (
            obstacle.name
            if obstacle.name
            else f"obstacle_{i}"
        )

        display_sphere(
            viewer=meshcat_viewer,
            name=name,
            center=obstacle.center,
            radius=obstacle.radius,
        )


# ============================================================
# Main example
# ============================================================


def main():

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame=(
            "end_effector_link"
        ),
    )

    print("=" * 70)
    print("KINOVA COLLISION GEOMETRY VISUALIZATION")
    print("=" * 70)

    print()
    print(robot)

    print()
    print(
        f"{len(KINOVA_CAPSULES)} "
        "capsules"
    )

    print(
        f"{len(OBSTACLES)} "
        "obstacles"
    )

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    print()
    print("Starting Meshcat...")

    robot.init_viewer(
        open_browser=True,
    )

    # --------------------------------------------------------
    # Initial joint configuration
    # --------------------------------------------------------

    theta = np.array([
        0.0,
        -0.35,
        0.0,
        1.15,
        0.0,
        0.70,
        0.0,
    ])

    q = (
        robot
        .joint_positions_to_configuration(
            theta
        )
    )

    robot.display(q)

    # --------------------------------------------------------
    # Collision geometry
    # --------------------------------------------------------

    display_obstacles(
        robot,
        OBSTACLES,
    )

    update_capsules(
        robot,
        q,
        KINOVA_CAPSULES,
    )

    print()
    print(
        "Initial collision geometry displayed."
    )

    time.sleep(2.0)

    # --------------------------------------------------------
    # Slow animation
    #
    # Useful to verify that the capsules follow the robot.
    # --------------------------------------------------------

    print()
    print(
        "Animating robot to verify capsule attachment..."
    )

    n_steps = 160

    for s in np.linspace(
        0.0,
        1.0,
        n_steps,
    ):

        theta_s = theta.copy()

        theta_s[0] += (
            0.60
            * np.sin(
                np.pi * s
            )
        )

        theta_s[1] += (
            -0.30
            * np.sin(
                2.0
                * np.pi
                * s
            )
        )

        theta_s[3] += (
            0.40
            * np.sin(
                np.pi * s
            )
        )

        theta_s[5] += (
            -0.25
            * np.sin(
                2.0
                * np.pi
                * s
            )
        )

        q_s = (
            robot
            .joint_positions_to_configuration(
                theta_s
            )
        )

        robot.display(
            q_s
        )

        update_capsules(
            robot,
            q_s,
            KINOVA_CAPSULES,
        )

        time.sleep(
            0.025
        )

    print()
    print("Done.")
    print(
        "Check that each capsule approximately "
        "covers the corresponding robot link."
    )

    input(
        "Press Enter to quit..."
    )


if __name__ == "__main__":
    main()

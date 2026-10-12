import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from splineopt.bspline import BSpline
from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
)
from splineopt.collision import (
    Capsule,
    SphereObstacle,
    CollisionEnvironment,
    CollisionConstraintBuilder,
    capsule_sphere_distance,
)
from splineopt.optimizer import (
    GaussNewtonSolver,
)
from splineopt.robotics import Robot

from examples.kinova_trajectory_projection import (
    URDF_PATH,
    Q_MIN,
    Q_MAX,
    V_MIN,
    V_MAX,
    A_MIN,
    A_MAX,
    _set_meshcat_line,
)

from examples.kinova_collision_visualization import (
    display_obstacles,
    update_capsules,
)


# =====================================================================
# Configuration
# =====================================================================

DEGREE = 3
N_CONTROL_POINTS = 14

T_START = 0.0
T_END = 5.0

N_TRACKING = 61
N_COLLISION = 50
N_DENSE = 400

LAMBDA_SMOOTH = 1e-5

DAMPING = 1e-3
MAX_ITERATIONS = 25

SAFETY_DISTANCE = 0.02

ANIMATION_DT = 0.025


# =====================================================================
# Kinova capsule approximation
# =====================================================================

# Slightly larger than in the first visualization example.

KINOVA_CAPSULES = [
    Capsule(
        name="upper_arm_1",
        start_frame="shoulder_link",
        end_frame="half_arm_1_link",
        radius=0.075,
    ),
    Capsule(
        name="upper_arm_2",
        start_frame="half_arm_1_link",
        end_frame="half_arm_2_link",
        radius=0.075,
    ),
    Capsule(
        name="forearm",
        start_frame="half_arm_2_link",
        end_frame="forearm_link",
        radius=0.070,
    ),
    Capsule(
        name="wrist_1",
        start_frame="forearm_link",
        end_frame="spherical_wrist_1_link",
        radius=0.060,
    ),
    Capsule(
        name="wrist_2",
        start_frame="spherical_wrist_1_link",
        end_frame="spherical_wrist_2_link",
        radius=0.055,
    ),
    Capsule(
        name="bracelet",
        start_frame="spherical_wrist_2_link",
        end_frame="bracelet_link",
        radius=0.055,
    ),
    Capsule(
        name="tool",
        start_frame="bracelet_link",
        end_frame="end_effector_link",
        radius=0.050,
    ),
]


# =====================================================================
# Helpers
# =====================================================================


def make_spline(
    t_start,
    t_end,
    n_dof,
):
    """Create a cubic open-clamped B-spline."""

    n_internal_knots = (
        N_CONTROL_POINTS
        - DEGREE
        - 1
    )

    internal_knots = np.linspace(
        t_start,
        t_end,
        n_internal_knots + 2,
    )[1:-1]

    knots = np.concatenate([
        np.full(
            DEGREE + 1,
            t_start,
        ),
        internal_knots,
        np.full(
            DEGREE + 1,
            t_end,
        ),
    ])

    return BSpline(
        knots=knots,
        degree=DEGREE,
        n_dof=n_dof,
    )


def smoothstep5(x):
    """
    Quintic smoothstep.

    s(0) = 0
    s(1) = 1

    with zero first and second derivatives
    at both endpoints.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        10.0 * x**3
        - 15.0 * x**4
        + 6.0 * x**5
    )


def build_joint_reference(
    times,
    q_start,
    q_goal,
):
    """Smooth virtual joint-space reference."""

    u = (
        times - times[0]
    ) / (
        times[-1]
        - times[0]
    )

    s = smoothstep5(u)

    return (
        q_start[None, :]
        + s[:, None]
        * (
            q_goal
            - q_start
        )[None, :]
    )


def build_target_poses(
    robot,
    theta_reference,
):
    target_poses = []

    for theta in theta_reference:

        q_pin = (
            robot
            .joint_positions_to_configuration(
                theta
            )
        )

        target_poses.append(
            robot.frame_pose(
                q_pin
            )
        )

    return target_poses


def evaluate_joint_trajectory(
    spline,
    control_points,
    times,
):
    return np.array([
        spline.position(
            control_points,
            t,
        )
        for t in times
    ])


def forward_kinematics_positions(
    robot,
    theta_trajectory,
):
    positions = []

    for theta in theta_trajectory:

        q_pin = (
            robot
            .joint_positions_to_configuration(
                theta
            )
        )

        positions.append(
            robot.frame_position(
                q_pin
            )
        )

    return np.asarray(
        positions
    )


def capsule_endpoints(
    robot,
    theta,
    capsule,
):
    q_pin = (
        robot
        .joint_positions_to_configuration(
            theta
        )
    )

    p0 = robot.frame_position(
        q_pin,
        frame=capsule.start_frame,
    )

    p1 = robot.frame_position(
        q_pin,
        frame=capsule.end_frame,
    )

    return p0, p1


def point_clearance_from_robot(
    robot,
    theta,
    point,
    capsules,
):
    """
    Minimum distance from a point to the surface of all
    robot capsules.
    """

    minimum = np.inf

    for capsule in capsules:

        p0, p1 = capsule_endpoints(
            robot,
            theta,
            capsule,
        )

        distance, _, _, _ = (
            capsule_sphere_distance(
                segment_start=p0,
                segment_end=p1,
                capsule_radius=(
                    capsule.radius
                ),
                sphere_center=point,
                sphere_radius=0.0,
            )
        )

        minimum = min(
            minimum,
            distance,
        )

    return minimum


def automatically_place_obstacle(
    robot,
    spline,
    baseline_control_points,
    q_initial,
    capsules,
):
    """
    Choose an obstacle location automatically.

    We search for a point lying on one of the moving capsules
    along the baseline trajectory while remaining far from the
    robot in its initial configuration.

    Therefore:

        - initial constant trajectory is collision-free,
        - baseline optimized trajectory collides.
    """

    search_times = np.linspace(
        T_START + 0.30 * (
            T_END - T_START
        ),
        T_START + 0.85 * (
            T_END - T_START
        ),
        60,
    )

    # Avoid placing the obstacle directly on the tool.
    candidate_names = {
        "forearm",
        "wrist_1",
        "wrist_2",
        "bracelet",
    }

    best = None

    for t in search_times:

        theta = spline.position(
            baseline_control_points,
            t,
        )

        for capsule in capsules:

            if (
                capsule.name
                not in candidate_names
            ):
                continue

            p0, p1 = capsule_endpoints(
                robot,
                theta,
                capsule,
            )

            center = (
                0.5
                * (
                    p0 + p1
                )
            )

            initial_clearance = (
                point_clearance_from_robot(
                    robot=robot,
                    theta=q_initial,
                    point=center,
                    capsules=capsules,
                )
            )

            score = initial_clearance

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "score": score,
                    "center": center.copy(),
                    "time": float(t),
                    "capsule": capsule,
                }

    if best is None:
        raise RuntimeError(
            "Could not find an obstacle location."
        )

    # Maximum obstacle radius that preserves:
    #
    # initial distance >= safety distance + buffer.
    buffer = 0.015

    maximum_radius = (
        best["score"]
        - SAFETY_DISTANCE
        - buffer
    )

    if maximum_radius <= 0.02:
        raise RuntimeError(
            "Could not place a sufficiently large obstacle "
            "while keeping the initial trajectory collision-free."
        )

    obstacle_radius = min(
        0.07,
        0.8 * maximum_radius,
    )

    obstacle = SphereObstacle(
        name="automatic_obstacle",
        center=best[
            "center"
        ],
        radius=obstacle_radius,
    )

    return obstacle, best


def minimum_distance_trajectory(
    builder,
    control_points,
):
    distances = builder.distances(
        control_points
    )

    if distances.shape[1] == 0:
        return np.full(
            distances.shape[0],
            np.inf,
        )

    return np.min(
        distances,
        axis=1,
    )


# =====================================================================
# Main
# =====================================================================


def main():

    print("=" * 78)
    print("KINOVA COLLISION-AWARE B-SPLINE OPTIMIZATION")
    print("=" * 78)

    # =================================================================
    # Robot
    # =================================================================

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame=(
            "end_effector_link"
        ),
    )

    print()
    print(robot)

    # =================================================================
    # Reference trajectory
    # =================================================================

    q_start = np.array([
        0.00,
        0.45,
        0.15,
        -1.35,
        -0.15,
        0.75,
        0.10,
    ])

    q_goal = np.array([
        0.90,
        1.40,
        -0.60,
        -1.70,
        0.80,
        1.00,
        -0.60,
    ])

    tracking_times = np.linspace(
        T_START,
        T_END,
        N_TRACKING,
    )

    theta_reference = (
        build_joint_reference(
            tracking_times,
            q_start,
            q_goal,
        )
    )

    target_poses = (
        build_target_poses(
            robot,
            theta_reference,
        )
    )

    # =================================================================
    # B-spline
    # =================================================================

    spline = make_spline(
        t_start=T_START,
        t_end=T_END,
        n_dof=robot.nv,
    )

    # -------------------------------------------------------------
    # Initial trajectory:
    #
    # constant at q_start.
    #
    # This satisfies:
    #
    #   q(0)    = q_start
    #   qdot(0) = 0
    #   qddot(0)= 0
    #   qdot(tf)= 0
    # -------------------------------------------------------------

    initial_control_points = np.tile(
        q_start,
        (
            spline.n_control_points,
            1,
        ),
    )

    # =================================================================
    # Solver
    # =================================================================

    solver = GaussNewtonSolver(
        damping_mode="adaptive",
        damping=DAMPING,
        max_iterations=MAX_ITERATIONS,
        max_damping_retries=4,
    )

    # =================================================================
    # 1) Baseline without obstacle
    # =================================================================

    print()
    print("=" * 78)
    print("1) BASELINE WITHOUT COLLISION CONSTRAINTS")
    print("=" * 78)

    baseline_problem = (
        CartesianTrajectoryProblem(
            robot=robot,
            spline=spline,
            tracking_times=(
                tracking_times
            ),
            target_poses=(
                target_poses
            ),
            q0=q_start,
            v0=np.zeros(
                robot.nv
            ),
            a0=np.zeros(
                robot.nv
            ),
            vf=np.zeros(
                robot.nv
            ),
            q_min=Q_MIN,
            q_max=Q_MAX,
            v_min=V_MIN,
            v_max=V_MAX,
            a_min=A_MIN,
            a_max=A_MAX,
            orientation_scale=0.0,
            lambda_smooth=(
                LAMBDA_SMOOTH
            ),
            lambda_terminal=1.0,
        )
    )

    baseline_result = solver.solve(
        problem=baseline_problem,
        initial_control_points=(
            initial_control_points
        ),
        verbose=True,
    )

    baseline_control_points = (
        baseline_result.control_points
    )

    print()
    print(
        "Baseline status:",
        baseline_result.status,
    )

    print(
        "Baseline iterations:",
        baseline_result.iterations,
    )

    # =================================================================
    # 2) Automatically place an obstacle on the baseline
    # =================================================================

    obstacle, obstacle_info = (
        automatically_place_obstacle(
            robot=robot,
            spline=spline,
            baseline_control_points=(
                baseline_control_points
            ),
            q_initial=q_start,
            capsules=KINOVA_CAPSULES,
        )
    )

    print()
    print("=" * 78)
    print("2) AUTOMATIC OBSTACLE")
    print("=" * 78)

    print(
        f"Selected capsule: "
        f"{obstacle_info['capsule'].name}"
    )

    print(
        f"Selected time: "
        f"{obstacle_info['time']:.3f} s"
    )

    print(
        "Obstacle center:",
        np.round(
            obstacle.center,
            4,
        ),
    )

    print(
        f"Obstacle radius: "
        f"{obstacle.radius:.3f} m"
    )

    print(
        f"Initial point clearance before "
        f"adding sphere radius: "
        f"{obstacle_info['score']:.3f} m"
    )

    # =================================================================
    # Collision environment
    # =================================================================

    environment = (
        CollisionEnvironment(
            robot=robot,
            capsules=KINOVA_CAPSULES,
            obstacles=[
                obstacle
            ],
        )
    )

    collision_times = np.linspace(
        T_START,
        T_END,
        N_COLLISION,
    )

    collision_builder = (
        CollisionConstraintBuilder(
            spline=spline,
            environment=environment,
            collocation_times=(
                collision_times
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
        )
    )

    # =================================================================
    # Check initial and baseline distances
    # =================================================================

    initial_min_distance = (
        collision_builder
        .minimum_distance(
            initial_control_points
        )
    )

    baseline_min_distance = (
        collision_builder
        .minimum_distance(
            baseline_control_points
        )
    )

    print()
    print(
        f"Safety distance: "
        f"{SAFETY_DISTANCE:.3f} m"
    )

    print(
        f"Initial trajectory minimum distance: "
        f"{initial_min_distance:.3f} m"
    )

    print(
        f"Baseline trajectory minimum distance: "
        f"{baseline_min_distance:.3f} m"
    )

    if (
        initial_min_distance
        < SAFETY_DISTANCE
    ):
        raise RuntimeError(
            "Initial trajectory is unexpectedly "
            "in collision."
        )

    if (
        baseline_min_distance
        >= SAFETY_DISTANCE
    ):
        raise RuntimeError(
            "Automatically generated obstacle does "
            "not intersect the baseline trajectory."
        )

    # =================================================================
    # 3) Collision-aware optimization
    # =================================================================

    print()
    print("=" * 78)
    print("3) COLLISION-AWARE OPTIMIZATION")
    print("=" * 78)

    collision_problem = (
        CartesianTrajectoryProblem(
            robot=robot,
            spline=spline,
            tracking_times=(
                tracking_times
            ),
            target_poses=(
                target_poses
            ),
            q0=q_start,
            v0=np.zeros(
                robot.nv
            ),
            a0=np.zeros(
                robot.nv
            ),
            vf=np.zeros(
                robot.nv
            ),
            q_min=Q_MIN,
            q_max=Q_MAX,
            v_min=V_MIN,
            v_max=V_MAX,
            a_min=A_MIN,
            a_max=A_MAX,
            orientation_scale=0.0,
            lambda_smooth=(
                LAMBDA_SMOOTH
            ),
            lambda_terminal=1.0,
            collision_constraints=(
                collision_builder
            ),
        )
    )

    collision_result = (
        solver.solve(
            problem=collision_problem,
            initial_control_points=(
                initial_control_points
            ),
            verbose=True,
        )
    )

    collision_control_points = (
        collision_result.control_points
    )

    print()
    print(
        "Collision-aware status:",
        collision_result.status,
    )

    print(
        "Collision-aware iterations:",
        collision_result.iterations,
    )

    # =================================================================
    # 4) Dense validation
    # =================================================================

    dense_times = np.linspace(
        T_START,
        T_END,
        N_DENSE,
    )

    dense_builder = (
        CollisionConstraintBuilder(
            spline=spline,
            environment=environment,
            collocation_times=(
                dense_times
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
        )
    )

    baseline_distance_dense = (
        minimum_distance_trajectory(
            dense_builder,
            baseline_control_points,
        )
    )

    collision_distance_dense = (
        minimum_distance_trajectory(
            dense_builder,
            collision_control_points,
        )
    )

    min_baseline_dense = float(
        np.min(
            baseline_distance_dense
        )
    )

    min_collision_dense = float(
        np.min(
            collision_distance_dense
        )
    )

    print()
    print("=" * 78)
    print("DENSE COLLISION VALIDATION")
    print("=" * 78)

    print(
        f"Baseline dense minimum distance: "
        f"{min_baseline_dense:.4f} m"
    )

    print(
        f"Collision-aware dense minimum distance: "
        f"{min_collision_dense:.4f} m"
    )

    print(
        f"Required safety distance: "
        f"{SAFETY_DISTANCE:.4f} m"
    )

    print(
        "Dense collision feasibility:",
        (
            "PASS"
            if (
                min_collision_dense
                >= SAFETY_DISTANCE
                - 1e-3
            )
            else "FAIL"
        ),
    )

    # =================================================================
    # Joint and Cartesian dense trajectories
    # =================================================================

    theta_baseline_dense = (
        evaluate_joint_trajectory(
            spline,
            baseline_control_points,
            dense_times,
        )
    )

    theta_collision_dense = (
        evaluate_joint_trajectory(
            spline,
            collision_control_points,
            dense_times,
        )
    )

    theta_initial_dense = (
        evaluate_joint_trajectory(
            spline,
            initial_control_points,
            dense_times,
        )
    )

    cart_baseline_dense = (
        forward_kinematics_positions(
            robot,
            theta_baseline_dense,
        )
    )

    cart_collision_dense = (
        forward_kinematics_positions(
            robot,
            theta_collision_dense,
        )
    )

    cart_target_dense = (
        forward_kinematics_positions(
            robot,
            build_joint_reference(
                dense_times,
                q_start,
                q_goal,
            ),
        )
    )

    # =================================================================
    # 5) Plots
    # =================================================================

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        cart_target_dense[:, 0],
        cart_target_dense[:, 1],
        cart_target_dense[:, 2],
        "--",
        label="Cartesian target",
    )

    ax.plot(
        cart_baseline_dense[:, 0],
        cart_baseline_dense[:, 1],
        cart_baseline_dense[:, 2],
        label="Baseline",
    )

    ax.plot(
        cart_collision_dense[:, 0],
        cart_collision_dense[:, 1],
        cart_collision_dense[:, 2],
        label="Collision-aware",
    )

    # -------------------------------------------------------------
    # Draw spherical obstacle
    # -------------------------------------------------------------

    u = np.linspace(
        0.0,
        2.0 * np.pi,
        30,
    )

    v = np.linspace(
        0.0,
        np.pi,
        20,
    )

    x = (
        obstacle.center[0]
        + obstacle.radius
        * np.outer(
            np.cos(u),
            np.sin(v),
        )
    )

    y = (
        obstacle.center[1]
        + obstacle.radius
        * np.outer(
            np.sin(u),
            np.sin(v),
        )
    )

    z = (
        obstacle.center[2]
        + obstacle.radius
        * np.outer(
            np.ones_like(u),
            np.cos(v),
        )
    )

    ax.plot_surface(
        x,
        y,
        z,
        alpha=0.25,
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    ax.set_zlabel(
        "z [m]"
    )

    ax.set_title(
        "End-effector trajectories"
    )

    ax.legend()

    plt.tight_layout()

    # -------------------------------------------------------------
    # Distance plot
    # -------------------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.plot(
        dense_times,
        baseline_distance_dense,
        label="Baseline",
    )

    ax2.plot(
        dense_times,
        collision_distance_dense,
        label="Collision-aware",
    )

    ax2.axhline(
        SAFETY_DISTANCE,
        linestyle="--",
        label="Safety distance",
    )

    ax2.axhline(
        0.0,
        linestyle=":",
        label="Contact",
    )

    ax2.set_xlabel(
        "Time [s]"
    )

    ax2.set_ylabel(
        "Minimum capsule-obstacle distance [m]"
    )

    ax2.set_title(
        "Collision clearance"
    )

    ax2.legend()

    plt.tight_layout()

    plt.show(
        block=False
    )

    # =================================================================
    # 6) Meshcat
    # =================================================================

    print()
    print("=" * 78)
    print("MESHCAT VISUALIZATION")
    print("=" * 78)

    robot.init_viewer(
        open_browser=True,
    )

    display_obstacles(
        robot,
        [
            obstacle
        ],
    )

    _set_meshcat_line(
        robot,
        cart_target_dense,
        name="target",
        color=0x888888,
        linewidth=3.0,
    )

    _set_meshcat_line(
        robot,
        cart_baseline_dense,
        name="baseline",
        color=0xFF8800,
        linewidth=4.0,
    )

    _set_meshcat_line(
        robot,
        cart_collision_dense,
        name="collision_aware",
        color=0x0077FF,
        linewidth=4.0,
    )

    print()
    print(
        "Meshcat:"
    )

    print(
        "  grey   = target"
    )

    print(
        "  orange = unconstrained baseline"
    )

    print(
        "  blue   = collision-aware trajectory"
    )

    print()
    print(
        "Animating collision-aware trajectory..."
    )

    time.sleep(
        1.0
    )

    while True:

        for theta in (
            theta_collision_dense
        ):

            q_pin = (
                robot
                .joint_positions_to_configuration(
                    theta
                )
            )

            robot.display(
                q_pin
            )

            update_capsules(
                robot,
                q_pin,
                KINOVA_CAPSULES,
            )

            time.sleep(
                ANIMATION_DT
            )

        answer = input(
            "\nReplay animation? [y/N]: "
        )

        if answer.lower() != "y":
            break


if __name__ == "__main__":
    main()

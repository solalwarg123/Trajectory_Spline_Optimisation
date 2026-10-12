import time

import matplotlib.pyplot as plt
import numpy as np

from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
)

from splineopt.collision import (
    Capsule,
    SphereObstacle,
    CollisionEnvironment,
    CollisionConstraintBuilder,
)

from splineopt.optimizer import (
    GaussNewtonSolver,
)

from splineopt.ipopt_solver import (
    IpoptSolver,
)

from splineopt.robotics import Robot


# =====================================================================
# Existing Kinova example helpers
# =====================================================================

from examples.kinova_trajectory_projection import (
    URDF_PATH,
    Q_MIN,
    Q_MAX,
    V_MIN,
    V_MAX,
    A_MIN,
    A_MAX,
    LAMBDA_SMOOTH,
    N_DENSE_SAMPLES,
    ANIMATION_SPEED,
    build_virtual_joint_reference,
    build_cartesian_targets_from_virtual_reference,
    solve_bspline_projection,
    evaluate_spline,
    interpolate_joint_trajectory,
    numerical_derivatives,
    forward_kinematics_trajectory,
    full_feasible_mask,
    position_feasible_mask,
    display_meshcat_nominal_feasibility_path,
    _set_meshcat_line,
)


# =====================================================================
# Existing comparison helpers
# =====================================================================

from examples.kinova_ipopt_vs_gauss_newton import (
    pose_tracking_errors,
    plot_pose_tracking_errors,
    plot_cartesian_dynamics_comparison,
    plot_joint_quantity_comparison,
    plot_feasibility_ratios_comparison,
    print_pose_tracking_comparison,
    print_constraint_comparison,
    print_smoothness_comparison,
    print_solver_comparison,
    print_iteration_summary,
)


# =====================================================================
# Collision visualization helpers
# =====================================================================

from examples.kinova_collision_visualization import (
    display_obstacles,
    update_capsules,
)


# =====================================================================
# Configuration
# =====================================================================

GAUSS_NEWTON_DAMPING = 1e-3
GAUSS_NEWTON_MAX_ITERATIONS = 20

IPOPT_TOLERANCE = 1e-6
IPOPT_MAX_ITERATIONS = 200

CARTESIAN_TERMINAL_WEIGHT = 1.0

ORIENTATION_SCALE = 0.10

# Collision constraints imposed at these collocation points.
N_COLLISION_POINTS = 35

# Dense validation of collisions.
N_COLLISION_VALIDATION = 500

# Required clearance.
SAFETY_DISTANCE = 0.020

# Dense feasibility tolerance.
FEASIBILITY_TOL = 1e-3


# =====================================================================
# Desired benchmark geometry
# =====================================================================

# We want the INITIAL QP trajectory to penetrate the sphere only
# slightly.
TARGET_INITIAL_COLLISION = -0.006  # -6 mm

# Sphere radius.
OBSTACLE_RADIUS = 0.050  # 50 mm

# Search around the middle of the trajectory.
OBSTACLE_TIME_FRACTION = 0.55

# Capsule around which the obstacle is initially constructed.
OBSTACLE_CAPSULE = "bracelet"

# The fixed initial configuration should remain comfortably safe.
INITIAL_CLEARANCE_REQUIRED = (
    SAFETY_DISTANCE
    + 0.020
)

# Actual minimum should occur approximately in the middle.
MIN_COLLISION_TIME_FRACTION = 0.35
MAX_COLLISION_TIME_FRACTION = 0.75


# =====================================================================
# Plot colors
# =====================================================================

COLOR_NOMINAL = "C1"
COLOR_INITIAL = "0.45"
COLOR_GN = "C3"
COLOR_IPOPT = "C0"


# =====================================================================
# Kinova capsule approximation
# =====================================================================

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
# Utilities
# =====================================================================

def spline_configuration(
    spline,
    control_points,
    t,
):
    c = np.asarray(
        control_points,
        dtype=float,
    ).reshape(-1)

    A = spline.evaluation_matrix(
        t,
        derivative_order=0,
    )

    return np.asarray(
        A @ c,
        dtype=float,
    ).reshape(-1)


def find_capsule(
    name,
):
    for capsule in KINOVA_CAPSULES:

        if capsule.name == name:
            return capsule

    raise ValueError(
        f"Unknown capsule: {name}"
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

    return (
        np.asarray(
            p0,
            dtype=float,
        ).copy(),
        np.asarray(
            p1,
            dtype=float,
        ).copy(),
    )


def perpendicular_direction(
    axis,
    angle,
):
    """
    Return a unit vector perpendicular to axis.

    angle rotates the vector around the capsule axis.
    """

    axis = np.asarray(
        axis,
        dtype=float,
    ).reshape(3)

    axis_norm = np.linalg.norm(
        axis
    )

    if axis_norm < 1e-12:
        raise RuntimeError(
            "Degenerate capsule axis."
        )

    axis = axis / axis_norm

    basis = np.eye(3)

    reference = basis[
        np.argmin(
            np.abs(
                axis
            )
        )
    ]

    normal_1 = np.cross(
        axis,
        reference,
    )

    normal_1 /= np.linalg.norm(
        normal_1
    )

    normal_2 = np.cross(
        axis,
        normal_1,
    )

    normal_2 /= np.linalg.norm(
        normal_2
    )

    return (
        np.cos(angle)
        * normal_1
        + np.sin(angle)
        * normal_2
    )


def plot_objective_convergence(
    gn_result,
    ipopt_result,
):
    gn_iterations = np.array(
        [
            item.iteration + 1
            for item in gn_result.history
        ],
        dtype=int,
    )

    gn_objectives = np.array(
        [
            item.objective
            for item in gn_result.history
        ],
        dtype=float,
    )

    ipopt_objectives = np.asarray(
        ipopt_result.objective_history,
        dtype=float,
    )

    ipopt_iterations = np.arange(
        len(ipopt_objectives)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        gn_iterations,
        gn_objectives,
        marker="o",
        markersize=3,
        linewidth=2.0,
        label="Gauss-Newton / SQP",
    )

    ax.plot(
        ipopt_iterations,
        ipopt_objectives,
        linewidth=2.0,
        label="IPOPT",
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        "Objective"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_title(
        "Objective convergence"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

# =====================================================================
# Precompute robot capsule geometry along QP initialization
# =====================================================================


def precompute_capsule_geometry(
    robot,
    spline,
    control_points,
    times,
):
    """
    Precompute the endpoints of every capsule along a trajectory.

    Returns
    -------
    starts :
        shape (n_times, n_capsules, 3)

    ends :
        shape (n_times, n_capsules, 3)
    """

    starts = np.zeros(
        (
            len(times),
            len(KINOVA_CAPSULES),
            3,
        )
    )

    ends = np.zeros_like(
        starts
    )

    for i, t in enumerate(
        times
    ):

        theta = spline_configuration(
            spline,
            control_points,
            t,
        )

        for j, capsule in enumerate(
            KINOVA_CAPSULES
        ):

            p0, p1 = capsule_endpoints(
                robot,
                theta,
                capsule,
            )

            starts[i, j] = p0
            ends[i, j] = p1

    return (
        starts,
        ends,
    )


def sphere_clearance_from_geometry(
    center,
    sphere_radius,
    times,
    starts,
    ends,
):
    """
    Compute signed clearance between one sphere and all capsules.

    Positive:
        separated.

    Zero:
        geometric contact.

    Negative:
        penetration.
    """

    center = np.asarray(
        center,
        dtype=float,
    ).reshape(3)

    segments = (
        ends
        - starts
    )

    segment_norm_squared = np.sum(
        segments * segments,
        axis=2,
    )

    point_vectors = (
        center[None, None, :]
        - starts
    )

    numerator = np.sum(
        point_vectors
        * segments,
        axis=2,
    )

    alpha = np.zeros_like(
        numerator
    )

    valid = (
        segment_norm_squared
        > 1e-14
    )

    alpha[valid] = (
        numerator[valid]
        / segment_norm_squared[valid]
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )

    closest_points = (
        starts
        + alpha[:, :, None]
        * segments
    )

    center_distances = np.linalg.norm(
        closest_points
        - center[None, None, :],
        axis=2,
    )

    capsule_radii = np.asarray(
        [
            capsule.radius
            for capsule
            in KINOVA_CAPSULES
        ],
        dtype=float,
    )

    clearances = (
        center_distances
        - capsule_radii[None, :]
        - sphere_radius
    )

    flat_index = int(
        np.argmin(
            clearances
        )
    )

    time_index, capsule_index = (
        np.unravel_index(
            flat_index,
            clearances.shape,
        )
    )

    return {
        "minimum": float(
            clearances[
                time_index,
                capsule_index,
            ]
        ),

        "time": float(
            times[
                time_index
            ]
        ),

        "capsule_index": (
            capsule_index
        ),

        "capsule": (
            KINOVA_CAPSULES[
                capsule_index
            ]
        ),

        "initial_clearance": float(
            np.min(
                clearances[0]
            )
        ),
    }


# =====================================================================
# Automatic SINGLE obstacle placement
# =====================================================================

def build_single_obstacle(
    robot,
    spline,
    initial_control_points,
):
    """
    Automatically place one spherical obstacle.

    Desired benchmark:

      - initial robot configuration safely away from obstacle,
      - QP trajectory slightly penetrates obstacle,
      - penetration occurs around the middle of the motion,
      - global distance is checked against ALL robot capsules.
    """

    target_time = (
        spline.t_start
        + OBSTACLE_TIME_FRACTION
        * (
            spline.t_end
            - spline.t_start
        )
    )

    target_capsule = find_capsule(
        OBSTACLE_CAPSULE
    )

    theta_target = spline_configuration(
        spline,
        initial_control_points,
        target_time,
    )

    p0, p1 = capsule_endpoints(
        robot,
        theta_target,
        target_capsule,
    )

    axis = (
        p1
        - p0
    )

    midpoint = (
        0.5
        * (
            p0
            + p1
        )
    )

    # -------------------------------------------------------------
    # Geometry used to evaluate candidates quickly.
    # -------------------------------------------------------------

    search_times = np.linspace(
        spline.t_start,
        spline.t_end,
        121,
    )

    # Make sure target_time is represented exactly.
    search_times = np.unique(
        np.concatenate([
            search_times,
            [target_time],
        ])
    )

    starts, ends = (
        precompute_capsule_geometry(
            robot,
            spline,
            initial_control_points,
            search_times,
        )
    )

    # -------------------------------------------------------------
    # Search obstacle orientations around the selected capsule and
    # several radial clearances.
    #
    # The final selection is based on GLOBAL robot clearance, not
    # only the selected capsule.
    # -------------------------------------------------------------

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        72,
        endpoint=False,
    )

    # Clearance relative to selected capsule at target_time.
    radial_clearances = np.linspace(
        -0.010,
        0.160,
        80,
    )

    candidates = []

    duration = (
        spline.t_end
        - spline.t_start
    )

    min_allowed_time = (
        spline.t_start
        + MIN_COLLISION_TIME_FRACTION
        * duration
    )

    max_allowed_time = (
        spline.t_start
        + MAX_COLLISION_TIME_FRACTION
        * duration
    )

    for angle in angles:

        normal = perpendicular_direction(
            axis,
            angle,
        )

        for radial_clearance in (
            radial_clearances
        ):

            centerline_distance = (
                target_capsule.radius
                + OBSTACLE_RADIUS
                + radial_clearance
            )

            if (
                centerline_distance
                <= 0.0
            ):
                continue

            center = (
                midpoint
                + centerline_distance
                * normal
            )

            information = (
                sphere_clearance_from_geometry(
                    center=center,
                    sphere_radius=(
                        OBSTACLE_RADIUS
                    ),
                    times=search_times,
                    starts=starts,
                    ends=ends,
                )
            )

            # Fixed initial configuration must be safely feasible.
            if (
                information[
                    "initial_clearance"
                ]
                < INITIAL_CLEARANCE_REQUIRED
            ):
                continue

            # Collision should occur around the middle, not at the
            # beginning or end.
            if not (
                min_allowed_time
                <= information["time"]
                <= max_allowed_time
            ):
                continue

            # We specifically want a SMALL true penetration.
            if not (
                -0.015
                <= information["minimum"]
                <= -0.001
            ):
                continue

            score = (
                abs(
                    information["minimum"]
                    - TARGET_INITIAL_COLLISION
                )
                + 0.02
                * abs(
                    information["time"]
                    - target_time
                )
                / max(
                    duration,
                    1e-12,
                )
            )

            candidates.append({
                "score": score,
                "center": center.copy(),
                "angle": float(angle),
                "radial_clearance": float(
                    radial_clearance
                ),
                **information,
            })

    if len(candidates) == 0:

        raise RuntimeError(
            "Could not find a suitable single obstacle. "
            "Try increasing the radial search range."
        )

    best = min(
        candidates,
        key=lambda item: item["score"],
    )

    obstacle = SphereObstacle(
        name="obstacle",
        center=best["center"],
        radius=OBSTACLE_RADIUS,
    )

    print()
    print("=" * 82)
    print("AUTOMATIC SINGLE SPHERICAL OBSTACLE")
    print("=" * 82)

    print(
        f"Construction capsule: "
        f"{target_capsule.name}"
    )

    print(
        f"Construction time: "
        f"{target_time:.3f} s"
    )

    print(
        f"Obstacle radius: "
        f"{1000.0 * obstacle.radius:.1f} mm"
    )

    print(
        f"Obstacle center: "
        f"{np.round(obstacle.center, 3)}"
    )

    print(
        f"Selected angle: "
        f"{best['angle']:.3f} rad"
    )

    print(
        f"Initial configuration clearance: "
        f"{1000.0 * best['initial_clearance']:.2f} mm"
    )

    print(
        f"QP global minimum clearance: "
        f"{1000.0 * best['minimum']:.2f} mm"
    )

    print(
        f"QP minimum at t = "
        f"{best['time']:.3f} s"
    )

    print(
        f"Closest capsule at minimum: "
        f"{best['capsule'].name}"
    )

    return obstacle


# =====================================================================
# Collision evaluation
# =====================================================================

def minimum_collision_distance_over_time(
    builder,
    control_points,
):
    """
    Minimum capsule-obstacle clearance for each collocation time.
    """

    distances = np.asarray(
        builder.distances(
            control_points
        ),
        dtype=float,
    )

    if distances.ndim == 1:
        distances = distances.reshape(
            -1,
            builder.environment.n_capsules,
        )

    return np.min(
        distances,
        axis=1,
    )


# =====================================================================
# Sphere drawing
# =====================================================================

def draw_sphere(
    ax,
    obstacle,
):
    u = np.linspace(
        0.0,
        2.0 * np.pi,
        32,
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


# =====================================================================
# Collision plot
# =====================================================================

def plot_collision_clearance(
    times,
    initial_distance,
    gn_distance,
    ipopt_distance,
):
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        times,
        1000.0 * initial_distance,
        "--",
        linewidth=2.0,
        label="QP initialization",
    )

    ax.plot(
        times,
        1000.0 * gn_distance,
        linewidth=2.3,
        label="Gauss-Newton / SQP",
    )

    ax.plot(
        times,
        1000.0 * ipopt_distance,
        linewidth=2.3,
        label="IPOPT",
    )

    ax.axhline(
        1000.0 * SAFETY_DISTANCE,
        linestyle="--",
        linewidth=1.5,
        label="Safety distance",
    )

    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1.2,
        label="Geometric contact",
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Minimum capsule-obstacle clearance [mm]"
    )

    ax.set_title(
        "Collision clearance"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()


# =====================================================================
# Cartesian 3D plot
# =====================================================================

def plot_cartesian_workspace(
    cart_nominal,
    cart_initial,
    cart_gn,
    cart_ipopt,
    obstacle,
):
    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        cart_nominal[:, 0],
        cart_nominal[:, 1],
        cart_nominal[:, 2],
        "--",
        linewidth=2.0,
        label="Nominal Cartesian reference",
    )

    ax.plot(
        cart_initial[:, 0],
        cart_initial[:, 1],
        cart_initial[:, 2],
        linewidth=2.0,
        label="QP initialization",
    )

    ax.plot(
        cart_gn[:, 0],
        cart_gn[:, 1],
        cart_gn[:, 2],
        linewidth=2.5,
        label="Gauss-Newton / SQP",
    )

    ax.plot(
        cart_ipopt[:, 0],
        cart_ipopt[:, 1],
        cart_ipopt[:, 2],
        linewidth=2.5,
        label="IPOPT",
    )

    draw_sphere(
        ax,
        obstacle,
    )

    ax.scatter(
        cart_nominal[0, 0],
        cart_nominal[0, 1],
        cart_nominal[0, 2],
        s=60,
        marker="o",
        label="Start",
    )

    ax.scatter(
        cart_nominal[-1, 0],
        cart_nominal[-1, 1],
        cart_nominal[-1, 2],
        s=80,
        marker="x",
        label="End",
    )

    all_points = np.vstack([
        cart_nominal,
        cart_initial,
        cart_gn,
        cart_ipopt,
        obstacle.center.reshape(
            1,
            3,
        ),
    ])

    xyz_min = np.min(
        all_points,
        axis=0,
    )

    xyz_max = np.max(
        all_points,
        axis=0,
    )

    margin = 0.10

    ax.set_xlim(
        xyz_min[0] - margin,
        xyz_max[0] + margin,
    )

    ax.set_ylim(
        xyz_min[1] - margin,
        xyz_max[1] + margin,
    )

    ax.set_zlim(
        xyz_min[2] - margin,
        xyz_max[2] + margin,
    )

    ranges = np.maximum(
        xyz_max - xyz_min,
        1e-3,
    )

    ax.set_box_aspect(
        ranges
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
        "Cartesian trajectories and spherical obstacle"
    )

    ax.legend()

    fig.tight_layout()


# =====================================================================
# Meshcat animation
# =====================================================================

def animate_with_capsules(
    robot,
    q_pin_trajectory,
    capsules,
    dt,
):
    for q_pin in q_pin_trajectory:

        robot.display(
            q_pin
        )

        update_capsules(
            robot,
            q_pin,
            capsules,
        )

        time.sleep(
            dt
        )


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 82)
    print(
        "KINOVA COLLISION-AWARE TRAJECTORY OPTIMIZATION"
    )
    print(
        "ONE SPHERICAL OBSTACLE — GAUSS-NEWTON / SQP vs IPOPT"
    )
    print("=" * 82)

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

    print(
        f"nq = {robot.nq}, "
        f"nv = {robot.nv}"
    )

    # =================================================================
    # Cartesian reference
    # =================================================================

    print()
    print(
        "Building Cartesian reference..."
    )

    (
        times,
        theta_virtual,
        waypoint_times,
        joint_waypoints,
    ) = build_virtual_joint_reference(
        robot
    )

    (
        q_pin_virtual,
        target_poses,
        target_positions,
    ) = (
        build_cartesian_targets_from_virtual_reference(
            robot,
            theta_virtual,
        )
    )

    # =================================================================
    # Sequential IK
    # =================================================================

    print()
    print(
        "Solving sequential IK..."
    )

    ik_result = robot.ik_trajectory(
        target_poses,
        q0=q_pin_virtual[0],
        tolerance=1e-5,
        max_iterations=500,
        step_size=0.1,
        damping=1e-6,
        max_step_norm=1.0,
    )

    if not ik_result.success:
        raise RuntimeError(
            "Sequential IK did not converge."
        )

    theta_ref = (
        ik_result.theta
    )

    print(
        f"Maximum IK error: "
        f"{np.max(ik_result.error_norms):.3e}"
    )

    print(
        f"Nominal IK max q2: "
        f"{np.max(theta_ref[:, 1]):.3f} rad "
        f"(limit {Q_MAX[1]:.3f})"
    )

    print(
        f"Nominal IK min q4: "
        f"{np.min(theta_ref[:, 3]):.3f} rad "
        f"(limit {Q_MIN[3]:.3f})"
    )

    # =================================================================
    # Joint-space QP initialization
    # =================================================================

    print()
    print(
        "Building feasible joint-space "
        "B-spline initialization..."
    )

    (
        spline,
        initial_control_points,
        osqp_result,
    ) = solve_bspline_projection(
        robot,
        times,
        theta_ref,
    )

    print(
        f"Initial joint-space QP status: "
        f"{osqp_result.info.status}"
    )

    # =================================================================
    # Build ONE obstacle
    # =================================================================

    obstacle = build_single_obstacle(
        robot,
        spline,
        initial_control_points,
    )

    obstacles = [
        obstacle
    ]

    environment = (
        CollisionEnvironment(
            robot=robot,
            capsules=KINOVA_CAPSULES,
            obstacles=obstacles,
        )
    )

    collision_times = np.linspace(
        spline.t_start,
        spline.t_end,
        N_COLLISION_POINTS,
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
    # Shared nonlinear problem
    # =================================================================

    problem = (
        CartesianTrajectoryProblem(
            robot=robot,
            spline=spline,

            tracking_times=times,
            target_poses=target_poses,

            q0=theta_ref[0],

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

            orientation_scale=(
                ORIENTATION_SCALE
            ),

            lambda_smooth=(
                LAMBDA_SMOOTH
            ),

            lambda_terminal=(
                CARTESIAN_TERMINAL_WEIGHT
            ),

            collision_constraints=(
                collision_builder
            ),
        )
    )

    # =================================================================
    # Initial metrics
    # =================================================================

    initial_metrics = (
        problem.error_metrics(
            initial_control_points
        )
    )

    initial_collision_min = (
        problem
        .minimum_collision_distance(
            initial_control_points
        )
    )

    print()
    print(
        "Initial joint-feasible B-spline:"
    )

    print(
        f"  position RMS = "
        f"{1e3 * initial_metrics['position_rms']:.2f} mm"
    )

    print(
        f"  position max = "
        f"{1e3 * initial_metrics['position_max']:.2f} mm"
    )

    print(
        f"  orientation RMS = "
        f"{np.degrees(initial_metrics['orientation_rms']):.2f} deg"
    )

    print(
        f"  orientation max = "
        f"{np.degrees(initial_metrics['orientation_max']):.2f} deg"
    )

    print(
        f"  minimum collision distance = "
        f"{1000.0 * initial_collision_min:.2f} mm"
    )

    print(
        f"  required safety distance = "
        f"{1000.0 * SAFETY_DISTANCE:.2f} mm"
    )

    print(
        f"  collision feasible = "
        f"{problem.collision_feasible(initial_control_points)}"
    )

    # =================================================================
    # Gauss-Newton / SQP
    # =================================================================

    print()
    print("=" * 82)
    print(
        "Solving with collision-aware "
        "GAUSS-NEWTON / SQP..."
    )
    print("=" * 82)

    gn_solver = (
        GaussNewtonSolver(
            damping_mode="adaptive",

            damping=(
                GAUSS_NEWTON_DAMPING
            ),

            max_iterations=(
                GAUSS_NEWTON_MAX_ITERATIONS
            ),

            max_damping_retries=5,

            collision_slack_weight=1e4,
            collision_merit_weight=1e4,
        )
    )

    gn_start = (
        time.perf_counter()
    )

    gn_result = (
        gn_solver.solve(
            problem=problem,
            initial_control_points=(
                initial_control_points
            ),
            verbose=True,
        )
    )

    gn_time = (
        time.perf_counter()
        - gn_start
    )

    gn_control_points = (
        gn_result.control_points
    )

    print()
    print(
        f"Gauss-Newton status: "
        f"{gn_result.status}"
    )

    print(
        f"Gauss-Newton success: "
        f"{gn_result.success}"
    )

    print(
        f"Gauss-Newton iterations: "
        f"{gn_result.iterations}"
    )

    print(
        f"Gauss-Newton total time: "
        f"{gn_time:.4f} s"
    )

    # =================================================================
    # IPOPT
    # =================================================================

    print()
    print("=" * 82)
    print(
        "Solving with collision-aware "
        "IPOPT + L-BFGS..."
    )
    print("=" * 82)

    ipopt_solver = (
        IpoptSolver(
            tolerance=(
                IPOPT_TOLERANCE
            ),

            max_iterations=(
                IPOPT_MAX_ITERATIONS
            ),

            print_level=0,

            options={
                "sb": "yes",
            },
        )
    )

    ipopt_result = (
        ipopt_solver.solve(
            problem=problem,
            initial_control_points=(
                initial_control_points
            ),
            verbose=True,
        )
    )

    ipopt_control_points = (
        ipopt_result.control_points
    )

    print()
    print(
        f"IPOPT success: "
        f"{ipopt_result.success}"
    )

    print(
        f"IPOPT status: "
        f"{ipopt_result.status_message}"
    )

    print(
        f"IPOPT iterations: "
        f"{ipopt_result.iterations}"
    )

    print(
        f"IPOPT objective: "
        f"{ipopt_result.objective:.6e}"
    )

    print(
        f"IPOPT total time: "
        f"{ipopt_result.solve_time:.4f} s"
    )

    # =================================================================
    # Dense trajectory evaluation
    # =================================================================

    dense_times = np.linspace(
        times[0],
        times[-1],
        N_DENSE_SAMPLES,
    )

    # Nominal IK

    q_ref_dense = (
        interpolate_joint_trajectory(
            times,
            theta_ref,
            dense_times,
        )
    )

    (
        v_ref_dense,
        a_ref_dense,
    ) = numerical_derivatives(
        q_ref_dense,
        dense_times,
    )

    (
        q_pin_ref_dense,
        cart_ref_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_ref_dense,
    )

    nominal_position_feasible = (
        position_feasible_mask(
            q_ref_dense
        )
    )

    # QP initialization

    (
        q_initial_dense,
        v_initial_dense,
        a_initial_dense,
    ) = evaluate_spline(
        spline,
        initial_control_points,
        dense_times,
    )

    (
        q_pin_initial_dense,
        cart_initial_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_initial_dense,
    )

    # GN

    (
        q_gn_dense,
        v_gn_dense,
        a_gn_dense,
    ) = evaluate_spline(
        spline,
        gn_control_points,
        dense_times,
    )

    (
        q_pin_gn_dense,
        cart_gn_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_gn_dense,
    )

    # IPOPT

    (
        q_ipopt_dense,
        v_ipopt_dense,
        a_ipopt_dense,
    ) = evaluate_spline(
        spline,
        ipopt_control_points,
        dense_times,
    )

    (
        q_pin_ipopt_dense,
        cart_ipopt_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_ipopt_dense,
    )

    # =================================================================
    # Dense joint feasibility
    # =================================================================

    gn_joint_feasible = (
        full_feasible_mask(
            q_gn_dense,
            v_gn_dense,
            a_gn_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    ipopt_joint_feasible = (
        full_feasible_mask(
            q_ipopt_dense,
            v_ipopt_dense,
            a_ipopt_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    # =================================================================
    # Dense collision validation
    # =================================================================

    collision_validation_times = (
        np.linspace(
            spline.t_start,
            spline.t_end,
            N_COLLISION_VALIDATION,
        )
    )

    dense_collision_builder = (
        CollisionConstraintBuilder(
            spline=spline,
            environment=environment,
            collocation_times=(
                collision_validation_times
            ),
            safety_distance=(
                SAFETY_DISTANCE
            ),
        )
    )

    initial_collision_dense = (
        minimum_collision_distance_over_time(
            dense_collision_builder,
            initial_control_points,
        )
    )

    gn_collision_dense = (
        minimum_collision_distance_over_time(
            dense_collision_builder,
            gn_control_points,
        )
    )

    ipopt_collision_dense = (
        minimum_collision_distance_over_time(
            dense_collision_builder,
            ipopt_control_points,
        )
    )

    min_initial_collision = float(
        np.min(
            initial_collision_dense
        )
    )

    min_gn_collision = float(
        np.min(
            gn_collision_dense
        )
    )

    min_ipopt_collision = float(
        np.min(
            ipopt_collision_dense
        )
    )

    print()
    print("=" * 82)
    print(
        "DENSE COLLISION VALIDATION"
    )
    print("=" * 82)

    print(
        f"QP initialization: "
        f"{1000.0 * min_initial_collision:.2f} mm"
    )

    print(
        f"Gauss-Newton:      "
        f"{1000.0 * min_gn_collision:.2f} mm"
    )

    print(
        f"IPOPT:             "
        f"{1000.0 * min_ipopt_collision:.2f} mm"
    )

    print(
        f"Required:          "
        f"{1000.0 * SAFETY_DISTANCE:.2f} mm"
    )

    print()

    print(
        "GN collision feasible:",
        (
            min_gn_collision
            >= SAFETY_DISTANCE
            - FEASIBILITY_TOL
        ),
    )

    print(
        "IPOPT collision feasible:",
        (
            min_ipopt_collision
            >= SAFETY_DISTANCE
            - FEASIBILITY_TOL
        ),
    )

    print()

    print(
        "GN joint q/v/a feasible:",
        bool(
            np.all(
                gn_joint_feasible
            )
        ),
    )

    print(
        "IPOPT joint q/v/a feasible:",
        bool(
            np.all(
                ipopt_joint_feasible
            )
        ),
    )

    # =================================================================
    # Tracking errors
    # =================================================================

    (
        position_error_gn,
        orientation_error_gn,
    ) = pose_tracking_errors(
        robot,
        q_pin_ref_dense,
        q_pin_gn_dense,
    )

    (
        position_error_ipopt,
        orientation_error_ipopt,
    ) = pose_tracking_errors(
        robot,
        q_pin_ref_dense,
        q_pin_ipopt_dense,
    )

    # =================================================================
    # Numerical comparison
    # =================================================================

    print_pose_tracking_comparison(
        position_error_gn,
        orientation_error_gn,
        position_error_ipopt,
        orientation_error_ipopt,
    )

    print_constraint_comparison(
        q_gn_dense,
        v_gn_dense,
        a_gn_dense,
        q_ipopt_dense,
        v_ipopt_dense,
        a_ipopt_dense,
    )

    print_smoothness_comparison(
        dense_times,
        a_gn_dense,
        a_ipopt_dense,
    )

    print_solver_comparison(
        gn_result,
        gn_time,
        ipopt_result,
    )

    print_iteration_summary(
        gn_result
    )

    # =================================================================
    # Figures
    # =================================================================

    plot_cartesian_workspace(
        cart_ref_dense,
        cart_initial_dense,
        cart_gn_dense,
        cart_ipopt_dense,
        obstacle,
    )

    plot_collision_clearance(
        collision_validation_times,
        initial_collision_dense,
        gn_collision_dense,
        ipopt_collision_dense,
    )

    plot_pose_tracking_errors(
        dense_times,
        position_error_gn,
        orientation_error_gn,
        position_error_ipopt,
        orientation_error_ipopt,
        nominal_position_feasible,
    )

    plot_cartesian_dynamics_comparison(
        dense_times,
        cart_ref_dense,
        cart_gn_dense,
        cart_ipopt_dense,
        nominal_position_feasible,
    )

    plot_joint_quantity_comparison(
        dense_times,
        q_ref_dense,
        q_gn_dense,
        q_ipopt_dense,
        Q_MIN,
        Q_MAX,
        ylabel="Position [rad]",
        title="Joint positions",
    )

    plot_joint_quantity_comparison(
        dense_times,
        v_ref_dense,
        v_gn_dense,
        v_ipopt_dense,
        V_MIN,
        V_MAX,
        ylabel="Velocity [rad/s]",
        title="Joint velocities",
    )

    plot_joint_quantity_comparison(
        dense_times,
        a_ref_dense,
        a_gn_dense,
        a_ipopt_dense,
        A_MIN,
        A_MAX,
        ylabel="Acceleration [rad/s²]",
        title="Joint accelerations",
    )

    plot_feasibility_ratios_comparison(
        dense_times,

        q_ref_dense,
        q_gn_dense,
        q_ipopt_dense,

        v_ref_dense,
        v_gn_dense,
        v_ipopt_dense,

        a_ref_dense,
        a_gn_dense,
        a_ipopt_dense,
    )

    plot_objective_convergence(
        gn_result,
        ipopt_result,
    )

    plt.show(
        block=False
    )

    plt.pause(
        0.2
    )

    # =================================================================
    # Meshcat
    # =================================================================

    print()
    print("=" * 82)
    print(
        "MESHCAT VISUALIZATION"
    )
    print("=" * 82)

    robot.init_viewer(
        open_browser=True
    )

    robot.display_target(
        target_poses[-1],
        name="final_target",
        scale=0.08,
    )

    display_obstacles(
        robot,
        obstacles,
    )

    display_meshcat_nominal_feasibility_path(
        robot,
        cart_ref_dense,
        nominal_position_feasible,
        name="nominal",
    )

    _set_meshcat_line(
        robot,
        cart_initial_dense,
        name="initial_qp",
        color=0x888888,
        linewidth=3.0,
    )

    _set_meshcat_line(
        robot,
        cart_gn_dense,
        name="gauss_newton",
        color=0xAA33CC,
        linewidth=4.0,
    )

    _set_meshcat_line(
        robot,
        cart_ipopt_dense,
        name="ipopt",
        color=0x0077FF,
        linewidth=4.0,
    )

    # =================================================================
    # Animation trajectories
    # =================================================================

    q_initial_animation, _, _ = (
        evaluate_spline(
            spline,
            initial_control_points,
            times,
        )
    )

    q_gn_animation, _, _ = (
        evaluate_spline(
            spline,
            gn_control_points,
            times,
        )
    )

    q_ipopt_animation, _, _ = (
        evaluate_spline(
            spline,
            ipopt_control_points,
            times,
        )
    )

    q_pin_initial_animation = (
        robot
        .joint_trajectory_to_configurations(
            q_initial_animation
        )
    )

    q_pin_gn_animation = (
        robot
        .joint_trajectory_to_configurations(
            q_gn_animation
        )
    )

    q_pin_ipopt_animation = (
        robot
        .joint_trajectory_to_configurations(
            q_ipopt_animation
        )
    )

    animation_dt = (
        (
            times[-1]
            - times[0]
        )
        / (
            len(times)
            - 1
        )
        / ANIMATION_SPEED
    )

    print()
    print(
        "Meshcat colors:"
    )

    print(
        "  green/red = nominal IK"
    )

    print(
        "  grey      = QP initialization"
    )

    print(
        "  purple    = Gauss-Newton / SQP"
    )

    print(
        "  blue      = IPOPT"
    )

    print(
        "  sphere    = static obstacle"
    )

    input(
        "\nPress Enter to animate "
        "the QP INITIALIZATION..."
    )

    animate_with_capsules(
        robot,
        q_pin_initial_animation,
        KINOVA_CAPSULES,
        animation_dt,
    )

    input(
        "\nPress Enter to animate "
        "the GAUSS-NEWTON trajectory..."
    )

    animate_with_capsules(
        robot,
        q_pin_gn_animation,
        KINOVA_CAPSULES,
        animation_dt,
    )

    input(
        "\nPress Enter to animate "
        "the IPOPT trajectory..."
    )

    animate_with_capsules(
        robot,
        q_pin_ipopt_animation,
        KINOVA_CAPSULES,
        animation_dt,
    )

    print()
    print(
        "Done."
    )

    input(
        "Press Enter to quit..."
    )


if __name__ == "__main__":
    main()

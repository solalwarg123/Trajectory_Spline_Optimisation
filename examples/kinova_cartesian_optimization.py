import time

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from mpl_toolkits.mplot3d.art3d import Line3DCollection

from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
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
    position_usage,
    symmetric_usage,
    shade_time_feasibility,
    display_meshcat_nominal_feasibility_path,
    _set_meshcat_line,
)


# =====================================================================
# Configuration
# =====================================================================

CARTESIAN_DAMPING = 1e-3
CARTESIAN_MAX_ITERATIONS = 20

CARTESIAN_TERMINAL_WEIGHT = 1.0

# Equivalent translation scale of one radian of orientation error.
#
# orientation cost:
#
#     orientation_scale^2 * ||rotation_error||^2
#
ORIENTATION_SCALE = 0.10

# Only used for dense numerical feasibility reporting.
FEASIBILITY_TOL = 1e-3


# =====================================================================
# Plot colors
# =====================================================================

COLOR_NOMINAL = "C1"
COLOR_JOINT = "C2"

COLOR_GN_POSITION = "C0"
COLOR_GN_POSE = "C3"

COLOR_FEASIBLE = "#2ca02c"
COLOR_INFEASIBLE = "#d62728"
COLOR_LIMIT = "#555555"


# =====================================================================
# Generic metrics
# =====================================================================

def rms(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def acceleration_cost(
    acceleration,
    times,
):
    """
    Integral of squared joint acceleration.

        integral ||qddot(t)||² dt
    """

    return float(
        np.trapezoid(
            np.sum(
                acceleration ** 2,
                axis=1,
            ),
            times,
        )
    )


def pose_tracking_errors(
    robot,
    q_pin_reference,
    q_pin_candidate,
):
    """
    Point-wise end-effector tracking errors.

    Returns
    -------
    position_errors :
        Euclidean translation errors [m].

    orientation_errors :
        Geodesic SO(3) orientation errors [rad].
    """

    position_errors = []
    orientation_errors = []

    for (
        q_reference,
        q_candidate,
    ) in zip(
        q_pin_reference,
        q_pin_candidate,
    ):

        M_reference = robot.frame_pose(
            q_reference
        )

        M_candidate = robot.frame_pose(
            q_candidate
        )

        # ---------------------------------------------------------
        # Translation
        # ---------------------------------------------------------

        position_error = np.linalg.norm(
            M_candidate.translation
            - M_reference.translation
        )

        # ---------------------------------------------------------
        # Rotation
        #
        # Geodesic SO(3) distance:
        #
        #     || log3(R_candidate^T R_reference) ||
        # ---------------------------------------------------------

        R_error = (
            M_candidate.rotation.T
            @ M_reference.rotation
        )

        orientation_error = np.linalg.norm(
            pin.log3(
                R_error
            )
        )

        position_errors.append(
            position_error
        )

        orientation_errors.append(
            orientation_error
        )

    return (
        np.asarray(
            position_errors
        ),
        np.asarray(
            orientation_errors
        ),
    )


# =====================================================================
# Plot styling helper
# =====================================================================

def shade_horizontal_admissible_region(
    ax,
    lower,
    upper,
    trajectories,
):
    """
    Shade admissible and forbidden joint-space regions.
    """

    all_values = np.concatenate(
        [
            np.asarray(values).reshape(-1)
            for values in trajectories
        ]
    )

    finite_values = all_values[
        np.isfinite(all_values)
    ]

    if np.isfinite(lower):
        finite_values = np.append(
            finite_values,
            lower,
        )

    if np.isfinite(upper):
        finite_values = np.append(
            finite_values,
            upper,
        )

    if finite_values.size == 0:
        return

    value_min = np.min(
        finite_values
    )

    value_max = np.max(
        finite_values
    )

    span = max(
        value_max - value_min,
        0.1,
    )

    y_min = (
        value_min
        - 0.12 * span
    )

    y_max = (
        value_max
        + 0.12 * span
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    # -------------------------------------------------------------
    # Two-sided limits
    # -------------------------------------------------------------

    if (
        np.isfinite(lower)
        and np.isfinite(upper)
    ):

        ax.axhspan(
            lower,
            upper,
            color=COLOR_FEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        if y_min < lower:

            ax.axhspan(
                y_min,
                lower,
                color=COLOR_INFEASIBLE,
                alpha=0.08,
                zorder=0,
            )

        if upper < y_max:

            ax.axhspan(
                upper,
                y_max,
                color=COLOR_INFEASIBLE,
                alpha=0.08,
                zorder=0,
            )

        ax.axhline(
            lower,
            color=COLOR_LIMIT,
            linestyle=":",
            linewidth=1.1,
            zorder=1,
        )

        ax.axhline(
            upper,
            color=COLOR_LIMIT,
            linestyle=":",
            linewidth=1.1,
            zorder=1,
        )

    # -------------------------------------------------------------
    # Lower bound only
    # -------------------------------------------------------------

    elif np.isfinite(lower):

        ax.axhspan(
            lower,
            y_max,
            color=COLOR_FEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        ax.axhspan(
            y_min,
            lower,
            color=COLOR_INFEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        ax.axhline(
            lower,
            color=COLOR_LIMIT,
            linestyle=":",
            linewidth=1.1,
            zorder=1,
        )

    # -------------------------------------------------------------
    # Upper bound only
    # -------------------------------------------------------------

    elif np.isfinite(upper):

        ax.axhspan(
            y_min,
            upper,
            color=COLOR_FEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        ax.axhspan(
            upper,
            y_max,
            color=COLOR_INFEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        ax.axhline(
            upper,
            color=COLOR_LIMIT,
            linestyle=":",
            linewidth=1.1,
            zorder=1,
        )

    # -------------------------------------------------------------
    # Continuous joint
    # -------------------------------------------------------------

    else:

        ax.axhspan(
            y_min,
            y_max,
            color=COLOR_FEASIBLE,
            alpha=0.05,
            zorder=0,
        )


# =====================================================================
# Cartesian 3D comparison
# =====================================================================

def plot_cartesian_3d_comparison(
    cart_nominal,
    cart_joint,
    cart_position,
    cart_pose,
    nominal_position_feasible,
):
    """
    3D Cartesian path comparison.

    Nominal:
        green = feasible joint positions
        red   = infeasible joint positions

    Joint-space QP:
        orange

    Cartesian GN position-only:
        blue

    Cartesian GN full pose:
        purple
    """

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # -------------------------------------------------------------
    # Nominal path
    # -------------------------------------------------------------

    segments = np.stack(
        [
            cart_nominal[:-1],
            cart_nominal[1:],
        ],
        axis=1,
    )

    segment_feasible = (
        nominal_position_feasible[:-1]
        & nominal_position_feasible[1:]
    )

    segment_colors = [
        (
            COLOR_FEASIBLE
            if feasible
            else COLOR_INFEASIBLE
        )
        for feasible
        in segment_feasible
    ]

    collection = Line3DCollection(
        segments,
        colors=segment_colors,
        linewidths=3.0,
        zorder=10,
    )

    ax.add_collection3d(
        collection
    )

    # -------------------------------------------------------------
    # Joint-space baseline
    # -------------------------------------------------------------

    ax.plot(
        cart_joint[:, 0],
        cart_joint[:, 1],
        cart_joint[:, 2],
        color=COLOR_JOINT,
        linewidth=2.3,
        label="Joint-space QP",
    )

    # -------------------------------------------------------------
    # Position-only GN
    # -------------------------------------------------------------

    ax.plot(
        cart_position[:, 0],
        cart_position[:, 1],
        cart_position[:, 2],
        color=COLOR_GN_POSITION,
        linewidth=2.5,
        label="Cartesian GN — position only",
    )

    # -------------------------------------------------------------
    # Full pose GN
    # -------------------------------------------------------------

    ax.plot(
        cart_pose[:, 0],
        cart_pose[:, 1],
        cart_pose[:, 2],
        color=COLOR_GN_POSE,
        linewidth=2.5,
        label="Cartesian GN — full pose",
    )

    # Nominal legend handles

    ax.plot(
        [],
        [],
        [],
        color=COLOR_FEASIBLE,
        linewidth=3.0,
        label="Nominal: joint-position feasible",
    )

    ax.plot(
        [],
        [],
        [],
        color=COLOR_INFEASIBLE,
        linewidth=3.0,
        label="Nominal: joint-position infeasible",
    )

    # Start / end

    ax.scatter(
        cart_nominal[0, 0],
        cart_nominal[0, 1],
        cart_nominal[0, 2],
        s=65,
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

    # -------------------------------------------------------------
    # Axis limits
    # -------------------------------------------------------------

    all_points = np.vstack(
        [
            cart_nominal,
            cart_joint,
            cart_position,
            cart_pose,
        ]
    )

    xyz_min = np.min(
        all_points,
        axis=0,
    )

    xyz_max = np.max(
        all_points,
        axis=0,
    )

    ax.set_xlim(
        xyz_min[0],
        xyz_max[0],
    )

    ax.set_ylim(
        xyz_min[1],
        xyz_max[1],
    )

    ax.set_zlim(
        xyz_min[2],
        xyz_max[2],
    )

    ranges = np.maximum(
        xyz_max - xyz_min,
        1e-3,
    )

    ax.set_box_aspect(
        ranges
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.set_title(
        "Cartesian end-effector trajectories"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()


# =====================================================================
# Pose tracking errors
# =====================================================================

def plot_pose_tracking_errors(
    times,
    position_error_joint,
    orientation_error_joint,
    position_error_position,
    orientation_error_position,
    position_error_pose,
    orientation_error_pose,
    nominal_position_feasible,
):
    """
    Compare translation and orientation tracking errors.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
    )

    for ax in axes:

        shade_time_feasibility(
            ax,
            times,
            nominal_position_feasible,
            alpha=0.06,
        )

    # =============================================================
    # Position error
    # =============================================================

    axes[0].plot(
        times,
        1000.0 * position_error_joint,
        color=COLOR_JOINT,
        linewidth=2.0,
        label="Joint-space QP",
        zorder=4,
    )

    axes[0].plot(
        times,
        1000.0 * position_error_position,
        color=COLOR_GN_POSITION,
        linewidth=2.3,
        label="Cartesian GN — position only",
        zorder=5,
    )

    axes[0].plot(
        times,
        1000.0 * position_error_pose,
        color=COLOR_GN_POSE,
        linewidth=2.3,
        label="Cartesian GN — full pose",
        zorder=6,
    )

    axes[0].axhline(
        0.0,
        color=COLOR_NOMINAL,
        linestyle="--",
        linewidth=1.8,
        label="Nominal IK",
        zorder=10,
    )

    axes[0].set_ylabel(
        "Position error [mm]"
    )

    axes[0].set_title(
        "End-effector position tracking error"
    )

    axes[0].legend(
        loc="best"
    )

    # =============================================================
    # Orientation error
    # =============================================================

    axes[1].plot(
        times,
        np.degrees(
            orientation_error_joint
        ),
        color=COLOR_JOINT,
        linewidth=2.0,
        label="Joint-space QP",
        zorder=4,
    )

    axes[1].plot(
        times,
        np.degrees(
            orientation_error_position
        ),
        color=COLOR_GN_POSITION,
        linewidth=2.3,
        label="Cartesian GN — position only",
        zorder=5,
    )

    axes[1].plot(
        times,
        np.degrees(
            orientation_error_pose
        ),
        color=COLOR_GN_POSE,
        linewidth=2.3,
        label="Cartesian GN — full pose",
        zorder=6,
    )

    axes[1].axhline(
        0.0,
        color=COLOR_NOMINAL,
        linestyle="--",
        linewidth=1.8,
        label="Nominal IK",
        zorder=10,
    )

    axes[1].set_ylabel(
        "Orientation error [deg]"
    )

    axes[1].set_xlabel(
        "Time [s]"
    )

    axes[1].set_title(
        "End-effector orientation tracking error"
    )

    axes[1].legend(
        loc="best"
    )

    fig.tight_layout()


# =====================================================================
# Cartesian dynamics
# =====================================================================

def plot_cartesian_dynamics_comparison(
    times,
    cart_nominal,
    cart_joint,
    cart_position,
    cart_pose,
    nominal_position_feasible,
):
    """
    Compare Cartesian speed and acceleration.
    """

    (
        v_nominal,
        a_nominal,
    ) = numerical_derivatives(
        cart_nominal,
        times,
    )

    (
        v_joint,
        a_joint,
    ) = numerical_derivatives(
        cart_joint,
        times,
    )

    (
        v_position,
        a_position,
    ) = numerical_derivatives(
        cart_position,
        times,
    )

    (
        v_pose,
        a_pose,
    ) = numerical_derivatives(
        cart_pose,
        times,
    )

    speed_nominal = np.linalg.norm(
        v_nominal,
        axis=1,
    )

    speed_joint = np.linalg.norm(
        v_joint,
        axis=1,
    )

    speed_position = np.linalg.norm(
        v_position,
        axis=1,
    )

    speed_pose = np.linalg.norm(
        v_pose,
        axis=1,
    )

    accel_nominal = np.linalg.norm(
        a_nominal,
        axis=1,
    )

    accel_joint = np.linalg.norm(
        a_joint,
        axis=1,
    )

    accel_position = np.linalg.norm(
        a_position,
        axis=1,
    )

    accel_pose = np.linalg.norm(
        a_pose,
        axis=1,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
    )

    for ax in axes:

        shade_time_feasibility(
            ax,
            times,
            nominal_position_feasible,
        )

    # =============================================================
    # Speed
    # =============================================================

    axes[0].plot(
        times,
        speed_joint,
        color=COLOR_JOINT,
        linewidth=2.0,
        label="Joint-space QP",
        zorder=4,
    )

    axes[0].plot(
        times,
        speed_position,
        color=COLOR_GN_POSITION,
        linewidth=2.2,
        label="Cartesian GN — position only",
        zorder=5,
    )

    axes[0].plot(
        times,
        speed_pose,
        color=COLOR_GN_POSE,
        linewidth=2.2,
        label="Cartesian GN — full pose",
        zorder=6,
    )

    # Plot nominal last and above everything else.

    axes[0].plot(
        times,
        speed_nominal,
        "--",
        color=COLOR_NOMINAL,
        linewidth=1.8,
        label="Nominal IK",
        zorder=10,
    )

    axes[0].set_ylabel(
        r"$\|\dot p\|$ [m/s]"
    )

    axes[0].set_title(
        "Cartesian speed"
    )

    axes[0].legend(
        loc="best"
    )

    # =============================================================
    # Acceleration
    # =============================================================

    axes[1].plot(
        times,
        accel_joint,
        color=COLOR_JOINT,
        linewidth=2.0,
        label="Joint-space QP",
        zorder=4,
    )

    axes[1].plot(
        times,
        accel_position,
        color=COLOR_GN_POSITION,
        linewidth=2.2,
        label="Cartesian GN — position only",
        zorder=5,
    )

    axes[1].plot(
        times,
        accel_pose,
        color=COLOR_GN_POSE,
        linewidth=2.2,
        label="Cartesian GN — full pose",
        zorder=6,
    )

    axes[1].plot(
        times,
        accel_nominal,
        "--",
        color=COLOR_NOMINAL,
        linewidth=1.8,
        label="Nominal IK",
        zorder=10,
    )

    axes[1].set_ylabel(
        r"$\|\ddot p\|$ [m/s²]"
    )

    axes[1].set_xlabel(
        "Time [s]"
    )

    axes[1].set_title(
        "Cartesian acceleration magnitude"
    )

    axes[1].legend(
        loc="best"
    )

    fig.tight_layout()


# =====================================================================
# Joint-space plots
# =====================================================================

def plot_joint_quantity_comparison(
    times,
    nominal,
    joint_space,
    position_gn,
    pose_gn,
    lower,
    upper,
    ylabel,
    title,
):
    """
    Compare all four trajectories for the seven robot joints.
    """

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13, 11),
        sharex=True,
    )

    axes = axes.ravel()

    for j in range(7):

        ax = axes[j]

        shade_horizontal_admissible_region(
            ax,
            lower[j],
            upper[j],
            trajectories=[
                nominal[:, j],
                joint_space[:, j],
                position_gn[:, j],
                pose_gn[:, j],
            ],
        )

        # ---------------------------------------------------------
        # Optimized trajectories
        # ---------------------------------------------------------

        ax.plot(
            times,
            joint_space[:, j],
            color=COLOR_JOINT,
            linewidth=2.0,
            label="Joint-space QP",
            zorder=4,
        )

        ax.plot(
            times,
            position_gn[:, j],
            color=COLOR_GN_POSITION,
            linewidth=2.2,
            label="Cartesian GN — position",
            zorder=5,
        )

        ax.plot(
            times,
            pose_gn[:, j],
            color=COLOR_GN_POSE,
            linewidth=2.2,
            label="Cartesian GN — full pose",
            zorder=6,
        )

        # ---------------------------------------------------------
        # Nominal on top
        # ---------------------------------------------------------

        ax.plot(
            times,
            nominal[:, j],
            "--",
            color=COLOR_NOMINAL,
            linewidth=1.8,
            label="Nominal IK",
            zorder=10,
        )

        ax.set_title(
            f"Joint {j + 1}"
        )

        ax.set_ylabel(
            ylabel
        )

        if j >= 6:

            ax.set_xlabel(
                "Time [s]"
            )

        if (
            not np.isfinite(
                lower[j]
            )
            and not np.isfinite(
                upper[j]
            )
        ):

            ax.text(
                0.02,
                0.92,
                "continuous joint",
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.7,
                va="top",
            )

    axes[0].legend(
        loc="best"
    )

    axes[-1].axis(
        "off"
    )

    fig.suptitle(
        title
        + "\nGreen = admissible, red = forbidden",
        y=0.995,
    )

    fig.tight_layout()


# =====================================================================
# Constraint ratio plots
# =====================================================================

def plot_feasibility_ratios_comparison(
    times,
    q_nominal,
    q_joint,
    q_position,
    q_pose,
    v_nominal,
    v_joint,
    v_position,
    v_pose,
    a_nominal,
    a_joint,
    a_position,
    a_pose,
):
    """
    Compare normalized q / qdot / qddot constraint usage.
    """

    curves = [
        (
            position_usage(
                q_nominal
            ),
            position_usage(
                q_joint
            ),
            position_usage(
                q_position
            ),
            position_usage(
                q_pose
            ),
            r"$\rho_q$",
            "Joint-position limit usage",
        ),
        (
            symmetric_usage(
                v_nominal,
                V_MAX,
            ),
            symmetric_usage(
                v_joint,
                V_MAX,
            ),
            symmetric_usage(
                v_position,
                V_MAX,
            ),
            symmetric_usage(
                v_pose,
                V_MAX,
            ),
            r"$\rho_v$",
            "Joint-velocity limit usage",
        ),
        (
            symmetric_usage(
                a_nominal,
                A_MAX,
            ),
            symmetric_usage(
                a_joint,
                A_MAX,
            ),
            symmetric_usage(
                a_position,
                A_MAX,
            ),
            symmetric_usage(
                a_pose,
                A_MAX,
            ),
            r"$\rho_a$",
            "Joint-acceleration limit usage",
        ),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    for (
        ax,
        (
            nominal,
            joint_space,
            position_gn,
            pose_gn,
            ylabel,
            title,
        ),
    ) in zip(
        axes,
        curves,
    ):

        upper = max(
            1.15,
            1.08 * np.max(
                nominal
            ),
            1.08 * np.max(
                joint_space
            ),
            1.08 * np.max(
                position_gn
            ),
            1.08 * np.max(
                pose_gn
            ),
        )

        ax.axhspan(
            0.0,
            1.0,
            color=COLOR_FEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        ax.axhspan(
            1.0,
            upper,
            color=COLOR_INFEASIBLE,
            alpha=0.08,
            zorder=0,
        )

        ax.axhline(
            1.0,
            color=COLOR_LIMIT,
            linestyle=":",
            linewidth=1.3,
            label="Constraint boundary",
            zorder=2,
        )

        ax.plot(
            times,
            joint_space,
            color=COLOR_JOINT,
            linewidth=2.0,
            label="Joint-space QP",
            zorder=4,
        )

        ax.plot(
            times,
            position_gn,
            color=COLOR_GN_POSITION,
            linewidth=2.2,
            label="Cartesian GN — position",
            zorder=5,
        )

        ax.plot(
            times,
            pose_gn,
            color=COLOR_GN_POSE,
            linewidth=2.2,
            label="Cartesian GN — full pose",
            zorder=6,
        )

        ax.plot(
            times,
            nominal,
            "--",
            color=COLOR_NOMINAL,
            linewidth=1.8,
            label="Nominal IK",
            zorder=10,
        )

        ax.set_ylim(
            0.0,
            upper,
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title
        )

    axes[0].legend(
        loc="best"
    )

    axes[-1].set_xlabel(
        "Time [s]"
    )

    fig.suptitle(
        "Normalized feasibility "
        "(<= 1 is admissible)"
    )

    fig.tight_layout()


# =====================================================================
# Numerical reporting
# =====================================================================

def print_pose_tracking_comparison(
    position_error_joint,
    orientation_error_joint,
    position_error_position,
    orientation_error_position,
    position_error_pose,
    orientation_error_pose,
):
    """
    Compact pose-tracking comparison.
    """

    print()
    print("=" * 94)
    print(
        "END-EFFECTOR TRACKING COMPARISON"
    )
    print("=" * 94)

    print(
        f"{'Method':<30}"
        f"{'pos RMS [mm]':>15}"
        f"{'pos max [mm]':>15}"
        f"{'ori RMS [deg]':>16}"
        f"{'ori max [deg]':>16}"
    )

    print("-" * 94)

    rows = [
        (
            "Joint-space QP",
            position_error_joint,
            orientation_error_joint,
        ),
        (
            "Cartesian GN - position",
            position_error_position,
            orientation_error_position,
        ),
        (
            "Cartesian GN - full pose",
            position_error_pose,
            orientation_error_pose,
        ),
    ]

    for (
        name,
        position_error,
        orientation_error,
    ) in rows:

        print(
            f"{name:<30}"
            f"{1000.0 * rms(position_error):>15.2f}"
            f"{1000.0 * np.max(position_error):>15.2f}"
            f"{np.degrees(rms(orientation_error)):>16.2f}"
            f"{np.degrees(np.max(orientation_error)):>16.2f}"
        )


def print_constraint_comparison(
    q_joint,
    v_joint,
    a_joint,
    q_position,
    v_position,
    a_position,
    q_pose,
    v_pose,
    a_pose,
):
    """
    Print maximum normalized constraint usage.
    """

    print()
    print("=" * 80)
    print(
        "CONSTRAINT USAGE"
    )
    print("=" * 80)

    print(
        f"{'Method':<30}"
        f"{'rho_q':>14}"
        f"{'rho_v':>14}"
        f"{'rho_a':>14}"
    )

    print("-" * 80)

    rows = [
        (
            "Joint-space QP",
            q_joint,
            v_joint,
            a_joint,
        ),
        (
            "Cartesian GN - position",
            q_position,
            v_position,
            a_position,
        ),
        (
            "Cartesian GN - full pose",
            q_pose,
            v_pose,
            a_pose,
        ),
    ]

    for (
        name,
        q,
        v,
        a,
    ) in rows:

        print(
            f"{name:<30}"
            f"{np.max(position_usage(q)):>14.6f}"
            f"{np.max(symmetric_usage(v, V_MAX)):>14.6f}"
            f"{np.max(symmetric_usage(a, A_MAX)):>14.6f}"
        )


def print_smoothness_comparison(
    dense_times,
    a_joint,
    a_position,
    a_pose,
):
    """
    Print integrated squared joint acceleration.
    """

    print()
    print("=" * 70)
    print(
        "JOINT-SPACE SMOOTHNESS"
    )
    print("=" * 70)

    print(
        f"{'Method':<30}"
        f"{'integral ||qddot||²':>25}"
    )

    print("-" * 70)

    rows = [
        (
            "Joint-space QP",
            a_joint,
        ),
        (
            "Cartesian GN - position",
            a_position,
        ),
        (
            "Cartesian GN - full pose",
            a_pose,
        ),
    ]

    for name, acceleration in rows:

        print(
            f"{name:<30}"
            f"{acceleration_cost(acceleration, dense_times):>25.3f}"
        )


def print_iteration_summary(
    result,
    title,
):
    """
    Print one Gauss-Newton iteration history.
    """

    print()
    print("=" * 110)
    print(
        title
    )
    print("=" * 110)

    if not result.history:

        print(
            "No accepted Gauss-Newton iterations."
        )

        return

    for item in result.history:

        print(
            f"{item.iteration:02d} | "
            f"J={item.objective:.6e} | "
            f"pos RMS={1e3 * item.position_rms:.3f} mm | "
            f"pos max={1e3 * item.position_max:.3f} mm | "
            f"ori RMS={np.degrees(item.orientation_rms):.3f} deg | "
            f"ori max={np.degrees(item.orientation_max):.3f} deg | "
            f"step={item.step_norm:.3e} | "
            f"alpha={item.alpha:.3f} | "
            f"OSQP={1e3 * item.osqp_solve_time:.3f} ms"
        )


# =====================================================================
# Main
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Robot
    # -----------------------------------------------------------------

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    print(
        robot
    )

    print(
        f"nq = {robot.nq}, "
        f"nv = {robot.nv}"
    )

    # =================================================================
    # 1) Cartesian reference
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
    ) = build_cartesian_targets_from_virtual_reference(
        robot,
        theta_virtual,
    )

    # =================================================================
    # 2) Sequential full-pose IK
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
    # 3) V1 joint-space constrained projection
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
    # 4) Cartesian problems and Gauss-Newton solver
    # =================================================================

    common_problem_kwargs = {
        "robot": robot,
        "spline": spline,
        "tracking_times": times,
        "target_poses": target_poses,

        "q0": theta_ref[0],
        "v0": np.zeros(robot.nv),
        "a0": np.zeros(robot.nv),
        "vf": np.zeros(robot.nv),

        "q_min": Q_MIN,
        "q_max": Q_MAX,
        "v_min": V_MIN,
        "v_max": V_MAX,
        "a_min": A_MIN,
        "a_max": A_MAX,

        "lambda_smooth": LAMBDA_SMOOTH,
        "lambda_terminal": CARTESIAN_TERMINAL_WEIGHT,
    }

    position_problem = CartesianTrajectoryProblem(
        **common_problem_kwargs,
        orientation_scale=0.0,
    )

    pose_problem = CartesianTrajectoryProblem(
        **common_problem_kwargs,
        orientation_scale=ORIENTATION_SCALE,
    )

    solver = GaussNewtonSolver(
        damping_mode="adaptive",
        damping=CARTESIAN_DAMPING,
        max_iterations=CARTESIAN_MAX_ITERATIONS,
    )

    # -----------------------------------------------------------------
    # Initial pose error
    # -----------------------------------------------------------------

    initial_metrics = (
        pose_problem.error_metrics(
            initial_control_points
        )
    )

    print()
    print(
        "Initial feasible B-spline pose tracking:"
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
        f"  orientation scale = "
        f"{ORIENTATION_SCALE:.3f} m/rad"
    )

    # =================================================================
    # 5) Cartesian GN — position only
    # =================================================================

    print()
    print("=" * 76)
    print(
        "Solving Cartesian B-spline optimization "
        "(POSITION ONLY)..."
    )
    print("=" * 76)

    position_result = solver.solve(
        problem=position_problem,
        initial_control_points=initial_control_points,
        verbose=True,
    )

    position_control_points = (
        position_result.control_points
    )

    print()
    print(
        f"Position-only optimizer status: "
        f"{position_result.status}"
    )

    print(
        f"Position-only optimizer iterations: "
        f"{position_result.iterations}"
    )

    # =================================================================
    # 6) Cartesian GN — full SE(3) pose
    # =================================================================

    print()
    print("=" * 76)
    print(
        "Solving Cartesian B-spline optimization "
        "(FULL SE(3) POSE)..."
    )
    print("=" * 76)

    pose_result = solver.solve(
        problem=pose_problem,
        initial_control_points=initial_control_points,
        verbose=True,
    )

    pose_control_points = (
        pose_result.control_points
    )

    print()
    print(
        f"Full-pose optimizer status: "
        f"{pose_result.status}"
    )

    print(
        f"Full-pose optimizer iterations: "
        f"{pose_result.iterations}"
    )

    # =================================================================
    # 7) Dense evaluation
    # =================================================================

    dense_times = np.linspace(
        times[0],
        times[-1],
        N_DENSE_SAMPLES,
    )

    # -----------------------------------------------------------------
    # Nominal IK
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # V1 joint-space QP
    # -----------------------------------------------------------------

    (
        q_joint_dense,
        v_joint_dense,
        a_joint_dense,
    ) = evaluate_spline(
        spline,
        initial_control_points,
        dense_times,
    )

    (
        q_pin_joint_dense,
        cart_joint_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_joint_dense,
    )

    # -----------------------------------------------------------------
    # V2a Cartesian GN position-only
    # -----------------------------------------------------------------

    (
        q_position_dense,
        v_position_dense,
        a_position_dense,
    ) = evaluate_spline(
        spline,
        position_control_points,
        dense_times,
    )

    (
        q_pin_position_dense,
        cart_position_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_position_dense,
    )

    # -----------------------------------------------------------------
    # V2b Cartesian GN full pose
    # -----------------------------------------------------------------

    (
        q_pose_dense,
        v_pose_dense,
        a_pose_dense,
    ) = evaluate_spline(
        spline,
        pose_control_points,
        dense_times,
    )

    (
        q_pin_pose_dense,
        cart_pose_dense,
    ) = forward_kinematics_trajectory(
        robot,
        q_pose_dense,
    )

    # =================================================================
    # 8) Dense feasibility
    # =================================================================

    joint_feasible = (
        full_feasible_mask(
            q_joint_dense,
            v_joint_dense,
            a_joint_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    position_feasible = (
        full_feasible_mask(
            q_position_dense,
            v_position_dense,
            a_position_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    pose_feasible = (
        full_feasible_mask(
            q_pose_dense,
            v_pose_dense,
            a_pose_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    print()
    print(
        "Dense feasibility:"
    )

    print(
        f"  Joint-space QP:          "
        f"{bool(np.all(joint_feasible))}"
    )

    print(
        f"  Cartesian GN position:   "
        f"{bool(np.all(position_feasible))}"
    )

    print(
        f"  Cartesian GN full pose:  "
        f"{bool(np.all(pose_feasible))}"
    )

    # =================================================================
    # 9) Dense pose tracking errors
    # =================================================================

    (
        position_error_joint,
        orientation_error_joint,
    ) = pose_tracking_errors(
        robot,
        q_pin_ref_dense,
        q_pin_joint_dense,
    )

    (
        position_error_position,
        orientation_error_position,
    ) = pose_tracking_errors(
        robot,
        q_pin_ref_dense,
        q_pin_position_dense,
    )

    (
        position_error_pose,
        orientation_error_pose,
    ) = pose_tracking_errors(
        robot,
        q_pin_ref_dense,
        q_pin_pose_dense,
    )

    # =================================================================
    # 10) Numerical comparison
    # =================================================================

    print_pose_tracking_comparison(
        position_error_joint,
        orientation_error_joint,
        position_error_position,
        orientation_error_position,
        position_error_pose,
        orientation_error_pose,
    )

    print_constraint_comparison(
        q_joint_dense,
        v_joint_dense,
        a_joint_dense,
        q_position_dense,
        v_position_dense,
        a_position_dense,
        q_pose_dense,
        v_pose_dense,
        a_pose_dense,
    )

    print_smoothness_comparison(
        dense_times,
        a_joint_dense,
        a_position_dense,
        a_pose_dense,
    )

    # =================================================================
    # 11) Gauss-Newton iteration histories
    # =================================================================

    print_iteration_summary(
        position_result,
        (
            "GAUSS-NEWTON ITERATIONS "
            "— POSITION ONLY"
        ),
    )

    print_iteration_summary(
        pose_result,
        (
            "GAUSS-NEWTON ITERATIONS "
            "— FULL SE(3) POSE"
        ),
    )

    # =================================================================
    # 12) Figures
    # =================================================================

    # -----------------------------------------------------------------
    # Cartesian paths
    # -----------------------------------------------------------------

    plot_cartesian_3d_comparison(
        cart_ref_dense,
        cart_joint_dense,
        cart_position_dense,
        cart_pose_dense,
        nominal_position_feasible,
    )

    # -----------------------------------------------------------------
    # Position + orientation tracking errors
    # -----------------------------------------------------------------

    plot_pose_tracking_errors(
        dense_times,
        position_error_joint,
        orientation_error_joint,
        position_error_position,
        orientation_error_position,
        position_error_pose,
        orientation_error_pose,
        nominal_position_feasible,
    )

    # -----------------------------------------------------------------
    # Cartesian speed / acceleration
    # -----------------------------------------------------------------

    plot_cartesian_dynamics_comparison(
        dense_times,
        cart_ref_dense,
        cart_joint_dense,
        cart_position_dense,
        cart_pose_dense,
        nominal_position_feasible,
    )

    # -----------------------------------------------------------------
    # Joint positions
    # -----------------------------------------------------------------

    plot_joint_quantity_comparison(
        dense_times,
        q_ref_dense,
        q_joint_dense,
        q_position_dense,
        q_pose_dense,
        Q_MIN,
        Q_MAX,
        ylabel="Position [rad]",
        title="Joint positions",
    )

    # -----------------------------------------------------------------
    # Joint velocities
    # -----------------------------------------------------------------

    plot_joint_quantity_comparison(
        dense_times,
        v_ref_dense,
        v_joint_dense,
        v_position_dense,
        v_pose_dense,
        V_MIN,
        V_MAX,
        ylabel="Velocity [rad/s]",
        title="Joint velocities",
    )

    # -----------------------------------------------------------------
    # Joint accelerations
    # -----------------------------------------------------------------

    plot_joint_quantity_comparison(
        dense_times,
        a_ref_dense,
        a_joint_dense,
        a_position_dense,
        a_pose_dense,
        A_MIN,
        A_MAX,
        ylabel="Acceleration [rad/s²]",
        title="Joint accelerations",
    )

    # -----------------------------------------------------------------
    # Normalized feasibility
    # -----------------------------------------------------------------

    plot_feasibility_ratios_comparison(
        dense_times,
        q_ref_dense,
        q_joint_dense,
        q_position_dense,
        q_pose_dense,
        v_ref_dense,
        v_joint_dense,
        v_position_dense,
        v_pose_dense,
        a_ref_dense,
        a_joint_dense,
        a_position_dense,
        a_pose_dense,
    )

    plt.show(
        block=False
    )

    plt.pause(
        0.2
    )

    # =================================================================
    # 13) Meshcat
    # =================================================================

    print()
    print(
        "Opening Meshcat..."
    )

    robot.init_viewer(
        open_browser=True
    )

    robot.display_target(
        target_poses[-1],
        name="final_target",
        scale=0.08,
    )

    # -----------------------------------------------------------------
    # Nominal
    # -----------------------------------------------------------------

    display_meshcat_nominal_feasibility_path(
        robot,
        cart_ref_dense,
        nominal_position_feasible,
        name="nominal",
    )

    # -----------------------------------------------------------------
    # Joint-space QP
    # -----------------------------------------------------------------

    _set_meshcat_line(
        robot,
        cart_joint_dense,
        name="joint_space_baseline",
        color=0xFF8C00,
        linewidth=4.0,
    )

    # -----------------------------------------------------------------
    # Position-only Cartesian GN
    # -----------------------------------------------------------------

    _set_meshcat_line(
        robot,
        cart_position_dense,
        name="cartesian_position_only",
        color=0x0077FF,
        linewidth=4.0,
    )

    # -----------------------------------------------------------------
    # Full-pose Cartesian GN
    # -----------------------------------------------------------------

    _set_meshcat_line(
        robot,
        cart_pose_dense,
        name="cartesian_full_pose",
        color=0xAA33CC,
        linewidth=4.0,
    )

    # =================================================================
    # 14) Animation trajectories
    # =================================================================

    q_joint_animation, _, _ = (
        evaluate_spline(
            spline,
            initial_control_points,
            times,
        )
    )

    q_position_animation, _, _ = (
        evaluate_spline(
            spline,
            position_control_points,
            times,
        )
    )

    q_pose_animation, _, _ = (
        evaluate_spline(
            spline,
            pose_control_points,
            times,
        )
    )

    q_pin_joint_animation = (
        robot
        .joint_trajectory_to_configurations(
            q_joint_animation
        )
    )

    q_pin_position_animation = (
        robot
        .joint_trajectory_to_configurations(
            q_position_animation
        )
    )

    q_pin_pose_animation = (
        robot
        .joint_trajectory_to_configurations(
            q_pose_animation
        )
    )

    animation_dt = (
        (times[-1] - times[0])
        / (len(times) - 1)
        / ANIMATION_SPEED
    )

    robot.display(
        ik_result.q_pin[0]
    )

    print()
    print(
        "Meshcat colors:"
    )

    print(
        "  green/red = nominal IK trajectory"
    )

    print(
        "  orange    = joint-space B-spline baseline"
    )

    print(
        "  blue      = Cartesian GN position-only"
    )

    print(
        "  purple    = Cartesian GN full SE(3) pose"
    )

    # -----------------------------------------------------------------
    # Nominal
    # -----------------------------------------------------------------

    input(
        "\nPress Enter to animate "
        "the NOMINAL IK trajectory..."
    )

    robot.animate(
        ik_result.q_pin,
        dt=animation_dt,
    )

    # -----------------------------------------------------------------
    # Joint-space QP
    # -----------------------------------------------------------------

    input(
        "\nPress Enter to animate "
        "the JOINT-SPACE B-spline trajectory..."
    )

    robot.display(
        q_pin_joint_animation[0]
    )

    time.sleep(
        0.5
    )

    robot.animate(
        q_pin_joint_animation,
        dt=animation_dt,
    )

    # -----------------------------------------------------------------
    # Position-only GN
    # -----------------------------------------------------------------

    input(
        "\nPress Enter to animate "
        "the CARTESIAN POSITION-ONLY trajectory..."
    )

    robot.display(
        q_pin_position_animation[0]
    )

    time.sleep(
        0.5
    )

    robot.animate(
        q_pin_position_animation,
        dt=animation_dt,
    )

    # -----------------------------------------------------------------
    # Full-pose GN
    # -----------------------------------------------------------------

    input(
        "\nPress Enter to animate "
        "the FULL SE(3) POSE trajectory..."
    )

    robot.display(
        q_pin_pose_animation[0]
    )

    time.sleep(
        0.5
    )

    robot.animate(
        q_pin_pose_animation,
        dt=animation_dt,
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

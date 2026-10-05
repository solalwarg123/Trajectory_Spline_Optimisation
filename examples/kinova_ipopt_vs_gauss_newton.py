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

from splineopt.ipopt_solver import (
    IpoptSolver,
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


GAUSS_NEWTON_DAMPING = 1e-3
GAUSS_NEWTON_MAX_ITERATIONS = 20

IPOPT_TOLERANCE = 1e-6
IPOPT_MAX_ITERATIONS = 200

CARTESIAN_TERMINAL_WEIGHT = 1.0

# Equivalent translation scale of one radian of orientation error.
ORIENTATION_SCALE = 0.10

# Used for dense feasibility reporting.
FEASIBILITY_TOL = 1e-3


# =====================================================================
# Plot colors
# =====================================================================


COLOR_NOMINAL = "C1"
COLOR_GN = "C3"
COLOR_IPOPT = "C0"

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
    """Integral of squared joint acceleration."""

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
    Point-wise end-effector pose tracking errors.

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

        position_error = np.linalg.norm(
            M_candidate.translation
            - M_reference.translation
        )

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
# Plot helper
# =====================================================================


def shade_horizontal_admissible_region(
    ax,
    lower,
    upper,
    trajectories,
):
    """Shade admissible and forbidden joint-space regions."""

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
    cart_gn,
    cart_ipopt,
    nominal_position_feasible,
):

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
    # Optimized trajectories
    # -------------------------------------------------------------

    ax.plot(
        cart_gn[:, 0],
        cart_gn[:, 1],
        cart_gn[:, 2],
        color=COLOR_GN,
        linewidth=2.5,
        label="Gauss-Newton — full pose",
    )

    ax.plot(
        cart_ipopt[:, 0],
        cart_ipopt[:, 1],
        cart_ipopt[:, 2],
        color=COLOR_IPOPT,
        linewidth=2.5,
        label="IPOPT — full pose",
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

    all_points = np.vstack(
        [
            cart_nominal,
            cart_gn,
            cart_ipopt,
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
    position_error_gn,
    orientation_error_gn,
    position_error_ipopt,
    orientation_error_ipopt,
    nominal_position_feasible,
):

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

    # Position

    axes[0].plot(
        times,
        1000.0 * position_error_gn,
        color=COLOR_GN,
        linewidth=2.3,
        label="Gauss-Newton",
    )

    axes[0].plot(
        times,
        1000.0 * position_error_ipopt,
        color=COLOR_IPOPT,
        linewidth=2.3,
        label="IPOPT",
    )

    axes[0].axhline(
        0.0,
        color=COLOR_NOMINAL,
        linestyle="--",
        linewidth=1.8,
        label="Nominal IK",
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

    # Orientation

    axes[1].plot(
        times,
        np.degrees(
            orientation_error_gn
        ),
        color=COLOR_GN,
        linewidth=2.3,
        label="Gauss-Newton",
    )

    axes[1].plot(
        times,
        np.degrees(
            orientation_error_ipopt
        ),
        color=COLOR_IPOPT,
        linewidth=2.3,
        label="IPOPT",
    )

    axes[1].axhline(
        0.0,
        color=COLOR_NOMINAL,
        linestyle="--",
        linewidth=1.8,
        label="Nominal IK",
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
    cart_gn,
    cart_ipopt,
    nominal_position_feasible,
):

    (
        v_nominal,
        a_nominal,
    ) = numerical_derivatives(
        cart_nominal,
        times,
    )

    (
        v_gn,
        a_gn,
    ) = numerical_derivatives(
        cart_gn,
        times,
    )

    (
        v_ipopt,
        a_ipopt,
    ) = numerical_derivatives(
        cart_ipopt,
        times,
    )

    speed_nominal = np.linalg.norm(
        v_nominal,
        axis=1,
    )

    speed_gn = np.linalg.norm(
        v_gn,
        axis=1,
    )

    speed_ipopt = np.linalg.norm(
        v_ipopt,
        axis=1,
    )

    accel_nominal = np.linalg.norm(
        a_nominal,
        axis=1,
    )

    accel_gn = np.linalg.norm(
        a_gn,
        axis=1,
    )

    accel_ipopt = np.linalg.norm(
        a_ipopt,
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

    axes[0].plot(
        times,
        speed_gn,
        color=COLOR_GN,
        linewidth=2.2,
        label="Gauss-Newton",
    )

    axes[0].plot(
        times,
        speed_ipopt,
        color=COLOR_IPOPT,
        linewidth=2.2,
        label="IPOPT",
    )

    axes[0].plot(
        times,
        speed_nominal,
        "--",
        color=COLOR_NOMINAL,
        linewidth=1.8,
        label="Nominal IK",
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

    axes[1].plot(
        times,
        accel_gn,
        color=COLOR_GN,
        linewidth=2.2,
        label="Gauss-Newton",
    )

    axes[1].plot(
        times,
        accel_ipopt,
        color=COLOR_IPOPT,
        linewidth=2.2,
        label="IPOPT",
    )

    axes[1].plot(
        times,
        accel_nominal,
        "--",
        color=COLOR_NOMINAL,
        linewidth=1.8,
        label="Nominal IK",
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
    gn,
    ipopt,
    lower,
    upper,
    ylabel,
    title,
):

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
                gn[:, j],
                ipopt[:, j],
            ],
        )

        ax.plot(
            times,
            gn[:, j],
            color=COLOR_GN,
            linewidth=2.2,
            label="Gauss-Newton",
            zorder=5,
        )

        ax.plot(
            times,
            ipopt[:, j],
            color=COLOR_IPOPT,
            linewidth=2.2,
            label="IPOPT",
            zorder=6,
        )

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
# Constraint ratios
# =====================================================================


def plot_feasibility_ratios_comparison(
    times,
    q_nominal,
    q_gn,
    q_ipopt,
    v_nominal,
    v_gn,
    v_ipopt,
    a_nominal,
    a_gn,
    a_ipopt,
):

    curves = [
        (
            position_usage(
                q_nominal
            ),
            position_usage(
                q_gn
            ),
            position_usage(
                q_ipopt
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
                v_gn,
                V_MAX,
            ),
            symmetric_usage(
                v_ipopt,
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
                a_gn,
                A_MAX,
            ),
            symmetric_usage(
                a_ipopt,
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
            gn,
            ipopt,
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
                gn
            ),
            1.08 * np.max(
                ipopt
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
            gn,
            color=COLOR_GN,
            linewidth=2.2,
            label="Gauss-Newton",
            zorder=5,
        )

        ax.plot(
            times,
            ipopt,
            color=COLOR_IPOPT,
            linewidth=2.2,
            label="IPOPT",
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
# Reporting
# =====================================================================


def print_pose_tracking_comparison(
    position_error_gn,
    orientation_error_gn,
    position_error_ipopt,
    orientation_error_ipopt,
):

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
            "Gauss-Newton",
            position_error_gn,
            orientation_error_gn,
        ),
        (
            "IPOPT",
            position_error_ipopt,
            orientation_error_ipopt,
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
    q_gn,
    v_gn,
    a_gn,
    q_ipopt,
    v_ipopt,
    a_ipopt,
):

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
            "Gauss-Newton",
            q_gn,
            v_gn,
            a_gn,
        ),
        (
            "IPOPT",
            q_ipopt,
            v_ipopt,
            a_ipopt,
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
    a_gn,
    a_ipopt,
):

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
            "Gauss-Newton",
            a_gn,
        ),
        (
            "IPOPT",
            a_ipopt,
        ),
    ]

    for (
        name,
        acceleration,
    ) in rows:

        print(
            f"{name:<30}"
            f"{acceleration_cost(acceleration, dense_times):>25.3f}"
        )


def print_solver_comparison(
    gn_result,
    gn_time,
    ipopt_result,
):

    print()
    print("=" * 82)
    print(
        "SOLVER COMPARISON"
    )
    print("=" * 82)

    print(
        f"{'Method':<22}"
        f"{'success':>10}"
        f"{'iterations':>14}"
        f"{'objective':>18}"
        f"{'time [s]':>14}"
    )

    print("-" * 82)

    print(
        f"{'Gauss-Newton':<22}"
        f"{str(gn_result.success):>10}"
        f"{gn_result.iterations:>14}"
        f"{gn_result.history[-1].objective if gn_result.history else np.nan:>18.6e}"
        f"{gn_time:>14.4f}"
    )

    print(
        f"{'IPOPT':<22}"
        f"{str(ipopt_result.success):>10}"
        f"{ipopt_result.iterations:>14}"
        f"{ipopt_result.objective:>18.6e}"
        f"{ipopt_result.solve_time:>14.4f}"
    )

    print()

    print(
        f"Gauss-Newton status: "
        f"{gn_result.status}"
    )

    print(
        f"IPOPT status: "
        f"{ipopt_result.status_message}"
    )


def print_iteration_summary(
    result,
):

    print()
    print("=" * 110)
    print(
        "GAUSS-NEWTON ITERATIONS — FULL SE(3) POSE"
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
    # 3) Feasible B-spline initialization
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
    # 4) Common full-pose optimization problem
    # =================================================================

    problem = CartesianTrajectoryProblem(
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

        orientation_scale=ORIENTATION_SCALE,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_terminal=CARTESIAN_TERMINAL_WEIGHT,
    )

    initial_metrics = (
        problem.error_metrics(
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
    # 5) Static Gauss-Newton
    # =================================================================

    print()
    print("=" * 76)
    print(
        "Solving with STATIC GAUSS-NEWTON..."
    )
    print("=" * 76)

    gn_solver = GaussNewtonSolver(
        damping_mode="static",
        damping=GAUSS_NEWTON_DAMPING,
        max_iterations=(
            GAUSS_NEWTON_MAX_ITERATIONS
        ),
    )

    gn_start = time.perf_counter()

    gn_result = gn_solver.solve(
        problem=problem,
        initial_control_points=(
            initial_control_points
        ),
        verbose=True,
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
        f"Gauss-Newton iterations: "
        f"{gn_result.iterations}"
    )

    print(
        f"Gauss-Newton total time: "
        f"{gn_time:.4f} s"
    )

    # =================================================================
    # 6) IPOPT
    # =================================================================

    print()
    print("=" * 76)
    print(
        "Solving with IPOPT + L-BFGS..."
    )
    print("=" * 76)

    ipopt_solver = IpoptSolver(
        tolerance=IPOPT_TOLERANCE,
        max_iterations=IPOPT_MAX_ITERATIONS,
        print_level=0,
    )

    ipopt_result = ipopt_solver.solve(
        problem=problem,
        initial_control_points=(
            initial_control_points
        ),
        verbose=True,
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
    # 7) Dense evaluation
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

    # Gauss-Newton

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
    # 8) Dense feasibility
    # =================================================================

    gn_feasible = (
        full_feasible_mask(
            q_gn_dense,
            v_gn_dense,
            a_gn_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    ipopt_feasible = (
        full_feasible_mask(
            q_ipopt_dense,
            v_ipopt_dense,
            a_ipopt_dense,
            atol=FEASIBILITY_TOL,
        )
    )

    print()
    print(
        "Dense feasibility:"
    )

    print(
        f"  Gauss-Newton: "
        f"{bool(np.all(gn_feasible))}"
    )

    print(
        f"  IPOPT:        "
        f"{bool(np.all(ipopt_feasible))}"
    )

    # =================================================================
    # 9) Dense pose tracking errors
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
    # 10) Numerical comparison
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
    # 11) Figures
    # =================================================================

    plot_cartesian_3d_comparison(
        cart_ref_dense,
        cart_gn_dense,
        cart_ipopt_dense,
        nominal_position_feasible,
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

    plt.show(
        block=False
    )

    plt.pause(
        0.2
    )

    # =================================================================
    # 12) Meshcat
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

    display_meshcat_nominal_feasibility_path(
        robot,
        cart_ref_dense,
        nominal_position_feasible,
        name="nominal",
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
    # 13) Animation trajectories
    # =================================================================

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
        "  purple    = static Gauss-Newton"
    )

    print(
        "  blue      = IPOPT"
    )

    # Nominal

    input(
        "\nPress Enter to animate "
        "the NOMINAL IK trajectory..."
    )

    robot.animate(
        ik_result.q_pin,
        dt=animation_dt,
    )

    # Gauss-Newton

    input(
        "\nPress Enter to animate "
        "the GAUSS-NEWTON trajectory..."
    )

    robot.display(
        q_pin_gn_animation[0]
    )

    time.sleep(
        0.5
    )

    robot.animate(
        q_pin_gn_animation,
        dt=animation_dt,
    )

    # IPOPT

    input(
        "\nPress Enter to animate "
        "the IPOPT trajectory..."
    )

    robot.display(
        q_pin_ipopt_animation[0]
    )

    time.sleep(
        0.5
    )

    robot.animate(
        q_pin_ipopt_animation,
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

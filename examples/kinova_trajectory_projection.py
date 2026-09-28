from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import meshcat.geometry as mg
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from splineopt.bspline import BSpline
from splineopt.optimizer import JointTrajectoryOptimizer
from splineopt.robotics import Robot


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)

# B-spline
DEGREE = 3
N_CONTROL_POINTS = 30
LAMBDA_SMOOTH = 1e-5

# Sampling
N_REFERENCE_SAMPLES = 201
N_DENSE_SAMPLES = 700

# Animation speed multiplier (> 1 = faster than physical time)
ANIMATION_SPEED = 2.0

# Numerical tolerance consistent with OSQP solver accuracy.
SOLVER_FEASIBILITY_TOL = 1e-3

# ---------------------------------------------------------------------
# Kinova Gen3 limits used in this project
# ---------------------------------------------------------------------
# Position limits from the URDF. Joints 1, 3, 5 and 7 are continuous.

Q_MIN = np.array([
    -np.inf,
    -2.24,
    -np.inf,
    -2.57,
    -np.inf,
    -2.09,
    -np.inf,
])

Q_MAX = np.array([
    np.inf,
    2.24,
    np.inf,
    2.57,
    np.inf,
    2.09,
    np.inf,
])

# Limits from kortex_description/arms/gen3/7dof/config/joint_limits.yaml
V_MAX = np.full(7, 0.8727)
A_MAX = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

V_MIN = -V_MAX
A_MIN = -A_MAX

# Plot colors
COLOR_NOMINAL = "C1"
COLOR_OPTIMIZED = "C0"
COLOR_FEASIBLE = "#2ca02c"
COLOR_INFEASIBLE = "#d62728"
COLOR_LIMIT = "#555555"

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# =====================================================================
# Controlled demonstration trajectory
# =====================================================================

def build_virtual_joint_reference(robot):
    """
    Build a deliberately aggressive joint-space reference.

    The trajectory is used only to generate a reproducible Cartesian
    target through forward kinematics. The target is then reconstructed
    with sequential IK, so the actual nominal input to the B-spline QP
    remains an IK trajectory.

    Design goals:
      - initial and final holds -> zero boundary velocity,
      - sharp changes of direction -> large acceleration variations,
      - temporary violations of joint 2 and joint 4 position limits,
      - return inside the admissible joint region afterwards.
    """

    if robot.nv != 7:
        raise ValueError(
            f"This demonstration expects a 7-DoF robot, got nv={robot.nv}."
        )

    waypoint_times = np.array([
        0.0,
        0.8,
        2.5,
        4.0,
        5.4,
        7.2,
        9.0,
        10.0,
    ])

    #                    q1    q2    q3    q4     q5    q6    q7
    waypoints = np.array([
        [0.00, 0.45,  0.15, -1.35, -0.15, 0.75,  0.10],
        [0.00, 0.45,  0.15, -1.35, -0.15, 0.75,  0.10],  # hold
        [0.35, 1.30,  0.45, -1.05, -0.45, 0.85,  0.30],
        [0.70, 2.72,  0.75, -2.72, -0.65, 1.10,  0.55],  # q2/q4 infeasible
        [0.95, 2.82,  0.20, -2.70,  0.20, 1.40, -0.30],  # q2/q4 infeasible
        [0.55, 1.65, -0.40, -1.55,  0.55, 1.15, -0.50],
        [0.10, 0.65, -0.15, -1.25,  0.00, 0.80,  0.00],
        [0.10, 0.65, -0.15, -1.25,  0.00, 0.80,  0.00],  # hold
    ])

    times = np.linspace(
        waypoint_times[0],
        waypoint_times[-1],
        N_REFERENCE_SAMPLES,
    )

    theta_virtual = np.column_stack([
        np.interp(times, waypoint_times, waypoints[:, j])
        for j in range(robot.nv)
    ])

    return times, theta_virtual, waypoint_times, waypoints


def build_cartesian_targets_from_virtual_reference(robot, theta_virtual):
    """
    Generate Cartesian SE(3) targets from the virtual joint reference.
    """

    q_pin_virtual = robot.joint_trajectory_to_configurations(theta_virtual)

    target_poses = [
        robot.frame_pose(q)
        for q in q_pin_virtual
    ]

    target_positions = np.array([
        pose.translation.copy()
        for pose in target_poses
    ])

    return q_pin_virtual, target_poses, target_positions


# =====================================================================
# B-spline + optimization
# =====================================================================

def make_spline(t_start, t_end, n_dof):
    """Create a cubic open-clamped B-spline."""

    n_internal_knots = N_CONTROL_POINTS - DEGREE - 1

    internal_knots = np.linspace(
        t_start,
        t_end,
        n_internal_knots + 2,
    )[1:-1]

    knots = np.concatenate([
        np.full(DEGREE + 1, t_start),
        internal_knots,
        np.full(DEGREE + 1, t_end),
    ])

    return BSpline(
        knots=knots,
        degree=DEGREE,
        n_dof=n_dof,
    )


def solve_bspline_projection(robot, times, theta_ref):
    """Project the IK trajectory onto the admissible B-spline set."""

    spline = make_spline(
        t_start=times[0],
        t_end=times[-1],
        n_dof=robot.nv,
    )

    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=LAMBDA_SMOOTH,
    )

    optimizer_result, osqp_result = optimizer.solve(
        tracking_times=times,
        q_ref=theta_ref,
        q0=theta_ref[0],
        v0=np.zeros(robot.nv),
        a0=np.zeros(robot.nv),
        vf=np.zeros(robot.nv),
        q_min=Q_MIN,
        q_max=Q_MAX,
        v_min=V_MIN,
        v_max=V_MAX,
        a_min=A_MIN,
        a_max=A_MAX,
    )

    return spline, optimizer_result, osqp_result


def evaluate_spline(spline, control_points, times):
    """Evaluate q, qdot and qddot analytically."""

    q = np.array([
        spline.position(control_points, t)
        for t in times
    ])

    v = np.array([
        spline.velocity(control_points, t)
        for t in times
    ])

    a = np.array([
        spline.acceleration(control_points, t)
        for t in times
    ])

    return q, v, a


# =====================================================================
# Kinematic / numerical helpers
# =====================================================================

def interpolate_joint_trajectory(times, theta, query_times):
    return np.column_stack([
        np.interp(query_times, times, theta[:, j])
        for j in range(theta.shape[1])
    ])


def numerical_derivatives(values, times):
    velocity = np.gradient(
        values,
        times,
        axis=0,
        edge_order=2,
    )

    acceleration = np.gradient(
        velocity,
        times,
        axis=0,
        edge_order=2,
    )

    return velocity, acceleration


def forward_kinematics_trajectory(robot, theta):
    q_pin = robot.joint_trajectory_to_configurations(theta)

    positions = np.array([
        robot.frame_position(q)
        for q in q_pin
    ])

    return q_pin, positions


# =====================================================================
# Feasibility helpers
# =====================================================================

def position_feasible_mask(theta, atol=1e-9):
    lower_ok = np.logical_or(
        ~np.isfinite(Q_MIN)[None, :],
        theta >= Q_MIN[None, :] - atol,
    )

    upper_ok = np.logical_or(
        ~np.isfinite(Q_MAX)[None, :],
        theta <= Q_MAX[None, :] + atol,
    )

    return np.all(lower_ok & upper_ok, axis=1)


def full_feasible_mask(theta, velocity, acceleration, atol=1e-9):
    q_ok = position_feasible_mask(theta, atol=atol)

    v_ok = np.all(
        (velocity >= V_MIN[None, :] - atol)
        & (velocity <= V_MAX[None, :] + atol),
        axis=1,
    )

    a_ok = np.all(
        (acceleration >= A_MIN[None, :] - atol)
        & (acceleration <= A_MAX[None, :] + atol),
        axis=1,
    )

    return q_ok & v_ok & a_ok


def position_usage(theta):
    """
    Normalized position-limit usage for bounded joints only.

    rho_q <= 1 means all bounded joint positions are admissible.
    """

    bounded = np.isfinite(Q_MIN) & np.isfinite(Q_MAX)

    center = 0.5 * (Q_MIN[bounded] + Q_MAX[bounded])
    half_range = 0.5 * (Q_MAX[bounded] - Q_MIN[bounded])

    normalized = np.abs(
        (theta[:, bounded] - center[None, :])
        / half_range[None, :]
    )

    return np.max(normalized, axis=1)


def symmetric_usage(values, positive_limits):
    return np.max(
        np.abs(values) / positive_limits[None, :],
        axis=1,
    )


def contiguous_intervals(times, mask):
    """Return contiguous [t_start, t_end, value] intervals."""

    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0:
        return []

    intervals = []
    start = 0

    for i in range(1, len(mask)):
        if mask[i] != mask[start]:
            intervals.append((times[start], times[i], bool(mask[start])))
            start = i

    intervals.append((times[start], times[-1], bool(mask[start])))
    return intervals


# =====================================================================
# Plot styling helpers
# =====================================================================

def shade_time_feasibility(ax, times, feasible_mask, alpha=0.045):
    """Light green/red background showing nominal feasibility in time."""

    for t0, t1, feasible in contiguous_intervals(times, feasible_mask):
        ax.axvspan(
            t0,
            t1,
            color=COLOR_FEASIBLE if feasible else COLOR_INFEASIBLE,
            alpha=alpha,
            zorder=0,
        )


def shade_horizontal_admissible_region(ax, lower, upper, nominal, optimized):
    """Shade admissible region green and forbidden region red."""

    all_values = np.concatenate([
        np.asarray(nominal).reshape(-1),
        np.asarray(optimized).reshape(-1),
    ])

    finite_candidates = all_values[np.isfinite(all_values)]

    if np.isfinite(lower):
        finite_candidates = np.append(finite_candidates, lower)
    if np.isfinite(upper):
        finite_candidates = np.append(finite_candidates, upper)

    if finite_candidates.size == 0:
        return

    value_min = np.min(finite_candidates)
    value_max = np.max(finite_candidates)

    span = max(value_max - value_min, 0.1)
    y_min = value_min - 0.12 * span
    y_max = value_max + 0.12 * span

    ax.set_ylim(y_min, y_max)

    if np.isfinite(lower) and np.isfinite(upper):
        ax.axhspan(lower, upper, color=COLOR_FEASIBLE, alpha=0.08, zorder=0)
        if y_min < lower:
            ax.axhspan(y_min, lower, color=COLOR_INFEASIBLE,
                       alpha=0.08, zorder=0)
        if upper < y_max:
            ax.axhspan(upper, y_max, color=COLOR_INFEASIBLE,
                       alpha=0.08, zorder=0)

        ax.axhline(lower, color=COLOR_LIMIT, linestyle=":", linewidth=1.1)
        ax.axhline(upper, color=COLOR_LIMIT, linestyle=":", linewidth=1.1)

    elif np.isfinite(lower):
        ax.axhspan(lower, y_max, color=COLOR_FEASIBLE, alpha=0.08, zorder=0)
        ax.axhspan(y_min, lower, color=COLOR_INFEASIBLE, alpha=0.08, zorder=0)
        ax.axhline(lower, color=COLOR_LIMIT, linestyle=":", linewidth=1.1)

    elif np.isfinite(upper):
        ax.axhspan(y_min, upper, color=COLOR_FEASIBLE, alpha=0.08, zorder=0)
        ax.axhspan(upper, y_max, color=COLOR_INFEASIBLE, alpha=0.08, zorder=0)
        ax.axhline(upper, color=COLOR_LIMIT, linestyle=":", linewidth=1.1)

    else:
        ax.axhspan(y_min, y_max, color=COLOR_FEASIBLE, alpha=0.05, zorder=0)


# =====================================================================
# Plots
# =====================================================================

def plot_cartesian_3d(cart_nominal, cart_optimized, nominal_position_feasible):
    """
    3D end-effector path.

    Nominal path is green while its IK configuration is within joint
    position limits and red while the nominal IK state violates at least
    one bounded joint position limit.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    segments = np.stack([
        cart_nominal[:-1],
        cart_nominal[1:],
    ], axis=1)

    segment_feasible = (
        nominal_position_feasible[:-1]
        & nominal_position_feasible[1:]
    )

    segment_colors = [
        COLOR_FEASIBLE if feasible else COLOR_INFEASIBLE
        for feasible in segment_feasible
    ]

    collection = Line3DCollection(
        segments,
        colors=segment_colors,
        linewidths=2.8,
    )
    ax.add_collection3d(collection)

    ax.plot(
        cart_optimized[:, 0],
        cart_optimized[:, 1],
        cart_optimized[:, 2],
        color=COLOR_OPTIMIZED,
        linewidth=3.0,
        label="Optimized feasible B-spline",
    )

    # Dummy handles for clear legend entries.
    ax.plot([], [], [], color=COLOR_FEASIBLE, linewidth=2.8,
            label="Nominal: joint-position feasible")
    ax.plot([], [], [], color=COLOR_INFEASIBLE, linewidth=2.8,
            label="Nominal: joint-position infeasible")

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

    all_points = np.vstack([cart_nominal, cart_optimized])
    xyz_min = np.min(all_points, axis=0)
    xyz_max = np.max(all_points, axis=0)

    # Needed because Line3DCollection does not always autoscale axes.
    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])

    ranges = np.maximum(xyz_max - xyz_min, 1e-3)
    ax.set_box_aspect(ranges)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Cartesian end-effector trajectory")
    ax.legend(loc="best")

    fig.tight_layout()


def plot_cartesian_dynamics(
    times,
    cart_nominal,
    cart_optimized,
    nominal_position_feasible,
):
    """Cartesian speed and acceleration magnitude."""

    v_nominal, a_nominal = numerical_derivatives(cart_nominal, times)
    v_optimized, a_optimized = numerical_derivatives(cart_optimized, times)

    speed_nominal = np.linalg.norm(v_nominal, axis=1)
    speed_optimized = np.linalg.norm(v_optimized, axis=1)

    accel_nominal = np.linalg.norm(a_nominal, axis=1)
    accel_optimized = np.linalg.norm(a_optimized, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax in axes:
        shade_time_feasibility(ax, times, nominal_position_feasible)

    axes[0].plot(times, speed_nominal, "--", color=COLOR_NOMINAL,
                 label="Nominal")
    axes[0].plot(times, speed_optimized, color=COLOR_OPTIMIZED,
                 label="Optimized")
    axes[0].set_ylabel(r"$\|\dot p\|$ [m/s]")
    axes[0].set_title("Cartesian speed")
    axes[0].legend()

    axes[1].plot(times, accel_nominal, "--", color=COLOR_NOMINAL,
                 label="Nominal")
    axes[1].plot(times, accel_optimized, color=COLOR_OPTIMIZED,
                 label="Optimized")
    axes[1].set_ylabel(r"$\|\ddot p\|$ [m/s²]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Cartesian acceleration magnitude")
    axes[1].legend()

    fig.tight_layout()


def plot_joint_quantity(
    times,
    nominal,
    optimized,
    lower,
    upper,
    ylabel,
    title,
):
    """Seven joint plots with explicit admissible / forbidden bands."""

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
            nominal[:, j],
            optimized[:, j],
        )

        ax.plot(
            times,
            nominal[:, j],
            "--",
            color=COLOR_NOMINAL,
            linewidth=1.7,
            label="Nominal IK",
            zorder=3,
        )

        ax.plot(
            times,
            optimized[:, j],
            color=COLOR_OPTIMIZED,
            linewidth=2.4,
            label="Optimized B-spline",
            zorder=4,
        )

        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel(ylabel)

        if j >= 6:
            ax.set_xlabel("Time [s]")

        if not np.isfinite(lower[j]) and not np.isfinite(upper[j]):
            ax.text(
                0.02,
                0.92,
                "continuous joint",
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.7,
                va="top",
            )

    axes[0].legend(loc="best")
    axes[-1].axis("off")

    fig.suptitle(
        title + "\nGreen = admissible, red = forbidden",
        y=0.995,
    )
    fig.tight_layout()


def plot_feasibility_ratios(
    times,
    q_nominal,
    q_optimized,
    v_nominal,
    v_optimized,
    a_nominal,
    a_optimized,
):
    """Compact normalized q / qdot / qddot feasibility diagnostic."""

    curves = [
        (
            position_usage(q_nominal),
            position_usage(q_optimized),
            r"$\rho_q$",
            "Joint-position limit usage",
        ),
        (
            symmetric_usage(v_nominal, V_MAX),
            symmetric_usage(v_optimized, V_MAX),
            r"$\rho_v$",
            "Joint-velocity limit usage",
        ),
        (
            symmetric_usage(a_nominal, A_MAX),
            symmetric_usage(a_optimized, A_MAX),
            r"$\rho_a$",
            "Joint-acceleration limit usage",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for ax, (nominal, optimized, ylabel, title) in zip(axes, curves):
        upper = max(1.15, 1.08 * np.max(nominal), 1.08 * np.max(optimized))

        ax.axhspan(0.0, 1.0, color=COLOR_FEASIBLE, alpha=0.08)
        ax.axhspan(1.0, upper, color=COLOR_INFEASIBLE, alpha=0.08)
        ax.axhline(1.0, color=COLOR_LIMIT, linestyle=":", linewidth=1.3,
                   label="Constraint boundary")

        ax.plot(times, nominal, "--", color=COLOR_NOMINAL,
                label="Nominal IK")
        ax.plot(times, optimized, color=COLOR_OPTIMIZED,
                label="Optimized B-spline")

        ax.set_ylim(0.0, upper)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time [s]")

    fig.suptitle("Normalized feasibility (<= 1 is admissible)")
    fig.tight_layout()


# =====================================================================
# Metrics
# =====================================================================

def print_metrics(
    times,
    theta_virtual,
    theta_ref,
    q_ref_dense,
    v_ref_dense,
    a_ref_dense,
    q_opt_dense,
    v_opt_dense,
    a_opt_dense,
    cart_ref_dense,
    cart_opt_dense,
    osqp_result,
):
    ik_branch_rms = np.sqrt(np.mean((theta_ref - theta_virtual) ** 2))

    cart_error = np.linalg.norm(
        cart_opt_dense - cart_ref_dense,
        axis=1,
    )

    nominal_smoothness = np.trapezoid(
        np.sum(a_ref_dense ** 2, axis=1),
        times,
    )

    optimized_smoothness = np.trapezoid(
        np.sum(a_opt_dense ** 2, axis=1),
        times,
    )

    print()
    print("=" * 76)
    print("KINOVA GEN3 — IK + CONSTRAINED B-SPLINE PROJECTION")
    print("=" * 76)
    print(f"OSQP status:                 {osqp_result.info.status}")
    print(
        f"OSQP solve time:             {osqp_result.info.solve_time * 1e3:.3f} ms")
    print()
    print(f"IK branch RMS vs generator:  {ik_branch_rms:.4f} rad")
    print()
    print(
        f"Nominal max rho_q:           {np.max(position_usage(q_ref_dense)):.3f}")
    print(
        f"Optimized max rho_q:         {np.max(position_usage(q_opt_dense)):.3f}")
    print(
        f"Nominal max rho_v:           {np.max(symmetric_usage(v_ref_dense, V_MAX)):.3f}")
    print(
        f"Optimized max rho_v:         {np.max(symmetric_usage(v_opt_dense, V_MAX)):.3f}")
    print(
        f"Nominal max rho_a:           {np.max(symmetric_usage(a_ref_dense, A_MAX)):.3f}")
    print(
        f"Optimized max rho_a:         {np.max(symmetric_usage(a_opt_dense, A_MAX)):.3f}")
    print()

    print(
        f"Cartesian RMS deviation:     {np.sqrt(np.mean(cart_error ** 2)) * 1000:.2f} mm")
    print(f"Cartesian max deviation:     {np.max(cart_error) * 1000:.2f} mm")
    print()
    print(f"Nominal integral ||qddot||²: {nominal_smoothness:.3f}")
    print(f"Optimized integral ||qddot||²: {optimized_smoothness:.3f}")


# =====================================================================
# Meshcat helpers
# =====================================================================

def _set_meshcat_line(robot, points, name, color, linewidth=3.0):
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return

    geometry = mg.PointsGeometry(points.T)
    material = mg.LineBasicMaterial(
        color=color,
        linewidth=linewidth,
    )

    robot.viewer.viewer["trajectories"][name].set_object(
        mg.Line(geometry, material)
    )


def display_meshcat_nominal_feasibility_path(
    robot,
    points,
    feasible_mask,
    name="nominal",
):
    """Display nominal path in green/red contiguous pieces."""

    segment_feasible = feasible_mask[:-1] & feasible_mask[1:]

    if len(segment_feasible) == 0:
        return

    run_start = 0
    run_value = bool(segment_feasible[0])
    run_index = 0

    for i in range(1, len(segment_feasible) + 1):
        run_ends = (
            i == len(segment_feasible)
            or bool(segment_feasible[i]) != run_value
        )

        if run_ends:
            run_points = points[run_start:i + 1]

            _set_meshcat_line(
                robot,
                run_points,
                name=f"{name}_{run_index}",
                color=0x2CA02C if run_value else 0xD62728,
                linewidth=4.0,
            )

            run_index += 1

            if i < len(segment_feasible):
                run_start = i
                run_value = bool(segment_feasible[i])


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

    print(robot)
    print(f"nq = {robot.nq}, nv = {robot.nv}")

    # -----------------------------------------------------------------
    # 1) Controlled virtual trajectory -> Cartesian target
    # -----------------------------------------------------------------

    (
        times,
        theta_virtual,
        waypoint_times,
        joint_waypoints,
    ) = build_virtual_joint_reference(robot)

    (
        q_pin_virtual,
        target_poses,
        target_positions,
    ) = build_cartesian_targets_from_virtual_reference(
        robot,
        theta_virtual,
    )

    # -----------------------------------------------------------------
    # 2) Sequential full-pose IK
    # -----------------------------------------------------------------

    print()
    print("Solving sequential IK...")

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
        raise RuntimeError("Sequential IK did not converge.")

    theta_ref = ik_result.theta

    print(f"Maximum IK error: {np.max(ik_result.error_norms):.3e}")
    print(f"Nominal IK max q2: {np.max(theta_ref[:, 1]):.3f} rad "
          f"(limit {Q_MAX[1]:.3f})")
    print(f"Nominal IK min q4: {np.min(theta_ref[:, 3]):.3f} rad "
          f"(limit {Q_MIN[3]:.3f})")

    # -----------------------------------------------------------------
    # 3) Constrained B-spline projection
    # -----------------------------------------------------------------

    print()
    print("Solving constrained B-spline projection...")

    spline, control_points, osqp_result = solve_bspline_projection(
        robot,
        times,
        theta_ref,
    )

    # -----------------------------------------------------------------
    # 4) Dense evaluation
    # -----------------------------------------------------------------

    dense_times = np.linspace(times[0], times[-1], N_DENSE_SAMPLES)

    q_ref_dense = interpolate_joint_trajectory(
        times,
        theta_ref,
        dense_times,
    )

    v_ref_dense, a_ref_dense = numerical_derivatives(
        q_ref_dense,
        dense_times,
    )

    q_opt_dense, v_opt_dense, a_opt_dense = evaluate_spline(
        spline,
        control_points,
        dense_times,
    )

    q_pin_ref_dense, cart_ref_dense = forward_kinematics_trajectory(
        robot,
        q_ref_dense,
    )

    q_pin_opt_dense, cart_opt_dense = forward_kinematics_trajectory(
        robot,
        q_opt_dense,
    )

    nominal_position_feasible = position_feasible_mask(q_ref_dense)
    optimized_full_feasible = full_feasible_mask(
        q_opt_dense,
        v_opt_dense,
        a_opt_dense,
        atol=SOLVER_FEASIBILITY_TOL,
    )

    print(
        f"Optimized dense trajectory fully feasible: "
        f"{bool(np.all(optimized_full_feasible))}"
    )

    # -----------------------------------------------------------------
    # 5) Metrics
    # -----------------------------------------------------------------

    print_metrics(
        times=dense_times,
        theta_virtual=theta_virtual,
        theta_ref=theta_ref,
        q_ref_dense=q_ref_dense,
        v_ref_dense=v_ref_dense,
        a_ref_dense=a_ref_dense,
        q_opt_dense=q_opt_dense,
        v_opt_dense=v_opt_dense,
        a_opt_dense=a_opt_dense,
        cart_ref_dense=cart_ref_dense,
        cart_opt_dense=cart_opt_dense,
        osqp_result=osqp_result,
    )

    # -----------------------------------------------------------------
    # 6) Figures
    # -----------------------------------------------------------------

    plot_cartesian_3d(
        cart_ref_dense,
        cart_opt_dense,
        nominal_position_feasible,
    )

    plot_cartesian_dynamics(
        dense_times,
        cart_ref_dense,
        cart_opt_dense,
        nominal_position_feasible,
    )

    plot_joint_quantity(
        dense_times,
        q_ref_dense,
        q_opt_dense,
        Q_MIN,
        Q_MAX,
        ylabel="Position [rad]",
        title="Joint positions",
    )

    plot_joint_quantity(
        dense_times,
        v_ref_dense,
        v_opt_dense,
        V_MIN,
        V_MAX,
        ylabel="Velocity [rad/s]",
        title="Joint velocities",
    )

    plot_joint_quantity(
        dense_times,
        a_ref_dense,
        a_opt_dense,
        A_MIN,
        A_MAX,
        ylabel="Acceleration [rad/s²]",
        title="Joint accelerations",
    )

    plot_feasibility_ratios(
        dense_times,
        q_ref_dense,
        q_opt_dense,
        v_ref_dense,
        v_opt_dense,
        a_ref_dense,
        a_opt_dense,
    )

    plt.show(block=False)
    plt.pause(0.2)

    # -----------------------------------------------------------------
    # 7) Meshcat
    # -----------------------------------------------------------------

    print("Opening Meshcat...")

    robot.init_viewer(open_browser=True)

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
        cart_opt_dense,
        name="optimized",
        color=0x0077FF,
        linewidth=4.0,
    )

    # Use the reference sampling for animation.
    q_opt_animation, _, _ = evaluate_spline(
        spline,
        control_points,
        times,
    )

    q_pin_opt_animation = robot.joint_trajectory_to_configurations(
        q_opt_animation
    )

    animation_dt = (
        (times[-1] - times[0])
        / (len(times) - 1)
        / ANIMATION_SPEED
    )

    robot.display(ik_result.q_pin[0])

    print()
    print("Meshcat colors:")
    print("  green = nominal segment generated by joint-feasible IK state")
    print("  red   = nominal segment generated by joint-limit-violating IK state")
    print("  blue  = optimized constrained B-spline path")

    input("\nPress Enter to animate the NOMINAL IK trajectory...")
    time.sleep(0.5)

    robot.animate(
        ik_result.q_pin,
        dt=animation_dt,
    )

    input("\nPress Enter to animate the OPTIMIZED B-spline trajectory...")

    robot.display(q_pin_opt_animation[0])
    time.sleep(0.5)

    robot.animate(
        q_pin_opt_animation,
        dt=animation_dt,
    )

    print()
    print("Done.")
    input("Press Enter to quit...")


if __name__ == "__main__":
    main()

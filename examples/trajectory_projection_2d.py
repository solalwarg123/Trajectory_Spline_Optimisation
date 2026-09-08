import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from splineopt.bspline import BSpline
from splineopt.optimizer import JointTrajectoryOptimizer


# ============================================================
# B-spline definition
# ============================================================

def make_spline():
    """
    Cubic open-clamped B-spline on [0, 8].

    Number of control points:
        Nc = 14
    """

    degree = 3
    n_control_points = 14

    t_start = 0.0
    t_end = 8.0

    n_internal_knots = (
        n_control_points
        - degree
        - 1
    )

    internal_knots = np.linspace(
        t_start,
        t_end,
        n_internal_knots + 2,
    )[1:-1]

    knots = np.concatenate([
        np.full(degree + 1, t_start),
        internal_knots,
        np.full(degree + 1, t_end),
    ])

    return BSpline(
        knots=knots,
        degree=degree,
        n_dof=2,
    )


# ============================================================
# Nominal control polygons
# ============================================================

def feasible_nominal_control_points():
    """
    Feasible nominal trajectory.

    Geometry:
    - horizontal part,
    - rounded corner,
    - vertical part,
    - diagonal return / finishing part.

    First two and last two points are repeated so that the
    clamped cubic spline starts and ends with zero velocity.
    """

    return np.array([
        [-0.45, -0.30],
        [-0.45, -0.30],

        [-0.35, -0.30],
        [-0.25, -0.30],
        [-0.15, -0.30],

        [-0.08, -0.22],
        [-0.05, -0.12],

        [-0.05,  0.05],
        [-0.05,  0.20],

        [0.15,  0.28],
        [0.32,  0.35],
        [0.42,  0.30],

        [0.30,  0.10],
        [0.30,  0.10],
    ])


def infeasible_nominal_control_points():
    """
    Nominal trajectory that temporarily leaves the admissible
    region, then comes back inside.

    This makes the projection effect visually clearer:
    - the optimizer can follow the beginning,
    - it must deviate while outside the admissible set,
    - then it can reconnect when the nominal returns inside.
    """

    return np.array([
        [-0.45, -0.30],
        [-0.45, -0.30],

        # Same beginning
        [-0.35, -0.30],
        [-0.25, -0.30],
        [-0.15, -0.30],

        # Rounded corner
        [-0.08, -0.22],
        [-0.05, -0.12],

        # Vertical section
        [-0.05,  0.05],
        [-0.05,  0.20],

        # Move toward the admissible boundary
        [0.30,  0.32],

        # Go outside in q1, then come back
        [0.78,  0.35],
        [0.78,  0.20],

        # Return inside
        [0.30,  0.10],
        [0.30,  0.10],
    ])


# ============================================================
# Spline evaluation
# ============================================================

def evaluate_spline(
    spline,
    control_points,
    times,
):
    """
    Evaluate position, velocity, and acceleration.
    """

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


# ============================================================
# Optimization
# ============================================================

def solve_projection(
    spline,
    nominal_control_points,
    tracking_times,
    limits,
    lambda_smooth=1e-6,
):
    """
    Build the nominal reference from a given control polygon
    and project it onto the admissible B-spline set.
    """

    q_ref, _, _ = evaluate_spline(
        spline,
        nominal_control_points,
        tracking_times,
    )

    q0 = spline.position(
        nominal_control_points,
        spline.t_start,
    )

    v0 = spline.velocity(
        nominal_control_points,
        spline.t_start,
    )

    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=lambda_smooth,
    )

    control_points_opt, result = optimizer.solve(
        tracking_times=tracking_times,
        q_ref=q_ref,
        q0=q0,
        v0=v0,
        q_min=limits["q_min"],
        q_max=limits["q_max"],
        v_min=limits["v_min"],
        v_max=limits["v_max"],
        a_min=limits["a_min"],
        a_max=limits["a_max"],
    )

    return control_points_opt, result


# ============================================================
# Metrics
# ============================================================

def print_metrics(
    title,
    result,
    q_ref,
    v_ref,
    a_ref,
    q_opt,
    v_opt,
    a_opt,
    limits,
):
    error = q_opt - q_ref

    rms_error = np.sqrt(
        np.mean(error**2, axis=0)
    )

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"OSQP status: {result.info.status}"
    )

    print(
        f"Solve time: "
        f"{result.info.solve_time * 1e3:.3f} ms"
    )

    print()

    for j in range(2):
        print(f"Joint {j + 1}")

        print(
            f"  RMS tracking error: "
            f"{rms_error[j]:.6f} rad"
        )

        print(
            f"  nominal q range: "
            f"[{np.min(q_ref[:, j]):.3f}, "
            f"{np.max(q_ref[:, j]):.3f}]"
        )

        print(
            f"  optimized q range: "
            f"[{np.min(q_opt[:, j]):.3f}, "
            f"{np.max(q_opt[:, j]):.3f}]"
        )

        print(
            f"  q limits: "
            f"[{limits['q_min'][j]:.3f}, "
            f"{limits['q_max'][j]:.3f}]"
        )

        print(
            f"  nominal max |v|: "
            f"{np.max(np.abs(v_ref[:, j])):.3f}"
        )

        print(
            f"  optimized max |v|: "
            f"{np.max(np.abs(v_opt[:, j])):.3f}"
        )

        print(
            f"  nominal max |a|: "
            f"{np.max(np.abs(a_ref[:, j])):.3f}"
        )

        print(
            f"  optimized max |a|: "
            f"{np.max(np.abs(a_opt[:, j])):.3f}"
        )

        print()


# ============================================================
# Plot styling helpers
# ============================================================

def add_limit_band(
    ax,
    lower,
    upper,
    color,
    alpha=0.08,
):
    """
    Add a faint admissible band between lower and upper.
    """
    ax.axhspan(
        lower,
        upper,
        color=color,
        alpha=alpha,
        zorder=0,
    )


def plot_joint_pair(
    ax,
    times,
    nominal,
    optimized,
    lower_limits,
    upper_limits,
    ylabel,
    nominal_suffix,
    optimized_suffix,
    colors,
    marker_step=70,
):
    """
    Plot q1 and q2 on a single axis, with a clear distinction
    between nominal and optimized trajectories.

    The nominal curve uses dashed lines + markers so it remains
    visible even in the feasible case when it overlaps with the
    optimized curve.
    """

    for j in range(2):

        # Colored admissible band
        add_limit_band(
            ax,
            lower_limits[j],
            upper_limits[j],
            color=colors[j],
            alpha=0.06,
        )

        # Limit lines
        ax.axhline(
            lower_limits[j],
            color=colors[j],
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
            zorder=1,
        )

        ax.axhline(
            upper_limits[j],
            color=colors[j],
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
            zorder=1,
        )

        # Optimized curve: thicker
        ax.plot(
            times,
            optimized[:, j],
            color=colors[j],
            linewidth=3.2,
            alpha=0.70,
            label=f"q{j + 1} {optimized_suffix}",
            zorder=2,
        )

        # Nominal curve: dashed + markers, on top
        ax.plot(
            times,
            nominal[:, j],
            color=colors[j],
            linestyle="--",
            linewidth=1.8,
            marker="o" if j == 0 else "s",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markevery=marker_step,
            label=f"q{j + 1} {nominal_suffix}",
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.4)


# ============================================================
# Time-domain comparison
# ============================================================

def plot_time_comparison(
    times,
    feasible_data,
    infeasible_data,
    limits,
):
    """
    One figure with:
                feasible      infeasible
    position        |             |
    velocity        |             |
    acceleration    |             |
    """

    (
        q_ref_f,
        v_ref_f,
        a_ref_f,
        q_opt_f,
        v_opt_f,
        a_opt_f,
    ) = feasible_data

    (
        q_ref_i,
        v_ref_i,
        a_ref_i,
        q_opt_i,
        v_opt_i,
        a_opt_i,
    ) = infeasible_data

    colors = [
        "tab:blue",
        "tab:orange",
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15, 10.5),
        sharex=True,
    )

    axes[0, 0].set_title(
        "Feasible reference"
    )
    axes[0, 1].set_title(
        "Infeasible reference"
    )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    plot_joint_pair(
        ax=axes[0, 0],
        times=times,
        nominal=q_ref_f,
        optimized=q_opt_f,
        lower_limits=limits["q_min"],
        upper_limits=limits["q_max"],
        ylabel="Position [rad]",
        nominal_suffix="nominal",
        optimized_suffix="optimized",
        colors=colors,
    )

    plot_joint_pair(
        ax=axes[0, 1],
        times=times,
        nominal=q_ref_i,
        optimized=q_opt_i,
        lower_limits=limits["q_min"],
        upper_limits=limits["q_max"],
        ylabel="Position [rad]",
        nominal_suffix="nominal",
        optimized_suffix="optimized",
        colors=colors,
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    plot_joint_pair(
        ax=axes[1, 0],
        times=times,
        nominal=v_ref_f,
        optimized=v_opt_f,
        lower_limits=limits["v_min"],
        upper_limits=limits["v_max"],
        ylabel="Velocity [rad/s]",
        nominal_suffix="nominal",
        optimized_suffix="optimized",
        colors=colors,
    )

    plot_joint_pair(
        ax=axes[1, 1],
        times=times,
        nominal=v_ref_i,
        optimized=v_opt_i,
        lower_limits=limits["v_min"],
        upper_limits=limits["v_max"],
        ylabel="Velocity [rad/s]",
        nominal_suffix="nominal",
        optimized_suffix="optimized",
        colors=colors,
    )

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    plot_joint_pair(
        ax=axes[2, 0],
        times=times,
        nominal=a_ref_f,
        optimized=a_opt_f,
        lower_limits=limits["a_min"],
        upper_limits=limits["a_max"],
        ylabel="Acceleration [rad/s²]",
        nominal_suffix="nominal",
        optimized_suffix="optimized",
        colors=colors,
    )

    plot_joint_pair(
        ax=axes[2, 1],
        times=times,
        nominal=a_ref_i,
        optimized=a_opt_i,
        lower_limits=limits["a_min"],
        upper_limits=limits["a_max"],
        ylabel="Acceleration [rad/s²]",
        nominal_suffix="nominal",
        optimized_suffix="optimized",
        colors=colors,
    )

    # X labels
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")

    # Legends only on top row to reduce clutter
    axes[0, 0].legend(
        loc="upper left",
        ncol=2,
    )
    axes[0, 1].legend(
        loc="upper left",
        ncol=2,
    )

    fig.suptitle(
        "2-DoF trajectory projection — time-domain comparison",
        fontsize=15,
    )

    plt.tight_layout()


# ============================================================
# Joint-space plot
# ============================================================

def plot_joint_space(
    q_ref_feasible,
    q_opt_feasible,
    q_ref_infeasible,
    q_opt_infeasible,
    limits,
):
    """
    Single joint-space figure containing all four trajectories,
    with a colored admissible set and colored forbidden regions.
    """

    q1_min = limits["q_min"][0]
    q1_max = limits["q_max"][0]

    q2_min = limits["q_min"][1]
    q2_max = limits["q_max"][1]

    # Plot bounds around admissible box
    margin_q1 = 0.18
    margin_q2 = 0.14

    x_plot_min = q1_min - margin_q1
    x_plot_max = max(
        q1_max + margin_q1,
        np.max(q_ref_infeasible[:, 0]) + 0.08,
    )

    y_plot_min = q2_min - margin_q2
    y_plot_max = q2_max + margin_q2

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # Forbidden regions in light red
    # --------------------------------------------------------

    # Left forbidden strip
    ax.add_patch(Rectangle(
        (x_plot_min, y_plot_min),
        q1_min - x_plot_min,
        y_plot_max - y_plot_min,
        facecolor="lightcoral",
        alpha=0.12,
        edgecolor="none",
        zorder=0,
    ))

    # Right forbidden strip
    ax.add_patch(Rectangle(
        (q1_max, y_plot_min),
        x_plot_max - q1_max,
        y_plot_max - y_plot_min,
        facecolor="lightcoral",
        alpha=0.12,
        edgecolor="none",
        zorder=0,
    ))

    # Bottom forbidden strip inside x admissible span
    ax.add_patch(Rectangle(
        (q1_min, y_plot_min),
        q1_max - q1_min,
        q2_min - y_plot_min,
        facecolor="lightcoral",
        alpha=0.12,
        edgecolor="none",
        zorder=0,
    ))

    # Top forbidden strip inside x admissible span
    ax.add_patch(Rectangle(
        (q1_min, q2_max),
        q1_max - q1_min,
        y_plot_max - q2_max,
        facecolor="lightcoral",
        alpha=0.12,
        edgecolor="none",
        zorder=0,
    ))

    # --------------------------------------------------------
    # Admissible box in light green
    # --------------------------------------------------------

    ax.add_patch(Rectangle(
        (q1_min, q2_min),
        q1_max - q1_min,
        q2_max - q2_min,
        facecolor="lightgreen",
        alpha=0.18,
        edgecolor="forestgreen",
        linewidth=2.0,
        linestyle="--",
        zorder=1,
        label="Admissible joint set",
    ))

    # --------------------------------------------------------
    # Feasible pair
    # --------------------------------------------------------

    ax.plot(
        q_opt_feasible[:, 0],
        q_opt_feasible[:, 1],
        color="tab:blue",
        linewidth=3.0,
        alpha=0.70,
        label="Feasible optimized",
        zorder=2,
    )

    ax.plot(
        q_ref_feasible[:, 0],
        q_ref_feasible[:, 1],
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        markevery=90,
        label="Feasible nominal",
        zorder=3,
    )

    # --------------------------------------------------------
    # Infeasible pair
    # --------------------------------------------------------

    ax.plot(
        q_opt_infeasible[:, 0],
        q_opt_infeasible[:, 1],
        color="tab:purple",
        linewidth=3.0,
        alpha=0.75,
        label="Infeasible optimized",
        zorder=2,
    )

    ax.plot(
        q_ref_infeasible[:, 0],
        q_ref_infeasible[:, 1],
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        markevery=90,
        label="Infeasible nominal",
        zorder=4,
    )

    # Start and end markers
    ax.scatter(
        q_ref_feasible[0, 0],
        q_ref_feasible[0, 1],
        marker="o",
        s=70,
        label="Start",
        zorder=5,
    )

    ax.scatter(
        q_ref_feasible[-1, 0],
        q_ref_feasible[-1, 1],
        marker="X",
        s=90,
        label="End",
        zorder=5,
    )

    ax.set_xlim(
        x_plot_min,
        x_plot_max,
    )

    ax.set_ylim(
        y_plot_min,
        y_plot_max,
    )

    ax.set_xlabel("q1 [rad]")
    ax.set_ylabel("q2 [rad]")
    ax.set_title(
        "Joint-space comparison"
    )
    ax.grid(alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    plt.tight_layout()


# ============================================================
# Main
# ============================================================

def main():

    spline = make_spline()

    # --------------------------------------------------------
    # Limits
    # --------------------------------------------------------

    limits = {
        "q_min": np.array([
            -0.55,
            -0.45,
        ]),
        "q_max": np.array([
            0.55,
            0.45,
        ]),
        # Loose enough so the main visual infeasibility comes
        # from joint position in the infeasible case.
        "v_min": np.array([
            -0.70,
            -0.70,
        ]),
        "v_max": np.array([
            0.70,
            0.70,
        ]),
        "a_min": np.array([
            -1.70,
            -1.70,
        ]),
        "a_max": np.array([
            1.70,
            1.70,
        ]),
    }

    nominal_feasible = (
        feasible_nominal_control_points()
    )

    nominal_infeasible = (
        infeasible_nominal_control_points()
    )

    # --------------------------------------------------------
    # Time grids
    # --------------------------------------------------------

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        320,
    )

    dense_times = np.linspace(
        spline.t_start,
        spline.t_end,
        1400,
    )

    # --------------------------------------------------------
    # Feasible case
    # --------------------------------------------------------

    optimized_feasible, result_feasible = (
        solve_projection(
            spline=spline,
            nominal_control_points=nominal_feasible,
            tracking_times=tracking_times,
            limits=limits,
            lambda_smooth=1e-6,
        )
    )

    q_ref_f, v_ref_f, a_ref_f = evaluate_spline(
        spline,
        nominal_feasible,
        dense_times,
    )

    q_opt_f, v_opt_f, a_opt_f = evaluate_spline(
        spline,
        optimized_feasible,
        dense_times,
    )

    # --------------------------------------------------------
    # Infeasible case
    # --------------------------------------------------------

    optimized_infeasible, result_infeasible = (
        solve_projection(
            spline=spline,
            nominal_control_points=nominal_infeasible,
            tracking_times=tracking_times,
            limits=limits,
            lambda_smooth=1e-6,
        )
    )

    q_ref_i, v_ref_i, a_ref_i = evaluate_spline(
        spline,
        nominal_infeasible,
        dense_times,
    )

    q_opt_i, v_opt_i, a_opt_i = evaluate_spline(
        spline,
        optimized_infeasible,
        dense_times,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print_metrics(
        title="FEASIBLE CASE",
        result=result_feasible,
        q_ref=q_ref_f,
        v_ref=v_ref_f,
        a_ref=a_ref_f,
        q_opt=q_opt_f,
        v_opt=v_opt_f,
        a_opt=a_opt_f,
        limits=limits,
    )

    print_metrics(
        title="INFEASIBLE CASE",
        result=result_infeasible,
        q_ref=q_ref_i,
        v_ref=v_ref_i,
        a_ref=a_ref_i,
        q_opt=q_opt_i,
        v_opt=v_opt_i,
        a_opt=a_opt_i,
        limits=limits,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    feasible_data = (
        q_ref_f,
        v_ref_f,
        a_ref_f,
        q_opt_f,
        v_opt_f,
        a_opt_f,
    )

    infeasible_data = (
        q_ref_i,
        v_ref_i,
        a_ref_i,
        q_opt_i,
        v_opt_i,
        a_opt_i,
    )

    plot_time_comparison(
        times=dense_times,
        feasible_data=feasible_data,
        infeasible_data=infeasible_data,
        limits=limits,
    )

    plot_joint_space(
        q_ref_feasible=q_ref_f,
        q_opt_feasible=q_opt_f,
        q_ref_infeasible=q_ref_i,
        q_opt_infeasible=q_opt_i,
        limits=limits,
    )

    plt.show()


if __name__ == "__main__":
    main()

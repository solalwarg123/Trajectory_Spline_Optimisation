import numpy as np
import matplotlib.pyplot as plt

from splineopt.bspline import BSpline
from splineopt.joint_optimizer import JointTrajectoryOptimizer


# ============================================================
# Spline
# ============================================================

def make_spline():
    """
    Cubic open clamped B-spline on [0, 3].
    """

    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0,
        2.0,
        3.0, 3.0, 3.0, 3.0,
    ])

    return BSpline(
        knots=knots,
        degree=3,
        n_dof=1,
    )


# ============================================================
# Nominal trajectory
# ============================================================

def rise_trajectory(
    times,
    q_target,
    rise_time,
):
    """
    Smooth half-cosine rise:

        q(t) =
            q_target / 2
            * (1 - cos(pi t / rise_time))

    for t <= rise_time.

    The trajectory remains constant afterwards.

    Returns position, velocity and acceleration.
    """

    times = np.asarray(times)

    q = np.full_like(
        times,
        q_target,
        dtype=float,
    )

    v = np.zeros_like(
        times,
        dtype=float,
    )

    a = np.zeros_like(
        times,
        dtype=float,
    )

    rising = times <= rise_time

    t = times[rising]

    q[rising] = (
        0.5
        * q_target
        * (
            1.0
            - np.cos(
                np.pi * t / rise_time
            )
        )
    )

    v[rising] = (
        0.5
        * q_target
        * np.pi
        / rise_time
        * np.sin(
            np.pi * t / rise_time
        )
    )

    a[rising] = (
        0.5
        * q_target
        * (
            np.pi / rise_time
        ) ** 2
        * np.cos(
            np.pi * t / rise_time
        )
    )

    return q, v, a


# ============================================================
# Solve one case
# ============================================================

def solve_case(
    spline,
    rise_time,
    q_target,
    limits,
):
    """
    Optimize one nominal trajectory.
    """

    tracking_times = np.linspace(
        spline.t_start,
        spline.t_end,
        100,
    )

    q_ref, _, _ = rise_trajectory(
        tracking_times,
        q_target=q_target,
        rise_time=rise_time,
    )

    q_ref = q_ref[:, None]

    optimizer = JointTrajectoryOptimizer(
        spline=spline,
        lambda_smooth=1e-3,
    )

    control_points, result = optimizer.solve(
        tracking_times=tracking_times,
        q_ref=q_ref,
        q0=np.array([0.0]),
        v0=np.array([0.0]),
        q_min=np.array([limits["q_min"]]),
        q_max=np.array([limits["q_max"]]),
        v_min=np.array([limits["v_min"]]),
        v_max=np.array([limits["v_max"]]),
        a_min=np.array([limits["a_min"]]),
        a_max=np.array([limits["a_max"]]),
    )

    return control_points, result


# ============================================================
# Evaluate optimized spline
# ============================================================

def evaluate_spline(
    spline,
    control_points,
    times,
):

    q = np.array([
        spline.position(
            control_points,
            t,
        )[0]
        for t in times
    ])

    v = np.array([
        spline.velocity(
            control_points,
            t,
        )[0]
        for t in times
    ])

    a = np.array([
        spline.acceleration(
            control_points,
            t,
        )[0]
        for t in times
    ])

    return q, v, a


# ============================================================
# Plot
# ============================================================

def plot_case(
    spline,
    control_points,
    rise_time,
    q_target,
    limits,
    title,
):

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        1000,
    )

    q_ref, v_ref, a_ref = rise_trajectory(
        times,
        q_target=q_target,
        rise_time=rise_time,
    )

    q_opt, v_opt, a_opt = evaluate_spline(
        spline,
        control_points,
        times,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    fig.suptitle(title)

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    axes[0].plot(
        times,
        q_ref,
        "--",
        label="Nominal trajectory",
    )

    axes[0].plot(
        times,
        q_opt,
        label="Optimized B-spline",
    )

    axes[0].axhline(
        limits["q_max"],
        linestyle=":",
        label="Limits",
    )

    axes[0].axhline(
        limits["q_min"],
        linestyle=":",
    )

    axes[0].set_ylabel(
        "Position [rad]"
    )

    axes[0].grid()
    axes[0].legend()

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    axes[1].plot(
        times,
        v_ref,
        "--",
        label="Nominal trajectory",
    )

    axes[1].plot(
        times,
        v_opt,
        label="Optimized B-spline",
    )

    axes[1].axhline(
        limits["v_max"],
        linestyle=":",
        label="Limits",
    )

    axes[1].axhline(
        limits["v_min"],
        linestyle=":",
    )

    axes[1].set_ylabel(
        "Velocity [rad/s]"
    )

    axes[1].grid()
    axes[1].legend()

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    axes[2].plot(
        times,
        a_ref,
        "--",
        label="Nominal trajectory",
    )

    axes[2].plot(
        times,
        a_opt,
        label="Optimized B-spline",
    )

    axes[2].axhline(
        limits["a_max"],
        linestyle=":",
        label="Limits",
    )

    axes[2].axhline(
        limits["a_min"],
        linestyle=":",
    )

    axes[2].set_ylabel(
        "Acceleration [rad/s²]"
    )

    axes[2].set_xlabel(
        "Time [s]"
    )

    axes[2].grid()
    axes[2].legend()

    plt.tight_layout()

    return q_ref, v_ref, a_ref, q_opt, v_opt, a_opt


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
):

    rms_error = np.sqrt(
        np.mean(
            (q_opt - q_ref) ** 2
        )
    )

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"OSQP status: "
        f"{result.info.status}"
    )

    print(
        f"Solve time: "
        f"{result.info.solve_time * 1e3:.3f} ms"
    )

    print(
        f"RMS tracking error: "
        f"{rms_error:.6f} rad"
    )

    print()

    print(
        f"Nominal max velocity: "
        f"{np.max(np.abs(v_ref)):.3f} rad/s"
    )

    print(
        f"Optimized max velocity: "
        f"{np.max(np.abs(v_opt)):.3f} rad/s"
    )

    print()

    print(
        f"Nominal max acceleration: "
        f"{np.max(np.abs(a_ref)):.3f} rad/s²"
    )

    print(
        f"Optimized max acceleration: "
        f"{np.max(np.abs(a_opt)):.3f} rad/s²"
    )


# ============================================================
# Main
# ============================================================

def main():

    spline = make_spline()

    # Same limits for both experiments
    limits = {
        "q_min": -1.0,
        "q_max": 1.0,
        "v_min": -0.8,
        "v_max": 0.8,
        "a_min": -1.5,
        "a_max": 1.5,
    }

    q_target = 0.8

    # ========================================================
    # Case 1: feasible nominal trajectory
    # ========================================================

    feasible_rise_time = 2.5

    feasible_control_points, feasible_result = (
        solve_case(
            spline=spline,
            rise_time=feasible_rise_time,
            q_target=q_target,
            limits=limits,
        )
    )

    feasible_data = plot_case(
        spline=spline,
        control_points=feasible_control_points,
        rise_time=feasible_rise_time,
        q_target=q_target,
        limits=limits,
        title="Feasible nominal trajectory",
    )

    print_metrics(
        "FEASIBLE CASE",
        feasible_result,
        *feasible_data,
    )

    # ========================================================
    # Case 2: infeasible nominal trajectory
    # ========================================================

    infeasible_rise_time = 0.5

    infeasible_control_points, infeasible_result = (
        solve_case(
            spline=spline,
            rise_time=infeasible_rise_time,
            q_target=q_target,
            limits=limits,
        )
    )

    infeasible_data = plot_case(
        spline=spline,
        control_points=infeasible_control_points,
        rise_time=infeasible_rise_time,
        q_target=q_target,
        limits=limits,
        title="Infeasible nominal trajectory",
    )

    print_metrics(
        "INFEASIBLE CASE",
        infeasible_result,
        *infeasible_data,
    )

    plt.show()


if __name__ == "__main__":
    main()

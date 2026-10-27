
"""Shared Kinova reference, metrics and visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from splineopt.bspline import BSpline
from splineopt.joint_optimizer import JointTrajectoryOptimizer


ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = ROOT / "assets/robots/kinova3_flat/gen3_7dof.urdf"

Q_MIN = np.array([-np.inf, -2.24, -np.inf, -2.57, -np.inf, -2.09, -np.inf])
Q_MAX = -Q_MIN
V_MAX = np.full(7, 0.8727)
V_MIN = -V_MAX
A_MAX = np.array([1., 1., 1., 1., 10., 10., 10.])
A_MIN = -A_MAX

LAMBDA_SMOOTH = 1e-5

COLORS = {
    "nominal": "tab:orange",
    "joint": "tab:blue",
    "gn-pos": "tab:green",
    "gn-pose": "tab:purple",
    "ipopt": "tab:red",
}

LABELS = {
    "nominal": "Nominal IK",
    "joint": "Joint-space QP",
    "gn-pos": "GN position",
    "gn-pose": "GN full pose",
    "ipopt": "IPOPT full pose",
}


# ============================================================
# Reference and B-spline construction
# ============================================================

def build_reference(robot, n_samples=201):
    """Construct the original deliberately infeasible Kinova reference."""
    if robot.nv != 7:
        raise ValueError("This example requires a 7-DoF robot.")

    waypoint_times = np.array([0., .8, 2.5, 4., 5.4, 7.2, 9., 10.])
    waypoints = np.array([
        [0., .45, .15, -1.35, -.15, .75, .10],
        [0., .45, .15, -1.35, -.15, .75, .10],
        [.35, 1.30, .45, -1.05, -.45, .85, .30],
        [.70, 2.72, .75, -2.72, -.65, 1.10, .55],
        [.95, 2.82, .20, -2.70, .20, 1.40, -.30],
        [.55, 1.65, -.40, -1.55, .55, 1.15, -.50],
        [.10, .65, -.15, -1.25, 0., .80, 0.],
        [.10, .65, -.15, -1.25, 0., .80, 0.],
    ])

    times = np.linspace(0., 10., n_samples)
    theta = np.column_stack([
        np.interp(times, waypoint_times, waypoints[:, j])
        for j in range(7)
    ])

    q_pin = robot.joint_trajectory_to_configurations(theta)
    poses = [robot.frame_pose(q) for q in q_pin]
    return times, theta, poses


def make_spline(times, n_dof=7, n_controls=30):
    degree = 3
    interior = np.linspace(
        times[0], times[-1], n_controls - degree + 1
    )[1:-1]
    knots = np.r_[
        np.full(degree + 1, times[0]),
        interior,
        np.full(degree + 1, times[-1]),
    ]
    return BSpline(knots=knots, degree=degree, n_dof=n_dof)


def joint_projection(robot, times, theta, n_controls=30):
    """Feasible joint-space QP, also used to initialize GN/IPOPT."""
    spline = make_spline(times, robot.nv, n_controls)
    optimizer = JointTrajectoryOptimizer(
        spline=spline, lambda_smooth=LAMBDA_SMOOTH
    )

    controls, result = optimizer.solve(
        tracking_times=times,
        q_ref=theta,
        q0=theta[0],
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

    if "solved" not in result.info.status.lower():
        raise RuntimeError(f"Joint QP failed: {result.info.status}")

    return spline, controls, result


def evaluate_spline(spline, controls, times):
    return tuple(
        np.array([fn(controls, t) for t in times])
        for fn in (
            spline.position,
            spline.velocity,
            spline.acceleration,
        )
    )


def interpolate_reference(source_times, theta, times):
    q = np.column_stack([
        np.interp(times, source_times, theta[:, j])
        for j in range(theta.shape[1])
    ])
    v = np.gradient(q, times, axis=0)
    a = np.gradient(v, times, axis=0)
    return q, v, a


def fk_trajectory(robot, theta):
    q_pin = robot.joint_trajectory_to_configurations(theta)
    poses = [robot.frame_pose(q) for q in q_pin]
    xyz = np.array([p.translation.copy() for p in poses])
    return q_pin, poses, xyz


# ============================================================
# Metrics
# ============================================================

def trajectory_metrics(robot, reference_poses, q, v, a, tol=1e-3):
    _, poses, xyz = fk_trajectory(robot, q)

    pos_errors = np.array([
        np.linalg.norm(p.translation - ref.translation)
        for p, ref in zip(poses, reference_poses)
    ])
    ori_errors = np.array([
        np.linalg.norm(pin.log3(p.rotation.T @ ref.rotation))
        for p, ref in zip(poses, reference_poses)
    ])

    violations = []
    for values, low, high in (
        (q, Q_MIN, Q_MAX),
        (v, V_MIN, V_MAX),
        (a, A_MIN, A_MAX),
    ):
        violations.append(max(
            0.,
            float(np.max(np.where(np.isfinite(high), values - high, -np.inf))),
            float(np.max(np.where(np.isfinite(low), low - values, -np.inf))),
        ))

    return {
        "q": q, "v": v, "a": a,
        "xyz": xyz,
        "pos_error": pos_errors,
        "ori_error": ori_errors,
        "pos_rms": float(np.sqrt(np.mean(pos_errors**2))),
        "ori_rms": float(np.sqrt(np.mean(ori_errors**2))),
        "violations": violations,
    }


def print_summary(results, times, timings):
    print("\n" + "=" * 99)
    print("TRAJECTORY COMPARISON")
    print("=" * 99)
    print(
        f"{'Method':<19} {'Pos RMS mm':>11} {'Ori RMS deg':>12} "
        f"{'Max q viol':>12} {'Max v viol':>12} "
        f"{'Max a viol':>12} {'Time s':>9}"
    )
    print("-" * 99)
    for name, data in results.items():
        dq, dv, da = data["violations"]
        print(
            f"{LABELS[name]:<19} "
            f"{1000 * data['pos_rms']:>11.3f} "
            f"{np.degrees(data['ori_rms']):>12.3f} "
            f"{dq:>12.2e} {dv:>12.2e} {da:>12.2e} "
            f"{timings.get(name, 0.):>9.3f}"
        )

    print("\nSmoothness (integral of squared acceleration):")
    for name, data in results.items():
        cost = np.trapezoid(np.sum(data["a"]**2, axis=1), times)
        print(f"  {LABELS[name]:<19}: {cost:.5f}")


# ============================================================
# Visualization
# ============================================================

def _style(ax, title, xlabel="Time [s]", ylabel=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.25)


def plot_cartesian_3d(results):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, data in results.items():
        p = data["xyz"]
        ax.plot(
            p[:, 0], p[:, 1], p[:, 2],
            color=COLORS[name], lw=2.3,
            linestyle="--" if name == "nominal" else "-",
            label=LABELS[name],
        )

    first = next(iter(results.values()))["xyz"]
    ax.scatter(*first[0], c="green", s=55, label="Start")
    ax.scatter(*first[-1], c="black", s=55, label="Target")

    _style(ax, "End-effector trajectories", "X [m]", "Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()


def plot_cartesian_errors(times, results):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Cartesian tracking errors", fontsize=14)

    for name, data in results.items():
        if name == "nominal":
            continue
        axes[0].plot(
            times, 1000 * data["pos_error"],
            color=COLORS[name], lw=2, label=LABELS[name],
        )
        axes[1].plot(
            times, np.degrees(data["ori_error"]),
            color=COLORS[name], lw=2, label=LABELS[name],
        )

    _style(axes[0], "Position error", ylabel="Error [mm]")
    _style(axes[1], "Orientation error", ylabel="Error [deg]")
    for ax in axes:
        ax.legend()
    fig.tight_layout()


def plot_joint_quantities(times, results):
    """One 7-row figure for q, v, a with shaded admissible regions."""
    quantities = [
        ("q", "Position", "rad", Q_MIN, Q_MAX),
        ("v", "Velocity", "rad/s", V_MIN, V_MAX),
        ("a", "Acceleration", "rad/s²", A_MIN, A_MAX),
    ]

    for key, title, unit, low, high in quantities:
        fig, axes = plt.subplots(
            7, 1, figsize=(13, 15), sharex=True
        )
        fig.suptitle(f"Joint {title.lower()} comparison", fontsize=14)

        for j, ax in enumerate(axes):
            if np.isfinite(low[j]) and np.isfinite(high[j]):
                ax.axhspan(
                    low[j], high[j],
                    color="tab:green", alpha=.07,
                )
                ax.axhline(low[j], color="tab:red", ls=":", lw=1)
                ax.axhline(high[j], color="tab:red", ls=":", lw=1)

            for name, data in results.items():
                ax.plot(
                    times, data[key][:, j],
                    color=COLORS[name],
                    ls="--" if name == "nominal" else "-",
                    lw=1.5,
                    label=LABELS[name],
                )

            ax.set_ylabel(f"J{j+1} [{unit}]")
            ax.grid(alpha=.25)

        axes[0].legend(ncol=min(5, len(results)), fontsize=8)
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()


def plot_feasibility_usage(times, results):
    """Velocity and acceleration usage relative to physical limits."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Joint limit utilization", fontsize=14)

    for name, data in results.items():
        for ax, key, bound in zip(
            axes, ("v", "a"), (V_MAX, A_MAX)
        ):
            usage = np.max(np.abs(data[key]) / bound, axis=1)
            ax.plot(
                times, usage,
                color=COLORS[name],
                linestyle="--" if name == "nominal" else "-",
                lw=2, label=LABELS[name],
            )

    for ax, title in zip(axes, ("Velocity", "Acceleration")):
        ax.axhspan(0, 1, color="tab:green", alpha=.07)
        ax.axhspan(1, max(1.5, ax.get_ylim()[1]),
                   color="tab:red", alpha=.05)
        ax.axhline(1, color="tab:red", ls=":", label="Limit")
        _style(ax, title, ylabel="Max utilization")
        ax.legend(ncol=3, fontsize=8)

    fig.tight_layout()


def show_plots(times, results):
    plot_cartesian_3d(results)
    plot_cartesian_errors(times, results)
    plot_joint_quantities(times, results)
    plot_feasibility_usage(times, results)
    plt.show()

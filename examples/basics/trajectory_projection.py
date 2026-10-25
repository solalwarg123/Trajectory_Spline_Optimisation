
"""Joint-space B-spline projection: dynamic and position constraints."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from splineopt.bspline import BSpline
from splineopt.joint_optimizer import JointTrajectoryOptimizer


# ============================================================
# Shared B-spline utilities
# ============================================================

def make_spline(duration, n_controls, n_dof):
    degree = 3
    internal = np.linspace(
        0, duration, n_controls - degree + 1
    )[1:-1]
    knots = np.r_[
        np.zeros(degree + 1),
        internal,
        np.full(degree + 1, duration),
    ]
    return BSpline(knots=knots, degree=degree, n_dof=n_dof)


def evaluate(spline, controls, times):
    return tuple(
        np.array([function(controls, t) for t in times])
        for function in (
            spline.position,
            spline.velocity,
            spline.acceleration,
        )
    )


def project(spline, tracking_times, q_ref, limits,
            q0, v0, smoothness):
    optimizer = JointTrajectoryOptimizer(
        spline=spline, lambda_smooth=smoothness
    )
    controls, result = optimizer.solve(
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
    if "solved" not in result.info.status.lower():
        raise RuntimeError(f"OSQP failed: {result.info.status}")
    return controls, result


def print_metrics(title, result, reference, projected, limits):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"OSQP status: {result.info.status}")

    error = reference[0] - projected[0]
    print(f"Position tracking RMS: {np.sqrt(np.mean(error**2)):.6f} rad")

    for k, name in enumerate(("q", "v", "a")):
        lo, hi = limits[f"{name}_min"], limits[f"{name}_max"]
        for label, values in (
            ("Nominal", reference[k]),
            ("Projected", projected[k]),
        ):
            violation = max(
                0.0,
                float(np.max(values - hi)),
                float(np.max(lo - values)),
            )
            print(f"{label:9s} {name} max violation: {violation:.3e}")


# ============================================================
# Scenario 1: dynamic infeasibility (1D)
# ============================================================

def rise_trajectory(times, target, rise_time):
    times = np.asarray(times)
    tau = np.minimum(times, rise_time)

    q = 0.5 * target * (1 - np.cos(np.pi * tau / rise_time))
    v = np.where(
        times <= rise_time,
        0.5 * target * np.pi / rise_time
        * np.sin(np.pi * tau / rise_time),
        0.0,
    )
    a = np.where(
        times <= rise_time,
        0.5 * target * (np.pi / rise_time)**2
        * np.cos(np.pi * tau / rise_time),
        0.0,
    )
    return q[:, None], v[:, None], a[:, None]


def run_dynamics():
    spline = make_spline(3.0, 6, 1)
    limits = {
        "q_min": np.array([-1.0]), "q_max": np.array([1.0]),
        "v_min": np.array([-0.8]), "v_max": np.array([0.8]),
        "a_min": np.array([-1.5]), "a_max": np.array([1.5]),
    }

    tracking = np.linspace(0, 3, 100)
    dense = np.linspace(0, 3, 1000)
    cases = [
        ("Feasible nominal trajectory", 2.5),
        ("Infeasible nominal trajectory", 0.5),
    ]

    results = []
    for title, rise_time in cases:
        q_ref = rise_trajectory(tracking, 0.8, rise_time)[0]
        controls, result = project(
            spline, tracking, q_ref, limits,
            q0=np.array([0.0]),
            v0=np.array([0.0]),
            smoothness=1e-3,
        )
        nominal = rise_trajectory(dense, 0.8, rise_time)
        optimized = evaluate(spline, controls, dense)
        print_metrics(title, result, nominal, optimized, limits)
        results.append((title, nominal, optimized))

    plot_dynamics(dense, results, limits)


def plot_dynamics(times, results, limits):
    # Same 3 quantities and feasible/infeasible comparison as
    # the original 1D example, combined into one figure.
    fig, axes = plt.subplots(
        3, 2, figsize=(13, 9), sharex=True
    )
    fig.suptitle(
        "1D B-spline projection — dynamic constraints",
        fontsize=15,
    )

    quantities = [
        ("q", "Position [rad]"),
        ("v", "Velocity [rad/s]"),
        ("a", "Acceleration [rad/s²]"),
    ]

    for col, (title, nominal, optimized) in enumerate(results):
        axes[0, col].set_title(title, fontweight="bold")

        for row, (key, ylabel) in enumerate(quantities):
            ax = axes[row, col]
            low = limits[f"{key}_min"][0]
            high = limits[f"{key}_max"][0]

            ax.axhspan(
                low, high, color="tab:green", alpha=0.08,
                label="Admissible range",
            )
            ax.axhline(high, color="tab:red", linestyle=":", lw=1.4)
            ax.axhline(low, color="tab:red", linestyle=":", lw=1.4)

            ax.plot(
                times, nominal[row][:, 0],
                "--", color="tab:orange", lw=2,
                label="Nominal",
            )
            ax.plot(
                times, optimized[row][:, 0],
                color="tab:blue", lw=2.2,
                label="Optimized B-spline",
            )

            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")

        axes[-1, col].set_xlabel("Time [s]")

    fig.tight_layout()


# ============================================================
# Scenario 2: position infeasibility (2D)
# ============================================================

def nominal_2d_controls(infeasible=False):
    controls = np.array([
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
    if infeasible:
        controls[9:12] = [
            [0.30, 0.32],
            [0.78, 0.35],
            [0.78, 0.20],
        ]
    return controls


def run_position():
    spline = make_spline(8.0, 14, 2)
    limits = {
        "q_min": np.array([-0.55, -0.45]),
        "q_max": np.array([0.55,  0.45]),
        "v_min": np.array([-0.70, -0.70]),
        "v_max": np.array([0.70,  0.70]),
        "a_min": np.array([-1.70, -1.70]),
        "a_max": np.array([1.70,  1.70]),
    }

    tracking = np.linspace(0, 8, 320)
    dense = np.linspace(0, 8, 1400)

    results = []
    for title, infeasible in (
        ("Feasible nominal trajectory", False),
        ("Infeasible nominal trajectory", True),
    ):
        nominal_controls = nominal_2d_controls(infeasible)
        q_tracking = evaluate(spline, nominal_controls, tracking)[0]

        controls, result = project(
            spline, tracking, q_tracking, limits,
            q0=spline.position(nominal_controls, spline.t_start),
            v0=spline.velocity(nominal_controls, spline.t_start),
            smoothness=1e-6,
        )
        nominal = evaluate(spline, nominal_controls, dense)
        optimized = evaluate(spline, controls, dense)
        print_metrics(title, result, nominal, optimized, limits)
        results.append((title, nominal, optimized))

    plot_position_time(dense, results, limits)
    plot_joint_space(results, limits)


def plot_position_time(times, results, limits):
    # Original layout: 3 rows x 2 cases, joint colors,
    # nominal dashed / projected solid, feasible bands.
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 10), sharex=True
    )
    fig.suptitle(
        "2D B-spline projection — joint constraints",
        fontsize=15,
    )

    quantities = [
        ("q", "Position [rad]"),
        ("v", "Velocity [rad/s]"),
        ("a", "Acceleration [rad/s²]"),
    ]

    colors = ("tab:blue", "tab:orange")

    for col, (title, nominal, optimized) in enumerate(results):
        axes[0, col].set_title(title, fontweight="bold")

        for row, (key, ylabel) in enumerate(quantities):
            ax = axes[row, col]
            lower = limits[f"{key}_min"]
            upper = limits[f"{key}_max"]

            for j, color in enumerate(colors):
                ax.axhspan(
                    lower[j], upper[j],
                    color=color, alpha=0.045,
                )
                ax.axhline(
                    lower[j], color=color, ls=":", alpha=0.55,
                )
                ax.axhline(
                    upper[j], color=color, ls=":", alpha=0.55,
                )

                ax.plot(
                    times, nominal[row][:, j],
                    "--", color=color, lw=1.7, alpha=0.8,
                    label=f"Joint {j+1} nominal",
                )
                ax.plot(
                    times, optimized[row][:, j],
                    "-", color=color, lw=2.3,
                    label=f"Joint {j+1} projected",
                )

            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)

        axes[2, col].set_xlabel("Time [s]")

    fig.tight_layout()


def plot_joint_space(results, limits):
    # Dedicated joint-space figure with forbidden regions
    # shaded red, as in the original 2D example.
    qmin, qmax = limits["q_min"], limits["q_max"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Joint-space projection — admissible and forbidden regions",
        fontsize=15,
    )

    xlim = (qmin[0] - 0.25, 0.95)
    ylim = (qmin[1] - 0.20, qmax[1] + 0.20)

    for ax, (title, nominal, optimized) in zip(axes, results):
        ax.set_facecolor("#fce8e8")

        rectangle = Rectangle(
            (qmin[0], qmin[1]),
            qmax[0] - qmin[0],
            qmax[1] - qmin[1],
            facecolor="#e1f4e3",
            edgecolor="tab:green",
            linewidth=2,
            label="Admissible joint region",
        )
        ax.add_patch(rectangle)

        q_nominal = nominal[0]
        q_optimized = optimized[0]

        ax.plot(
            q_nominal[:, 0], q_nominal[:, 1],
            "--", color="tab:orange", lw=2,
            label="Nominal trajectory",
        )
        ax.plot(
            q_optimized[:, 0], q_optimized[:, 1],
            "-", color="tab:blue", lw=2.5,
            label="Projected B-spline",
        )

        ax.scatter(
            *q_nominal[0], marker="o", s=55,
            color="tab:green", zorder=5,
        )
        ax.scatter(
            *q_nominal[-1], marker="s", s=55,
            color="tab:red", zorder=5,
        )

        ax.set(
            title=title,
            xlabel="Joint 1 [rad]",
            ylabel="Joint 2 [rad]",
            xlim=xlim,
            ylim=ylim,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()


# ============================================================
# Command-line entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("all", "dynamics", "position"),
        default="all",
        help="Run 1D dynamics, 2D position, or both (default).",
    )
    args = parser.parse_args()

    if args.scenario in ("all", "dynamics"):
        run_dynamics()

    if args.scenario in ("all", "position"):
        run_position()

    plt.show()


if __name__ == "__main__":
    main()

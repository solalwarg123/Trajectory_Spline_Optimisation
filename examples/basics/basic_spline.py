"""Basic cubic B-spline: control points, trajectory and derivatives."""

import numpy as np
import matplotlib.pyplot as plt

from splineopt.bspline import BSpline


def main():
    # --------------------------------------------------
    # 1. Define a cubic B-spline with 2 degrees of freedom
    # --------------------------------------------------
    degree = 3

    knots = np.array([
        0, 0, 0, 0,
        1, 2, 3,
        4, 4, 4, 4,
    ], dtype=float)

    # Each row defines one control point in joint space
    control_points = np.array([
        [0.0,  0.0],
        [0.3, -0.2],
        [1.0,  0.5],
        [1.5,  1.0],
        [1.0,  1.4],
        [0.5,  0.8],
        [0.0,  0.2],
    ])

    spline = BSpline(knots=knots, degree=degree, n_dof=2)

    # --------------------------------------------------
    # 2. Evaluate position, velocity and acceleration
    # --------------------------------------------------
    times = np.linspace(spline.t_start, spline.t_end, 500)

    q = np.array([spline.position(control_points, t) for t in times])
    dq = np.array([spline.velocity(control_points, t) for t in times])
    ddq = np.array([spline.acceleration(control_points, t) for t in times])

    print("Cubic B-spline example")
    print(f"Degree: {degree}")
    print(f"Control points: {len(control_points)}")
    print(f"Duration: {spline.t_end - spline.t_start:.1f} s")

    # --------------------------------------------------
    # 3. Visualization
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Cubic B-spline — trajectory and derivatives", fontsize=14)

    # Joint-space trajectory and control polygon
    ax = axes[0, 0]
    ax.plot(q[:, 0], q[:, 1], lw=2.5, label="B-spline trajectory")
    ax.plot(
        control_points[:, 0], control_points[:, 1],
        "o--", color="gray", alpha=0.7, label="Control polygon"
    )
    ax.scatter(q[0, 0], q[0, 1], color="green", s=70, label="Start", zorder=5)
    ax.scatter(q[-1, 0], q[-1, 1], color="red", s=70, label="End", zorder=5)
    ax.set(title="Trajectory in joint space",
           xlabel="Joint 1 [rad]", ylabel="Joint 2 [rad]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    # Time derivatives
    quantities = [
        (axes[0, 1], q, "Position", "Position [rad]"),
        (axes[1, 0], dq, "Velocity", "Velocity [rad/s]"),
        (axes[1, 1], ddq, "Acceleration", "Acceleration [rad/s²]"),
    ]

    internal_knots = np.unique(knots[degree + 1: -degree - 1])

    for ax, values, title, ylabel in quantities:
        for j in range(2):
            ax.plot(times, values[:, j], lw=2, label=f"Joint {j + 1}")

        for knot in internal_knots:
            ax.axvline(knot, color="gray", ls=":", alpha=0.5)

        ax.set(title=title, xlabel="Time [s]", ylabel=ylabel)
        ax.legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

from splineopt.bspline import BSpline


# ============================================================
# B-spline definition
# ============================================================

degree = 3
n_dof = 2

# Open clamped knot vector
knots = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 3.0,
    4.0, 4.0, 4.0, 4.0,
])

# Number of control points:
#
# Nc = len(knots) - degree - 1 = 7
#
# Each row corresponds to one control point c_i.
# Each column corresponds to one joint.
control_points = np.array([
    [0.0,  0.0],
    [0.3, -0.2],
    [1.0,  0.5],
    [1.5,  1.0],
    [1.0,  1.4],
    [0.5,  0.8],
    [0.0,  0.2],
])


# ============================================================
# Create spline
# ============================================================

spline = BSpline(
    knots=knots,
    degree=degree,
    n_dof=n_dof,
)


# ============================================================
# Evaluate trajectory
# ============================================================

times = np.linspace(
    spline.t_start,
    spline.t_end,
    500,
)

q = np.array([
    spline.position(control_points, t)
    for t in times
])

dq = np.array([
    spline.velocity(control_points, t)
    for t in times
])

ddq = np.array([
    spline.acceleration(control_points, t)
    for t in times
])


# ============================================================
# Position
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(
    times,
    q[:, 0],
    label="Joint 1",
)

plt.plot(
    times,
    q[:, 1],
    label="Joint 2",
)

plt.xlabel("Time [s]")
plt.ylabel("Position [rad]")
plt.title("B-spline joint positions")

plt.grid()
plt.legend()
plt.tight_layout()


# ============================================================
# Velocity
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(
    times,
    dq[:, 0],
    label="Joint 1",
)

plt.plot(
    times,
    dq[:, 1],
    label="Joint 2",
)

plt.xlabel("Time [s]")
plt.ylabel("Velocity [rad/s]")
plt.title("B-spline joint velocities")

plt.grid()
plt.legend()
plt.tight_layout()


# ============================================================
# Acceleration
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(
    times,
    ddq[:, 0],
    label="Joint 1",
)

plt.plot(
    times,
    ddq[:, 1],
    label="Joint 2",
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [rad/s²]")
plt.title("B-spline joint accelerations")

plt.grid()
plt.legend()
plt.tight_layout()


# ============================================================
# Joint-space trajectory
# ============================================================

plt.figure(figsize=(6, 6))

plt.plot(
    q[:, 0],
    q[:, 1],
    label="B-spline trajectory",
)

plt.scatter(
    control_points[:, 0],
    control_points[:, 1],
    marker="x",
    label="Control points",
)

plt.plot(
    control_points[:, 0],
    control_points[:, 1],
    linestyle="--",
    alpha=0.5,
    label="Control polygon",
)

plt.xlabel("Joint 1 [rad]")
plt.ylabel("Joint 2 [rad]")
plt.title("Trajectory in joint space")

plt.grid()
plt.legend()
plt.axis("equal")
plt.tight_layout()


plt.show()

"""Kinova Gen3 closed-loop Cartesian MPC demonstration (no collisions).

Run from repository root:
    python -m examples.kinova_mpc
    python -m examples.kinova_mpc --dt 0.0333333 --steps 90 --no-animation

Requires splineopt/mpc.py with CartesianMPC.step(q0, v0, target_poses).
Plots and CSV are saved to outputs/kinova_mpc/.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from splineopt.bspline import BSpline
from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.mpc import CartesianMPC
from splineopt.robotics import Robot
from examples.kinova_trajectory_projection import (
    URDF_PATH, Q_MIN, Q_MAX, V_MIN, V_MAX, A_MIN, A_MAX,
)


START = np.array([0.00, 0.45, 0.15, -1.35, -0.15, 0.75, 0.10])
GOAL = np.array([0.35, 1.05, 0.35, -1.60, -0.30, 1.00, 0.20])
REFERENCE_DURATION = 3.0  # seconds; reference stays at GOAL afterwards


def make_spline(horizon, ndof, n_control_points):
    degree = 3
    inner = np.linspace(0.0, horizon, n_control_points - degree + 1)[1:-1]
    knots = np.r_[np.zeros(degree + 1), inner, np.full(degree + 1, horizon)]
    return BSpline(knots=knots, degree=degree, n_dof=ndof)


def reference_joint(t):
    """Smooth start and stop; deterministic and safely within the joint limits."""
    x = np.clip(np.asarray(t, dtype=float) / REFERENCE_DURATION, 0.0, 1.0)
    smooth = 10*x**3 - 15*x**4 + 6*x**5
    return START + smooth[..., None] * (GOAL - START)


def pose_from_theta(robot, theta):
    return robot.frame_pose(robot.joint_positions_to_configuration(theta))


def pose_errors(actual, desired):
    p = np.linalg.norm(actual.translation - desired.translation)
    r = np.linalg.norm(pin.log3(actual.rotation.T @ desired.rotation))
    return float(p), float(r)


def enforce_initial_state_on_guess(spline, control_points, q0, v0):
    """Cheap algebraic q(0)/v(0) correction for an open-clamped B-spline.

    No QP and no temporal shift. Keeps remaining control points unchanged.
    The optimizer still enforces all joint constraints.
    """
    c = np.asarray(control_points, dtype=float).reshape(
        spline.n_control_points, spline.n_dof).copy()
    p = spline.degree
    h = spline.knots[p + 1] - spline.knots[0]
    if h <= 0:
        raise ValueError('Expected a positive first B-spline segment')
    c[0] = q0
    c[1] = q0 + (h / p) * v0
    return c


def max_linear_violation(problem, control_points):
    A, lower, upper = problem.linear_constraints()
    values = np.asarray(A @ np.asarray(control_points).reshape(-1)).reshape(-1)
    return float(max(np.max(lower - values), np.max(values - upper), 0.0))


# ================================================================
# Plotting — consistent with other Kinova examples
# ================================================================
COLOR_REFERENCE = "C1"
COLOR_MPC = "C3"
COLOR_FEASIBLE = "#2ca02c"
COLOR_INFEASIBLE = "#d62728"
COLOR_LIMIT = "#555555"


def shade_joint_limits(ax, lower, upper, trajectories):
    """Display admissible and forbidden joint-space regions."""

    values = np.concatenate([
        np.asarray(x).ravel() for x in trajectories
    ])
    values = values[np.isfinite(values)]

    if np.isfinite(lower):
        values = np.append(values, lower)
    if np.isfinite(upper):
        values = np.append(values, upper)

    if values.size == 0:
        return

    span = max(np.ptp(values), 0.1)
    ymin = np.min(values) - 0.12 * span
    ymax = np.max(values) + 0.12 * span

    ax.set_ylim(ymin, ymax)

    if np.isfinite(lower) and np.isfinite(upper):
        ax.axhspan(lower, upper,
                   color=COLOR_FEASIBLE, alpha=0.08, zorder=0)
    elif np.isfinite(lower):
        ax.axhspan(lower, ymax,
                   color=COLOR_FEASIBLE, alpha=0.08, zorder=0)
    elif np.isfinite(upper):
        ax.axhspan(ymin, upper,
                   color=COLOR_FEASIBLE, alpha=0.08, zorder=0)

    if np.isfinite(lower):
        ax.axhspan(ymin, lower,
                   color=COLOR_INFEASIBLE, alpha=0.08, zorder=0)
        ax.axhline(lower, color=COLOR_LIMIT,
                   linestyle=":", linewidth=1.1)

    if np.isfinite(upper):
        ax.axhspan(upper, ymax,
                   color=COLOR_INFEASIBLE, alpha=0.08, zorder=0)
        ax.axhline(upper, color=COLOR_LIMIT,
                   linestyle=":", linewidth=1.1)


def plot_joint_comparison(times, reference, executed,
                          lower, upper, ylabel, title):
    """Seven joint plots in the same 4x2 layout as offline examples."""

    fig, axes = plt.subplots(
        4, 2, figsize=(13, 11), sharex=True
    )
    axes = axes.ravel()

    for j in range(7):
        ax = axes[j]

        shade_joint_limits(
            ax, lower[j], upper[j],
            trajectories=[reference[:, j], executed[:, j]]
        )

        ax.plot(
            times, executed[:, j],
            color=COLOR_MPC, linewidth=2.3,
            label="MPC executed", zorder=5
        )
        ax.plot(
            times, reference[:, j],
            "--", color=COLOR_REFERENCE, linewidth=1.8,
            label="Nominal reference", zorder=6
        )

        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if j >= 6:
            ax.set_xlabel("Time [s]")

        if not np.isfinite(lower[j]) and not np.isfinite(upper[j]):
            ax.text(
                0.02, 0.92, "continuous joint",
                transform=ax.transAxes,
                fontsize=8, alpha=0.7, va="top"
            )

    axes[0].legend(loc="best")
    axes[-1].axis("off")

    fig.suptitle(
        title + "\nGreen = admissible, red = forbidden",
        y=0.995
    )
    fig.tight_layout()
    return fig


def save_plots(folder, times, actual_xyz, reference_xyz, ep, eo,
               cycle_ms, iterations, statuses, q_hist, v_hist,
               violations, dt):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times)
    actual_xyz = np.asarray(actual_xyz)
    reference_xyz = np.asarray(reference_xyz)
    ep = np.asarray(ep)
    eo = np.asarray(eo)
    cycle_ms = np.asarray(cycle_ms)
    iterations = np.asarray(iterations)
    violations = np.asarray(violations)

    full_times = np.r_[0.0, times]

    # Joint-space nominal reference
    q_ref = reference_joint(full_times)

    # Reference velocity: numerical differentiation
    v_ref = np.gradient(q_ref, full_times, axis=0)

    # ============================================================
    # Figure 1 — Cartesian path
    # ============================================================

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        reference_xyz[:, 0],
        reference_xyz[:, 1],
        reference_xyz[:, 2],
        "--", color=COLOR_REFERENCE,
        linewidth=2.2, label="Nominal reference"
    )

    ax.plot(
        actual_xyz[:, 0],
        actual_xyz[:, 1],
        actual_xyz[:, 2],
        color=COLOR_MPC,
        linewidth=2.6, label="MPC executed"
    )

    ax.scatter(
        *actual_xyz[0], color=COLOR_FEASIBLE,
        s=65, label="First executed state", zorder=10
    )
    ax.scatter(
        *actual_xyz[-1], color=COLOR_MPC,
        s=65, marker="X", label="Final executed state", zorder=10
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Cartesian trajectory — closed-loop MPC")
    ax.legend()

    # Equal aspect ratio to avoid distorted 3D trajectories
    points = np.vstack([actual_xyz, reference_xyz])
    span = np.ptp(points, axis=0)
    ax.set_box_aspect(np.maximum(span, 1e-3))

    fig.tight_layout()
    fig.savefig(folder / "cartesian_path.png",
                dpi=180, bbox_inches="tight")

    # ============================================================
    # Figure 2 — Cartesian tracking errors
    # ============================================================

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )

    axes[0].plot(
        times, 1000.0 * ep,
        color=COLOR_MPC, linewidth=2.3,
        label="MPC position error"
    )
    axes[0].axhline(
        0, color=COLOR_REFERENCE,
        linestyle="--", linewidth=1.5
    )
    axes[0].set_ylabel("Position error [mm]")
    axes[0].set_title("End-effector position tracking error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        times, np.degrees(eo),
        color=COLOR_MPC, linewidth=2.3,
        label="MPC orientation error"
    )
    axes[1].axhline(
        0, color=COLOR_REFERENCE,
        linestyle="--", linewidth=1.5
    )
    axes[1].set_ylabel("Orientation error [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("End-effector orientation tracking error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(folder / "tracking_errors.png",
                dpi=180, bbox_inches="tight")

    # ============================================================
    # Figure 3 — Joint positions
    # ============================================================

    fig = plot_joint_comparison(
        full_times,
        q_ref,
        q_hist,
        Q_MIN,
        Q_MAX,
        ylabel="Position [rad]",
        title="Joint positions — nominal vs MPC"
    )
    fig.savefig(folder / "joint_positions.png",
                dpi=180, bbox_inches="tight")

    # ============================================================
    # Figure 4 — Joint velocities
    # ============================================================

    fig = plot_joint_comparison(
        full_times,
        v_ref,
        v_hist,
        V_MIN,
        V_MAX,
        ylabel="Velocity [rad/s]",
        title="Joint velocities — nominal vs MPC"
    )
    fig.savefig(folder / "joint_velocities.png",
                dpi=180, bbox_inches="tight")

    # ============================================================
    # Figure 5 — Optimization timing and GN iterations
    # ============================================================

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )

    deadline_ms = 1000.0 * dt

    axes[0].plot(
        times, cycle_ms,
        ".-", color="C0",
        linewidth=2.0, markersize=5,
        label="MPC computation"
    )

    axes[0].axhline(
        deadline_ms,
        color=COLOR_INFEASIBLE,
        linestyle="--", linewidth=1.8,
        label=f"Deadline ({deadline_ms:.1f} ms)"
    )

    axes[0].axhline(
        np.median(cycle_ms),
        color=COLOR_FEASIBLE,
        linestyle=":", linewidth=1.8,
        label=f"Median ({np.median(cycle_ms):.1f} ms)"
    )

    axes[0].fill_between(
        times, cycle_ms, deadline_ms,
        where=cycle_ms > deadline_ms,
        color=COLOR_INFEASIBLE,
        alpha=0.15,
        interpolate=True
    )

    axes[0].set_ylabel("Computation time [ms]")
    axes[0].set_title("MPC computation time per cycle")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].step(
        times, iterations,
        where="mid", color="C0",
        linewidth=2.0,
        label="GN/SQP iterations"
    )

    axes[1].set_ylabel("Iterations")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Gauss-Newton iterations per MPC cycle")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(folder / "computation_times.png",
                dpi=180, bbox_inches="tight")

    # ============================================================
    # Figure 6 — Joint constraint usage
    # ============================================================

    def usage(values, lower, upper):
        values = np.asarray(values)
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        result = np.zeros(values.shape[0])

        for j in range(values.shape[1]):
            limits = []

            if np.isfinite(lower[j]) and lower[j] < 0:
                limits.append(
                    np.maximum(0, -values[:, j]) / abs(lower[j])
                )

            if np.isfinite(upper[j]) and upper[j] > 0:
                limits.append(
                    np.maximum(0, values[:, j]) / upper[j]
                )

            if limits:
                result = np.maximum(
                    result,
                    np.maximum.reduce(limits)
                )

        return result

    position_usage = usage(q_hist, Q_MIN, Q_MAX)
    velocity_usage = usage(v_hist, V_MIN, V_MAX)

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )

    for ax, values, title in [
        (axes[0], position_usage, "Joint-position limit usage"),
        (axes[1], velocity_usage, "Joint-velocity limit usage"),
    ]:
        ax.plot(
            full_times, values,
            color=COLOR_MPC, linewidth=2.3,
            label="MPC executed"
        )
        ax.axhline(
            1.0, color=COLOR_INFEASIBLE,
            linestyle="--", linewidth=1.8,
            label="Limit"
        )
        ax.set_ylabel("Normalized usage")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(folder / "constraint_usage.png",
                dpi=180, bbox_inches="tight")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=0.1,
                        help='MPC control period [s]')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--horizon', type=float, default=1.5)
    parser.add_argument('--tracking-points', type=int, default=15)
    parser.add_argument('--control-points', type=int, default=12)
    parser.add_argument('--max-iterations', type=int, default=20)
    parser.add_argument('--orientation-scale', type=float, default=0.1)
    parser.add_argument('--no-animation', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/kinova_mpc'))
    args = parser.parse_args()
    if args.dt <= 0 or args.dt >= args.horizon:
        parser.error('Require 0 < dt < horizon')
    if args.steps < 1 or args.tracking_points < 2 or args.control_points < 5:
        parser.error('Invalid discretization / step counts')
    args.output.mkdir(parents=True, exist_ok=True)

    robot = Robot(urdf_path=URDF_PATH, end_effector_frame='end_effector_link')
    spline = make_spline(args.horizon, robot.nv, args.control_points)
    grid = np.linspace(0.0, args.horizon, args.tracking_points)
    target_poses = [pose_from_theta(robot, reference_joint(t)) for t in grid]

    # First solve: constant trajectory at the initial configuration.
    # No a0/vf boundary conditions: they would impose a rest state at every horizon.
    c0 = np.tile(START, (spline.n_control_points, 1))
    q = START.copy()
    v = np.zeros(robot.nv)
    problem = CartesianTrajectoryProblem(
        robot=robot, spline=spline, tracking_times=grid,
        target_poses=target_poses, q0=q, v0=v,
        q_min=Q_MIN, q_max=Q_MAX, v_min=V_MIN, v_max=V_MAX,
        a_min=A_MIN, a_max=A_MAX,
        orientation_scale=args.orientation_scale,
        lambda_smooth=1e-5,
        lambda_terminal=1.0,
    )
    solver = GaussNewtonSolver(
        damping_mode='adaptive', damping=1e-3,
        max_iterations=args.max_iterations,
    )
    mpc = CartesianMPC(problem, solver, c0)

    times, cycle_ms, iterations, statuses, violations = [], [], [], [], []
    position_errors, orientation_errors, actual_xyz, reference_xyz = [], [], [], []
    q_history, v_history = [q.copy()], [v.copy()]

    print(f'MPC: dt={args.dt:.4f}s, horizon={args.horizon:.2f}s, steps={args.steps}, '
          f'iterations <= {args.max_iterations}')
    for k in range(args.steps):
        t = k * args.dt
        targets = [pose_from_theta(
            robot, reference_joint(t + tau)) for tau in grid]
        tic = time.perf_counter()
        # Correct start boundary conditions without shifting the trajectory
        # or running an additional QP. This helps GN line search maintain
        # equality feasibility when the measured state changes.
        mpc.previous_control_points = enforce_initial_state_on_guess(
            spline, mpc.previous_control_points, q, v)
        result = mpc.step(q0=q, v0=v, target_poses=targets)
        elapsed_ms = 1000.0 * (time.perf_counter() - tic)
        c = result.control_points
        violation = max_linear_violation(problem, c)
        if not np.all(np.isfinite(c)) or violation > 1e-4:
            raise RuntimeError(
                f'Cycle {k}: infeasible solution (status={result.status}, '
                f'max linear violation={violation:.3g}). '
                'This can occur when GN line search does not enforce changed initial equalities. '
                'Do not execute an infeasible candidate.'
            )
        # max_iterations is a budget exhaustion, not necessarily a bad trajectory.
        # For a feasible candidate, keep it as the next warm start.
        if not result.success and result.status == 'max_iterations':
            mpc.previous_control_points = np.asarray(c).copy()
        elif not result.success:
            raise RuntimeError(
                f'Cycle {k}: GN failed ({result.status}); no command executed')

        # Perfect tracking for the FIRST experiment: apply the planned state at dt.
        # Tracking disturbances will later be inserted here, not in CartesianMPC.
        q = spline.position(c, args.dt)
        v = spline.velocity(c, args.dt)
        time_now = t + args.dt
        actual = pose_from_theta(robot, q)
        desired = pose_from_theta(robot, reference_joint(time_now))
        ep, eo = pose_errors(actual, desired)

        times.append(time_now)
        cycle_ms.append(elapsed_ms)
        iterations.append(result.iterations)
        statuses.append(result.status)
        violations.append(violation)
        position_errors.append(ep)
        orientation_errors.append(eo)
        actual_xyz.append(actual.translation.copy())
        reference_xyz.append(desired.translation.copy())
        q_history.append(q.copy())
        v_history.append(v.copy())
        print(f'[{k+1:03d}/{args.steps:03d}] t={time_now:.2f}s  '
              f'{elapsed_ms:8.1f}ms  GN={result.iterations:2d}  '
              f'error={ep*1000:7.2f}mm  status={result.status}')

    times = np.asarray(times)
    cycle_ms = np.asarray(cycle_ms)
    iterations = np.asarray(iterations)
    position_errors = np.asarray(position_errors)
    orientation_errors = np.asarray(orientation_errors)
    violations = np.asarray(violations)
    actual_xyz = np.asarray(actual_xyz)
    reference_xyz = np.asarray(reference_xyz)
    q_history = np.asarray(q_history)
    v_history = np.asarray(v_history)

    print('\n' + '=' * 65)
    print('MPC STATISTICS (solve + update; excludes reference generation, plotting, Meshcat)')
    print('=' * 65)
    print(f'Compute time [ms]: mean={cycle_ms.mean():.2f}, median={np.median(cycle_ms):.2f}, '
          f'p95={np.percentile(cycle_ms,95):.2f}, max={cycle_ms.max():.2f}')
    print(f'Deadline ({1000*args.dt:.2f} ms) missed: '
          f'{np.count_nonzero(cycle_ms > 1000*args.dt)}/{len(cycle_ms)} '
          f'({100*np.mean(cycle_ms > 1000*args.dt):.1f}%)')
    print(f'GN iterations: mean={iterations.mean():.1f}, '
          f'median={np.median(iterations):.1f}, max={iterations.max()}')
    print(f'Position error: RMS={1000*np.sqrt(np.mean(position_errors**2)):.2f}mm, '
          f'max={1000*position_errors.max():.2f}mm')
    print(f'Orientation error: RMS={np.rad2deg(np.sqrt(np.mean(orientation_errors**2))):.2f}deg, '
          f'max={np.rad2deg(orientation_errors.max()):.2f}deg')
    print(
        f'Max linear constraint violation at planning grid: {violations.max():.3e}')
    print('Status counts:', {s: statuses.count(s)
          for s in sorted(set(statuses))})

    with (args.output / 'cycles.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['t_s', 'solve_ms', 'gn_iterations', 'status', 'position_error_m',
                         'orientation_error_rad', 'max_linear_violation'])
        writer.writerows(zip(times, cycle_ms, iterations, statuses,
                             position_errors, orientation_errors, violations))
    save_plots(args.output, times, actual_xyz, reference_xyz,
               position_errors, orientation_errors, cycle_ms, iterations,
               statuses, q_history, v_history, violations, args.dt)
    print(f'Plots + CSV saved in: {args.output.resolve()}')

    if not args.no_animation:
        print('Opening Meshcat; replaying executed movement (excluded from timing)...')
        robot.init_viewer(open_browser=True)
        robot.display_target(pose_from_theta(robot, reference_joint(args.steps*args.dt)),
                             name='final_reference')
        configs = robot.joint_trajectory_to_configurations(q_history)
        robot.animate(configs, dt=args.dt)
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()


"""Kinova Gen3 offline B-spline trajectory optimization.

Compare joint-space QP, Cartesian Gauss-Newton and IPOPT.
"""

import argparse
import time

import numpy as np

from splineopt.robotics import Robot
from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.ipopt_solver import IpoptSolver

from examples.utils.kinova_helpers import (
    URDF_PATH,
    Q_MIN, Q_MAX, V_MIN, V_MAX, A_MIN, A_MAX,
    LAMBDA_SMOOTH,
    build_reference,
    joint_projection,
    evaluate_spline,
    interpolate_reference,
    fk_trajectory,
    trajectory_metrics,
    print_summary,
    plot_cartesian_3d,
    plot_cartesian_errors,
    plot_joint_quantities,
    plot_feasibility_usage,
)

from examples.utils.kinova_visualization import (
    animate_kinova_trajectories,
)

METHODS = ("joint", "gn-pos", "gn-pose", "ipopt")


def solve_method(name, problem, initial_controls, args):
    """Run one Cartesian optimizer from the same QP initialization."""

    if name.startswith("gn-"):
        solver = GaussNewtonSolver(
            damping_mode="adaptive",
            damping=1e-3,
            max_iterations=args.gn_iters,
        )

    elif name == "ipopt":
        solver = IpoptSolver(
            tolerance=1e-6,
            max_iterations=args.ipopt_iters,
            print_level=0,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")

    start = time.perf_counter()

    result = solver.solve(
        problem=problem,
        initial_control_points=initial_controls,
        verbose=args.verbose,
    )

    elapsed = time.perf_counter() - start

    print(f"\n{name}:")
    if name == "ipopt":
        print(f"  status: {result.status_message}")
        print(f"  success: {result.success}")
    else:
        print(f"  status: {result.status}")

    print(f"  iterations: {result.iterations}")
    print(f"  solve time: {elapsed:.3f} s")

    return result.control_points, elapsed


def run(args):
    # --------------------------------------------------------
    # 1. Robot and Cartesian reference
    # --------------------------------------------------------
    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    print(f"Robot: {robot.nv} DoF")
    print("Building Cartesian target trajectory...")

    times, theta_virtual, target_poses = build_reference(robot)

    # --------------------------------------------------------
    # 2. Sequential IK (nominal joint trajectory)
    # --------------------------------------------------------
    q_pin_virtual = robot.joint_trajectory_to_configurations(
        theta_virtual
    )

    ik = robot.ik_trajectory(
        target_poses,
        q0=q_pin_virtual[0],
        tolerance=1e-5,
        max_iterations=500,
        step_size=0.1,
        damping=1e-6,
        max_step_norm=1.0,
    )

    if not ik.success:
        raise RuntimeError("Sequential IK failed.")

    theta_ref = ik.theta

    print(f"IK maximum error: {np.max(ik.error_norms):.3e}")
    print(f"Nominal q2 max: {np.max(theta_ref[:, 1]):.3f} rad")
    print(f"Nominal q4 min: {np.min(theta_ref[:, 3]):.3f} rad")

    # --------------------------------------------------------
    # 3. Joint-space QP initialization
    # --------------------------------------------------------
    print("\nSolving joint-space QP...")

    start = time.perf_counter()
    spline, initial_controls, qp_result = joint_projection(
        robot, times, theta_ref,
        n_controls=args.control_points,
    )
    qp_time = time.perf_counter() - start

    print(f"QP status: {qp_result.info.status}")
    print(f"QP time: {qp_time:.3f} s")

    # --------------------------------------------------------
    # 4. Cartesian problems
    # --------------------------------------------------------
    common = dict(
        robot=robot,
        spline=spline,
        tracking_times=times,
        target_poses=target_poses,

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

        lambda_smooth=LAMBDA_SMOOTH,
        lambda_terminal=1.0,
    )

    problems = {}

    if "gn-pos" in args.methods:
        problems["gn-pos"] = CartesianTrajectoryProblem(
            **common, orientation_scale=0.0,
        )

    if "gn-pose" in args.methods or "ipopt" in args.methods:
        pose_problem = CartesianTrajectoryProblem(
            **common, orientation_scale=args.orientation_scale,
        )
        problems["gn-pose"] = pose_problem
        problems["ipopt"] = pose_problem

    # --------------------------------------------------------
    # 5. Solve selected methods
    # --------------------------------------------------------
    controls = {"joint": initial_controls}
    timings = {"joint": qp_time}

    for name in args.methods:
        if name == "joint":
            continue

        print(f"\nSolving {name}...")

        controls[name], timings[name] = solve_method(
            name, problems[name], initial_controls, args,
        )

    # --------------------------------------------------------
    # 6. Dense evaluation and metrics
    # --------------------------------------------------------
    dense_times = np.linspace(
        times[0], times[-1], args.dense_samples,
    )

    nominal_q, nominal_v, nominal_a = interpolate_reference(
        times, theta_ref, dense_times,
    )

    _, target_poses_dense, _ = fk_trajectory(
        robot, nominal_q,
    )

    results = {
        "nominal": trajectory_metrics(
            robot, target_poses_dense,
            nominal_q, nominal_v, nominal_a,
        ),
    }

    # QP is always computed, but only plotted if selected
    for name in args.methods:
        q, v, a = evaluate_spline(
            spline, controls[name], dense_times,
        )
        results[name] = trajectory_metrics(
            robot, target_poses_dense, q, v, a,
        )

    print_summary(results, dense_times, timings)

    # --------------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------------
    import matplotlib.pyplot as plt

    if not args.no_plots:
        # Generate all figures, then display them without blocking.
        plot_cartesian_3d(results)
        plot_cartesian_errors(dense_times, results)
        plot_joint_quantities(dense_times, results)
        plot_feasibility_usage(dense_times, results)

        plt.show(block=False)
        plt.pause(0.5)

    if not args.no_animation:
        animate_kinova_trajectories(
            robot=robot,
            trajectories={
                name: {
                    "q": result["q"],
                    "times": dense_times,
                    "label": name,
                }
                for name, result in results.items()
            },
            speed=args.animation_speed,
            title="KINOVA OFFLINE OPTIMIZATION",
        )

    elif not args.no_plots:
        # Without animation, keep the figures open until closed.
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--methods", nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Optimizers to compare.",
    )
    parser.add_argument(
        "--gn-iters", type=int, default=20,
    )
    parser.add_argument(
        "--ipopt-iters", type=int, default=200,
    )
    parser.add_argument(
        "--control-points", type=int, default=30,
    )
    parser.add_argument(
        "--dense-samples", type=int, default=700,
    )
    parser.add_argument(
        "--orientation-scale", type=float, default=0.10,
    )
    parser.add_argument(
        "--animation-speed", type=float, default=2.0,
    )
    parser.add_argument(
        "--no-animation", action="store_true",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )

    args = parser.parse_args()

    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate methods are not allowed.")

    run(args)


if __name__ == "__main__":
    main()

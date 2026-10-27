
"""Kinova collision-aware trajectory optimization.

Scenarios:
    avoidance : FREE vs collision-aware GN
    solvers   : collision-aware GN vs IPOPT

All geometry, metrics and visualization live in examples.utils.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from splineopt.bspline import BSpline
from splineopt.robotics import Robot
from splineopt.cartesian_problem import CartesianTrajectoryProblem
from splineopt.optimizer import GaussNewtonSolver
from splineopt.ipopt_solver import IpoptSolver

from examples.utils.kinova_helpers import (
    URDF_PATH, Q_MIN, Q_MAX, V_MIN, V_MAX, A_MIN, A_MAX,
    build_reference, joint_projection,
)
from examples.utils.collision_visualization import (
    make_collision_builder,
    place_avoidance_obstacle,
    place_solver_obstacle,
    evaluate_collision_results,
    print_collision_metrics,
    plot_collision_results,
    CAPSULES,
)

from examples.utils.kinova_visualization import (
    animate_kinova_trajectories,
)


def make_spline(duration, n_controls, n_dof=7):
    degree = 3
    count = n_controls-degree-1
    interior = np.linspace(0, duration, count+2)[1:-1]
    knots = np.r_[
        np.zeros(degree+1),
        interior,
        np.full(degree+1, duration),
    ]
    return BSpline(knots=knots, degree=degree, n_dof=n_dof)


def build_scenario(robot, scenario, args):
    """Construct the scenario without importing any old example."""

    if scenario == "avoidance":
        duration = 5.0
        times = np.linspace(0, duration, 61)

        q_start = np.array([
            0., .45, .15, -1.35, -.15, .75, .10,
        ])
        q_goal = np.array([
            .90, 1.40, -.60, -1.70, .80, 1.00, -.60,
        ])

        u = times / duration
        s = 10*u**3 - 15*u**4 + 6*u**5
        theta_ref = q_start + s[:, None]*(q_goal-q_start)

        q_pin = robot.joint_trajectory_to_configurations(theta_ref)
        poses = [robot.frame_pose(q) for q in q_pin]

        spline = make_spline(duration, 14, robot.nv)
        initial = np.tile(
            q_start, (spline.n_control_points, 1)
        )

        return {
            "times": times,
            "targets": poses,
            "spline": spline,
            "initial": initial,
            "q0": q_start,
            "orientation_scale": 0.0,
            "smoothness": 1e-5,
            "collision_samples": 50,
        }

    # Same 10-second Cartesian reference family as the
    # unified Kinova optimization example.
    times, theta_virtual, poses = build_reference(robot)

    q_pin = robot.joint_trajectory_to_configurations(theta_virtual)
    ik = robot.ik_trajectory(
        poses,
        q0=q_pin[0],
        tolerance=1e-5,
        max_iterations=500,
        step_size=0.1,
        damping=1e-6,
        max_step_norm=1.0,
    )
    if not ik.success:
        raise RuntimeError("Sequential IK failed.")

    spline, initial, qp_result = joint_projection(
        robot, times, ik.theta,
        n_controls=args.control_points,
    )
    print(f"QP initialization: {qp_result.info.status}")

    return {
        "times": times,
        "targets": poses,
        "spline": spline,
        "initial": initial,
        "q0": ik.theta[0],
        "orientation_scale": args.orientation_scale,
        "smoothness": 1e-5,
        "collision_samples": 35,
    }


def make_problem(robot, data, collision_builder=None):
    return CartesianTrajectoryProblem(
        robot=robot,
        spline=data["spline"],
        tracking_times=data["times"],
        target_poses=data["targets"],
        q0=data["q0"],
        v0=np.zeros(robot.nv),
        a0=np.zeros(robot.nv),
        vf=np.zeros(robot.nv),
        q_min=Q_MIN, q_max=Q_MAX,
        v_min=V_MIN, v_max=V_MAX,
        a_min=A_MIN, a_max=A_MAX,
        orientation_scale=data["orientation_scale"],
        lambda_smooth=data["smoothness"],
        lambda_terminal=1.0,
        collision_constraints=collision_builder,
    )


def solve_gn(problem, initial, iterations, verbose):
    solver = GaussNewtonSolver(
        damping_mode="adaptive",
        damping=1e-3,
        max_iterations=iterations,
        max_damping_retries=5,
        collision_slack_weight=1e4,
        collision_merit_weight=1e4,
    )

    start = time.perf_counter()
    result = solver.solve(
        problem=problem,
        initial_control_points=initial.copy(),
        verbose=verbose,
    )
    elapsed = time.perf_counter()-start

    print(
        f"GN: {result.status}, "
        f"{result.iterations} iterations, {elapsed:.3f} s"
    )
    return result.control_points, elapsed


def solve_ipopt(problem, initial, iterations, verbose):
    solver = IpoptSolver(
        tolerance=1e-6,
        max_iterations=iterations,
        print_level=5 if verbose else 0,
    )

    start = time.perf_counter()
    result = solver.solve(
        problem=problem,
        initial_control_points=initial.copy(),
        verbose=verbose,
    )
    elapsed = time.perf_counter()-start

    print(
        f"IPOPT: {result.status_message}, "
        f"{result.iterations} iterations, {elapsed:.3f} s"
    )
    return result.control_points, elapsed


def run(args):
    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    data = build_scenario(robot, args.scenario, args)
    spline = data["spline"]
    initial = data["initial"]

    solutions = {}
    timings = {}

    print("\n" + "="*75)
    print(f"COLLISION OPTIMIZATION — {args.scenario.upper()}")
    print("="*75)

    if args.scenario == "avoidance":
        print("\n1. Solving collision-free baseline")
        free_problem = make_problem(robot, data)
        free, elapsed = solve_gn(
            free_problem, initial, args.gn_iters, args.verbose
        )
        solutions["free"] = free
        timings["free"] = elapsed

        obstacle, info = place_avoidance_obstacle(
            robot, spline, free, data["q0"],
            safety=args.safety,
        )
    else:
        solutions["initial"] = initial
        timings["initial"] = 0.0

        obstacle, info = place_solver_obstacle(
            robot, spline, initial,
            safety=args.safety,
            radius=args.obstacle_radius,
        )

    print("\n2. Obstacle")
    print(f"  Capsule: {info['capsule']}")
    print(f"  Time: {info['time']:.3f} s")
    print(f"  Center: {np.round(obstacle.center, 4)}")
    print(f"  Radius: {obstacle.radius:.3f} m")
    print(f"  Safety distance: {args.safety:.3f} m")

    builder = make_collision_builder(
        robot, spline, [obstacle],
        safety=args.safety,
        samples=data["collision_samples"],
    )

    problem = make_problem(robot, data, builder)

    if "gn" in args.methods:
        print("\n3. Solving collision-aware GN")
        solutions["gn"], timings["gn"] = solve_gn(
            problem, initial, args.gn_iters, args.verbose
        )

    if "ipopt" in args.methods:
        print("\n4. Solving collision-aware IPOPT")
        solutions["ipopt"], timings["ipopt"] = solve_ipopt(
            problem, initial, args.ipopt_iters, args.verbose
        )

    dense_times, reference, results = evaluate_collision_results(
        robot=robot,
        spline=spline,
        controls_by_name=solutions,
        target_times=data["times"],
        target_poses=data["targets"],
        obstacle=obstacle,
        safety=args.safety,
        dense_samples=args.dense_samples,
    )

    print_collision_metrics(results, timings, args.safety)

    if not args.no_plots:
        plot_collision_results(
            dense_times, reference, results,
            obstacle, args.safety,
        )
        plt.show(block=False)

    if not args.no_animation:
        animate_kinova_trajectories(
            robot=robot,
            trajectories={
                name: {
                    "q": result["q"],
                    "times": dense_times,
                    "label": name.upper(),
                }
                for name, result in results.items()
            },
            obstacles=[obstacle],
            capsules=CAPSULES,
            speed=args.animation_speed,
            title="KINOVA COLLISION OPTIMIZATION",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--scenario",
        choices=("avoidance", "solvers"),
        default="avoidance",
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=("gn", "ipopt"),
        default=None,
    )
    parser.add_argument("--gn-iters", type=int, default=25)
    parser.add_argument("--ipopt-iters", type=int, default=200)
    parser.add_argument("--control-points", type=int, default=30)
    parser.add_argument("--orientation-scale", type=float, default=0.10)
    parser.add_argument("--dense-samples", type=int, default=400)
    parser.add_argument("--safety", type=float, default=0.020)
    parser.add_argument("--obstacle-radius", type=float, default=0.050)
    parser.add_argument("--animation-speed", type=float, default=2.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.methods is None:
        args.methods = (
            ["gn"] if args.scenario == "avoidance"
            else ["gn", "ipopt"]
        )

    if args.gn_iters <= 0 or args.ipopt_iters <= 0:
        parser.error("Iteration counts must be positive.")
    if args.safety < 0 or args.obstacle_radius <= 0:
        parser.error("Invalid collision parameters.")
    if args.animation_speed <= 0:
        parser.error("Animation speed must be positive.")

    run(args)


if __name__ == "__main__":
    main()

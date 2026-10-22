import time

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin


from splineopt.cartesian_problem import (
    CartesianTrajectoryProblem,
)

from splineopt.optimizer import (
    GaussNewtonSolver,
)
from splineopt.joint_optimizer import JointTrajectoryOptimizer

from splineopt.ipopt_solver import (
    IpoptSolver,
)

from splineopt.robotics import Robot


from examples.kinova_trajectory_projection import (
    URDF_PATH,
    Q_MIN,
    Q_MAX,
    V_MIN,
    V_MAX,
    A_MIN,
    A_MAX,
    LAMBDA_SMOOTH,
    N_DENSE_SAMPLES,
    build_virtual_joint_reference,
    build_cartesian_targets_from_virtual_reference,
    solve_bspline_projection,
    evaluate_spline,
    interpolate_joint_trajectory,
    forward_kinematics_trajectory,
    full_feasible_mask,
)


# =====================================================================
# Benchmark configuration
# =====================================================================


GAUSS_NEWTON_DAMPING = 1e-3
GAUSS_NEWTON_MAX_ITERATIONS = 20

IPOPT_TOLERANCE = 1e-6

IPOPT_SHORT_MAX_ITERATIONS = 20
IPOPT_LONG_MAX_ITERATIONS = 200

CARTESIAN_TERMINAL_WEIGHT = 1.0
ORIENTATION_SCALE = 0.10

FEASIBILITY_TOL = 1e-3


# ---------------------------------------------------------------------
# Timing repetitions
#
# During development:
#
#     TIMING_REPEATS = 1
#
# Final benchmark:
#
#     TIMING_REPEATS = 3
#
# Each solver/scenario pair is rerun TIMING_REPEATS times.
# ---------------------------------------------------------------------


TIMING_REPEATS = 1


# =====================================================================
# Methods
# =====================================================================


METHODS = [
    (
        "joint",
        "Joint-space QP",
    ),
    (
        "gn",
        "GN-20",
    ),
    (
        "ipopt20",
        "IPOPT-20",
    ),
    (
        "ipopt200",
        "IPOPT-200",
    ),
]


COLOR_NOMINAL = "C1"


METHOD_COLORS = {
    "Joint-space QP": "C2",
    "GN-20": "C3",
    "IPOPT-20": "C4",
    "IPOPT-200": "C0",
}


METHOD_LINESTYLES = {
    "Joint-space QP": "-",
    "GN-20": "-",
    "IPOPT-20": "--",
    "IPOPT-200": "-",
}


# =====================================================================
# Benchmark scenarios
# =====================================================================


SCENARIOS = [

    {
        "name": "S1 — baseline",

        "short_name": "S1",

        "perturbations": [],

        "q2_tighten": 0.00,

        "q4_tighten": 0.00,
    },

    {
        "name": "S2 — q2 constrained",

        "short_name": "S2",

        "perturbations": [
            (0, +0.10, 1),
        ],

        "q2_tighten": 0.10,

        "q4_tighten": 0.00,
    },

    {
        "name": "S3 — q4 constrained",

        "short_name": "S3",

        "perturbations": [
            (2, -0.10, 1),
        ],

        "q2_tighten": 0.00,

        "q4_tighten": 0.10,
    },

    {
        "name": "S4 — combined constraints",

        "short_name": "S4",

        "perturbations": [
            (0, +0.10, 1),
            (2, -0.10, 2),
            (4, +0.08, 1),
        ],

        "q2_tighten": 0.15,

        "q4_tighten": 0.15,
    },

    {
        "name": "S5 — difficult",

        "short_name": "S5",

        "perturbations": [
            (0, +0.15, 1),
            (2, -0.15, 2),
            (4, +0.12, 1),
            (5, -0.10, 2),
        ],

        "q2_tighten": 0.25,

        "q4_tighten": 0.20,
    },

]


# =====================================================================
# Generic utilities
# =====================================================================


def rms(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def acceleration_cost(
    acceleration,
    times,
):

    return float(
        np.trapezoid(
            np.sum(
                acceleration ** 2,
                axis=1,
            ),
            times,
        )
    )


def pose_tracking_errors(
    robot,
    q_pin_reference,
    q_pin_candidate,
):

    position_errors = []
    orientation_errors = []

    for (
        q_reference,
        q_candidate,
    ) in zip(
        q_pin_reference,
        q_pin_candidate,
    ):

        M_reference = (
            robot.frame_pose(
                q_reference
            )
        )

        M_candidate = (
            robot.frame_pose(
                q_candidate
            )
        )

        position_error = np.linalg.norm(
            M_candidate.translation
            - M_reference.translation
        )

        R_error = (
            M_candidate.rotation.T
            @ M_reference.rotation
        )

        orientation_error = np.linalg.norm(
            pin.log3(
                R_error
            )
        )

        position_errors.append(
            position_error
        )

        orientation_errors.append(
            orientation_error
        )

    return (
        np.asarray(
            position_errors
        ),
        np.asarray(
            orientation_errors
        ),
    )


# =====================================================================
# Timing
# =====================================================================


def timed_solve(
    solve_function,
    repeats=1,
):

    durations = []

    result = None

    for _ in range(
        repeats
    ):

        start = (
            time.perf_counter()
        )

        result = (
            solve_function()
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        durations.append(
            elapsed
        )

    durations = np.asarray(
        durations,
        dtype=float,
    )

    representative_time = float(
        np.median(
            durations
        )
    )

    return (
        result,
        representative_time,
        durations,
    )


# =====================================================================
# Scenario generation
# =====================================================================


def perturb_virtual_trajectory(
    times,
    theta_virtual,
    perturbations,
):

    theta = np.asarray(
        theta_virtual,
        dtype=float,
    ).copy()

    times = np.asarray(
        times,
        dtype=float,
    )

    tau = (
        (times - times[0])
        / (
            times[-1]
            - times[0]
        )
    )

    for (
        joint,
        amplitude,
        harmonic,
    ) in perturbations:

        profile = (
            amplitude
            * np.sin(
                harmonic
                * np.pi
                * tau
            )
        )

        theta[:, joint] += (
            profile
        )

    return theta


def build_scenario_limits(
    q0,
    q2_tighten,
    q4_tighten,
):

    q_min = np.asarray(
        Q_MIN,
        dtype=float,
    ).copy()

    q_max = np.asarray(
        Q_MAX,
        dtype=float,
    ).copy()

    q0 = np.asarray(
        q0,
        dtype=float,
    )

    boundary_margin = 0.05

    # Joint 2 upper bound

    requested_q2_max = (
        Q_MAX[1]
        - q2_tighten
    )

    q_max[1] = max(
        requested_q2_max,
        q0[1]
        + boundary_margin,
    )

    # Joint 4 lower bound

    requested_q4_min = (
        Q_MIN[3]
        + q4_tighten
    )

    q_min[3] = min(
        requested_q4_min,
        q0[3]
        - boundary_margin,
    )

    return (
        q_min,
        q_max,
    )


# =====================================================================
# Dense evaluation
# =====================================================================


def evaluate_solution(
    robot,
    spline,
    control_points,
    dense_times,
    q_pin_reference,
):

    (
        q,
        v,
        a,
    ) = evaluate_spline(
        spline,
        control_points,
        dense_times,
    )

    (
        q_pin,
        cartesian,
    ) = forward_kinematics_trajectory(
        robot,
        q,
    )

    (
        position_error,
        orientation_error,
    ) = pose_tracking_errors(
        robot,
        q_pin_reference,
        q_pin,
    )

    feasible_mask = (
        full_feasible_mask(
            q,
            v,
            a,
            atol=FEASIBILITY_TOL,
        )
    )

    return {

        "control_points":
            control_points,

        "q":
            q,

        "v":
            v,

        "a":
            a,

        "q_pin":
            q_pin,

        "cartesian":
            cartesian,

        "position_error":
            position_error,

        "orientation_error":
            orientation_error,

        "position_rms":
            rms(
                position_error
            ),

        "position_max":
            float(
                np.max(
                    position_error
                )
            ),

        "orientation_rms":
            rms(
                orientation_error
            ),

        "orientation_max":
            float(
                np.max(
                    orientation_error
                )
            ),

        "smoothness":
            acceleration_cost(
                a,
                dense_times,
            ),

        "feasible":
            bool(
                np.all(
                    feasible_mask
                )
            ),
    }


# =====================================================================
# Run one scenario
# =====================================================================


def run_scenario(
    robot,
    scenario,
):

    # =========================================================
    # Base reference
    # =========================================================

    (
        times,
        theta_virtual_base,
        _,
        _,
    ) = build_virtual_joint_reference(
        robot
    )

    # =========================================================
    # Scenario reference
    # =========================================================

    theta_virtual = (
        perturb_virtual_trajectory(
            times,
            theta_virtual_base,
            scenario[
                "perturbations"
            ],
        )
    )

    (
        q_pin_virtual,
        target_poses,
        _,
    ) = (
        build_cartesian_targets_from_virtual_reference(
            robot,
            theta_virtual,
        )
    )

    # =========================================================
    # Sequential IK
    # =========================================================

    ik_result = (
        robot.ik_trajectory(
            target_poses,
            q0=q_pin_virtual[0],
            tolerance=1e-5,
            max_iterations=500,
            step_size=0.1,
            damping=1e-6,
            max_step_norm=1.0,
        )
    )

    if not ik_result.success:

        raise RuntimeError(
            f"{scenario['name']}: "
            "sequential IK failed."
        )

    theta_ref = (
        ik_result.theta
    )

    # =========================================================
    # Scenario joint limits
    # =========================================================

    (
        q_min,
        q_max,
    ) = build_scenario_limits(
        q0=theta_ref[0],
        q2_tighten=(
            scenario[
                "q2_tighten"
            ]
        ),
        q4_tighten=(
            scenario[
                "q4_tighten"
            ]
        ),
    )

    # =========================================================
    # Build B-spline structure
    #
    # This setup is not timed.
    # =========================================================

    (
        spline,
        _,
        _,
    ) = solve_bspline_projection(
        robot,
        times,
        theta_ref,
    )

    # =========================================================
    # 1) Joint-space QP
    # =========================================================

    joint_solver = (
        JointTrajectoryOptimizer(
            spline=spline,
            lambda_smooth=(
                LAMBDA_SMOOTH
            ),
        )
    )

    def solve_joint():

        return joint_solver.solve(
            tracking_times=times,
            q_ref=theta_ref,

            q0=theta_ref[0],

            v0=np.zeros(
                robot.nv
            ),

            a0=np.zeros(
                robot.nv
            ),

            vf=np.zeros(
                robot.nv
            ),

            q_min=q_min,
            q_max=q_max,

            v_min=V_MIN,
            v_max=V_MAX,

            a_min=A_MIN,
            a_max=A_MAX,

            verbose=False,
        )

    (
        joint_result_tuple,
        joint_time,
        joint_timings,
    ) = timed_solve(
        solve_joint,
        repeats=TIMING_REPEATS,
    )

    (
        joint_control_points,
        joint_osqp_result,
    ) = joint_result_tuple

    # =========================================================
    # Shared Cartesian problem
    # =========================================================

    problem = (
        CartesianTrajectoryProblem(
            robot=robot,
            spline=spline,

            tracking_times=times,
            target_poses=target_poses,

            q0=theta_ref[0],

            v0=np.zeros(
                robot.nv
            ),

            a0=np.zeros(
                robot.nv
            ),

            vf=np.zeros(
                robot.nv
            ),

            q_min=q_min,
            q_max=q_max,

            v_min=V_MIN,
            v_max=V_MAX,

            a_min=A_MIN,
            a_max=A_MAX,

            orientation_scale=(
                ORIENTATION_SCALE
            ),

            lambda_smooth=(
                LAMBDA_SMOOTH
            ),

            lambda_terminal=(
                CARTESIAN_TERMINAL_WEIGHT
            ),
        )
    )

    # =========================================================
    # 2) Static Gauss-Newton
    # =========================================================

    gn_solver = (
        GaussNewtonSolver(
            damping_mode="static",

            damping=(
                GAUSS_NEWTON_DAMPING
            ),

            max_iterations=(
                GAUSS_NEWTON_MAX_ITERATIONS
            ),
        )
    )

    def solve_gn():

        return gn_solver.solve(
            problem=problem,

            initial_control_points=(
                joint_control_points
            ),

            verbose=False,
        )

    (
        gn_result,
        gn_time,
        gn_timings,
    ) = timed_solve(
        solve_gn,
        repeats=TIMING_REPEATS,
    )

    # =========================================================
    # 3) IPOPT — max 20 iterations
    # =========================================================

    ipopt20_solver = (
        IpoptSolver(
            tolerance=(
                IPOPT_TOLERANCE
            ),

            max_iterations=(
                IPOPT_SHORT_MAX_ITERATIONS
            ),

            print_level=0,

            options={
                "sb": "yes",
            },
        )
    )

    def solve_ipopt20():

        return ipopt20_solver.solve(
            problem=problem,

            initial_control_points=(
                joint_control_points
            ),

            verbose=False,
        )

    (
        ipopt20_result,
        ipopt20_time,
        ipopt20_timings,
    ) = timed_solve(
        solve_ipopt20,
        repeats=TIMING_REPEATS,
    )

    # =========================================================
    # 4) IPOPT — max 200 iterations
    # =========================================================

    ipopt200_solver = (
        IpoptSolver(
            tolerance=(
                IPOPT_TOLERANCE
            ),

            max_iterations=(
                IPOPT_LONG_MAX_ITERATIONS
            ),

            print_level=0,

            options={
                "sb": "yes",
            },
        )
    )

    def solve_ipopt200():

        return ipopt200_solver.solve(
            problem=problem,

            initial_control_points=(
                joint_control_points
            ),

            verbose=False,
        )

    (
        ipopt200_result,
        ipopt200_time,
        ipopt200_timings,
    ) = timed_solve(
        solve_ipopt200,
        repeats=TIMING_REPEATS,
    )

    # =========================================================
    # Dense reference
    # =========================================================

    dense_times = np.linspace(
        times[0],
        times[-1],
        N_DENSE_SAMPLES,
    )

    q_ref_dense = (
        interpolate_joint_trajectory(
            times,
            theta_ref,
            dense_times,
        )
    )

    (
        q_pin_ref_dense,
        cart_ref_dense,
    ) = (
        forward_kinematics_trajectory(
            robot,
            q_ref_dense,
        )
    )

    # =========================================================
    # Dense evaluation
    # =========================================================

    joint_data = (
        evaluate_solution(
            robot=robot,
            spline=spline,

            control_points=(
                joint_control_points
            ),

            dense_times=(
                dense_times
            ),

            q_pin_reference=(
                q_pin_ref_dense
            ),
        )
    )

    gn_data = (
        evaluate_solution(
            robot=robot,
            spline=spline,

            control_points=(
                gn_result.control_points
            ),

            dense_times=(
                dense_times
            ),

            q_pin_reference=(
                q_pin_ref_dense
            ),
        )
    )

    ipopt20_data = (
        evaluate_solution(
            robot=robot,
            spline=spline,

            control_points=(
                ipopt20_result.control_points
            ),

            dense_times=(
                dense_times
            ),

            q_pin_reference=(
                q_pin_ref_dense
            ),
        )
    )

    ipopt200_data = (
        evaluate_solution(
            robot=robot,
            spline=spline,

            control_points=(
                ipopt200_result.control_points
            ),

            dense_times=(
                dense_times
            ),

            q_pin_reference=(
                q_pin_ref_dense
            ),
        )
    )

    # =========================================================
    # Solver information
    # =========================================================

    joint_data.update({

        "solve_time":
            joint_time,

        "timings":
            joint_timings,

        "iterations":
            1,

        "status":
            joint_osqp_result.info.status,
    })

    gn_data.update({

        "solve_time":
            gn_time,

        "timings":
            gn_timings,

        "iterations":
            gn_result.iterations,

        "status":
            gn_result.status,
    })

    ipopt20_data.update({

        "solve_time":
            ipopt20_time,

        "timings":
            ipopt20_timings,

        "iterations":
            ipopt20_result.iterations,

        "status":
            ipopt20_result.status_message,
    })

    ipopt200_data.update({

        "solve_time":
            ipopt200_time,

        "timings":
            ipopt200_timings,

        "iterations":
            ipopt200_result.iterations,

        "status":
            ipopt200_result.status_message,
    })

    return {

        "name":
            scenario[
                "name"
            ],

        "short_name":
            scenario[
                "short_name"
            ],

        "times":
            times,

        "dense_times":
            dense_times,

        "theta_ref":
            theta_ref,

        "q_ref_dense":
            q_ref_dense,

        "cart_ref_dense":
            cart_ref_dense,

        "q_min":
            q_min,

        "q_max":
            q_max,

        "joint":
            joint_data,

        "gn":
            gn_data,

        "ipopt20":
            ipopt20_data,

        "ipopt200":
            ipopt200_data,
    }


# =====================================================================
# Console reporting
# =====================================================================


def print_constraint_summary(
    benchmark_results,
):

    print()
    print("=" * 82)
    print(
        "SCENARIO ANGULAR CONSTRAINTS"
    )
    print("=" * 82)

    print(
        f"{'Scenario':<32}"
        f"{'q2 max [rad]':>18}"
        f"{'q4 min [rad]':>18}"
    )

    print("-" * 82)

    for scenario in (
        benchmark_results
    ):

        print(
            f"{scenario['name']:<32}"
            f"{scenario['q_max'][1]:>18.3f}"
            f"{scenario['q_min'][3]:>18.3f}"
        )


def print_quality_table(
    benchmark_results,
):

    print()
    print("=" * 122)
    print(
        "TRACKING QUALITY"
    )
    print("=" * 122)

    print(
        f"{'Scenario':<27}"
        f"{'Method':<18}"
        f"{'pos RMS [mm]':>14}"
        f"{'pos max [mm]':>14}"
        f"{'ori RMS [deg]':>15}"
        f"{'ori max [deg]':>15}"
        f"{'smoothness':>13}"
        f"{'feasible':>10}"
    )

    print("-" * 122)

    for scenario in (
        benchmark_results
    ):

        first = True

        for (
            key,
            method_name,
        ) in METHODS:

            data = (
                scenario[
                    key
                ]
            )

            scenario_name = (
                scenario[
                    "name"
                ]
                if first
                else ""
            )

            first = False

            print(
                f"{scenario_name:<27}"
                f"{method_name:<18}"
                f"{1e3 * data['position_rms']:>14.2f}"
                f"{1e3 * data['position_max']:>14.2f}"
                f"{np.degrees(data['orientation_rms']):>15.2f}"
                f"{np.degrees(data['orientation_max']):>15.2f}"
                f"{data['smoothness']:>13.3f}"
                f"{str(data['feasible']):>10}"
            )

        print("-" * 122)


def print_solver_table(
    benchmark_results,
):

    print()
    print("=" * 102)
    print(
        "SOLVER RESULTS"
    )
    print("=" * 102)

    print(
        f"{'Scenario':<27}"
        f"{'Method':<18}"
        f"{'time [s]':>12}"
        f"{'iterations':>13}"
        f"{'status':>31}"
    )

    print("-" * 102)

    for scenario in (
        benchmark_results
    ):

        first = True

        for (
            key,
            method_name,
        ) in METHODS:

            data = (
                scenario[
                    key
                ]
            )

            scenario_name = (
                scenario[
                    "name"
                ]
                if first
                else ""
            )

            first = False

            status = str(
                data[
                    "status"
                ]
            )

            if len(
                status
            ) > 29:

                status = (
                    status[:26]
                    + "..."
                )

            print(
                f"{scenario_name:<27}"
                f"{method_name:<18}"
                f"{data['solve_time']:>12.4f}"
                f"{data['iterations']:>13}"
                f"{status:>31}"
            )

        print("-" * 102)


def print_timing_statistics(
    benchmark_results,
):

    print()
    print("=" * 94)
    print(
        "SOLVE-TIME STATISTICS ACROSS SCENARIOS"
    )
    print("=" * 94)

    print(
        f"{'Method':<22}"
        f"{'mean [s]':>14}"
        f"{'median [s]':>14}"
        f"{'std [s]':>14}"
        f"{'min [s]':>14}"
        f"{'max [s]':>14}"
    )

    print("-" * 94)

    for (
        key,
        method_name,
    ) in METHODS:

        values = np.asarray(
            [
                scenario[
                    key
                ][
                    "solve_time"
                ]

                for scenario
                in benchmark_results
            ],
            dtype=float,
        )

        print(
            f"{method_name:<22}"
            f"{np.mean(values):>14.4f}"
            f"{np.median(values):>14.4f}"
            f"{np.std(values):>14.4f}"
            f"{np.min(values):>14.4f}"
            f"{np.max(values):>14.4f}"
        )


# =====================================================================
# 3D trajectories
# =====================================================================


def plot_3d_trajectories(
    benchmark_results,
):

    fig = plt.figure(
        figsize=(18, 10)
    )

    for (
        index,
        scenario,
    ) in enumerate(
        benchmark_results
    ):

        ax = fig.add_subplot(
            2,
            3,
            index + 1,
            projection="3d",
        )

        reference = (
            scenario[
                "cart_ref_dense"
            ]
        )

        ax.plot(
            reference[:, 0],
            reference[:, 1],
            reference[:, 2],
            "--",
            linewidth=2.0,
            color=COLOR_NOMINAL,
            label="Nominal",
        )

        for (
            key,
            method_name,
        ) in METHODS:

            trajectory = (
                scenario[
                    key
                ][
                    "cartesian"
                ]
            )

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                linewidth=2.0,
                linestyle=(
                    METHOD_LINESTYLES[
                        method_name
                    ]
                ),
                color=(
                    METHOD_COLORS[
                        method_name
                    ]
                ),
                label=method_name,
            )

        all_points = np.vstack(
            [reference]
            + [
                scenario[
                    key
                ][
                    "cartesian"
                ]

                for (
                    key,
                    _
                ) in METHODS
            ]
        )

        xyz_min = np.min(
            all_points,
            axis=0,
        )

        xyz_max = np.max(
            all_points,
            axis=0,
        )

        ranges = np.maximum(
            xyz_max
            - xyz_min,
            1e-3,
        )

        center = (
            0.5
            * (
                xyz_min
                + xyz_max
            )
        )

        half_range = (
            0.55
            * np.max(
                ranges
            )
        )

        ax.set_xlim(
            center[0]
            - half_range,
            center[0]
            + half_range,
        )

        ax.set_ylim(
            center[1]
            - half_range,
            center[1]
            + half_range,
        )

        ax.set_zlim(
            center[2]
            - half_range,
            center[2]
            + half_range,
        )

        ax.set_box_aspect(
            (1, 1, 1)
        )

        ax.set_xlabel(
            "x [m]"
        )

        ax.set_ylabel(
            "y [m]"
        )

        ax.set_zlabel(
            "z [m]"
        )

        ax.set_title(
            scenario[
                "name"
            ]
        )

        if index == 0:

            ax.legend(
                fontsize=8,
                loc="best",
            )

    if len(
        benchmark_results
    ) < 6:

        ax_unused = (
            fig.add_subplot(
                2,
                3,
                6,
            )
        )

        ax_unused.axis(
            "off"
        )

    fig.suptitle(
        "Cartesian trajectory benchmark",
        fontsize=16,
    )

    fig.tight_layout()


# =====================================================================
# Tracking errors
# =====================================================================


def plot_tracking_errors(
    benchmark_results,
):

    n_scenarios = len(
        benchmark_results
    )

    fig, axes = plt.subplots(
        n_scenarios,
        2,

        figsize=(
            15,
            2.5
            * n_scenarios,
        ),

        sharex="col",
    )

    if n_scenarios == 1:

        axes = np.asarray(
            [axes]
        )

    axes[0, 0].set_title(
        "Position tracking error",
        fontsize=13,
    )

    axes[0, 1].set_title(
        "Orientation tracking error",
        fontsize=13,
    )

    legend_handles = None
    legend_labels = None

    for (
        row,
        scenario,
    ) in enumerate(
        benchmark_results
    ):

        times = (
            scenario[
                "dense_times"
            ]
        )

        position_ax = (
            axes[
                row,
                0
            ]
        )

        orientation_ax = (
            axes[
                row,
                1
            ]
        )

        for (
            key,
            method_name,
        ) in METHODS:

            data = (
                scenario[
                    key
                ]
            )

            position_ax.plot(
                times,

                1000.0
                * data[
                    "position_error"
                ],

                linewidth=1.7,

                linestyle=(
                    METHOD_LINESTYLES[
                        method_name
                    ]
                ),

                color=(
                    METHOD_COLORS[
                        method_name
                    ]
                ),

                label=method_name,
            )

            orientation_ax.plot(
                times,

                np.degrees(
                    data[
                        "orientation_error"
                    ]
                ),

                linewidth=1.7,

                linestyle=(
                    METHOD_LINESTYLES[
                        method_name
                    ]
                ),

                color=(
                    METHOD_COLORS[
                        method_name
                    ]
                ),

                label=method_name,
            )

        position_ax.text(
            -0.13,
            0.5,

            scenario[
                "short_name"
            ],

            transform=(
                position_ax.transAxes
            ),

            rotation=90,

            va="center",

            ha="center",

            fontsize=11,

            fontweight="bold",
        )

        position_ax.set_ylabel(
            "Error [mm]"
        )

        orientation_ax.set_ylabel(
            "Error [deg]"
        )

        position_ax.grid(
            alpha=0.25
        )

        orientation_ax.grid(
            alpha=0.25
        )

        if row == (
            n_scenarios
            - 1
        ):

            position_ax.set_xlabel(
                "Time [s]"
            )

            orientation_ax.set_xlabel(
                "Time [s]"
            )

        if row == 0:

            (
                legend_handles,
                legend_labels,
            ) = (
                position_ax
                .get_legend_handles_labels()
            )

    fig.legend(
        legend_handles,
        legend_labels,

        loc="upper center",

        ncol=len(
            METHODS
        ),

        bbox_to_anchor=(
            0.5,
            0.995,
        ),

        frameon=True,
    )

    fig.subplots_adjust(
        top=0.93,
        bottom=0.06,
        left=0.09,
        right=0.98,
        hspace=0.20,
        wspace=0.16,
    )


# =====================================================================
# Timing boxplot
# =====================================================================


def plot_solve_time_boxplot(
    benchmark_results,
):

    labels = []

    values = []

    colors = []

    for (
        key,
        method_name,
    ) in METHODS:

        method_times = np.asarray(
            [
                scenario[
                    key
                ][
                    "solve_time"
                ]

                for scenario
                in benchmark_results
            ],
            dtype=float,
        )

        labels.append(
            method_name
        )

        values.append(
            method_times
        )

        colors.append(
            METHOD_COLORS[
                method_name
            ]
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    box = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        widths=0.55,
    )

    for (
        patch,
        color,
    ) in zip(
        box[
            "boxes"
        ],
        colors,
    ):

        patch.set_facecolor(
            color
        )

        patch.set_alpha(
            0.35
        )

    # ---------------------------------------------------------
    # Overlay individual scenario values
    # ---------------------------------------------------------

    for index, (
        key,
        method_name,
    ) in enumerate(
        METHODS,
        start=1,
    ):

        times = np.asarray(
            [
                scenario[
                    key
                ][
                    "solve_time"
                ]

                for scenario
                in benchmark_results
            ]
        )

        ax.scatter(
            np.full(
                len(times),
                index,
            ),
            times,
            color=(
                METHOD_COLORS[
                    method_name
                ]
            ),
            s=35,
            zorder=5,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_ylabel(
        "Solve wall time [s]"
    )

    ax.set_title(
        "Solver computation time across scenarios"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()


# =====================================================================
# Tracking quality summary
# =====================================================================


def plot_quality_summary(
    benchmark_results,
):

    scenario_names = [
        scenario[
            "short_name"
        ]

        for scenario
        in benchmark_results
    ]

    x = np.arange(
        len(
            benchmark_results
        )
    )

    width = 0.18

    offsets = np.linspace(
        -1.5 * width,
        +1.5 * width,
        len(
            METHODS
        ),
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for (
        offset,
        (
            key,
            method_name,
        ),
    ) in zip(
        offsets,
        METHODS,
    ):

        position_values = [
            1000.0
            * scenario[
                key
            ][
                "position_rms"
            ]

            for scenario
            in benchmark_results
        ]

        orientation_values = [
            np.degrees(
                scenario[
                    key
                ][
                    "orientation_rms"
                ]
            )

            for scenario
            in benchmark_results
        ]

        axes[0].bar(
            x + offset,
            position_values,

            width=width,

            label=method_name,

            color=(
                METHOD_COLORS[
                    method_name
                ]
            ),
        )

        axes[1].bar(
            x + offset,
            orientation_values,

            width=width,

            label=method_name,

            color=(
                METHOD_COLORS[
                    method_name
                ]
            ),
        )

    axes[0].set_ylabel(
        "Position RMS [mm]"
    )

    axes[0].set_title(
        "Position tracking"
    )

    axes[1].set_ylabel(
        "Orientation RMS [deg]"
    )

    axes[1].set_title(
        "Orientation tracking"
    )

    axes[1].set_xlabel(
        "Scenario"
    )

    axes[1].set_xticks(
        x
    )

    axes[1].set_xticklabels(
        scenario_names
    )

    axes[0].legend(
        loc="upper center",
        ncol=len(
            METHODS
        ),
    )

    for ax in axes:

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Tracking quality summary"
    )

    fig.tight_layout()


# =====================================================================
# Quality / computation trade-off
# =====================================================================


def plot_quality_time_tradeoff(
    benchmark_results,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    for (
        key,
        method_name,
    ) in METHODS:

        solve_times = np.asarray(
            [
                scenario[
                    key
                ][
                    "solve_time"
                ]

                for scenario
                in benchmark_results
            ]
        )

        position_rms = np.asarray(
            [
                1000.0
                * scenario[
                    key
                ][
                    "position_rms"
                ]

                for scenario
                in benchmark_results
            ]
        )

        orientation_rms = np.asarray(
            [
                np.degrees(
                    scenario[
                        key
                    ][
                        "orientation_rms"
                    ]
                )

                for scenario
                in benchmark_results
            ]
        )

        median_time = float(
            np.median(
                solve_times
            )
        )

        mean_position = float(
            np.mean(
                position_rms
            )
        )

        mean_orientation = float(
            np.mean(
                orientation_rms
            )
        )

        axes[0].scatter(
            median_time,
            mean_position,

            s=90,

            color=(
                METHOD_COLORS[
                    method_name
                ]
            ),

            label=method_name,
        )

        axes[0].annotate(
            method_name,

            (
                median_time,
                mean_position,
            ),

            xytext=(
                6,
                5,
            ),

            textcoords=(
                "offset points"
            ),
        )

        axes[1].scatter(
            median_time,
            mean_orientation,

            s=90,

            color=(
                METHOD_COLORS[
                    method_name
                ]
            ),

            label=method_name,
        )

        axes[1].annotate(
            method_name,

            (
                median_time,
                mean_orientation,
            ),

            xytext=(
                6,
                5,
            ),

            textcoords=(
                "offset points"
            ),
        )

    for ax in axes:

        ax.set_xscale(
            "log"
        )

        ax.set_xlabel(
            "Median solve wall time [s]"
        )

        ax.grid(
            alpha=0.25
        )

    axes[0].set_ylabel(
        "Mean position RMS [mm]"
    )

    axes[0].set_title(
        "Position accuracy vs computation"
    )

    axes[1].set_ylabel(
        "Mean orientation RMS [deg]"
    )

    axes[1].set_title(
        "Orientation accuracy vs computation"
    )

    fig.suptitle(
        "Tracking quality / computation trade-off"
    )

    fig.tight_layout()


# =====================================================================
# Main
# =====================================================================


def main():

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame=(
            "end_effector_link"
        ),
    )

    print()
    print(
        "Kinova solver benchmark"
    )

    print(
        "======================="
    )

    print(
        f"Scenarios: "
        f"{len(SCENARIOS)}"
    )

    print(
        f"Timing repeats: "
        f"{TIMING_REPEATS}"
    )

    print()
    print(
        "Methods:"
    )

    for (
        _,
        method_name,
    ) in METHODS:

        print(
            f"  - {method_name}"
        )

    print()

    benchmark_results = []

    for (
        index,
        scenario,
    ) in enumerate(
        SCENARIOS
    ):

        print(
            f"[{index + 1}/"
            f"{len(SCENARIOS)}] "
            f"{scenario['name']}"
        )

        result = (
            run_scenario(
                robot,
                scenario,
            )
        )

        benchmark_results.append(
            result
        )

    # =========================================================
    # Console reporting
    # =========================================================

    print_constraint_summary(
        benchmark_results
    )

    print_quality_table(
        benchmark_results
    )

    print_solver_table(
        benchmark_results
    )

    print_timing_statistics(
        benchmark_results
    )

    # =========================================================
    # Figures
    # =========================================================

    plot_3d_trajectories(
        benchmark_results
    )

    plot_tracking_errors(
        benchmark_results
    )

    plot_solve_time_boxplot(
        benchmark_results
    )

    plot_quality_summary(
        benchmark_results
    )

    plot_quality_time_tradeoff(
        benchmark_results
    )

    plt.show()


if __name__ == "__main__":
    main()

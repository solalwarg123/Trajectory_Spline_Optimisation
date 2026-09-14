from pathlib import Path
import time

import numpy as np

from splineopt.robotics import Robot


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


def main():

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    # ------------------------------------------------------------------
    # Starting configuration
    # ------------------------------------------------------------------

    q0 = robot.neutral()

    offset = np.zeros(robot.nv)

    offset[:] = [
        0.20,
        -0.30,
        0.15,
        -0.25,
        0.20,
        0.15,
        -0.10,
    ]

    q_start = robot.integrate(
        q0,
        offset,
    )

    # ------------------------------------------------------------------
    # Construct a reachable target
    # ------------------------------------------------------------------

    delta = np.zeros(robot.nv)

    delta[:] = [
        0.20,
        -0.15,
        0.10,
        0.10,
        -0.10,
        0.10,
        0.05,
    ]

    q_target = robot.integrate(
        q_start,
        delta,
    )

    target_pose = robot.frame_pose(
        q_target
    )

    print("Target position:")
    print(target_pose.translation)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    result = robot.ik(
        target_pose,
        q0=q_start,
    )

    print()
    print(f"Success:    {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Error:      {result.error_norm:.3e}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    robot.init_viewer()

    print()
    print("Displaying initial configuration...")

    robot.display(q_start)

    time.sleep(2.0)

    print("Displaying IK solution...")

    robot.display(result.q)

    input(
        "Press Enter to quit..."
    )


if __name__ == "__main__":
    main()

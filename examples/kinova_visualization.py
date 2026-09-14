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

    # ------------------------------------------------------------------
    # Load robot
    # ------------------------------------------------------------------

    robot = Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )

    print("=" * 60)
    print("KINOVA GEN3")
    print("=" * 60)

    print()
    print(robot)

    print()
    print(f"nq = {robot.nq}")
    print(f"nv = {robot.nv}")

    # ------------------------------------------------------------------
    # Joints
    # ------------------------------------------------------------------

    print()
    print("Joints:")

    for i, name in enumerate(robot.joint_names):
        print(f"  {i}: {name}")

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    print()
    print("Frames:")

    for i, name in enumerate(robot.frame_names):
        print(f"  {i}: {name}")

    # ------------------------------------------------------------------
    # Neutral configuration
    # ------------------------------------------------------------------

    q0 = robot.neutral()

    print()
    print("Neutral configuration:")
    print(q0)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    pose = robot.frame_pose(q0)

    print()
    print("End-effector position:")
    print(pose.translation)

    print()
    print("End-effector rotation:")
    print(pose.rotation)

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    J = robot.frame_jacobian(q0)

    print()
    print("Jacobian shape:")
    print(J.shape)

    # ------------------------------------------------------------------
    # Start Meshcat
    # ------------------------------------------------------------------

    print()
    print("Starting Meshcat...")

    robot.init_viewer(
        open_browser=True,
    )

    robot.display(q0)

    time.sleep(1.0)

    # ------------------------------------------------------------------
    # Move first joint
    # ------------------------------------------------------------------

    print("Moving joint 1...")

    for angle in np.linspace(
        0.0,
        np.deg2rad(45.0),
        100,
    ):
        v = np.zeros(robot.nv)
        v[0] = angle

        q = robot.integrate(
            q0,
            v,
        )

        robot.display(q)

        time.sleep(0.01)

    q1 = q.copy()

    # ------------------------------------------------------------------
    # Move second joint
    # ------------------------------------------------------------------

    print("Moving joint 2...")

    for angle in np.linspace(
        0.0,
        np.deg2rad(-45.0),
        100,
    ):
        v = np.zeros(robot.nv)
        v[1] = angle

        q = robot.integrate(
            q1,
            v,
        )

        robot.display(q)

        time.sleep(0.01)

    # ------------------------------------------------------------------
    # Final pose
    # ------------------------------------------------------------------

    pose_final = robot.frame_pose(q)

    print()
    print("Final end-effector position:")
    print(pose_final.translation)

    print()
    print("Done.")

    input("Press Enter to quit...")


if __name__ == "__main__":
    main()

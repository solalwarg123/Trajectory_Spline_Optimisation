import time

import numpy as np
import pinocchio as pin
import meshcat.geometry as geometry

from pinocchio.visualize import MeshcatVisualizer


def create_meshcat_viewer(
    robot,
    open_browser=True,
):
    """
    Create and initialize a Meshcat viewer for a Robot.
    """

    viewer = MeshcatVisualizer(
        robot.model,
        robot.collision_model,
        robot.visual_model,
    )

    viewer.initViewer(
        open=open_browser,
    )

    viewer.loadViewerModel(
        rootNodeName="robot",
    )

    viewer.displayCollisions(False)
    viewer.displayVisuals(True)

    return viewer


def display_pose(
    viewer,
    pose,
    name="target",
    scale=0.12,
):
    """
    Display an SE(3) pose as an XYZ coordinate frame in Meshcat.

    Parameters
    ----------
    viewer:
        Pinocchio MeshcatVisualizer.

    pose:
        Pose to display as pin.SE3.

    name:
        Name of the object in the Meshcat scene.

    scale:
        Length of the XYZ axes.
    """

    if not isinstance(pose, pin.SE3):
        raise TypeError(
            "pose must be a pinocchio.SE3 object."
        )

    node = viewer.viewer["targets"][name]

    node.set_object(
        geometry.triad(scale)
    )

    node.set_transform(
        pose.homogeneous
    )


def remove_pose(
    viewer,
    name="target",
):
    """
    Remove a displayed pose from Meshcat.
    """

    viewer.viewer["targets"][name].delete()


def animate_configurations(
    viewer,
    q_sequence,
    dt=0.03,
):
    """
    Sequentially display robot configurations.

    Parameters
    ----------
    viewer:
        Pinocchio MeshcatVisualizer.

    q_sequence:
        Array-like of shape (N, nq).

    dt:
        Delay between configurations in seconds.
    """

    q_sequence = np.asarray(
        q_sequence,
        dtype=float,
    )

    if q_sequence.ndim != 2:
        raise ValueError(
            "q_sequence must have shape (N, nq)."
        )

    for q in q_sequence:
        viewer.display(q)
        time.sleep(dt)

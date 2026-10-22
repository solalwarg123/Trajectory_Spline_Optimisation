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
        Positive length of the XYZ axes, in meters.
    """

    if not isinstance(pose, pin.SE3):
        raise TypeError(
            "pose must be a pinocchio.SE3 object."
        )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")

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
    Animate a sequence of robot configurations in Meshcat.

    Parameters
    ----------
    viewer : pinocchio.visualize.MeshcatVisualizer
        Initialized robot viewer.
    q_sequence : array-like, shape (N, nq)
        Sequence of configurations in Pinocchio coordinates.
    dt : float
        Delay between consecutive configurations in seconds.

    Notes
    -----
    Animation uses time.sleep and is therefore blocking.
    It is intended for visualization, not real-time control.
    """
    q_sequence = np.asarray(q_sequence, dtype=float)

    if q_sequence.ndim != 2:
        raise ValueError("q_sequence must have shape (N, nq).")

    if q_sequence.shape[1] != viewer.model.nq:
        raise ValueError(
            f"Expected {viewer.model.nq} configuration values, "
            f"got {q_sequence.shape[1]}."
        )

    if not np.all(np.isfinite(q_sequence)):
        raise ValueError("q_sequence must contain finite values.")

    dt = float(dt)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative.")

    for q in q_sequence:
        viewer.display(q)
        time.sleep(dt)

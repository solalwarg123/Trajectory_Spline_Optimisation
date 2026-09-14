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

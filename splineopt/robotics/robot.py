from pathlib import Path

import numpy as np
import pinocchio as pin


class Robot:
    """
    Robot model based on Pinocchio.

    Provides basic kinematic operations and optional Meshcat visualization.
    """

    def __init__(
        self,
        urdf_path,
        end_effector_frame=None,
    ):
        self.urdf_path = Path(urdf_path).resolve()

        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found:\n{self.urdf_path}"
            )

        # ------------------------------------------------------------------
        # Load Pinocchio model + geometry
        # ------------------------------------------------------------------

        mesh_dir = self.urdf_path.parent

        (
            self.model,
            self.collision_model,
            self.visual_model,
        ) = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            str(mesh_dir),
        )

        (
            self.data,
            self.collision_data,
            self.visual_data,
        ) = pin.createDatas(
            self.model,
            self.collision_model,
            self.visual_model,
        )

        # ------------------------------------------------------------------
        # Basic information
        # ------------------------------------------------------------------

        self.nq = self.model.nq
        self.nv = self.model.nv

        self.q_lower = self.model.lowerPositionLimit.copy()
        self.q_upper = self.model.upperPositionLimit.copy()
        self.v_limit = self.model.velocityLimit.copy()

        self.joint_names = list(self.model.names)[1:]

        self.frame_names = [
            frame.name
            for frame in self.model.frames
        ]

        # ------------------------------------------------------------------
        # Default end-effector
        # ------------------------------------------------------------------

        self.end_effector_frame = end_effector_frame

        if end_effector_frame is not None:
            if end_effector_frame not in self.frame_names:
                raise ValueError(
                    f"Unknown end-effector frame: {end_effector_frame}\n"
                    f"Available frames:\n"
                    + "\n".join(self.frame_names)
                )

            self.end_effector_frame_id = self.model.getFrameId(
                end_effector_frame
            )

        else:
            self.end_effector_frame_id = None

        # ------------------------------------------------------------------
        # Visualization
        # ------------------------------------------------------------------

        self.viewer = None

    # ======================================================================
    # Configurations
    # ======================================================================

    def neutral(self):
        """Return the neutral configuration."""
        return pin.neutral(self.model)

    def integrate(self, q, v, dt=1.0):
        """
        Integrate a velocity / tangent displacement from q.

        Do not assume nq == nv.
        """

        q = self._check_configuration(q)

        v = np.asarray(v, dtype=float).reshape(-1)

        if v.shape != (self.nv,):
            raise ValueError(
                f"Expected v shape ({self.nv},), got {v.shape}."
            )

        return pin.integrate(
            self.model,
            q,
            dt * v,
        )

    # ======================================================================
    # Frames
    # ======================================================================

    def frame_id(self, frame=None):
        """
        Return the Pinocchio id of a frame.

        If frame is None, use the default end-effector.
        """

        if frame is None:
            if self.end_effector_frame_id is None:
                raise ValueError(
                    "No frame specified and no default "
                    "end-effector configured."
                )

            return self.end_effector_frame_id

        if frame not in self.frame_names:
            raise ValueError(
                f"Unknown frame: {frame}"
            )

        return self.model.getFrameId(frame)

    # ======================================================================
    # Forward kinematics
    # ======================================================================

    def frame_pose(self, q, frame=None):
        """
        Return the world pose of a frame as a pin.SE3.
        """

        q = self._check_configuration(q)
        frame_id = self.frame_id(frame)

        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        return self.data.oMf[frame_id].copy()

    def frame_position(self, q, frame=None):
        """
        Return the XYZ position of a frame.
        """

        return self.frame_pose(
            q,
            frame,
        ).translation.copy()

    # ======================================================================
    # Jacobian
    # ======================================================================

    def frame_jacobian(
        self,
        q,
        frame=None,
        reference_frame=pin.LOCAL_WORLD_ALIGNED,
    ):
        """
        Return the 6 x nv Jacobian of a frame.
        """

        q = self._check_configuration(q)
        frame_id = self.frame_id(frame)

        return pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            frame_id,
            reference_frame,
        ).copy()

    # ======================================================================
    # Inverse kinematics
    # ======================================================================

    def ik(
        self,
        target_pose,
        q0=None,
        frame=None,
        **kwargs,
    ):
        """
        Solve full pose inverse kinematics.

        Parameters
        ----------
        target_pose:
            Desired pose as pin.SE3.

        q0:
            Initial configuration.
            If None, the neutral configuration is used.

        frame:
            Frame to control.
            If None, the default end-effector is used.

        kwargs:
            Additional parameters passed to solve_pose_ik().
        """

        from .ik import solve_pose_ik

        return solve_pose_ik(
            robot=self,
            target_pose=target_pose,
            q0=q0,
            frame=frame,
            **kwargs,
        )

    # ======================================================================
    # Visualization
    # ======================================================================

    def init_viewer(self, open_browser=True):
        """
        Initialize a Meshcat viewer.
        """

        from .visualization import create_meshcat_viewer

        self.viewer = create_meshcat_viewer(
            self,
            open_browser=open_browser,
        )

        return self.viewer

    def display(self, q):
        """
        Display configuration q in Meshcat.

        The viewer is automatically initialized if needed.
        """

        q = self._check_configuration(q)

        if self.viewer is None:
            self.init_viewer()

        self.viewer.display(q)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _check_configuration(self, q):
        """
        Convert q to a 1D numpy array and check its dimension.
        """

        q = np.asarray(q, dtype=float).reshape(-1)

        if q.shape != (self.nq,):
            raise ValueError(
                f"Expected q shape ({self.nq},), got {q.shape}."
            )

        return q

    def __repr__(self):
        return (
            f"Robot("
            f"name={self.model.name!r}, "
            f"nq={self.nq}, "
            f"nv={self.nv}, "
            f"end_effector={self.end_effector_frame!r}"
            f")"
        )

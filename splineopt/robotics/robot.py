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
    # Joint-space representation
    # ======================================================================

    def joint_positions_to_configuration(self, theta):
        """
        Convert joint coordinates theta in R^nv to a Pinocchio
        configuration q in the configuration manifold.

        For the fixed-base Kinova Gen3, nv = 7 and theta contains
        the seven joint angles.

        The conversion is performed by integrating theta from the
        neutral configuration.
        """

        theta = np.asarray(
            theta,
            dtype=float,
        ).reshape(-1)

        if theta.shape != (self.nv,):
            raise ValueError(
                f"Expected theta shape ({self.nv},), "
                f"got {theta.shape}."
            )

        return pin.integrate(
            self.model,
            self.neutral(),
            theta,
        )

    def configuration_to_joint_positions(self, q):
        """
        Convert one Pinocchio configuration to joint coordinates.

        Note:
        for continuous joints, this returns the shortest displacement
        from the neutral configuration. For a trajectory, use
        configuration_trajectory_to_joint_positions() instead so that
        angles are unwrapped continuously.
        """

        q = self._check_configuration(q)

        return pin.difference(
            self.model,
            self.neutral(),
            q,
        )

    def configuration_trajectory_to_joint_positions(
        self,
        q_sequence,
    ):
        """
        Convert a sequence of Pinocchio configurations to a continuous
        joint trajectory theta(t) in R^nv.

        Successive Pinocchio differences are accumulated. This prevents
        artificial +/- pi or +/- 2*pi jumps on continuous joints.
        """

        q_sequence = np.asarray(
            q_sequence,
            dtype=float,
        )

        if q_sequence.ndim != 2:
            raise ValueError(
                "q_sequence must have shape (N, nq)."
            )

        if q_sequence.shape[1] != self.nq:
            raise ValueError(
                f"Expected q_sequence shape (N, {self.nq}), "
                f"got {q_sequence.shape}."
            )

        n_samples = len(q_sequence)

        if n_samples == 0:
            return np.empty(
                (0, self.nv)
            )

        theta = np.zeros(
            (n_samples, self.nv)
        )

        # First configuration relative to neutral.
        theta[0] = pin.difference(
            self.model,
            self.neutral(),
            q_sequence[0],
        )

        # Then accumulate local differences.
        for k in range(1, n_samples):

            dtheta = pin.difference(
                self.model,
                q_sequence[k - 1],
                q_sequence[k],
            )

            theta[k] = (
                theta[k - 1]
                + dtheta
            )

        return theta

    def joint_trajectory_to_configurations(
        self,
        theta_sequence,
    ):
        """
        Convert an (N, nv) joint trajectory to Pinocchio configurations.
        """

        theta_sequence = np.asarray(
            theta_sequence,
            dtype=float,
        )

        if theta_sequence.ndim != 2:
            raise ValueError(
                "theta_sequence must have shape (N, nv)."
            )

        if theta_sequence.shape[1] != self.nv:
            raise ValueError(
                f"Expected theta_sequence shape (N, {self.nv}), "
                f"got {theta_sequence.shape}."
            )

        return np.asarray([
            self.joint_positions_to_configuration(theta)
            for theta in theta_sequence
        ])

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

    def ik_trajectory(
        self,
        target_poses,
        q0=None,
        frame=None,
        **kwargs,
    ):
        """
        Solve sequential IK along a Cartesian trajectory.
        """

        from .ik import solve_pose_ik_trajectory

        return solve_pose_ik_trajectory(
            robot=self,
            target_poses=target_poses,
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

    def display_target(
        self,
        target_pose,
        name="target",
        scale=0.12,
    ):
        """
        Display a target SE(3) pose in Meshcat.
        """

        if self.viewer is None:
            self.init_viewer()

        from .visualization import display_pose

        display_pose(
            self.viewer,
            target_pose,
            name=name,
            scale=scale,
        )

    def remove_target(
        self,
        name="target",
    ):
        """
        Remove a target pose from Meshcat.
        """

        if self.viewer is None:
            return

        from .visualization import remove_pose

        remove_pose(
            self.viewer,
            name=name,
        )

    def animate(
        self,
        q_sequence,
        dt=0.03,
    ):
        """
        Animate a sequence of robot configurations.
        """

        if self.viewer is None:
            self.init_viewer()

        q_sequence = np.asarray(
            q_sequence,
            dtype=float,
        )

        if q_sequence.ndim != 2:
            raise ValueError(
                "q_sequence must have shape (N, nq)."
            )

        if q_sequence.shape[1] != self.nq:
            raise ValueError(
                f"Expected configurations with dimension "
                f"{self.nq}, got {q_sequence.shape[1]}."
            )

        from .visualization import animate_configurations

        animate_configurations(
            self.viewer,
            q_sequence,
            dt=dt,
        )

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

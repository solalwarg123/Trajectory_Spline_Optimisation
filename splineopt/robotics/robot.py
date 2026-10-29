from pathlib import Path

import numpy as np
import pinocchio as pin


class Robot:
    """
    Pinocchio-based robot model and kinematics interface.

    Loads a robot from URDF and provides:
        - Configuration manifold operations.
        - Forward kinematics and frame Jacobians.
        - Conversion between Pinocchio configurations and joint coordinates.
        - Single-pose and sequential inverse kinematics.
        - Optional Meshcat visualization.

    Pinocchio distinguishes:
        nq: Dimension of the configuration representation.
        nv: Dimension of the tangent space.

    These dimensions may differ, particularly for continuous joints.

    Kinematic computations reuse Pinocchio data structures for efficiency.

    Parameters
    ----------
    urdf_path:
        Path to the robot URDF file.
    end_effector_frame:
        Optional default frame used for kinematics and IK.
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

        # Load Pinocchio kinematic and geometry models

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

        # Expose configuration dimensions and joint limits

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
        self._frame_ids = {
            frame.name: i
            for i, frame in enumerate(self.model.frames)
        }

        # Validate and store the default end-effector frame

        self.end_effector_frame = end_effector_frame

        if end_effector_frame is not None:
            if end_effector_frame not in self.frame_names:
                raise ValueError(
                    f"Unknown end-effector frame: {end_effector_frame}\n"
                    f"Available frames:\n"
                    + "\n".join(self.frame_names)
                )

            self.end_effector_frame_id = self._frame_ids[end_effector_frame]

        else:
            self.end_effector_frame_id = None

        # Meshcat viewer is initialized lazily

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

        if not np.all(np.isfinite(v)):
            raise ValueError("v must contain finite values.")

        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

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
        Resolve a frame name to its Pinocchio frame ID.

        If frame is None, use the configured default end-effector.
        """
        if frame is None:
            if self.end_effector_frame_id is None:
                raise ValueError(
                    "No frame specified and no default "
                    "end-effector configured."
                )
            return self.end_effector_frame_id

        try:
            return self._frame_ids[frame]
        except (KeyError, TypeError):
            raise ValueError(f"Unknown frame: {frame}") from None

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
        Convert tangent-space joint coordinates to a Pinocchio configuration.

        The coordinates theta have dimension nv and are integrated from
        the robot's neutral configuration:

            q = integrate(model, neutral(), theta)

        For the fixed-base Kinova Gen3, theta represents the seven
        joint angles. Continuous joints are represented internally
        by Pinocchio using their manifold representation.

        Parameters
        ----------
        theta:
            Joint coordinate vector, shape (nv,).

        Returns
        -------
        np.ndarray:
            Pinocchio configuration, shape (nq,).

        Notes
        -----
        This representation is intended for fixed-base manipulators.
        For general configuration manifolds, theta is not necessarily
        a globally unique Euclidean parameterization.
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
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain finite values.")

        return pin.integrate(
            self.model,
            self.neutral(),
            theta,
        )

    def configuration_to_joint_positions(self, q):
        """
        Convert a Pinocchio configuration to joint coordinates.

        Computes the shortest tangent-space displacement from the
        neutral configuration:

            theta = difference(model, neutral(), q)

        For continuous joints, this may wrap angles to their principal
        representation.

        For a trajectory, use
        configuration_trajectory_to_joint_positions() to accumulate
        successive displacements and preserve angular continuity.
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
        Convert a Pinocchio configuration sequence to joint coordinates.

        The initial coordinates are computed relative to neutral().
        Subsequent coordinates are obtained by accumulating consecutive
        Pinocchio configuration differences:

            theta[0] = difference(neutral(), q[0])
            theta[k] = theta[k-1] + difference(q[k-1], q[k])

        For fixed-base manipulators, this unwraps continuous joint angles
        along the trajectory, avoiding artificial jumps at angle boundaries.

        Parameters
        ----------
        q_sequence:
            Configuration array of shape (N, nq).

        Returns
        -------
        np.ndarray:
            Continuous joint coordinates of shape (N, nv).
            An empty input produces an empty array with shape (0, nv).

        Notes
        -----
        The accumulation assumes that consecutive configurations are
        sufficiently close for the shortest manifold displacement to
        represent the intended motion. It is not a general-purpose global
        coordinate chart for arbitrary robot configuration manifolds.
        """

        q_sequence = np.asarray(
            q_sequence,
            dtype=float,
        )

        if q_sequence.ndim != 2:
            raise ValueError(
                "q_sequence must have shape (N, nq)."
            )

        if not np.all(np.isfinite(q_sequence)):
            raise ValueError("q_sequence must contain finite values.")

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
        Convert a sequence of joint coordinates to Pinocchio configurations.

        Each row is integrated independently from the neutral configuration.

        Parameters
        ----------
        theta_sequence:
            Joint coordinates of shape (N, nv).

        Returns
        -------
        np.ndarray:
            Pinocchio configurations of shape (N, nq).
        """

        theta_sequence = np.asarray(
            theta_sequence,
            dtype=float,
        )

        if theta_sequence.ndim != 2:
            raise ValueError(
                "theta_sequence must have shape (N, nv)."
            )

        if not np.all(np.isfinite(theta_sequence)):
            raise ValueError("theta_sequence must contain finite values.")

        if theta_sequence.shape[1] != self.nv:
            raise ValueError(
                f"Expected theta_sequence shape (N, {self.nv}), "
                f"got {theta_sequence.shape}."
            )

        if len(theta_sequence) == 0:
            return np.empty((0, self.nq), dtype=float)

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

    def animate(self, q_sequence, dt=0.03):
        """
        Animate a sequence of Pinocchio configurations in Meshcat.

        The viewer is initialized automatically if necessary.
        Validation of the trajectory and animation delay is delegated
        to animate_configurations().
        """
        from .visualization import animate_configurations

        if self.viewer is None:
            self.init_viewer()

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
        Validate a Pinocchio configuration vector.

        Checks its dimension and numerical finiteness without
        enforcing joint limits or normalizing the configuration.
        """
        q = np.asarray(q, dtype=float).reshape(-1)

        if q.shape != (self.nq,):
            raise ValueError(
                f"Expected q shape ({self.nq},), got {q.shape}."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError("Configuration must contain finite values.")

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

    def frame_kinematics(
        self,
        q,
        frames,
        reference_frame=pin.LOCAL_WORLD_ALIGNED,
    ):
        """Compute positions and Jacobians of several frames efficiently.

        A single Pinocchio forward-kinematics / Jacobian update is
        performed for all requested frames.

        Parameters
        ----------
        q :
            Pinocchio configuration vector.

        frames :
            Iterable of frame names.

        reference_frame :
            Pinocchio Jacobian reference frame.

        Returns
        -------
        dict
            Mapping

                frame_name -> (position, jacobian)

            where position has shape (3,) and jacobian has shape (6, nv).

        Notes
        -----
        A single kinematic update is shared across all requested frames,
        avoiding redundant Pinocchio computations when multiple frame
        positions and Jacobians are needed.

        Repeated frame names are evaluated only once.

        The returned arrays are copies and do not alias internal
        Pinocchio data.
        """

        q = self._check_configuration(q)

        frame_names = list(
            dict.fromkeys(frames)
        )

        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )

        pin.computeJointJacobians(
            self.model,
            self.data,
            q,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        result = {}

        for frame_name in frame_names:

            frame_id = self.frame_id(
                frame_name
            )

            position = (
                self.data.oMf[
                    frame_id
                ]
                .translation
                .copy()
            )

            jacobian = (
                pin.getFrameJacobian(
                    self.model,
                    self.data,
                    frame_id,
                    reference_frame,
                )
                .copy()
            )

            result[frame_name] = (
                position,
                jacobian,
            )

        return result

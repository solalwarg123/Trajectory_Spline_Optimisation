"""Collision geometry and linearized collision constraints.

This module implements a lightweight collision model for trajectory
optimization:

    - robot links approximated by capsules,
    - static spherical obstacles,
    - capsule-sphere signed distances,
    - analytical distance Jacobians,
    - B-spline collocation constraints,
    - first-order collision constraint linearization.

The module deliberately does not contain any optimization algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# =====================================================================
# Collision primitives
# =====================================================================


@dataclass(frozen=True)
class Capsule:
    """Capsule attached between two robot frames.

    Parameters
    ----------
    name:
        Human-readable name.

    start_frame:
        Name of the frame defining the first endpoint.

    end_frame:
        Name of the frame defining the second endpoint.

    radius:
        Capsule radius in meters.
    """

    name: str
    start_frame: str
    end_frame: str
    radius: float

    def __post_init__(self):
        if not self.name:
            raise ValueError("Capsule name must not be empty.")

        if not self.start_frame:
            raise ValueError("start_frame must not be empty.")

        if not self.end_frame:
            raise ValueError("end_frame must not be empty.")

        if self.start_frame == self.end_frame:
            raise ValueError("Capsule endpoints must use " "different frames.")

        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Capsule radius must be finite and positive.")


@dataclass(frozen=True)
class SphereObstacle:
    """
    Static spherical obstacle in the world frame.

    Attributes:
        center: Sphere center in world coordinates, shape (3,).
        radius: Sphere radius in meters.
        name: Optional identifier used in collision diagnostics.

    The center is converted to a NumPy array and copied during
    initialization.
    """

    center: np.ndarray
    radius: float
    name: str = ""

    def __post_init__(self):
        center = np.asarray(
            self.center,
            dtype=float,
        ).reshape(-1)

        if center.shape != (3,):
            raise ValueError("SphereObstacle.center must have " "shape (3,).")

        if not np.all(np.isfinite(center)):
            raise ValueError("SphereObstacle.center must contain " "finite values.")

        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("SphereObstacle radius must be finite and positive.")

        # Dataclass is frozen, so use object.__setattr__.
        object.__setattr__(
            self,
            "center",
            center.copy(),
        )


# =====================================================================
# Low-level geometry
# =====================================================================


def _closest_point_on_segment(
    point,
    segment_start,
    segment_end,
):
    """Return the closest point on a segment to a point.

    Returns
    -------
    closest_point :
        Shape (3,).

    alpha :
        Segment parameter in [0, 1]:

            p_closest
                = (1-alpha) * start
                + alpha * end
    """

    point = np.asarray(
        point,
        dtype=float,
    ).reshape(3)

    a = np.asarray(
        segment_start,
        dtype=float,
    ).reshape(3)

    b = np.asarray(
        segment_end,
        dtype=float,
    ).reshape(3)

    direction = b - a

    length_squared = float(direction @ direction)

    if length_squared <= 1e-16:
        raise ValueError("Capsule axis has essentially zero length.")

    alpha = float((point - a) @ direction / length_squared)

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    closest = (1.0 - alpha) * a + alpha * b

    return closest, alpha


def _fallback_normal(axis):
    """Return one deterministic unit vector normal to an axis.

    This is only used in the degenerate case where the sphere center
    lies exactly on the capsule axis and the Euclidean distance
    gradient is not uniquely defined.
    """

    axis = np.asarray(
        axis,
        dtype=float,
    ).reshape(3)

    axis_norm = np.linalg.norm(axis)

    if axis_norm <= 1e-16:
        return np.array([1.0, 0.0, 0.0])

    axis = axis / axis_norm

    basis_vectors = np.eye(3)

    # Choose the coordinate direction least aligned
    # with the capsule axis.
    index = int(np.argmin(np.abs(axis)))

    normal = np.cross(
        axis,
        basis_vectors[index],
    )

    normal_norm = np.linalg.norm(normal)

    if normal_norm <= 1e-16:
        raise RuntimeError("Could not construct a fallback normal.")

    return normal / normal_norm


def capsule_sphere_distance(
    segment_start,
    segment_end,
    capsule_radius,
    sphere_center,
    sphere_radius,
):
    """Signed separation distance between a capsule and a sphere.

    Positive:
        separated.

    Zero:
        touching.

    Negative:
        penetrating.

    Returns
    -------
    distance :
        Signed surface-to-surface distance.

    alpha :
        Closest-point parameter on the capsule axis.

    closest_point :
        Closest point on the capsule central segment.

    normal :
        Unit direction from the sphere center toward the capsule axis.
    """

    a = np.asarray(
        segment_start,
        dtype=float,
    ).reshape(3)

    b = np.asarray(
        segment_end,
        dtype=float,
    ).reshape(3)

    center = np.asarray(
        sphere_center,
        dtype=float,
    ).reshape(3)

    closest, alpha = _closest_point_on_segment(
        point=center,
        segment_start=a,
        segment_end=b,
    )

    delta = closest - center

    centerline_distance = float(np.linalg.norm(delta))

    if centerline_distance > 1e-12:
        normal = delta / centerline_distance
    else:
        # The distance function is non-differentiable here.
        # Use one deterministic subgradient direction.
        normal = _fallback_normal(b - a)

    signed_distance = centerline_distance - float(capsule_radius) - float(sphere_radius)

    return (
        signed_distance,
        alpha,
        closest,
        normal,
    )


# =====================================================================
# Collision environment
# =====================================================================


class CollisionEnvironment:
    """Robot capsules and static spherical obstacles.

    The environment evaluates all capsule-obstacle pairs.

    If no obstacle is provided, all returned arrays naturally have
    zero rows.
    """

    def __init__(
        self,
        robot,
        capsules,
        obstacles=None,
    ):
        self.robot = robot

        self.capsules = list(capsules)

        self.obstacles = [] if obstacles is None else list(obstacles)

        self._validate()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate(self):
        """Validate spline compatibility, safety distance and collocation times."""
        for capsule in self.capsules:
            if not isinstance(
                capsule,
                Capsule,
            ):
                raise TypeError("capsules must contain " "Capsule objects.")

            # Fail early if a frame name is invalid.
            self.robot.frame_id(capsule.start_frame)

            self.robot.frame_id(capsule.end_frame)

        for obstacle in self.obstacles:
            if not isinstance(
                obstacle,
                SphereObstacle,
            ):
                raise TypeError("obstacles must contain " "SphereObstacle objects.")

    # -----------------------------------------------------------------
    # Basic information
    # -----------------------------------------------------------------

    @property
    def n_capsules(self):
        return len(self.capsules)

    @property
    def n_obstacles(self):
        return len(self.obstacles)

    @property
    def n_pairs(self):
        return self.n_capsules * self.n_obstacles

    @property
    def has_obstacles(self):
        return self.n_obstacles > 0

    def pair_names(self):
        """Human-readable names for all collision pairs."""

        names = []

        for capsule in self.capsules:
            for j, obstacle in enumerate(self.obstacles):
                obstacle_name = obstacle.name if obstacle.name else f"obstacle_{j}"

                names.append(
                    (
                        capsule.name,
                        obstacle_name,
                    )
                )

        return names

    # -----------------------------------------------------------------
    # Distance evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        theta,
        *,
        compute_jacobian=True,
    ):
        """Evaluate all capsule-obstacle distances.

        Parameters
        ----------
        theta:
            Joint-space coordinates in R^nv, i.e. the same coordinates
            used by the B-spline trajectory.

        compute_jacobian:
            If True, also return the derivative of each distance with
            respect to theta.

        Returns
        -------
        distances :
            Shape (n_pairs,).

        jacobians :
            Shape (n_pairs, nv) if compute_jacobian=True.
            Otherwise None.
        """

        theta = np.asarray(
            theta,
            dtype=float,
        ).reshape(-1)

        if theta.shape != (self.robot.nv,):
            raise ValueError(
                f"Expected theta shape " f"({self.robot.nv},), " f"got {theta.shape}."
            )

        # -------------------------------------------------------------
        # Empty environment
        # -------------------------------------------------------------

        if self.n_pairs == 0:
            distances = np.empty(
                0,
                dtype=float,
            )

            if compute_jacobian:
                jacobians = np.empty(
                    (
                        0,
                        self.robot.nv,
                    ),
                    dtype=float,
                )
            else:
                jacobians = None

            return (
                distances,
                jacobians,
            )

        # -------------------------------------------------------------
        # Convert spline joint coordinates to Pinocchio configuration
        # -------------------------------------------------------------

        q_pin = self.robot.joint_positions_to_configuration(theta)

        # -------------------------------------------------------------
        # Compute endpoint positions and Jacobians
        #
        # For now this uses the existing Robot API.
        # Later this can be replaced by one batched FK/Jacobian call
        # without changing the public collision interface.
        # -------------------------------------------------------------

        frame_names = set()

        for capsule in self.capsules:
            frame_names.add(capsule.start_frame)
            frame_names.add(capsule.end_frame)

        kinematics = self.robot.frame_kinematics(
            q_pin,
            frame_names,
        )

        positions = {
            frame_name: kinematics[frame_name][0] for frame_name in frame_names
        }

        if compute_jacobian:

            jacobians = {
                frame_name: kinematics[frame_name][1][:3, :].copy()
                for frame_name in frame_names
            }

        else:

            jacobians = {}

        # -------------------------------------------------------------
        # Evaluate every capsule-obstacle pair
        # -------------------------------------------------------------

        distances_out = np.empty(
            self.n_pairs,
            dtype=float,
        )

        if compute_jacobian:
            jacobians_out = np.empty(
                (
                    self.n_pairs,
                    self.robot.nv,
                ),
                dtype=float,
            )
        else:
            jacobians_out = None

        row = 0

        for capsule in self.capsules:

            p0 = positions[capsule.start_frame]

            p1 = positions[capsule.end_frame]

            if compute_jacobian:
                J0 = jacobians[capsule.start_frame]

                J1 = jacobians[capsule.end_frame]

            for obstacle in self.obstacles:

                (
                    distance,
                    alpha,
                    _,
                    normal,
                ) = capsule_sphere_distance(
                    segment_start=p0,
                    segment_end=p1,
                    capsule_radius=(capsule.radius),
                    sphere_center=(obstacle.center),
                    sphere_radius=(obstacle.radius),
                )

                distances_out[row] = distance

                if compute_jacobian:
                    # Closest point:
                    #
                    # p* =
                    #     (1-alpha) p0
                    #     + alpha p1
                    #
                    # Locally, the derivative of the minimum
                    # distance can be evaluated at the active
                    # closest point without differentiating alpha.
                    J_closest = (1.0 - alpha) * J0 + alpha * J1

                    jacobians_out[row] = normal @ J_closest

                row += 1

        return (
            distances_out,
            jacobians_out,
        )


# =====================================================================
# B-spline collision constraints
# =====================================================================


class CollisionConstraintBuilder:
    """
    Build sampled collision constraints for a B-spline trajectory.

    The robot is approximated by capsules and the environment by
    static spherical obstacles. Signed distances are evaluated at
    specified collocation times along the trajectory.

    At a linearization point c_s, the joint configuration is:

        q_i = A_i @ c_s

    The signed distance is approximated to first order:

        d(A_i @ c) ~= d_i + J_d,i @ A_i @ (c - c_s)

    Imposing a minimum safety distance gives:

        H_i @ c >= h_i

    where:
        H_i = J_d,i @ A_i
        h_i = safety_distance - d_i + H_i @ c_s

    The builder provides exact sampled distances, analytical
    Jacobians, linearized constraints and feasibility checks.

    Important:
        Collision constraints are enforced only at collocation
        times. They do not guarantee continuous-time collision
        avoidance between samples.
    """

    def __init__(
        self,
        spline,
        environment,
        collocation_times,
        *,
        safety_distance=0.0,
    ):
        self.spline = spline
        self.environment = environment

        self.collocation_times = np.asarray(
            collocation_times,
            dtype=float,
        ).reshape(-1)

        self.safety_distance = float(safety_distance)

        self._validate()

        self.n_variables = self.spline.n_control_points * self.spline.n_dof

        # One evaluation matrix per collocation point.
        self.A_collocation = [
            self.spline.evaluation_matrix(
                t,
                derivative_order=0,
            )
            for t in self.collocation_times
        ]

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate(self):
        if self.spline.n_dof != self.environment.robot.nv:
            raise ValueError("spline.n_dof must match " "environment.robot.nv.")

        if not np.isfinite(self.safety_distance) or self.safety_distance < 0.0:
            raise ValueError("safety_distance must be finite and non-negative.")

        if not np.all(np.isfinite(self.collocation_times)):
            raise ValueError("collocation_times must contain only finite values.")

        if np.any(self.collocation_times < self.spline.t_start) or np.any(
            self.collocation_times > self.spline.t_end
        ):
            raise ValueError(
                "All collision collocation times " "must lie inside the spline domain."
            )

        if self.collocation_times.size > 1 and np.any(
            np.diff(self.collocation_times) <= 0.0
        ):
            raise ValueError("collocation_times must be " "strictly increasing.")

    # -----------------------------------------------------------------
    # Basic information
    # -----------------------------------------------------------------

    @property
    def n_collocation_points(self):
        return self.collocation_times.size

    @property
    def n_constraints(self):
        return self.n_collocation_points * self.environment.n_pairs

    @property
    def has_constraints(self):
        return self.n_constraints > 0

    # -----------------------------------------------------------------
    # True nonlinear distances
    # -----------------------------------------------------------------

    def distances(
        self,
        control_points,
    ):
        """Evaluate the true nonlinear distances at collocations.

        Returns
        -------
        distances :
            Shape

                (n_collocation_points, n_pairs)
        """

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        if c.size != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} "
                "control-point values, "
                f"got {c.size}."
            )

        if not np.all(np.isfinite(c)):
            raise ValueError("Control points must be finite.")

        n_col = self.n_collocation_points

        n_pairs = self.environment.n_pairs

        if n_pairs == 0:
            return np.empty(
                (
                    n_col,
                    0,
                ),
                dtype=float,
            )

        distances = np.empty(
            (
                n_col,
                n_pairs,
            ),
            dtype=float,
        )

        for i, A_i in enumerate(self.A_collocation):
            theta_i = A_i @ c

            d_i, _ = self.environment.evaluate(
                theta_i,
                compute_jacobian=False,
            )

            distances[i] = d_i

        return distances

    def minimum_distance(
        self,
        control_points,
    ):
        """Minimum true distance over all collision collocations."""

        if not self.has_constraints:
            return np.inf

        return float(np.min(self.distances(control_points)))

    def is_feasible(
        self,
        control_points,
        *,
        tolerance=1e-9,
    ):
        """
        Check sampled nonlinear collision feasibility.

        A trajectory is feasible if every sampled signed distance
        satisfies:

            distance >= safety_distance - tolerance

        Args:
            control_points: B-spline control points.
            tolerance: Allowed numerical violation in meters.

        Returns:
            True if all sampled collision constraints are satisfied.
            Also returns True when no collision constraints exist.

        Note:
            This check does not guarantee collision avoidance
            between collocation times.
        """

        if not self.has_constraints:
            return True

        return self.minimum_distance(control_points) >= (
            self.safety_distance - float(tolerance)
        )

    # -----------------------------------------------------------------
    # Linearized constraints
    # -----------------------------------------------------------------

    def linearize(
        self,
        control_points,
    ):
        """
        Linearize the collision constraints around control_points.

        For the current trajectory c_s, the signed distances are
        approximated by their first-order Taylor expansion:

            d(c) ~= d(c_s) + H @ (c - c_s)

        The safety constraints become:

            H @ c >= h

        with:
            h = safety_distance - d(c_s) + H @ c_s

        Args:
            control_points: Current B-spline control points.

        Returns:
            H: Sparse constraint Jacobian of shape
            (n_constraints, n_variables).
            h: Lower bounds of shape (n_constraints,).

        If no collision constraints exist, empty arrays are returned.
        """

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        distances, H = self.values_and_jacobian(c)

        if distances.size == 0:
            return (
                H,
                np.empty(
                    0,
                    dtype=float,
                ),
            )

        h = self.safety_distance - distances + np.asarray(H @ c).reshape(-1)

        return H, h

    def values_and_jacobian(
        self,
        control_points,
    ):
        """
        Evaluate all nonlinear collision constraints and their
        Jacobian with respect to the stacked B-spline control points.

        Returns
        -------
        values : ndarray, shape (n_constraints,)
            Signed capsule-obstacle distances.

        jacobian : scipy.sparse.csc_matrix,
            shape (n_constraints, n_variables)

            Jacobian of the distances with respect to the stacked
            control-point vector.
        """

        c = np.asarray(
            control_points,
            dtype=float,
        ).reshape(-1)

        if c.size != self.n_variables:
            raise ValueError(
                "Invalid control-point vector size: "
                f"expected {self.n_variables}, "
                f"got {c.size}."
            )

        if not np.all(np.isfinite(c)):
            raise ValueError("Control points must be finite.")

        if not self.has_constraints:
            return (
                np.empty(
                    0,
                    dtype=float,
                ),
                sp.csc_matrix(
                    (
                        0,
                        self.n_variables,
                    )
                ),
            )

        values = []
        jacobian_rows = []

        for A_i in self.A_collocation:

            theta_i = A_i @ c

            distances_i, J_distance_i = self.environment.evaluate(
                theta_i,
                compute_jacobian=True,
            )

            # J_distance_i:
            #
            #     n_pairs x n_dof
            #
            # A_i:
            #
            #     n_dof x n_variables
            #
            # therefore:
            #
            #     J_distance_i @ A_i
            #
            # gives:
            #
            #     n_pairs x n_variables
            J_control_i = J_distance_i @ A_i

            values.append(distances_i)

            jacobian_rows.append(sp.csc_matrix(J_control_i))

        values = np.concatenate(values)

        jacobian = sp.vstack(
            jacobian_rows,
            format="csc",
        )

        return (
            values,
            jacobian,
        )

    def near_obstacle_indices(
        self,
        control_points,
        activation_distance,
        n_samples=11,
    ):
        """
        Return indices of obstacles close to a trajectory.

        All robot capsules are checked against all obstacles.
        An obstacle is selected if at least one capsule
        approaches it within activation_distance.

        This is a sampled geometric broad-phase, not a
        continuous collision certificate.
        """
        if not np.isfinite(activation_distance):
            raise ValueError("activation_distance must be finite.")

        if activation_distance < 0.0:
            raise ValueError("activation_distance must be non-negative.")

        if n_samples < 2:
            raise ValueError("n_samples must be >= 2.")

        env = self.environment
        n_obs = env.n_obstacles

        if n_obs == 0:
            return ()

        c = np.asarray(control_points, dtype=float).reshape(-1)

        if c.size != self.n_variables:
            raise ValueError("Invalid control point vector size.")

        if not np.all(np.isfinite(c)):
            raise ValueError("Control points must be finite.")

        active = np.zeros(n_obs, dtype=bool)

        times = np.linspace(
            self.spline.t_start,
            self.spline.t_end,
            n_samples,
        )

        for t in times:
            A = self.spline.evaluation_matrix(t, derivative_order=0)
            q = A @ c

            distances, _ = env.evaluate(q, compute_jacobian=False)

            # CollisionEnvironment orders pairs as:
            # capsule 0 / obstacle 0..N,
            # capsule 1 / obstacle 0..N, etc.
            distances = distances.reshape(env.n_capsules, n_obs)

            active |= np.any(
                distances <= activation_distance,
                axis=0,
            )

            if np.all(active):
                break

        return tuple(np.flatnonzero(active).tolist())

    def subset_for_obstacles(self, obstacle_indices):
        """
        Create constraints for selected obstacles only.

        All robot capsules are preserved.
        The same spline, collocation grid and safety
        distance are used.
        """
        indices = tuple(obstacle_indices)

        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate obstacle indices.")

        if any(i < 0 or i >= self.environment.n_obstacles for i in indices):
            raise ValueError("Invalid obstacle index.")

        environment = CollisionEnvironment(
            robot=self.environment.robot,
            capsules=self.environment.capsules,
            obstacles=[self.environment.obstacles[i] for i in indices],
        )

        return CollisionConstraintBuilder(
            spline=self.spline,
            environment=environment,
            collocation_times=self.collocation_times,
            safety_distance=self.safety_distance,
        )

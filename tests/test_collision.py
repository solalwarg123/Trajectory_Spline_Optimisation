from pathlib import Path

import numpy as np
import pytest

from splineopt.bspline import BSpline
from splineopt.collision import (
    Capsule,
    SphereObstacle,
    CollisionEnvironment,
    CollisionConstraintBuilder,
    capsule_sphere_distance,
)
from splineopt.robotics import Robot


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "kinova3_flat"
    / "gen3_7dof.urdf"
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def robot():
    return Robot(
        urdf_path=URDF_PATH,
        end_effector_frame="end_effector_link",
    )


@pytest.fixture
def capsules():
    """
    Preliminary Kinova capsule approximation.

    Radii are only working values for now.
    They will be checked visually with Meshcat next.
    """
    return [
        Capsule(
            name="upper_arm_1",
            start_frame="shoulder_link",
            end_frame="half_arm_1_link",
            radius=0.06,
        ),
        Capsule(
            name="upper_arm_2",
            start_frame="half_arm_1_link",
            end_frame="half_arm_2_link",
            radius=0.06,
        ),
        Capsule(
            name="forearm",
            start_frame="half_arm_2_link",
            end_frame="forearm_link",
            radius=0.055,
        ),
        Capsule(
            name="wrist_1",
            start_frame="forearm_link",
            end_frame="spherical_wrist_1_link",
            radius=0.05,
        ),
        Capsule(
            name="wrist_2",
            start_frame="spherical_wrist_1_link",
            end_frame="spherical_wrist_2_link",
            radius=0.045,
        ),
        Capsule(
            name="bracelet",
            start_frame="spherical_wrist_2_link",
            end_frame="bracelet_link",
            radius=0.045,
        ),
        Capsule(
            name="tool",
            start_frame="bracelet_link",
            end_frame="end_effector_link",
            radius=0.04,
        ),
    ]


@pytest.fixture
def obstacle():
    return SphereObstacle(
        name="obstacle",
        center=np.array([
            0.40,
            0.10,
            0.45,
        ]),
        radius=0.10,
    )


@pytest.fixture
def spline(robot):
    """
    Small cubic B-spline used only to test collision collocations.
    """

    degree = 3
    n_control_points = 8

    t_start = 0.0
    t_end = 2.0

    n_internal = (
        n_control_points
        - degree
        - 1
    )

    internal_knots = np.linspace(
        t_start,
        t_end,
        n_internal + 2,
    )[1:-1]

    knots = np.concatenate([
        np.full(
            degree + 1,
            t_start,
        ),
        internal_knots,
        np.full(
            degree + 1,
            t_end,
        ),
    ])

    return BSpline(
        knots=knots,
        degree=degree,
        n_dof=robot.nv,
    )


# ============================================================
# Primitive geometry
# ============================================================


def test_capsule_sphere_distance_outside():
    """
    Capsule axis:
        (0,0,0) -> (1,0,0)

    Sphere center:
        (0.5, 1, 0)

    Centerline distance = 1.

    Capsule radius = 0.1
    Sphere radius = 0.2

    Signed surface distance = 1 - 0.1 - 0.2 = 0.7.
    """

    distance, alpha, closest, normal = (
        capsule_sphere_distance(
            segment_start=np.array([
                0.0,
                0.0,
                0.0,
            ]),
            segment_end=np.array([
                1.0,
                0.0,
                0.0,
            ]),
            capsule_radius=0.1,
            sphere_center=np.array([
                0.5,
                1.0,
                0.0,
            ]),
            sphere_radius=0.2,
        )
    )

    np.testing.assert_allclose(
        distance,
        0.7,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        alpha,
        0.5,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        closest,
        np.array([
            0.5,
            0.0,
            0.0,
        ]),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        normal,
        np.array([
            0.0,
            -1.0,
            0.0,
        ]),
        atol=1e-12,
    )


def test_capsule_sphere_distance_endpoint():
    """
    Check that the closest point is correctly clamped to
    the segment endpoint.
    """

    distance, alpha, closest, _ = (
        capsule_sphere_distance(
            segment_start=np.array([
                0.0,
                0.0,
                0.0,
            ]),
            segment_end=np.array([
                1.0,
                0.0,
                0.0,
            ]),
            capsule_radius=0.1,
            sphere_center=np.array([
                2.0,
                0.0,
                0.0,
            ]),
            sphere_radius=0.2,
        )
    )

    assert np.isclose(
        alpha,
        1.0,
    )

    np.testing.assert_allclose(
        closest,
        np.array([
            1.0,
            0.0,
            0.0,
        ]),
    )

    np.testing.assert_allclose(
        distance,
        0.7,
    )


def test_capsule_sphere_penetration():
    """
    Negative distance means geometric penetration.
    """

    distance, _, _, _ = (
        capsule_sphere_distance(
            segment_start=np.array([
                0.0,
                0.0,
                0.0,
            ]),
            segment_end=np.array([
                1.0,
                0.0,
                0.0,
            ]),
            capsule_radius=0.2,
            sphere_center=np.array([
                0.5,
                0.1,
                0.0,
            ]),
            sphere_radius=0.2,
        )
    )

    # Centerline distance = 0.1
    # Total radius = 0.4
    # Signed distance = -0.3

    np.testing.assert_allclose(
        distance,
        -0.3,
        atol=1e-12,
    )


# ============================================================
# CollisionEnvironment
# ============================================================


def test_collision_environment_dimensions(
    robot,
    capsules,
    obstacle,
):
    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[obstacle],
    )

    theta = np.zeros(
        robot.nv
    )

    distances, jacobians = (
        environment.evaluate(
            theta,
            compute_jacobian=True,
        )
    )

    assert environment.n_capsules == 7
    assert environment.n_obstacles == 1
    assert environment.n_pairs == 7

    assert distances.shape == (7,)
    assert jacobians.shape == (
        7,
        robot.nv,
    )

    assert np.all(
        np.isfinite(distances)
    )

    assert np.all(
        np.isfinite(jacobians)
    )


def test_empty_obstacle_environment(
    robot,
    capsules,
):
    """
    Zero obstacles must be a normal use case, not an error.
    """

    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[],
    )

    theta = np.zeros(
        robot.nv
    )

    distances, jacobians = (
        environment.evaluate(
            theta,
            compute_jacobian=True,
        )
    )

    assert environment.n_obstacles == 0
    assert environment.n_pairs == 0
    assert not environment.has_obstacles

    assert distances.shape == (0,)
    assert jacobians.shape == (
        0,
        robot.nv,
    )


# ============================================================
# Distance Jacobian
# ============================================================


def test_collision_distance_jacobian_finite_difference(
    robot,
    capsules,
):
    """
    Validate

        J_d(theta) @ v

    against

        [d(theta + eps*v) - d(theta)] / eps.

    The obstacle is deliberately placed away from singular
    closest-point configurations.
    """

    obstacle = SphereObstacle(
        name="jacobian_test",
        center=np.array([
            0.35,
            0.22,
            0.52,
        ]),
        radius=0.08,
    )

    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[obstacle],
    )

    theta = np.array([
        0.20,
        -0.30,
        0.15,
        0.35,
        -0.20,
        0.18,
        0.10,
    ])

    direction = np.array([
        0.30,
        -0.20,
        0.10,
        0.15,
        -0.10,
        0.20,
        -0.15,
    ])

    direction /= np.linalg.norm(
        direction
    )

    distances_0, J = (
        environment.evaluate(
            theta,
            compute_jacobian=True,
        )
    )

    eps = 1e-7

    distances_1, _ = (
        environment.evaluate(
            theta + eps * direction,
            compute_jacobian=False,
        )
    )

    numerical = (
        distances_1
        - distances_0
    ) / eps

    analytical = (
        J @ direction
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        atol=2e-5,
        rtol=2e-4,
    )


# ============================================================
# CollisionConstraintBuilder
# ============================================================


def test_collision_constraint_builder_dimensions(
    robot,
    capsules,
    obstacle,
    spline,
):
    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[obstacle],
    )

    n_collocation = 20

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        n_collocation,
    )

    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=times,
        safety_distance=0.02,
    )

    n_variables = (
        spline.n_control_points
        * spline.n_dof
    )

    assert builder.n_collocation_points == (
        n_collocation
    )

    assert builder.n_constraints == (
        n_collocation
        * environment.n_pairs
    )

    # Simple constant-ish trajectory around a non-zero
    # joint configuration.
    theta = np.array([
        0.10,
        -0.20,
        0.10,
        0.25,
        -0.15,
        0.10,
        0.05,
    ])

    control_points = np.tile(
        theta,
        spline.n_control_points,
    )

    H, h = builder.linearize(
        control_points
    )

    expected_constraints = (
        n_collocation
        * len(capsules)
    )

    assert H.shape == (
        expected_constraints,
        n_variables,
    )

    assert h.shape == (
        expected_constraints,
    )

    assert np.all(
        np.isfinite(H.data)
    )

    assert np.all(
        np.isfinite(h)
    )


def test_empty_collision_constraint_builder(
    robot,
    capsules,
    spline,
):
    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[],
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        20,
    )

    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=times,
        safety_distance=0.02,
    )

    n_variables = (
        spline.n_control_points
        * spline.n_dof
    )

    control_points = np.zeros(
        n_variables
    )

    H, h = builder.linearize(
        control_points
    )

    assert not builder.has_constraints

    assert H.shape == (
        0,
        n_variables,
    )

    assert h.shape == (0,)

    assert builder.minimum_distance(
        control_points
    ) == np.inf

    assert builder.is_feasible(
        control_points
    )


# ============================================================
# Linearization consistency
# ============================================================


def test_collision_constraint_linearization(
    robot,
    capsules,
    spline,
):
    """
    Test the complete chain

        c
        -> q(t)
        -> distance

    against

        H @ delta_c.

    This checks more than the joint-space Jacobian test because it
    also validates the B-spline evaluation matrix in the chain rule.
    """

    obstacle = SphereObstacle(
        name="linearization_test",
        center=np.array([
            0.38,
            0.20,
            0.50,
        ]),
        radius=0.08,
    )

    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[obstacle],
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        8,
    )

    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=times,
        safety_distance=0.02,
    )

    theta = np.array([
        0.20,
        -0.25,
        0.15,
        0.30,
        -0.15,
        0.12,
        0.08,
    ])

    c = np.tile(
        theta,
        spline.n_control_points,
    )

    H, _ = builder.linearize(
        c
    )

    rng = np.random.default_rng(
        1234
    )

    delta_c = rng.normal(
        size=c.shape
    )

    delta_c /= np.linalg.norm(
        delta_c
    )

    eps = 1e-7

    d0 = (
        builder.distances(c)
        .reshape(-1)
    )

    d1 = (
        builder.distances(
            c + eps * delta_c
        )
        .reshape(-1)
    )

    numerical = (
        d1 - d0
    ) / eps

    analytical = (
        H @ delta_c
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        atol=3e-5,
        rtol=5e-4,
    )


def test_values_and_jacobian_matches_existing_collision_evaluation(
    robot,
    capsules,
    obstacle,
    spline,
):
    """
    values_and_jacobian() must return the same nonlinear
    distances as distances(), with one Jacobian row per
    collision constraint.
    """

    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[
            obstacle
        ],
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        8,
    )

    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=times,
        safety_distance=0.02,
    )

    theta = np.array([
        0.20,
        -0.25,
        0.15,
        0.30,
        -0.15,
        0.12,
        0.08,
    ])

    c = np.tile(
        theta,
        spline.n_control_points,
    )

    values, jacobian = (
        builder.values_and_jacobian(
            c
        )
    )

    distances = (
        builder.distances(
            c
        )
        .reshape(-1)
    )

    np.testing.assert_allclose(
        values,
        distances,
        atol=1e-12,
    )

    assert jacobian.shape == (
        builder.n_constraints,
        builder.n_variables,
    )

    assert np.all(
        np.isfinite(
            jacobian.data
        )
    )


def test_collision_global_jacobian_matches_finite_difference(
    robot,
    capsules,
    spline,
):
    """
    Validate the complete nonlinear collision Jacobian

        d(c)

    against a central finite difference in a random
    control-point direction.
    """

    obstacle = SphereObstacle(
        name="global_jacobian_test",
        center=np.array([
            0.38,
            0.20,
            0.50,
        ]),
        radius=0.08,
    )

    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[
            obstacle
        ],
    )

    times = np.linspace(
        spline.t_start,
        spline.t_end,
        8,
    )

    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=times,
        safety_distance=0.02,
    )

    theta = np.array([
        0.20,
        -0.25,
        0.15,
        0.30,
        -0.15,
        0.12,
        0.08,
    ])

    c = np.tile(
        theta,
        spline.n_control_points,
    )

    values, jacobian = (
        builder.values_and_jacobian(
            c
        )
    )

    rng = np.random.default_rng(
        123
    )

    direction = rng.normal(
        size=c.size
    )

    direction /= np.linalg.norm(
        direction
    )

    epsilon = 1e-6

    values_plus, _ = (
        builder.values_and_jacobian(
            c
            + epsilon
            * direction
        )
    )

    values_minus, _ = (
        builder.values_and_jacobian(
            c
            - epsilon
            * direction
        )
    )

    numerical = (
        values_plus
        - values_minus
    ) / (
        2.0
        * epsilon
    )

    analytical = (
        jacobian
        @ direction
    )

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=5e-4,
        atol=3e-5,
    )


@pytest.mark.parametrize("radius", [np.nan, np.inf, -np.inf, 0.0, -0.1])
def test_collision_primitives_reject_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        Capsule(
            name="test",
            start_frame="frame_a",
            end_frame="frame_b",
            radius=radius,
        )

    with pytest.raises(ValueError, match="radius"):
        SphereObstacle(
            center=np.zeros(3),
            radius=radius,
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_collision_builder_rejects_nonfinite_parameters(
    robot, capsules, obstacle, spline, invalid_value
):
    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[obstacle],
    )

    with pytest.raises(ValueError, match="safety_distance"):
        CollisionConstraintBuilder(
            spline=spline,
            environment=environment,
            collocation_times=[spline.t_start, spline.t_end],
            safety_distance=invalid_value,
        )

    with pytest.raises(ValueError, match="collocation_times"):
        CollisionConstraintBuilder(
            spline=spline,
            environment=environment,
            collocation_times=[invalid_value],
            safety_distance=0.02,
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_collision_builder_rejects_nonfinite_control_points(
    robot, capsules, obstacle, spline, invalid_value
):
    environment = CollisionEnvironment(
        robot=robot,
        capsules=capsules,
        obstacles=[obstacle],
    )

    builder = CollisionConstraintBuilder(
        spline=spline,
        environment=environment,
        collocation_times=np.linspace(
            spline.t_start,
            spline.t_end,
            5,
        ),
        safety_distance=0.02,
    )

    control_points = np.zeros(
        (spline.n_control_points, spline.n_dof)
    )
    control_points[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        builder.distances(control_points)

    with pytest.raises(ValueError, match="finite"):
        builder.values_and_jacobian(control_points)

    with pytest.raises(ValueError, match="finite"):
        builder.near_obstacle_indices(
            control_points,
            activation_distance=0.1,
        )

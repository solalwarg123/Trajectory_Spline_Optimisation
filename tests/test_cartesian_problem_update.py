
import numpy as np
import pinocchio as pin
import pytest
from test_cartesian_optimization import kinova_case
from splineopt.cartesian_problem import CartesianTrajectoryProblem


@pytest.mark.parametrize("orientation_scale", [0.0, 0.1])
def test_update_equivalent_to_rebuild(
    kinova_case,
    orientation_scale,
):
    case = kinova_case
    n = case["spline"].n_dof

    def make_problem(q0, v0, poses):
        return CartesianTrajectoryProblem(
            robot=case["robot"],
            spline=case["spline"],
            tracking_times=case["tracking_times"],
            target_poses=poses,
            q0=q0,
            v0=v0,
            q_min=np.full(n, -10.0),
            q_max=np.full(n, 10.0),
            v_min=np.full(n, -10.0),
            v_max=np.full(n, 10.0),
            a_min=np.full(n, -10.0),
            a_max=np.full(n, 10.0),
            a0=case["a0"],
            orientation_scale=orientation_scale,
        )

    # Original problem
    problem = make_problem(
        case["q0"],
        case["v0"],
        case["target_poses"],
    )

    # New MPC state and Cartesian reference
    q_new = case["q0"].copy()
    q_new[0] += 0.02

    v_new = np.full(n, 0.01)

    poses_new = [
        pin.SE3(
            pose.rotation.copy(),
            pose.translation + np.array([0.01, -0.02, 0.0]),
        )
        for pose in case["target_poses"]
    ]

    # Update the existing problem
    cached_matrix = problem._A_static
    cached_tracking = problem._A_tracking

    problem.update(q_new, v_new, poses_new)

    # Verify that expensive matrices were not rebuilt
    assert problem._A_static is cached_matrix
    assert problem._A_tracking is cached_tracking

    # Construct a fresh reference problem
    fresh = make_problem(q_new, v_new, poses_new)

    # Compare linear constraints
    A1, l1, u1 = problem.linear_constraints()
    A2, l2, u2 = fresh.linear_constraints()

    np.testing.assert_allclose((A1 - A2).toarray(), 0.0)
    np.testing.assert_allclose(l1, l2)
    np.testing.assert_allclose(u1, u2)

    # Compare nonlinear tracking
    c = case["initial_control_points"].reshape(-1)

    r1, J1 = problem.tracking_residual_and_jacobian(c)
    r2, J2 = fresh.tracking_residual_and_jacobian(c)

    np.testing.assert_allclose(r1, r2, atol=1e-10)
    np.testing.assert_allclose(J1, J2, atol=1e-10)

    # Ensure a subsequent update also works
    problem.update(
        case["q0"],
        case["v0"],
        case["target_poses"],
    )

    original = make_problem(
        case["q0"],
        case["v0"],
        case["target_poses"],
    )

    _, l1, u1 = problem.linear_constraints()
    _, l2, u2 = original.linear_constraints()

    np.testing.assert_allclose(l1, l2)
    np.testing.assert_allclose(u1, u2)

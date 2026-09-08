# B-Spline Trajectory Optimization for Robot Motion

A lightweight trajectory optimization layer that projects a nominal robot motion onto a smooth and feasible B-spline trajectory.

```text
High-level policy / planner
            ↓
     nominal trajectory
            ↓
   B-spline optimizer
            ↓
 smooth feasible trajectory
            ↓
      robot controller
```

The project currently focuses on **offline joint-space trajectory optimization**.

## Current formulation

A joint trajectory is represented as

$$
q(t)=A^{(0)}(t)c,
$$

with derivatives

$$
\dot q(t)=A^{(1)}(t)c,
\qquad
\ddot q(t)=A^{(2)}(t)c,
$$

where $c$ contains the B-spline control points.

The optimizer solves

$$
\min_c
\;
\|A_{\mathrm{track}}c-b_{\mathrm{track}}\|^2
+
\lambda c^\top R_2 c
$$

subject to the initial conditions

$$
A^{(0)}(t_0)c=q_0,
\qquad
A^{(1)}(t_0)c=\dot q_0.
$$

Joint constraints are enforced using the B-spline convex-hull property:

$$
q_{\min}\le c\le q_{\max},
$$

$$
v_{\min}\le D_1c\le v_{\max},
$$

$$
a_{\min}\le D_2c\le a_{\max}.
$$

This gives continuous-time joint bounds without collocation. The resulting problem is a **convex quadratic program**, built directly with NumPy / SciPy and solved with **OSQP**.

## Current results

The project currently includes:

- multi-DoF B-spline evaluation;
- analytical velocity and acceleration matrices;
- derivative control-point matrices $D_1$ and $D_2$;
- analytical smoothness matrices;
- joint-space trajectory projection with OSQP;
- joint position, velocity and acceleration constraints;
- initial position and velocity constraints;
- 1-DoF and 2-DoF examples.

The B-spline implementation is validated against `scipy.interpolate.BSpline`, and the optimizer is covered by unit tests.

### 2-DoF trajectory projection

The feasible reference is reproduced almost exactly.

The infeasible reference temporarily leaves the admissible joint-position region and later comes back inside. The optimized B-spline stays inside the admissible set, deviates only when necessary, and reconnects with the reference afterwards.

![Time-domain comparison](assets/trajectory_projection_time.png)

![Joint-space comparison](assets/trajectory_projection_joint_space.png)

## Run

```bash
python -m pytest -v
python -m examples.trajectory_projection_1d
python -m examples.trajectory_projection_2d
```

## Next steps

- compare control-point bounds with collocation;
- benchmark solve time and conservatism;
- reuse OSQP problems and warm starts;
- add terminal constraints;
- introduce Pinocchio and Cartesian tracking;
- move toward sequential QP / receding-horizon optimization.

The optimizer is intentionally kept independent of ROS2, LeRobot, and any specific robot-learning framework.

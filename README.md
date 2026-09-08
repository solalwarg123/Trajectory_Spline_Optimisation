# B-Spline Trajectory Optimization for Robot Motion

A lightweight trajectory optimization layer that projects a nominal robot motion onto a smooth, feasible B-spline trajectory.

The long-term goal is to place this optimizer between a high-level planner or learned policy and a robot controller:

```text
High-level policy / planner
            ↓
     nominal trajectory
            ↓
   B-Spline optimizer
            ↓
 smooth feasible trajectory
            ↓
      robot controller
```

The project is currently focused on **offline joint-space trajectory optimization**. It does not yet use MPC, robot kinematics, collision avoidance, ROS2, or LeRobot.

## Current formulation

The joint trajectory is represented as

\[
q(t)=A^{(0)}(t)c,
\]

with derivatives

\[
\dot q(t)=A^{(1)}(t)c,
\qquad
\ddot q(t)=A^{(2)}(t)c,
\]

where \(c\) contains the B-spline control points.

The optimizer solves

\[
\min_c
\|A_{\mathrm{track}}c-b_{\mathrm{track}}\|^2
+
\lambda c^\top R_2 c,
\]

where the first term tracks the nominal trajectory and

\[
c^\top R_2 c
=
\int \|\ddot q(t)\|^2 dt
\]

penalizes acceleration.

Initial position and velocity are imposed exactly:

\[
A^{(0)}(t_0)c=q_0,
\qquad
A^{(1)}(t_0)c=\dot q_0.
\]

Joint limits are enforced using the B-spline convex-hull property:

\[
q_{\min}\le c\le q_{\max},
\]

\[
v_{\min}\le D_1c\le v_{\max},
\]

\[
a_{\min}\le D_2c\le a_{\max}.
\]

This gives continuous-time bounds without using collocation points for joint constraints.

The QP is built directly with NumPy / SciPy and solved with **OSQP**.

## Current results

The current implementation supports:

- B-spline evaluation for multiple DoFs;
- analytical velocity and acceleration matrices;
- derivative control-point matrices \(D_1,D_2\);
- analytical smoothness matrices \(R_r\);
- joint-space QP trajectory projection;
- joint position, velocity and acceleration limits;
- initial state equality constraints;
- 1-DoF and 2-DoF examples.

The B-spline implementation is validated against `scipy.interpolate.BSpline`, and the optimizer is covered by unit tests.

### 2-DoF example

The first case uses an already feasible nominal trajectory. The optimizer reproduces it almost exactly.

The second case deliberately makes the nominal trajectory leave the admissible joint-position region before returning inside. The optimized B-spline stays inside the limits, deviates only when necessary, and reconnects with the reference afterwards.

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
- add optional terminal conditions;
- introduce Pinocchio and Cartesian tracking;
- move toward sequential QP / receding-horizon optimization;
- later add learned-policy inputs, collision avoidance and bimanual motion.

The core optimizer is intentionally kept independent of ROS2, LeRobot, and any specific robot-learning framework.

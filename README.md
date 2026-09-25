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

The project currently focuses on **offline joint-space trajectory optimization**, with integration and visualization on a **Kinova Gen3 7-DoF robot** using Pinocchio and Meshcat.

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
- continuous-time joint position, velocity and acceleration constraints;
- initial position and velocity constraints;
- 1-DoF and 2-DoF synthetic examples;
- Kinova Gen3 7-DoF URDF integration;
- Pinocchio forward kinematics and Jacobians;
- full-pose SE(3) inverse kinematics;
- sequential IK for Cartesian trajectories;
- Meshcat visualization and animation;
- constrained B-spline projection of a Kinova trajectory.

The B-spline implementation is validated against `scipy.interpolate.BSpline`, and the optimizer and robotics layers are covered by unit tests.

## 2-DoF trajectory projection

The feasible reference is reproduced almost exactly.

The infeasible reference temporarily leaves the admissible joint-position region and later comes back inside. The optimized B-spline stays inside the admissible set, deviates only when necessary, and reconnects with the reference afterwards.

<img width="1252" height="937" alt="2-DoF trajectory projection" src="https://github.com/user-attachments/assets/40ddc832-ad75-4ab0-aeae-c9d16b0c3848" />

<img width="1917" height="943" alt="2-DoF joint trajectories" src="https://github.com/user-attachments/assets/def39557-688f-4a58-99c3-5d19f149e954" />

## Kinova Gen3 trajectory projection

The main robot demonstration uses the following pipeline:

```text
Virtual joint trajectory
          ↓
 Pinocchio forward kinematics
          ↓
 Cartesian target trajectory
          ↓
   Sequential SE(3) IK
          ↓
 Nominal joint trajectory
          ↓
Constrained B-spline projection
          ↓
 Smooth feasible trajectory
          ↓
 Pinocchio forward kinematics
```

The nominal trajectory is deliberately designed to contain joint-limit violations, velocity-limit violations and strong acceleration peaks.

The B-spline optimizer projects this nominal motion onto a smooth trajectory satisfying the joint position, velocity and acceleration constraints.

### Constraint feasibility

The normalized quantities

$$
\rho_q,\qquad
\rho_v,\qquad
\rho_a
$$

measure the maximum usage of the position, velocity and acceleration limits.

A value

$$
\rho \le 1
$$

corresponds to an admissible trajectory.

The nominal trajectory violates all three constraint families, while the optimized trajectory remains on or within the admissible boundary up to the numerical tolerance of OSQP.

<img width="1897" height="937" alt="Kinova trajectory feasibility" src="https://github.com/user-attachments/assets/1089163f-a0ee-4b5d-b3fc-bcc69fe72546" />

### Joint positions

The nominal IK trajectory temporarily leaves the admissible joint-position region.

The optimized B-spline remains inside the joint limits while staying as close as possible to the nominal motion.

<img width="1917" height="942" alt="Kinova joint positions" src="https://github.com/user-attachments/assets/f7f7baab-84b4-4bce-9cdd-23fec38f7bd4" />

### Joint velocities

The same optimization simultaneously enforces the velocity limits.

<img width="1906" height="933" alt="Kinova joint velocities" src="https://github.com/user-attachments/assets/deb7cb92-b039-45b5-93c1-ac52495277d9" />

### Robot animation

The nominal and optimized trajectories are visualized on the Kinova Gen3 using Meshcat.

#### Nominal IK trajectory

The nominal trajectory follows the Cartesian reference but violates the selected joint-space constraints.

<img width="800" height="352" alt="Nominal Kinova trajectory" src="https://github.com/user-attachments/assets/8d2be1a3-ea82-4d20-89a3-9a79782a1b58" />

#### Optimized B-spline trajectory

The optimized trajectory respects the joint-space constraints and produces a substantially smoother motion.

<img width="800" height="351" alt="Optimized Kinova trajectory" src="https://github.com/user-attachments/assets/7fd45f77-54a0-4234-87b4-93e6acff9d26" />

For the current demonstration:

```text
Nominal max position usage:         1.259
Optimized max position usage:       1.000

Nominal max velocity usage:         1.275
Optimized max velocity usage:       1.000

Nominal max acceleration usage:    39.353
Optimized max acceleration usage:   1.001

Nominal integral ||qddot||²:      191.598
Optimized integral ||qddot||²:     14.153
```

The small residual constraint error is within the numerical tolerance of OSQP.

The optimized trajectory therefore reduces the integrated squared acceleration by more than an order of magnitude while enforcing the joint-space constraints.

The current optimizer minimizes tracking error in **joint space**. As a consequence, projecting the joint trajectory can introduce a non-negligible Cartesian deviation from the original end-effector path.

This limitation motivates the next stage of the project: direct Cartesian trajectory optimization.

## Run

Run the unit tests:

```bash
python -m pytest -v
```

Run the synthetic examples:

```bash
python -m examples.trajectory_projection_1d
python -m examples.trajectory_projection_2d
```

Run the Kinova Gen3 trajectory projection:

```bash
python -m examples.kinova_trajectory_projection
```

The Kinova example opens a Meshcat viewer and displays the nominal and optimized robot trajectories.

## Next steps

The current joint-space QP provides a baseline for a more general Cartesian optimization formulation.

The next objective is to optimize the B-spline control points directly with respect to a Cartesian reference:

$$
q(t;c)=A^{(0)}(t)c,
$$

followed by Pinocchio forward kinematics:

$$
M(t;c)=FK(q(t;c)).
$$

A Cartesian tracking objective can then be introduced directly in the optimization problem.

Planned extensions include:

- direct Cartesian B-spline tracking;
- Gauss-Newton / sequential quadratic programming;
- Cartesian constraints;
- collision avoidance;
- warm starts and repeated QP solves;
- receding-horizon / online trajectory optimization;
- dual-arm constraints;
- optional integration with ROS2 or robot-learning frameworks.

The core optimizer is intentionally kept independent of ROS2, LeRobot, and any specific robot-learning framework.

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

## Cartesian optimization

The joint-space projection provides a feasible B-spline trajectory, but minimizing joint-space tracking error does not directly minimize the deviation of the end-effector from the desired Cartesian path.

The Cartesian optimizer therefore optimizes the B-spline control points directly with respect to a Cartesian reference.

For a set of control points $c$, the joint trajectory remains

```math
q(t;c) = A^{(0)}(t)c.
```

At each tracking time $t_k$, Pinocchio forward kinematics maps the corresponding joint configuration to an end-effector pose:

```math
M_k(c) = FK(q(t_k;c)),
```

where $M_k(c) \in SE(3)$.

### Cartesian tracking residual

Two tracking modes are implemented.

For **position-only tracking**, the residual is

```math
r_k(c) = p(q(t_k;c)) - p_{\mathrm{ref},k}.
```

For **full SE(3) pose tracking**, the residual is defined using the logarithmic map:

```math
e_k(c)
=
\log_6
\left(
M_k(c)^{-1} M_{\mathrm{ref},k}
\right).
```

Translation and rotation are weighted separately using

```math
W
=
\mathrm{diag}
\left(
1,1,1,s_R,s_R,s_R
\right).
```

The parameter $s_R$ controls the relative importance of orientation tracking with respect to position tracking.

Setting $s_R = 0$ recovers the position-only formulation.

### Analytical Jacobian

The Cartesian residual is differentiated analytically with respect to the B-spline control points.

Since

```math
q_k = A_k^{(0)}c,
```

the chain rule gives

```math
\frac{\partial e_k}{\partial c}
=
\frac{\partial e_k}{\partial q_k}
A_k^{(0)}.
```

Pinocchio provides the robot frame Jacobian and the differential of the $SE(3)$ logarithmic map.

The resulting Jacobian directly relates variations of the B-spline control points to variations of the end-effector pose.

### Nonlinear objective

The Cartesian optimization problem minimizes

```math
J(c)
=
\sum_k
w_k
\left\|
W e_k(c)
\right\|^2
+
\lambda c^\top R_2 c.
```

The coefficients $w_k$ are trapezoidal integration weights, while

```math
c^\top R_2 c
```

penalizes the integrated squared joint acceleration.

The objective therefore combines Cartesian tracking and joint-space smoothness.

### Constrained Gauss-Newton optimization

Because forward kinematics makes the Cartesian objective nonlinear in the B-spline control points, the problem is solved iteratively using constrained Gauss-Newton steps.

At iteration $i$, the residual is linearized around the current control points $c_i$:

```math
r(c)
\approx
r_i + J_i(c-c_i).
```

The resulting local quadratic problem is

```math
\min_c
\;
\left\|
r_i + J_i(c-c_i)
\right\|^2
+
\lambda c^\top R_2 c
+
\mu \left\|c-c_i\right\|^2.
```

The parameter $\mu$ is a damping term used to regularize the local Gauss-Newton model.

The original B-spline joint constraints are preserved:

```math
q_{\min} \le c \le q_{\max},
```

```math
v_{\min} \le D_1 c \le v_{\max},
```

```math
a_{\min} \le D_2 c \le a_{\max}.
```

Initial position, velocity and optional acceleration constraints are also preserved.

Each Gauss-Newton iteration therefore produces a convex quadratic program that is solved with OSQP.

### Optimization pipeline

```text
Cartesian reference
        ↓
B-spline control points
        ↓
Joint trajectory q(t)
        ↓
Pinocchio forward kinematics
        ↓
Cartesian residuals + Jacobians
        ↓
Gauss-Newton quadratic model
        ↓
B-spline joint constraints
        ↓
OSQP
        ↓
Candidate control points
        ↓
Backtracking line search
        ↓
Updated B-spline trajectory
```

A candidate step is accepted only if it decreases the true nonlinear Cartesian objective.

The formulation therefore preserves the B-spline joint position, velocity and acceleration constraints while optimizing the trajectory directly according to the end-effector motion.

## Results in simulation

The Cartesian formulation is evaluated on the Kinova Gen3 trajectory introduced above.

Four trajectories are compared:

- the nominal trajectory obtained from sequential IK;
- the feasible joint-space B-spline projection;
- Cartesian Gauss-Newton optimization with position-only tracking;
- Cartesian Gauss-Newton optimization with full SE(3) pose tracking.

The nominal trajectory is deliberately aggressive and temporarily violates the joint-position, velocity and acceleration limits. The three optimized B-spline trajectories remain feasible.

### Cartesian tracking

<img width="1917" height="947" alt="Cartesian tracking error" src="https://github.com/user-attachments/assets/fba04aba-ac37-4e70-a7e4-c8215291def2" />

The joint-space QP provides a smooth and feasible trajectory, but minimizing joint-space deviation does not directly minimize the Cartesian error. This becomes particularly visible in the infeasible part of the reference trajectory, where its end-effector position error reaches approximately 200 mm.

Direct Cartesian optimization significantly reduces this deviation.

The **position-only Gauss-Newton formulation** achieves the lowest translational tracking error, but does so by exploiting the kinematic redundancy of the 7-DoF robot. As a consequence, the end-effector orientation can deviate strongly from the nominal motion.

The **full-pose formulation** explicitly penalizes both translation and rotation. It therefore accepts a slightly larger position error than the position-only formulation while maintaining a much smaller orientation error.
The terminal Cartesian cost and zero final joint-velocity condition also bring the optimized trajectories back close to the desired final pose while ending the motion at rest.

### Joint-space behavior

<img width="1917" height="945" alt="Joint trajectories" src="https://github.com/user-attachments/assets/6c998027-9d8a-48b3-a570-dfad5a734470" />

The Cartesian optimizer is free to exploit the robot redundancy instead of remaining close to the nominal IK solution in joint space.

This is particularly visible on the continuous joints, where the position-only and full-pose solutions can follow very different joint trajectories while producing similar end-effector translations.

For the bounded joints, the nominal IK trajectory temporarily exceeds the admissible range, notably on joints 2 and 4. The optimized trajectories remain inside the joint limits.

The difference between the position-only and full-pose solutions also shows why direct Cartesian optimization cannot generally be reproduced by simply projecting an IK trajectory in joint space: different joint configurations can realize similar end-effector positions, while producing very different orientations.

### Constraint satisfaction

<img width="1917" height="938" alt="Trajectory feasibility" src="https://github.com/user-attachments/assets/22b97e6c-277a-4677-9855-fe25fb2e352b" />

The normalized constraint ratios are defined so that

```math
\rho_q \le 1,
\qquad
\rho_v \le 1,
\qquad
\rho_a \le 1
```

correspond to admissible joint position, velocity and acceleration trajectories.

The nominal IK trajectory violates all three types of constraints. In particular, its sharp changes of direction generate very large acceleration peaks.

In contrast, the joint-space QP and both Cartesian Gauss-Newton solutions remain within the admissible region over the complete trajectory, up to numerical solver tolerance.

The experiment therefore highlights the main advantage of the B-spline formulation: Cartesian tracking can be improved significantly while preserving the continuous-time joint position, velocity and acceleration constraints imposed through the B-spline control points and their derivatives.

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

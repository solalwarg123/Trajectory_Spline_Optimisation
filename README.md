# B-Spline Trajectory Optimization for Robot Motion

A lightweight trajectory optimization framework for generating smooth and feasible robot motions using B-splines.

```text
High-level policy / planner
            ↓
     trajectory reference
            ↓
   B-spline representation
            ↓
 joint-space QP / Cartesian GN
            ↓
 smooth feasible trajectory
            ↓
      robot controller
```

The project focuses on **B-spline trajectory optimization in both joint and Cartesian space**, with integration and visualization on a **Kinova Gen3 7-DoF robot** using Pinocchio and Meshcat.

The current implementation includes:

```text
joint-space convex projection
        +
direct Cartesian optimization
        +
continuous-time joint constraints
```

The joint-space formulation provides a convex baseline, while the Cartesian formulation directly optimizes the B-spline control points according to end-effector tracking objectives.

---

## B-spline trajectory representation

A multi-DoF joint trajectory is represented as

```math
q(t)=A^{(0)}(t)c,
```

with derivatives

```math
\dot q(t)=A^{(1)}(t)c,
\qquad
\ddot q(t)=A^{(2)}(t)c,
```

where $c$ contains the stacked B-spline control points.

The same representation is used by both the joint-space and Cartesian optimizers.

The integrated squared joint acceleration can be written analytically as

```math
\int
\left\|
\ddot q(t)
\right\|^2 dt
=
c^\top R_2 c,
```

where $R_2$ is the B-spline smoothness matrix.

This term is used as a trajectory regularizer throughout the project.

---

## Joint-space QP baseline

The first optimization formulation projects a nominal joint trajectory onto the feasible B-spline set.

The objective is

```math
\min_c
\;
\left\|
A_{\mathrm{track}}c-b_{\mathrm{track}}
\right\|^2
+
\lambda c^\top R_2 c.
```

The first term tracks the nominal joint trajectory, while the second penalizes joint acceleration.

Initial conditions can be imposed as

```math
A^{(0)}(t_0)c=q_0,
```

```math
A^{(1)}(t_0)c=\dot q_0,
```

with an optional initial acceleration condition

```math
A^{(2)}(t_0)c=\ddot q_0.
```

A terminal joint velocity can also be imposed when the motion is required to end at rest:

```math
A^{(1)}(t_f)c=\dot q_f.
```

### Continuous-time joint constraints

Joint constraints are enforced using the B-spline convex-hull property.

Joint position limits are imposed directly on the control points:

```math
q_{\min}\le c\le q_{\max}.
```

Velocity bounds are imposed on the first derivative control points:

```math
v_{\min}\le D_1c\le v_{\max}.
```

Acceleration bounds are imposed on the second derivative control points:

```math
a_{\min}\le D_2c\le a_{\max}.
```

These conditions provide sufficient continuous-time guarantees without requiring dense trajectory collocation.

The resulting problem is a **convex quadratic program**, constructed directly with NumPy / SciPy sparse matrices and solved with **OSQP**.

---

## Current features

The project currently includes:

- multi-DoF B-spline evaluation;
- analytical velocity and acceleration matrices;
- derivative control-point matrices $D_1$ and $D_2$;
- analytical smoothness matrices;
- joint-space trajectory projection with OSQP;
- continuous-time joint position, velocity and acceleration constraints;
- initial position, velocity and optional acceleration constraints;
- optional terminal velocity constraints;
- 1-DoF and 2-DoF synthetic examples;
- Kinova Gen3 7-DoF URDF integration;
- Pinocchio forward kinematics and analytical Jacobians;
- full-pose SE(3) inverse kinematics;
- sequential IK for Cartesian trajectories;
- direct Cartesian B-spline optimization;
- position-only and full SE(3) tracking objectives;
- constrained Gauss-Newton iterations with OSQP subproblems;
- terminal Cartesian tracking cost;
- joint-space / Cartesian baseline comparison;
- Meshcat visualization and animation;
- regression tests for the B-spline, robotics and Cartesian optimization layers.

The B-spline implementation is validated against `scipy.interpolate.BSpline`, and the analytical Cartesian Jacobians are validated numerically using finite differences.

---

## 2-DoF trajectory projection

The synthetic 2-DoF examples illustrate the basic projection behavior before introducing robot kinematics.

A feasible reference is reproduced almost exactly.

An infeasible reference temporarily leaves the admissible joint-position region and later comes back inside. The optimized B-spline remains inside the admissible set, deviates only when necessary, and reconnects with the reference afterwards.

<img width="1252" height="937" alt="2-DoF trajectory projection" src="https://github.com/user-attachments/assets/40ddc832-ad75-4ab0-aeae-c9d16b0c3848" />

<img width="1917" height="943" alt="2-DoF joint trajectories" src="https://github.com/user-attachments/assets/def39557-688f-4a58-99c3-5d19f149e954" />

---

## Kinova Gen3 joint-space projection

The first Kinova demonstration uses the following pipeline:

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

The nominal trajectory is deliberately designed to contain:

```text
joint-position violations
velocity-limit violations
strong acceleration peaks
```

The joint-space B-spline optimizer projects this nominal motion onto a smooth trajectory satisfying the selected position, velocity and acceleration constraints.

### Constraint feasibility

The normalized quantities

```math
\rho_q,
\qquad
\rho_v,
\qquad
\rho_a
```

measure the maximum usage of the position, velocity and acceleration limits.

A value

```math
\rho \le 1
```

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

For this joint-space demonstration:

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

However, the joint-space objective minimizes deviation from the nominal **joint trajectory**. It does not directly minimize the resulting Cartesian deviation of the end-effector.

The joint-space projection is therefore used as a baseline for the direct Cartesian formulation presented below.

---

## Cartesian optimization

The joint-space projection provides a feasible B-spline trajectory, but minimizing joint-space tracking error does not directly minimize the deviation of the end-effector from the desired Cartesian path.

The Cartesian optimizer therefore optimizes the B-spline control points directly with respect to a Cartesian reference.

For a set of control points $c$, the joint trajectory remains

```math
q(t;c)=A^{(0)}(t)c.
```

At each tracking time $t_k$, Pinocchio forward kinematics maps the corresponding joint configuration to an end-effector pose:

```math
M_k(c)=FK(q(t_k;c)),
```

where $M_k(c)\in SE(3)$.

### Cartesian tracking residual

Two tracking modes are implemented.

For **position-only tracking**, the residual is

```math
r_k(c)
=
p(q(t_k;c))
-
p_{\mathrm{ref},k}.
```

For **full SE(3) pose tracking**, the residual is defined using the logarithmic map:

```math
e_k(c)
=
\log_6
\left(
M_k(c)^{-1}
M_{\mathrm{ref},k}
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

The parameter $s_R$ controls the relative importance of orientation tracking with respect to position tracking and has units of meters per radian.

In the implementation,

```python
orientation_scale = 0.0
```

explicitly selects the position-only residual.

A positive value activates full SE(3) tracking.

### Analytical Jacobian

The Cartesian residual is differentiated analytically with respect to the B-spline control points.

Since

```math
q_k=A_k^{(0)}c,
```

the chain rule gives

```math
\frac{\partial e_k}{\partial c}
=
\frac{\partial e_k}{\partial q_k}
A_k^{(0)}.
```

For position-only tracking, the translational part of the Pinocchio frame Jacobian is used.

For full SE(3) tracking, the derivative combines the local frame Jacobian with the differential of the SE(3) logarithmic map provided by Pinocchio.

The resulting Jacobian directly relates variations of the B-spline control points to variations of the end-effector pose.

### Nonlinear objective

The Cartesian tracking term is integrated numerically using trapezoidal quadrature.

An additional terminal Cartesian cost gives more importance to reaching the desired final pose.

The objective is

```math
J(c)
=
\sum_k
w_k
\left\|
W e_k(c)
\right\|^2
+
\lambda c^\top R_2 c
+
\lambda_f
\left\|
W e_f(c)
\right\|^2.
```

Here:

- $w_k$ are trapezoidal integration weights;
- $\lambda$ controls joint-space smoothness;
- $\lambda_f$ controls the additional terminal tracking cost.

For the position-only formulation, the corresponding translational residual is used instead of the full SE(3) residual.

The smoothness term

```math
c^\top R_2 c
```

penalizes the integrated squared joint acceleration.

The terminal cost improves final-pose tracking without imposing a particular final IK configuration, which is useful for a redundant 7-DoF manipulator.

### Terminal velocity

For motions that are expected to end at rest, the Cartesian optimization can additionally impose

```math
\dot q(t_f)=0.
```

This provides a physical terminal condition while leaving the final joint configuration free.

The combination used in the Kinova demonstration is therefore:

```text
strong final Cartesian tracking
            +
zero final joint velocity
            +
free final joint configuration
```

### Constrained Gauss-Newton optimization

Forward kinematics makes the Cartesian objective nonlinear in the B-spline control points.

The problem is solved iteratively using constrained Gauss-Newton steps.

At iteration $i$, the residual is linearized around the current control points $c_i$:

```math
r(c)
\approx
r_i+J_i(c-c_i).
```

The tracking residual used here includes both the time-integrated tracking terms and, when enabled, the additional terminal residual.

The resulting local quadratic problem has the form

```math
\min_c
\;
\left\|
r_i+J_i(c-c_i)
\right\|^2
+
\lambda c^\top R_2 c
+
\mu
\left\|
c-c_i
\right\|^2,
```

where $\mu$ is a damping parameter.

The original B-spline joint constraints are preserved:

```math
q_{\min}\le c\le q_{\max},
```

```math
v_{\min}\le D_1c\le v_{\max},
```

```math
a_{\min}\le D_2c\le a_{\max}.
```

Initial conditions and the optional terminal velocity constraint are included in the same linear constraint system.

Each Gauss-Newton iteration therefore produces a convex quadratic program solved with OSQP.

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

The current solver uses fixed damping and a backtracking line search.

A candidate step is accepted only if it decreases the true nonlinear Cartesian objective.

The formulation therefore optimizes end-effector tracking directly while preserving the continuous-time joint position, velocity and acceleration guarantees provided by the B-spline representation.

---

## Results in simulation

The Cartesian formulation is evaluated on the same Kinova Gen3 trajectory used for the joint-space baseline.

Four trajectories are compared:

- the nominal trajectory obtained from sequential IK;
- the feasible joint-space B-spline projection;
- Cartesian Gauss-Newton optimization with position-only tracking;
- Cartesian Gauss-Newton optimization with full SE(3) pose tracking.

The nominal trajectory is deliberately aggressive and temporarily violates the joint-position, velocity and acceleration limits.

The three optimized B-spline trajectories remain feasible.

### Cartesian tracking

<img width="1917" height="947" alt="Cartesian tracking error" src="https://github.com/user-attachments/assets/fba04aba-ac37-4e70-a7e4-c8215291def2" />

The joint-space QP provides a smooth and feasible trajectory, but minimizing joint-space deviation does not directly minimize Cartesian error.

This becomes particularly visible in the infeasible part of the reference trajectory, where the end-effector position error of the joint-space projection reaches approximately 200 mm.

Direct Cartesian optimization significantly reduces this deviation.

The **position-only Gauss-Newton formulation** achieves the lowest translational tracking error, but does so by exploiting the kinematic redundancy of the 7-DoF robot.

Because orientation is not present in this objective, the end-effector orientation can deviate strongly from the nominal motion.

The **full-pose formulation** explicitly penalizes both translation and rotation.

It therefore accepts a slightly larger position error than the position-only formulation while maintaining a much smaller orientation error.

This illustrates the expected trade-off:

```text
position-only optimization
        ↓
best position tracking
        ↓
orientation is free to drift
```

compared with

```text
full SE(3) optimization
        ↓
position + orientation compromise
        ↓
better preservation of the complete end-effector motion
```

The terminal Cartesian cost and zero final joint-velocity condition also bring the optimized trajectories back close to the desired final pose while ending the motion at rest.

### Joint-space behavior

<img width="1917" height="945" alt="Joint trajectories" src="https://github.com/user-attachments/assets/6c998027-9d8a-48b3-a570-dfad5a734470" />

The Cartesian optimizer is free to exploit the robot redundancy instead of remaining close to the nominal IK solution in joint space.

This is particularly visible on the continuous joints, where the position-only and full-pose solutions can follow very different joint trajectories while producing similar end-effector translations.

For the bounded joints, the nominal IK trajectory temporarily exceeds the admissible range, notably on joints 2 and 4.

The optimized trajectories remain inside the joint limits.

The difference between the position-only and full-pose solutions also shows why direct Cartesian optimization cannot generally be reproduced by simply projecting an IK trajectory in joint space:

```text
similar end-effector position
        ≠
unique joint configuration
```

Different joint configurations can realize similar end-effector positions while producing very different orientations.

### Constraint satisfaction

<img width="1917" height="938" alt="Trajectory feasibility" src="https://github.com/user-attachments/assets/22b97e6c-277a-4677-9855-fe25fb2e352b" />

The normalized constraint ratios are defined so that

```math
\rho_q\le1,
\qquad
\rho_v\le1,
\qquad
\rho_a\le1
```

correspond to admissible joint position, velocity and acceleration trajectories.

The nominal IK trajectory violates all three types of constraints.

In particular, its sharp changes of direction generate very large acceleration peaks.

In contrast, the joint-space QP and both Cartesian Gauss-Newton solutions remain within the admissible region over the complete trajectory, up to numerical solver tolerance.

The experiment therefore highlights the main advantage of the B-spline formulation:

```text
direct Cartesian tracking
          +
smooth joint motion
          +
continuous-time q / v / a constraints
```

---

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

Run the Kinova Gen3 joint-space projection:

```bash
python -m examples.kinova_trajectory_projection
```

Run the Cartesian optimization comparison:

```bash
python -m examples.kinova_cartesian_optimization
```

The Kinova examples use Pinocchio for robot kinematics and Meshcat for interactive visualization and animation.

---

## Next steps

The current implementation provides both a convex joint-space baseline and a direct constrained Cartesian optimizer.

The next development steps focus on improving solver robustness and extending the formulation toward more general nonlinear trajectory optimization problems.

### Solver improvements

The current Cartesian solver uses constrained Gauss-Newton steps with fixed damping and backtracking line search.

The next objectives are:

- adaptive Levenberg-Marquardt damping;
- more robust convergence criteria;
- clearer solver termination statuses;
- warm starts and more efficient repeated QP solves.

### Solver validation

The custom Gauss-Newton / SQP solver will be compared against IPOPT on the same nonlinear B-spline trajectory optimization problem.

The comparison will include:

- final objective value;
- Cartesian tracking accuracy;
- constraint satisfaction;
- trajectory smoothness;
- convergence behavior;
- computation time.

IPOPT will be used as an offline numerical reference rather than as a replacement for the custom solver.

### Nonlinear constraints

Once the solver is sufficiently robust, the formulation will be extended beyond the current linear B-spline joint constraints.

The first planned extension is **collision avoidance** through nonlinear distance constraints.

This naturally extends the current constrained Gauss-Newton formulation toward a more general **Sequential Quadratic Programming (SQP)** framework.

Longer-term extensions include:

- Cartesian task constraints;
- receding-horizon / online trajectory optimization;
- dual-arm coordination;

# B-Spline Trajectory Optimization for Robot Motion

A lightweight trajectory optimization framework for generating smooth, constrained robot motions using B-splines, from offline trajectory projection to receding-horizon Cartesian MPC.

```text
High-level policy / planner
        ↓
trajectory reference
        ↓
B-spline representation
        ↓
joint-space QP / Cartesian GN / MPC
        ↓
smooth feasible trajectory
        ↓
robot controller
```

The project covers **joint-space and Cartesian B-spline trajectory optimization**, nonlinear collision avoidance, and closed-loop Cartesian MPC, demonstrated on a **Kinova Gen3 7-DoF robot** using Pinocchio and Meshcat.

The current implementation includes:

```text
joint-space convex projection
        +
direct Cartesian optimization
        +
continuous-time joint constraints
        +
collision-aware receding-horizon MPC
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
- adaptive Levenberg–Marquardt damping and an IPOPT comparison baseline;
- capsule–sphere collision constraints with sampled collocation checks;
- cached collision-geometry computations and optional per-obstacle activation;
- receding-horizon Cartesian MPC with disturbance rejection;
- native OSQP projection of infeasible MPC warm starts onto updated linear constraints;
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

### Adaptive damping experiment

An adaptive damping strategy inspired by Levenberg–Marquardt was evaluated against the default fixed-damping Gauss–Newton solver. The damping parameter was updated according to the agreement between the nonlinear objective decrease and the decrease predicted by the local Gauss–Newton model.

On the current Kinova Cartesian tracking benchmark, the benefits were mixed. For full SE(3) tracking, adaptive damping reached a final dense tracking error of **13.06 mm RMS in position** and **5.85° RMS in orientation**, with a maximum orientation error of **15.94°**. This is only a modest improvement over the fixed-damping baseline, which achieved approximately **13.35 mm**, **5.81°**, and **16.97°**, respectively.

For position-only tracking, adaptive damping performed worse than the fixed strategy: the dense position RMS error was **10.41 mm**, compared with approximately **9.00 mm** using fixed damping.

Adaptive damping also introduces additional QP solves when the line search fails and the step is recomputed with increased damping. Given the limited and problem-dependent improvements observed so far, **fixed damping remains the default strategy**, while adaptive damping is kept as an experimental option. The MPC experiments use adaptive damping; its additional robustness must still be balanced against computation time.

### Solver benchmark

A small benchmark was conducted on five Kinova Gen3 trajectory scenarios with increasingly restrictive joint-angle constraints. The joint-space QP provides a very fast feasible baseline, but Cartesian tracking degrades significantly as the joint constraints become more restrictive. The static Gauss-Newton solver gives a much better tracking / computation trade-off, reducing the mean position tracking error from roughly 98 mm for the joint-space baseline to about 17 mm, with a median solve time of approximately 3.5 s.

IPOPT with a limited-memory BFGS Hessian approximation reaches slightly better final solutions when allowed to run for many iterations, especially for orientation tracking and the most constrained scenarios. However, this comes at a much larger computational cost: IPOPT with 200 iterations takes about 25 s on the current benchmark, compared with about 3.5 s for Gauss-Newton. With only 20 iterations, IPOPT is faster than Gauss-Newton but remains significantly less accurate, showing that the specialized Gauss-Newton formulation exploits the least-squares structure of the problem more efficiently during the first iterations.

These results motivate keeping Gauss-Newton as the main specialized solver while using IPOPT as a more general nonlinear optimization baseline. Collision avoidance and receding-horizon control are discussed in the following sections.

### Collision-aware Cartesian optimization

The Cartesian optimization layer also supports nonlinear robot–obstacle distance constraints. Robot links are approximated with **capsules**, and static obstacles with **spheres**. Distances are evaluated using Pinocchio-based kinematics, and collision constraints are incorporated into the iterative optimization at selected trajectory collocation times.

For a capsule–obstacle pair, a typical clearance requirement is

```math
d_{ij}(q(t_k;c)) \ge d_{\mathrm{safe}},
```

where $d_{ij}$ is the signed surface clearance and $d_{\mathrm{safe}}$ is a nonnegative safety margin. Unlike the joint-position, velocity, and acceleration constraints, **these nonlinear collision constraints are enforced on a finite collocation grid**; they do not constitute a continuous-time collision-avoidance guarantee.

The collision-aware optimizer has been demonstrated on a Kinova Gen3 arm navigating a nearby static obstacle. The nonlinear objective and constraints are handled through successive local optimization problems, while the B-spline representation retains its continuous-time joint-limit guarantees when the returned solution is linearly feasible.

For MPC, an optional sampled broad-phase identifies obstacles close to the candidate trajectory. The optimizer initially includes only those obstacles, checks the resulting trajectory for newly approached obstacles, and may re-solve with additional constraints or the full obstacle set. This reduces unnecessary nonlinear constraint evaluations but is **not a substitute for continuous collision checking**.

### Online Cartesian MPC

`CartesianMPC` extends the offline Cartesian optimizer to **receding-horizon tracking**. At each control cycle, it updates the measured joint position and velocity and the future Cartesian pose targets, solves a new B-spline optimization problem, and executes only the first part of the planned motion.

```text
Current robot state + future Cartesian targets
                      ↓
             Update MPC problem
                      ↓
         Rebase previous B-spline
                      ↓
    Project warm start if linearly infeasible
                      ↓
     Constrained Cartesian Gauss–Newton
          (+ optional collisions)
                      ↓
        Execute first control interval
                      ↓
              New robot state
```

The MPC maintains continuous-time joint position, velocity and acceleration bounds through derivative-control-point constraints, subject to the feasibility and numerical accuracy of each solve. Closed-loop performance is evaluated in simulation under injected acceleration disturbances.

#### Native feasible warm-start restoration

A previously optimized B-spline is not necessarily feasible after the MPC initial state changes. Simply imposing the new initial position and velocity can also violate the joint derivative bounds. Starting Gauss–Newton from an infeasible iterate can cause its line search to return a constraint-violating point even when an individual QP subproblem is feasible.

The MPC therefore **rebases the previous solution and, when needed, projects it onto the updated linear feasible set**:

```math
\min_c \; \frac{1}{2}\|c-c_{\mathrm{guess}}\|^2
\qquad
\mathrm{s.t.}\quad l_k \le A_k c \le u_k.
```

The projection uses **OSQP**, with a reusable solver workspace when the sparse constraint matrix is unchanged. This step is implemented in `CartesianMPC` (`project_warm_start=True` by default), rather than in individual examples. It restores *linear* feasibility when the feasible set is nonempty; it does not guarantee nonlinear convergence or collision feasibility.

#### Closed-loop cluttered-tracking benchmark

The benchmark compares open-loop execution (`OPEN`), unconstrained-by-obstacle Cartesian MPC (`FREE`) and collision-aware Cartesian MPC (`SAFE`) on the same **10-second** feasible joint trajectory. All runs use an MPC horizon of **1.5 s**, a control interval of **0.1 s (10 Hz)**, three static sphere obstacles, a **20 mm** safety margin, and Gaussian acceleration perturbations with standard deviation **0.35 rad/s²**. The suffix `GN2` or `GN5` denotes the maximum number of Gauss–Newton iterations per MPC solve.

A benchmark using an **example-local feasible warm-start projection** produced the following results:

| Method | Completed cycles | Mean solve time | Position RMS error | Minimum observed clearance |
|---|---:|---:|---:|---:|
| OPEN | 55/100 | — | 57.25 mm | −5.6 mm |
| FREE-GN2 | 100/100 | 16.49 ms | 3.59 mm | 25.6 mm |
| SAFE-GN2 | 100/100 | 160.94 ms | 3.59 mm | 25.5 mm |
| FREE-GN5 | 100/100 | 36.04 ms | 3.59 mm | 25.5 mm |
| SAFE-GN5 | 54/100 | 399.44 ms | 3.86 mm | 25.5 mm |

*The times and RMS errors for incomplete runs are computed over their successfully completed cycles. These are simulation timings, not hardware-in-the-loop measurements.*

Open-loop execution accumulates disturbances, violates the selected clearance margin, and eventually reaches a joint/velocity limit. Feedback MPC maintains millimeter-scale Cartesian position errors on this trial. With these particular obstacle placements, `FREE` already follows a collision-free path; accordingly, this benchmark demonstrates the **computational cost of collision constraints**, not a case in which obstacle avoidance is required for successful execution.

The same experiment was subsequently run with the projection **integrated directly in `CartesianMPC`**. The native version completed all 100 cycles for FREE-GN2 and FREE-GN5, but SAFE-GN2 encountered a linear acceleration-constraint violation of approximately **7.78 × 10⁻⁴** at cycle 29, despite a feasible projected initialization. The native warm-start rebasing also caused larger pre-projection violations and intermittent extra projection cost. Thus **the native MPC refactor is functional for the tested obstacle-free configurations, but the collision-aware path still needs robustness work**. Results from the two implementations should not be conflated.

The nominal controller period is 100 ms. FREE-GN2 is a promising candidate for faster experiments, but average solve time alone does not establish a 30 Hz real-time guarantee. Collision-aware MPC in this setup substantially exceeds the 100 ms cycle budget.

Run the simpler closed-loop tracking example:

```bash
python -m examples.kinova_mpc_safe_tracking --mode gaussian --noise-accel-std 0.05 --max-iterations 2 --no-animation --show-plots
```

Run the cluttered-tracking comparison:

```bash
python -m examples.kinova_mpc_cluttered_tracking --gn-iters 2 5 --noise-std 0.35 --show-plots
```

The cluttered-tracking example exports per-cycle metrics, summary CSV files, and figures to `outputs/kinova_mpc_cluttered_tracking/`. The `--debug-projection` option prints warm-start feasibility diagnostics. The `--no-project-warm-start` option disables the native projection for comparison.

---

## Run

Run the unit tests:

```bash
python -m pytest -q
```

The current test suite includes 93 passing tests, covering the B-spline, robotic kinematics and optimization layers, and basic MPC state handling. Dedicated regression tests for **native warm-start projection** are still planned.

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

The main goal is to demonstrate a **high-level motion policy coupled to a constrained, smooth, receding-horizon B-spline trajectory optimizer**. Planned work is deliberately focused on integration and reproducibility rather than indefinitely tuning the warm start or collision solver.

1. **MPC robustness and tests.** Add focused tests for native warm-start projection; ensure solver termination status and linear/collision feasibility are checked independently; address the remaining collision-aware failure modes. Avoid accepting a nonconverged result merely because its linear constraints are satisfied.
2. **Mock high-level policy.** Simulate asynchronously updated, noisy or abruptly changing Cartesian targets (including held targets and inference latency) and evaluate how the MPC filters these commands into feasible motions. Start with the simpler collision-free MPC.
3. **Reproducible benchmarks.** Compare tracking accuracy, constraint violations, success rate, median/P95 computation time and projection overhead over multiple disturbance seeds. Distinguish offline-simulation throughput from demonstrated real-time performance.
4. **Portfolio presentation.** Simplify example scripts, polish plots, document modeling assumptions and numerical limitations, and provide a concise demo video and reproducible command lines.

Potential later extensions include continuous collision validation, moving obstacles, higher-rate MPC, ROS 2 or LeRobot integration, and dual-arm coordination. These are not required for the initial portfolio release.

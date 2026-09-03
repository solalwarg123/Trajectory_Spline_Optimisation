# B-Spline Trajectory Optimization for Learned Robot Policies

## 1. Project overview

The goal of this project is to develop a generic trajectory optimization layer for robotic manipulators.

The long-term idea is to separate robot control into two levels:

1. **High-level trajectory generation**

   * Eventually produced by a learned policy, VLA, imitation-learning policy, or another planner.
   * The high-level module proposes what the robot should do.
   * It may generate either:

     * joint-space trajectories;
     * Cartesian end-effector trajectories;
     * B-spline control points directly.

2. **Low-level constrained trajectory optimization**

   * Converts the proposed trajectory into a smooth and physically feasible joint trajectory.
   * Attempts to remain as close as possible to the high-level proposal.
   * Enforces robot constraints.
   * Initially based on convex optimization.
   * Later extended to nonlinear Cartesian tracking and collision avoidance.

The central design principle is:

```text
High-level trajectory proposal
            ↓
   Trajectory optimization
            ↓
 Smooth and feasible B-spline
            ↓
          Robot
```

The project should remain independent of any specific robot-learning framework.

LeRobot, ROS2, and real robot integrations may be added later as interfaces around the core optimization library.

---

# 2. Main research idea

A learned policy can generate useful robot motions without necessarily respecting:

* joint position limits;
* joint velocity limits;
* joint acceleration limits;
* smoothness requirements;
* Cartesian constraints;
* self-collision constraints;
* collisions between two robot arms.

Instead of requiring the learned model itself to satisfy all these constraints, the objective is to introduce a **trajectory projection / optimization layer**.

Given a nominal trajectory

$$
\tau_{\mathrm{nom}},
$$

we compute a feasible trajectory

$$
\tau^\star
$$

which remains close to the nominal one while satisfying the robot constraints.

Conceptually:

$$
\tau^\star
=
\operatorname{argmin}_{\tau \in \mathcal{F}}
\left(
J_{\mathrm{tracking}}(\tau,\tau_{\mathrm{nom}})
+
\lambda J_{\mathrm{smooth}}(\tau)
\right),
$$

where \(\mathcal{F}\) denotes the set of feasible trajectories.

The trajectory is represented using B-splines.

---

# 3. Why B-splines?

B-splines provide:

* a compact finite-dimensional trajectory representation;
* local control through control points;
* analytical derivatives;
* smooth trajectories;
* sparse structure;
* convenient quadratic smoothness costs;
* useful convex-hull properties for continuous-time bounds;
* an efficient representation for repeated online optimization.

The optimization variables will primarily be the joint-space B-spline control points.

For a spline trajectory:

$$
q(t)=A^{(0)}(t)c,
$$

its derivatives satisfy:

$$
\dot q(t)=A^{(1)}(t)c,
$$

$$
\ddot q(t)=A^{(2)}(t)c,
$$

where \(c\) is the stacked vector of joint-space control points.

---

# 4. Target architecture

Long-term architecture:

```text
        Camera / proprioception / task
                    │
                    ▼
          High-level policy/planner
       ACT / VLA / BSP / other source
                    │
          nominal trajectory
                    │
                    ▼
       B-Spline Trajectory Optimizer
    ┌───────────────────────────────┐
    │ trajectory tracking           │
    │ acceleration / jerk penalty   │
    │ joint limits                  │
    │ velocity limits               │
    │ acceleration limits           │
    │ Cartesian constraints         │
    │ collision avoidance           │
    └───────────────┬───────────────┘
                    │
           feasible trajectory
                    │
                    ▼
            robot controller
```

The core optimization package should **not depend on LeRobot or ROS2**.

Future integrations should be treated as adapters:

```text
core/
integrations/
    lerobot/
    ros2/
```

---

# 5. Development philosophy

The project will be developed progressively.

Each stage should:

* be independently testable;
* produce useful results;
* reuse the previous stage;
* avoid adding nonlinear complexity before the simpler problem is validated;
* include numerical tests and visualizations.

The project should not initially focus on AI training.

Until a real learned policy is integrated, a synthetic trajectory generator or recorded trajectory can play the role of the high-level policy.

---

# 6. Phase 0 — B-spline mathematical core

## Goal

Implement and validate the B-spline trajectory representation.

## Features

Implement:

* B-spline knot vectors;
* control points;
* evaluation matrix \(A^{(0)}(t)\);
* velocity matrix \(A^{(1)}(t)\);
* acceleration matrix \(A^{(2)}(t)\);
* optionally jerk matrix \(A^{(3)}(t)\);
* acceleration regularization matrix \(R_2\).

The API should eventually allow:

```python
trajectory.position(t)
trajectory.velocity(t)
trajectory.acceleration(t)
```

## Validation

Use `scipy.interpolate.BSpline` as a numerical reference.

Tests should verify:

* position evaluation;
* first derivative;
* second derivative;
* dimensions;
* continuity;
* endpoint behavior;
* different knot vectors;
* multi-DoF trajectories.

## Deliverable

A reliable `BSplineTrajectory` abstraction independent of any optimizer.

---

# 7. Phase 1 — Joint-space convex trajectory projection

## Goal

Build the first complete trajectory optimizer.

The nominal trajectory is initially given directly in joint space:

$$
q_{\mathrm{ref}}(t).
$$

Solve:

$$
\min_c
J_{\mathrm{tracking}}(c)
+
\lambda_a J_{\mathrm{acc}}(c).
$$

For example:

$$
J_{\mathrm{tracking}}
=
\sum_i
\left\|
A^{(0)}(t_i)c-q_{\mathrm{ref}}(t_i)
\right\|_W^2,
$$

and:

$$
J_{\mathrm{acc}}
=
c^\top R_2c.
$$

Constraints:

$$
q_{\min}\le q(t)\le q_{\max},
$$

$$
\dot q_{\min}\le\dot q(t)\le\dot q_{\max},
$$

$$
\ddot q_{\min}\le\ddot q(t)\le\ddot q_{\max}.
$$

Initial conditions:

$$
q(t_0)=q_0,
$$

$$
\dot q(t_0)=\dot q_0.
$$

Optional:

$$
\ddot q(t_0)=\ddot q_0.
$$

## Solver

Start with:

```text
CVXPY + OSQP
```

The objective is clarity and correctness, not maximum performance.

Later, construct the QP matrices directly:

$$
\min_c
\frac12 c^\top P c + q^\top c
$$

subject to

$$
l\le Ac\le u.
$$

This will allow efficient sparse solving and warm starts.

---

# 8. Continuous-time joint constraints

Two approaches should be investigated.

## Method A — Collocation

Evaluate:

$$
q,\dot q,\ddot q
$$

at several points per spline segment.

Advantages:

* simple;
* flexible;
* directly compatible with arbitrary constraints.

Disadvantage:

* constraint satisfaction is only checked at selected points.

## Method B — B-spline control polygon bounds

Use derivative control points:

$$
c^{(1)}=D_1c,
$$

$$
c^{(2)}=D_2c.
$$

Then constrain:

$$
q_{\min}\le c\le q_{\max},
$$

$$
v_{\min}\le D_1c\le v_{\max},
$$

$$
a_{\min}\le D_2c\le a_{\max}.
$$

Using the convex-hull property, this can provide continuous-time guarantees.

Both approaches may eventually be supported and benchmarked.

---

# 9. First experimental milestone

Create an intentionally infeasible reference trajectory.

Example:

* 1 DoF;
* reference requires excessive velocity and/or acceleration.

Compare:

```text
Nominal trajectory
vs.
Projected trajectory
```

Plot:

* position;
* velocity;
* acceleration;
* tracking error.

Report:

* maximum position;
* maximum velocity;
* maximum acceleration;
* RMS deviation from nominal;
* objective value;
* solver time;
* constraint violations.

Then repeat on a multi-DoF artificial trajectory.

This milestone should demonstrate the core idea before introducing robot kinematics.

---

# 10. Phase 2 — Robot kinematics

## Goal

Introduce a real robot model while keeping the optimization architecture unchanged.

Preferred initial library:

```text
Pinocchio
```

Use a URDF to provide:

* forward kinematics;
* end-effector position;
* end-effector pose;
* Jacobians;
* eventually collision geometry.

Start with a single manipulator.

## Initial scope

First support Cartesian **position only**:

$$
x_{\mathrm{ref}}(t)\in\mathbb R^3.
$$

Only after this works reliably, extend to full poses:

$$
X_{\mathrm{ref}}(t)\in SE(3).
$$

---

# 11. Phase 2A — Pointwise IK baseline

Before implementing Cartesian trajectory optimization, construct a baseline:

```text
Cartesian reference
        ↓
pointwise inverse kinematics
        ↓
joint reference
        ↓
joint-space trajectory projector
```

This baseline is useful because it separates:

* IK;
* trajectory optimization.

It will also provide a comparison against the later integrated Cartesian optimizer.

Potential problems to observe:

* branch switching;
* discontinuous IK solutions;
* high joint velocities;
* poor use of redundancy;
* trajectories close to joint limits.

---

# 12. Phase 2B — Cartesian trajectory optimization

Instead of solving IK independently at every point, optimize the joint spline directly.

Keep the optimization variables:

$$
c.
$$

The joint trajectory remains:

$$
q(t)=A^{(0)}(t)c.
$$

The Cartesian trajectory is:

$$
x(t)=f(q(t)).
$$

Given Cartesian references \(x_{\mathrm{ref},i}\):

$$
J_{\mathrm{Cartesian}}
=
\sum_i
\left\|
f(A^{(0)}(t_i)c)
-
x_{\mathrm{ref},i}
\right\|^2.
$$

This introduces non-convexity.

Instead of immediately using a generic nonlinear solver, investigate a sequential approach.

At current iterate \(c^k\):

$$
q_i^k=A_i c^k.
$$

Linearize using the robot Jacobian:

$$
f(q_i^k+\delta q_i)
\approx
f(q_i^k)
+
J(q_i^k)\delta q_i.
$$

Since:

$$
\delta q_i=A_i\delta c,
$$

the local Cartesian tracking objective becomes quadratic in \(\delta c\).

The subproblem can therefore remain a QP.

Conceptually:

```text
current B-spline
      ↓
FK + Jacobians
      ↓
linearize Cartesian tracking
      ↓
solve convex QP
      ↓
update control points
      ↓
repeat
```

This corresponds to a Gauss-Newton / SQP / sequential convex optimization approach.

---

# 13. Full SE(3) trajectories

Once Cartesian position tracking works, extend the target to complete end-effector poses.

Use:

$$
X(t)\in SE(3)
$$

and define a pose error through the Lie algebra:

$$
e=
\log
\left(
X_{\mathrm{ref}}^{-1}X(q)
\right)
\in\mathbb R^6.
$$

Translation and orientation may require different weights.

This phase should only be attempted after the simpler XYZ implementation is stable.

---

# 14. Phase 3 — Online / receding-horizon execution

## Goal

Move from offline trajectory optimization to repeated trajectory updates.

A high-level source periodically produces a new trajectory chunk:

```text
time t0:
policy → trajectory A
           ↓
        optimizer
           ↓
        execute

time t1:
policy → trajectory B
           ↓
        optimizer
           ↓
        execute
```

Important questions:

* how often does the high-level planner run?
* how often does the trajectory optimizer run?
* how much of each optimized trajectory is executed?
* how should consecutive trajectory chunks be joined?

Important continuity constraints include:

$$
q_{\mathrm{new}}(0)=q_{\mathrm{current}},
$$

$$
\dot q_{\mathrm{new}}(0)=\dot q_{\mathrm{current}},
$$

and potentially:

$$
\ddot q_{\mathrm{new}}(0)=\ddot q_{\mathrm{current}}.
$$

This phase is closely related to receding-horizon control and should reuse warm starts from the previous trajectory.

---

# 15. Phase 4 — Learned trajectory sources

Only after the optimizer itself is stable should a learned policy be introduced.

Possible trajectory sources include:

* ACT;
* diffusion policies;
* VLA models;
* B-Spline Policy;
* recorded demonstrations;
* other LeRobot policies.

The optimizer should not need to know how the nominal trajectory was generated.

A generic interface should be sufficient:

```python
nominal_trajectory = trajectory_source.predict(observation)
optimized_trajectory = optimizer.project(
    nominal_trajectory,
    current_state
)
```

Learning how to train these policies is useful but is **not the central contribution of the project**.

---

# 16. Phase 5 — Collision avoidance

Collision avoidance should be treated as a later extension because it introduces additional non-convex geometry.

Possible collision types:

* robot/environment;
* self-collision;
* left-arm/right-arm collision in a bimanual robot.

For collision pair \((a,b)\):

$$
d_{a,b}(q)\ge d_{\min}.
$$

At the current trajectory \(q^k\), the distance constraint may be locally linearized:

$$
d(q^k+\delta q)
\approx
d(q^k)
+
\nabla d(q^k)^\top\delta q.
$$

Since:

$$
\delta q=A\delta c,
$$

the resulting local constraint can again be inserted into the convex QP.

The optimization architecture becomes:

```text
Nonlinear robot geometry
       │
       ├── FK / Cartesian errors
       │
       └── collision distances
       │
       ▼
   linearization
       │
       ▼
     QP solve
       │
       ▼
 trajectory update
       │
       └──────── repeat
```

Pinocchio and associated collision libraries may be used for geometry and distance computations.

---

# 17. Phase 6 — Bimanual robots

Extend the configuration to:

$$
q=
\begin{pmatrix}
q_L\\
q_R
\end{pmatrix}.
$$

Support:

* separate Cartesian trajectories for both end effectors;
* shared optimization of both arms;
* individual joint limits;
* inter-arm collision avoidance;
* potentially coordination objectives.

This phase should reuse the same B-spline and SQP infrastructure rather than introduce a separate solver.

---

# 18. Possible future LeRobot integration

LeRobot is **not required for the first stages**.

A future LeRobot adapter could:

1. retrieve observations from a robot;
2. obtain an action chunk or trajectory from a policy;
3. convert it into the generic trajectory representation;
4. optimize it;
5. execute the resulting trajectory.

LeRobot should therefore be treated as a trajectory source / robot backend rather than as a dependency of the optimization core.

---

# 19. Possible future ROS2 integration

ROS2 is also not necessary initially.

It may become useful for:

* robot drivers;
* URDF models;
* visualization;
* sensor integration;
* distributed components;
* collision checking;
* real robotic hardware.

The core trajectory optimization library should remain usable without ROS.

---

# 20. Initial software structure

Tentative structure:

```text
project/
│
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
│
├── src/
│   └── trajectory_optimizer/
│       │
│       ├── splines/
│       │   ├── basis.py
│       │   ├── trajectory.py
│       │   └── derivatives.py
│       │
│       ├── optimization/
│       │   ├── objectives.py
│       │   ├── constraints.py
│       │   ├── problem.py
│       │   └── result.py
│       │
│       ├── solvers/
│       │   ├── cvxpy_solver.py
│       │   └── osqp_solver.py
│       │
│       ├── kinematics/
│       │   └── pinocchio_model.py
│       │
│       └── integrations/
│           ├── lerobot/
│           └── ros2/
│
├── examples/
│   ├── joint_1d.py
│   ├── joint_multidof.py
│   └── cartesian_tracking.py
│
├── tests/
│
├── benchmarks/
│
├── docs/
│   └── formulation.md
│
└── assets/
```

This structure is provisional and should remain simple during early development.

---

# 21. Short-term roadmap

## Milestone 0 — Repository setup

* [ ] Create local repository
* [ ] Add `.gitignore`
* [ ] Add Python environment / `pyproject.toml`
* [ ] Add this `PROJECT_PLAN.md`
* [ ] Add initial mathematical formulation to `docs/formulation.md`
* [ ] Push private repository

## Milestone 1 — B-spline core

* [ ] Implement knot-vector handling
* [ ] Implement B-spline evaluation matrices
* [ ] Implement first derivative
* [ ] Implement second derivative
* [ ] Compute acceleration regularization matrix
* [ ] Compare against SciPy
* [ ] Add unit tests

## Milestone 2 — Joint-space QP

* [ ] Implement joint trajectory tracking cost
* [ ] Implement initial conditions
* [ ] Implement position constraints
* [ ] Implement velocity constraints
* [ ] Implement acceleration constraints
* [ ] Solve with CVXPY + OSQP
* [ ] Create 1-DoF example
* [ ] Create multi-DoF example
* [ ] Plot results
* [ ] Measure solve time

## Milestone 3 — Better constraint formulation

* [ ] Implement derivative control-point matrices
* [ ] Investigate continuous-time bounds
* [ ] Compare control-polygon constraints with collocation
* [ ] Benchmark conservatism and runtime

## Milestone 4 — Direct QP

* [ ] Explicitly build \(P,q,A,l,u\)
* [ ] Solve directly with OSQP
* [ ] Add warm start
* [ ] Benchmark against CVXPY formulation

## Milestone 5 — Robot kinematics

* [ ] Load a robot URDF with Pinocchio
* [ ] Implement FK
* [ ] Implement Jacobian
* [ ] Validate Jacobians numerically
* [ ] Implement Cartesian XYZ trajectory representation

## Milestone 6 — Cartesian optimization

* [ ] Implement pointwise IK baseline
* [ ] Implement Cartesian tracking residual
* [ ] Implement Gauss-Newton / SQP iteration
* [ ] Add trust region if necessary
* [ ] Compare with pointwise IK baseline
* [ ] Extend to SE(3)

## Milestone 7 — Online trajectory optimization

* [ ] Simulate repeated trajectory chunks
* [ ] Enforce continuity between chunks
* [ ] Warm-start from previous solution
* [ ] Measure online solve frequency
* [ ] Study robustness to abrupt reference changes

## Milestone 8 — Real trajectory source

* [ ] Integrate an existing learned policy
* [ ] Possibly integrate B-Spline Policy
* [ ] Possibly integrate LeRobot
* [ ] Compare nominal vs optimized execution

## Milestone 9 — Collision avoidance

* [ ] Add collision geometry
* [ ] Compute pairwise distances
* [ ] Compute / approximate distance gradients
* [ ] Add sequentially convexified collision constraints
* [ ] Test self-collision
* [ ] Test bimanual collision avoidance

---

# 22. Experimental metrics

The project should systematically measure:

### Tracking

* RMS trajectory error;
* maximum trajectory error;
* Cartesian error;
* orientation error.

### Motion quality

* maximum velocity;
* maximum acceleration;
* maximum jerk;
* integral squared acceleration;
* integral squared jerk.

### Feasibility

* joint-limit violations;
* velocity violations;
* acceleration violations;
* collision violations.

### Optimization

* mean solve time;
* median solve time;
* p95 solve time;
* maximum solve time;
* number of SQP iterations;
* convergence rate;
* effect of warm start.

### Task-level performance

Once real policies and hardware are introduced:

* task success rate;
* execution time;
* tracking error;
* failure modes;
* difference between nominal and optimized trajectories.

---

# 23. Important design principles

1. **Keep the B-spline core independent of robotics frameworks.**

2. **Use joint-space spline control points as the main optimization variables.**

3. **Treat Cartesian tracking as an objective on top of the joint trajectory rather than fixing an IK solution beforehand.**

4. **Use pointwise IK as a baseline, not necessarily as the final architecture.**

5. **Exploit convex structure whenever possible.**

6. **Do not introduce collision avoidance before the basic trajectory optimizer is validated.**

7. **Prefer sequential convexification / SQP before abandoning the QP structure for a fully generic nonlinear solver.**

8. **Benchmark everything that is supposed to run online.**

9. **Separate trajectory representation, optimization formulation, solver, robot model, and external integrations.**

10. **Each milestone should produce a working, testable result.**

---

# 24. What is explicitly out of scope for the beginning

Do **not** initially spend time on:

* training AI policies;
* ROS2 architecture;
* LeRobot internals;
* real-time hardware communication;
* collision checking;
* dual-arm coordination;
* visualization frameworks;
* GUI development;
* package publication;
* extensive documentation for external users.

The immediate priority is:

```text
B-spline representation
        ↓
joint-space constrained QP
        ↓
validated numerical results
```

Everything else can be built on top of that.

---

# 25. Current immediate task

The first implementation target is:

> Given a joint-space reference trajectory, compute the closest smooth B-spline trajectory satisfying joint position, velocity, and acceleration constraints.

The initial development sequence is therefore:

```text
1. Repository setup
2. B-spline matrices
3. SciPy validation
4. Smoothness matrix
5. Joint-space QP
6. Synthetic infeasible trajectory
7. Plots + tests + benchmarks
```

No robot model is required before this milestone works reliably.

---

# 26. Long-term objective

The eventual objective is a generic real-time trajectory layer capable of taking trajectories proposed by learned robotic policies and transforming them into smooth, dynamically reasonable, collision-aware motions for single-arm and bimanual robots.

The intended progression is:

```text
Joint-space convex projection
            ↓
Cartesian trajectory optimization
            ↓
Online / receding-horizon optimization
            ↓
Learned policy integration
            ↓
Collision avoidance
            ↓
Bimanual robotic execution
```

The project should remain useful even if some of the later stages are never reached: the joint-space and Cartesian constrained B-spline optimizer should already form a complete standalone contribution.

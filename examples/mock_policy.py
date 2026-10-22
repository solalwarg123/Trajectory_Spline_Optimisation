
"""Reproducible noisy Cartesian action chunks for the Kinova Gen3 example.

No inference is actually run: inference_latency is simulated by the
consumer (kinova_policy_mpc.py), which makes new chunks available later.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from examples.kinova_trajectory_projection import (
    Q_MIN, Q_MAX, V_MIN, V_MAX, A_MIN, A_MAX,
)


TIME_SCALE = 2.30

_BASE_TIMES = np.array([
    0.0, 0.5, 1.5, 2.6, 3.8, 5.0, 6.3, 7.7, 9.0
])

WAYPOINT_TIMES = TIME_SCALE * _BASE_TIMES
DURATION = float(WAYPOINT_TIMES[-1])

WAYPOINTS = np.array([
    [0.00, 0.45,  0.15, -1.35, -0.15, 0.75,  0.10],
    [0.00, 0.45,  0.15, -1.35, -0.15, 0.75,  0.10],
    [0.35, 1.25,  0.45, -1.05, -0.45, 0.85,  0.30],
    [0.70, 2.10,  0.75, -2.40, -0.65, 1.10,  0.55],
    [0.95, 2.15,  0.20, -2.35,  0.20, 1.40, -0.30],
    [0.55, 1.55, -0.40, -1.55,  0.55, 1.15, -0.50],
    [0.10, 0.65, -0.15, -1.25,  0.00, 0.80,  0.00],
    [0.10, 0.65, -0.15, -1.25,  0.00, 0.80,  0.00],
    [0.10, 0.65, -0.15, -1.25,  0.00, 0.80,  0.00],
])


def nominal_state(t):
    """Piecewise smoothstep reference: q, qdot, qddot.

    Continuous position/velocity, possibly discontinuous
    acceleration at waypoints. Supports scalar or vector t.
    """
    scalar = np.ndim(t) == 0

    tt = np.atleast_1d(np.asarray(t, dtype=float))
    tt = np.clip(tt, 0.0, DURATION)

    idx = np.minimum(
        np.searchsorted(
            WAYPOINT_TIMES[1:],
            tt,
            side="right",
        ),
        len(WAYPOINT_TIMES) - 2,
    )

    h = WAYPOINT_TIMES[idx + 1] - WAYPOINT_TIMES[idx]
    s = (tt - WAYPOINT_TIMES[idx]) / h

    d = WAYPOINTS[idx + 1] - WAYPOINTS[idx]

    q = (
        WAYPOINTS[idx]
        + (3*s*s - 2*s*s*s)[:, None] * d
    )

    v = (
        ((6*s - 6*s*s) / h)[:, None] * d
    )

    a = (
        ((6 - 12*s) / (h*h))[:, None] * d
    )

    if scalar:
        return q[0], v[0], a[0]

    return q, v, a


def normalized_usage(x, lower, upper):
    x = np.asarray(x, dtype=float)

    return np.maximum(
        np.maximum(x, 0.0) / upper,
        np.minimum(x, 0.0) / lower,
    )


def nominal_diagnostics(n_samples=5001):
    times = np.linspace(0.0, DURATION, n_samples)
    q, v, a = nominal_state(times)

    q_viol = np.maximum(
        np.maximum(Q_MIN - q, q - Q_MAX), 0.0
    )
    v_viol = np.maximum(
        np.maximum(V_MIN - v, v - V_MAX), 0.0
    )
    a_viol = np.maximum(
        np.maximum(A_MIN - a, a - A_MAX), 0.0
    )

    return {
        "times": times,
        "q": q,
        "v": v,
        "a": a,
        "max_q_violation": float(np.max(q_viol)),
        "max_v_violation": float(np.max(v_viol)),
        "max_a_violation": float(np.max(a_viol)),
        "accel_violation_pct": 100 * float(
            np.mean(np.any(a_viol > 1e-8, axis=1))
        ),
        "max_v_usage": np.max(
            normalized_usage(v, V_MIN, V_MAX),
            axis=0,
        ),
        "max_a_usage": np.max(
            normalized_usage(a, A_MIN, A_MAX),
            axis=0,
        ),
    }


@dataclass
class ActionChunk:
    start_time: float
    dt: float
    poses: list
    nominal_poses: list

    @property
    def size(self):
        return len(self.poses)

    @property
    def end_time(self):
        return (
            self.start_time
            + (self.size - 1) * self.dt
        )

    def pose_at(self, t):
        """Hold-pad the final prediction after chunk end."""
        i = int(np.floor(
            (float(t) - self.start_time) / self.dt
            + 1e-7
        ))

        i = int(np.clip(i, 0, self.size - 1))

        return self.poses[i]

    def horizon(self, t, relative_times):
        """Build an MPC horizon with final-pose hold padding."""
        return [
            self.pose_at(float(t) + float(tau))
            for tau in relative_times
        ]


class MockChunkPolicy:
    """Generate noisy Cartesian action chunks."""

    def __init__(
        self,
        robot,
        chunk_size=16,
        dt=0.1,
        noise_std=0.0025,
        bias_std=0.001,
        tremor_amplitude=0.002,
        tremor_frequency=3.5,
        seed=42,
    ):
        self.robot = robot
        self.chunk_size = int(chunk_size)
        self.dt = float(dt)

        self.noise_std = float(noise_std)
        self.bias_std = float(bias_std)

        self.tremor_amplitude = float(
            tremor_amplitude
        )
        self.tremor_frequency = float(
            tremor_frequency
        )

        self.rng = np.random.default_rng(seed)

        if self.chunk_size < 2 or self.dt <= 0:
            raise ValueError(
                "Invalid policy chunk size/dt"
            )

        if min(
            self.noise_std,
            self.bias_std,
            self.tremor_amplitude,
            self.tremor_frequency,
        ) < 0:
            raise ValueError(
                "Noise parameters must be nonnegative"
            )

    def predict(self, t):
        """Predict one chunk, sampling its noise only once."""
        start = float(t)

        bias = self.rng.normal(
            0.0, self.bias_std, 3
        )

        phases = self.rng.uniform(
            0.0, 2*np.pi, 3
        )

        poses = []
        nominal_poses = []

        for i in range(self.chunk_size):
            q, _, _ = nominal_state(
                start + i*self.dt
            )

            fk = self.robot.frame_pose(
                self.robot.joint_positions_to_configuration(q)
            )

            R = fk.rotation.copy()
            p = fk.translation.copy()

            nominal_poses.append(
                pin.SE3(R.copy(), p.copy())
            )

            eps = self.rng.normal(
                0.0, self.noise_std, 3
            )

            tremor = (
                self.tremor_amplitude
                * np.sin(
                    2*np.pi
                    * self.tremor_frequency
                    * i
                    * self.dt
                    + phases
                )
            )

            poses.append(
                pin.SE3(
                    R.copy(),
                    p + bias + eps + tremor,
                )
            )

        return ActionChunk(
            start,
            self.dt,
            poses,
            nominal_poses,
        )


def pregenerate_chunks(policy, duration=DURATION):
    """Generate identical predictions for all methods."""
    chunk_period = (
        policy.chunk_size - 1
    ) * policy.dt

    return [
        policy.predict(t)
        for t in np.arange(
            0.0,
            duration - 1e-9,
            chunk_period,
        )
    ]


def available_chunk_index(
    chunks,
    time_s,
    inference_latency,
):
    """Find the newest chunk actually available.

    The first chunk is preloaded at t=0.
    All later chunks become available at
    start_time + inference_latency.
    """
    times = np.array(
        [c.start_time for c in chunks],
        dtype=float,
    )

    times[1:] += float(inference_latency)

    idx = np.searchsorted(
        times,
        float(time_s) + 1e-9,
        side="right",
    ) - 1

    return int(np.clip(
        idx, 0, len(chunks) - 1
    ))

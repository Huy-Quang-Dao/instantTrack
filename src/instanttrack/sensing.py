"""Policy observation as the robot's sensors and state estimator produce it.

Observation layout (same as :meth:`instanttrack.mdp.MDP.obs`): q - q0 - offset, qd, angular velocity,
projected gravity, previous action, reference-relative root orientation (6D). The default
:class:`SensingCfg` is the paper's: true state plus uniform noise (Table A.1). A robot can add its
signal chain, as on a real robot's state estimator:

* IMU: mounting tilt and gyro bias (per episode, in xi), gyro/accelerometer low-pass and a
  complementary tilt estimate; the policy's attitude and angular velocity come from the estimate.
* Joint velocity: per-joint first-order low-pass with a per-episode time-constant scale and raw noise.
* Observation delay: 0 or 1 policy step on everything read from hardware, lag held with a probability.

These are stateful. :class:`ObsPipeline` steps them on the simulator every physics step (from
MuJoCo's IMU sensors) and in imagination once per control step (from world-model states); the raw
joint-velocity noise is rescaled so both give the same filtered variance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch

from .rotations import mat_to_6d, quat_to_mat, skew, so3_exp

if TYPE_CHECKING:
    from .dr import XiLayout
    from .mdp import MDP
    from .robot import RobotSpec

GRAVITY = 9.81


@dataclass(frozen=True)
class ImuCfg:
    site: str = "imu_in_pelvis"
    gyro_sensor: str = "imu_ang_vel"
    accel_sensor: str = "imu_lin_acc"
    lowpass_tau: float = 0.019
    """Gyro/accelerometer low-pass time constant (s)."""
    tilt_tau: float = 1.0
    """Time constant of the pull toward the accelerometer's gravity direction (s)."""
    tilt: float = 0.0
    """Mounting roll/pitch half-range (rad), per episode."""
    gyro_bias: float = 0.0
    """Per-axis gyro bias half-range (rad/s), per episode; the estimate does not remove it."""


@dataclass(frozen=True)
class JointVelFilterCfg:
    tau: dict[str, float] = field(default_factory=dict)
    """Joint regex -> low-pass time constant (s); other joints pass through."""
    tau_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-episode scale on every time constant."""
    raw_noise_std: float = 0.0
    """Gaussian noise on the raw velocity before the filter, per physics step (rad/s)."""


@dataclass(frozen=True)
class SensingCfg:
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.5
    noise_ang_vel: float = 0.2
    noise_gravity: float = 0.05
    noise_ref_ori: float = 0.05
    """Uniform observation-noise half-widths (defaults: paper Table A.1)."""
    delay_hold_prob: float | None = None
    """None: no observation delay. Otherwise a 0/1 policy-step lag per sensor group, kept with this
    probability each step and resampled otherwise."""
    imu: ImuCfg | None = None
    joint_vel_filter: JointVelFilterCfg | None = None

    @property
    def stateful(self) -> bool:
        return self.delay_hold_prob is not None or self.imu is not None or self.joint_vel_filter is not None


def obs_noise(cfg: SensingCfg, J: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.full((J,), cfg.noise_joint_pos),
            torch.full((J,), cfg.noise_joint_vel),
            torch.full((3,), cfg.noise_ang_vel),
            torch.full((3,), cfg.noise_gravity),
            torch.zeros(J),
            torch.full((6,), cfg.noise_ref_ori),
        )
    )


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest rotation taking unit vector a to unit vector b (Rodrigues, smooth near identity)."""
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True).unsqueeze(-1)
    K = skew(v)
    eye = torch.eye(3, device=a.device, dtype=a.dtype).expand_as(K)
    return eye + K + (K @ K) / (1.0 + c).clamp(min=1e-6)


class ObsPipeline:
    """Builds the policy observation from (true) state and the sensing state of each env.

    The sensing state is a dict: ``qd_f`` (N, J) filtered joint velocity, ``g_hat``/``w_f``/``a_f``
    (N, 3) IMU estimate of up, filtered gyro and accelerometer (IMU frame), ``prev_raw`` (N, obs_dim)
    last undelayed observation, ``lag`` (N, 4) current delay per sensor group.
    """

    def __init__(self, robot: RobotSpec, mdp: MDP, layout: XiLayout, device: str, physics_dt: float):
        self.cfg, self.mdp, self.layout = robot.sensing, mdp, layout
        self.physics_dt = physics_dt
        self.stateful = self.cfg.stateful
        J = robot.num_joints
        self.J, self.obs_dim, self.device = J, mdp.obs_dim, device
        # delay group of every observation dim: q, qd, angular velocity, attitude (gravity + ref ori); -1 = exact
        group = [0] * J + [1] * J + [2] * 3 + [3] * 3 + [-1] * J + [3] * 6
        self.group = torch.tensor([max(g, 0) for g in group], device=device)
        self.delayed = torch.tensor([g >= 0 for g in group], device=device)
        jf = self.cfg.joint_vel_filter
        tau = [0.0] * J
        if jf is not None:
            for i, n in enumerate(robot.joint_names):
                for pat, t in jf.tau.items():
                    if re.fullmatch(pat, n):
                        tau[i] = t
        self.qd_tau = torch.tensor(tau, device=device)
        imu = self.cfg.imu
        if imu is not None:
            m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
            sid = m.site(imu.site).id
            assert m.body(m.site_bodyid[sid]).name == robot.tracked_bodies[0], "IMU site must be on the root body"
            self.site_pos = torch.tensor(m.site_pos[sid], dtype=torch.float32, device=device)
            self.site_rot = quat_to_mat(torch.tensor(m.site_quat[sid], dtype=torch.float32, device=device))
        self.sizes = {"qd_f": J, "g_hat": 3, "w_f": 3, "a_f": 3, "prev_raw": self.obs_dim, "lag": 4}
        self.state_dim = sum(self.sizes.values())

    # ------------------------------------------------------------------ state storage
    def pack(self, st: dict) -> torch.Tensor:
        return torch.cat([st[k].float() for k in self.sizes], -1)

    def unpack(self, x: torch.Tensor) -> dict:
        return dict(zip(self.sizes, x.split(list(self.sizes.values()), -1)))

    # ------------------------------------------------------------------ IMU geometry
    def _mount(self, xi: torch.Tensor) -> torch.Tensor:
        """IMU rotation in the root frame: XML site rotation times the mounting tilt from xi."""
        if self.cfg.imu.tilt > 0:
            rp = self.layout.get(xi, "imu_tilt")
            tilt = so3_exp(torch.cat((rp, torch.zeros_like(rp[..., :1])), -1))
            return self.site_rot @ tilt
        return self.site_rot.expand(xi.shape[:-1] + (3, 3))

    def imu_from_states(self, S0, S1, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Gyro and accelerometer in the XML site frame from two consecutive states (imagination)."""
        p0, R0, v0, w0 = S0[:4]
        R1, v1, w1 = S1[1], S1[2], S1[3]
        r_w = (R1 @ self.site_pos.unsqueeze(-1)).squeeze(-1)
        acc = (v1 - v0) / dt + torch.cross((w1 - w0) / dt, r_w, dim=-1) + torch.cross(w1, torch.cross(w1, r_w, dim=-1), dim=-1)
        acc = acc + torch.tensor([0.0, 0.0, GRAVITY], device=acc.device)
        Rs_t = (R1 @ self.site_rot).transpose(-1, -2)
        return (Rs_t @ w1.unsqueeze(-1)).squeeze(-1), (Rs_t @ acc.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------ filters
    def init_state(self, S, xi, gyro_site: torch.Tensor, accel_site: torch.Tensor) -> dict:
        """Filters converged on the current values (as on a robot running since boot)."""
        n = xi.shape[0]
        st = {"qd_f": S[5].clone(), "prev_raw": torch.zeros(n, self.obs_dim, device=xi.device),
              "lag": torch.zeros(n, 4, device=xi.device)}  # fmt: skip
        if self.cfg.imu is not None:
            M = self._mount(xi)
            Mt = M.transpose(-1, -2)
            st["g_hat"] = (S[1] @ M)[..., 2, :]
            st["w_f"] = (Mt @ gyro_site.unsqueeze(-1)).squeeze(-1) + self.layout.get(xi, "gyro_bias")
            st["a_f"] = (Mt @ accel_site.unsqueeze(-1)).squeeze(-1)
        else:
            st["g_hat"] = st["w_f"] = st["a_f"] = torch.zeros(n, 3, device=xi.device)
        return st

    def filter_update(self, st: dict, qd, gyro_site, accel_site, xi, dt: float, noise: bool, ctrl_rate: bool = False) -> dict:
        st = dict(st)
        jf = self.cfg.joint_vel_filter
        if jf is not None:
            tau = self.qd_tau * self.layout.get(xi, "qd_tau_scale")
            k = torch.where(tau > 0, 1.0 - torch.exp(-dt / tau.clamp(min=1e-9)), 1.0)
            if noise and jf.raw_noise_std > 0:
                std = torch.full_like(k, jf.raw_noise_std)
                if ctrl_rate:  # same filtered noise variance as per-physics-step noise: var = std^2 k / (2 - k)
                    k_p = torch.where(tau > 0, 1.0 - torch.exp(-self.physics_dt / tau.clamp(min=1e-9)), 1.0)
                    std = std * torch.sqrt((k_p / (2 - k_p)) * ((2 - k) / k))
                qd = qd + torch.randn_like(qd) * std
            st["qd_f"] = st["qd_f"] + k * (qd - st["qd_f"])
        imu = self.cfg.imu
        if imu is not None:
            Mt = self._mount(xi).transpose(-1, -2)
            gyro = (Mt @ gyro_site.unsqueeze(-1)).squeeze(-1) + self.layout.get(xi, "gyro_bias")
            accel = (Mt @ accel_site.unsqueeze(-1)).squeeze(-1)
            k_lp = 1.0 - math.exp(-dt / imu.lowpass_tau)
            k_tilt = 1.0 - math.exp(-dt / imu.tilt_tau)
            w_f = st["w_f"] + k_lp * (gyro - st["w_f"])
            a_f = st["a_f"] + k_lp * (accel - st["a_f"])
            g = st["g_hat"] - torch.cross(w_f, st["g_hat"], dim=-1) * dt
            g = g + k_tilt * (_normalize(a_f) - g)
            st["w_f"], st["a_f"], st["g_hat"] = w_f, a_f, _normalize(g)
        return st

    # ------------------------------------------------------------------ observation
    def raw_obs(self, st: dict, S, a_prev, xi, f) -> torch.Tensor:
        """Undelayed, noise-free observation from the true state and the sensing state."""
        p, R, v, w, q, qd = S
        dq0 = self.layout.get(xi, "joint_offset")
        if self.cfg.imu is not None:
            R_site = R @ self._mount(xi)
            R_att = R_site @ _align(st["g_hat"], R_site[..., 2, :])  # estimated attitude: up direction = g_hat
            ang_vel = st["w_f"]
        else:
            R_att = R
            ang_vel = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        qd_obs = st["qd_f"] if self.cfg.joint_vel_filter is not None else qd
        g_b = -R_att[..., 2, :]
        rel = mat_to_6d(R_att.transpose(-1, -2) @ self.mdp.motion.root_R[f])
        return torch.cat((q - self.mdp.default_q - dq0, qd_obs, ang_vel, g_b, a_prev, rel), -1)

    def observe(self, st: dict, raw: torch.Tensor, noise: bool, delay: bool) -> tuple[torch.Tensor, dict]:
        st = dict(st)
        o = raw
        if delay and self.cfg.delay_hold_prob is not None:
            lag = st["lag"]
            resample = torch.rand_like(lag) >= self.cfg.delay_hold_prob
            lag = torch.where(resample, torch.randint_like(lag, 0, 2), lag)
            late = (lag[:, self.group] > 0.5) & self.delayed
            o = torch.where(late, st["prev_raw"], raw)
            st["lag"] = lag
        st["prev_raw"] = raw
        if noise:
            o = self.mdp.add_noise(o)
        return o, st

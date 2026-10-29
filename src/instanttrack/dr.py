"""Simulator domain randomization and the layout of the sim context xi.

Every randomized simulator parameter is part of xi, which the world model and critic see (the
policy does not). The defaults are the paper's G1 setup: friction once per env, joint (encoder)
offsets per episode, and a CoM offset of ``RobotSpec.com_body`` once per env. A robot can switch on
more terms in ``RobotSpec.dr``; each is applied with the corresponding mjlab ``dr`` function where
mjlab has one, and its values are read back from the model into xi.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from .robot import RobotSpec


@dataclass(frozen=True)
class DomainRandCfg:
    friction_range: tuple[float, float] = (0.3, 1.2)
    """Contact friction, sampled once per env."""
    friction_geoms: str | None = None
    """Regex of the robot geoms that get it (None: every robot geom). Robot geoms win over the
    ground either by MuJoCo's max-of-pair rule (ground at the bottom of the range) or by priority."""
    joint_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
    """Joint regex -> half-range (rad) of the encoder offset, per episode: the policy reads q + bias and
    the PD target moves by -bias, as on a robot whose encoder zero is off."""
    com_mass_range: tuple[float, float] | None = None
    """None: ``com_body`` CoM offset once per env. Otherwise ``com_body`` is randomized per episode by
    pseudo-inertia: mass scale in this range and CoM shift in ``RobotSpec.com_range``."""
    link_mass_range: tuple[float, float] | None = None
    """Per-link mass scale (inertia scaled with it, pseudo-inertia), per episode."""
    link_mass_exclude: tuple[str, ...] = ()
    """Bodies kept out of the link-mass term (``com_body`` always is)."""
    solref_geoms: str | None = None
    solref_timeconst: tuple[float, float] = (0.02, 0.02)
    solref_dampratio: tuple[float, float] = (1.0, 1.0)
    """Contact solref of ``solref_geoms`` (shared across them), per episode."""
    pd_gain_scale: tuple[tuple[tuple[int, ...], tuple[float, float], tuple[float, float]], ...] = ()
    """(actuator ids, kp scale range, kd scale range), per joint and episode."""
    armature_range: tuple[float, float] | None = None
    damping_range: tuple[float, float] | None = None
    """Per-joint scale of the rotor armature / passive joint damping, per episode."""
    push_interval_s: tuple[float, float] | None = None
    push_velocity: tuple[float, ...] = (0.0,) * 6
    """Root velocity kick (x, y, z, roll, pitch, yaw half-ranges) every push_interval_s."""
    reset_pos: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rot: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint_pos: float = 0.1
    reset_joint_vel: float = 0.0
    """Half-ranges of the offsets from the reference state at episode start."""


def _match(patterns: dict[str, float], names, default: float = 0.0) -> list[float]:
    out = []
    for n in names:
        val = default
        for pat, v in patterns.items():
            if re.fullmatch(pat, n):
                val = v
        out.append(val)
    return out


class XiLayout:
    """Slices of xi and its normalization (center, half-range) for the networks."""

    def __init__(self, robot: RobotSpec, device: str):
        c, s = robot.dr, robot.sensing
        J = robot.num_joints
        self.slices: dict[str, slice] = {}
        centers, scales = [], []

        def add(name: str, center: list[float], scale: list[float]):
            n = sum(sl.stop - sl.start for sl in self.slices.values())
            self.slices[name] = slice(n, n + len(center))
            centers.extend(center), scales.extend(scale)

        lo, hi = c.friction_range
        add("friction", [(lo + hi) / 2], [(hi - lo) / 2])
        add("joint_offset", [0.0] * J, [max(h, 1e-6) for h in _match(c.joint_offset, robot.joint_names)])
        add("com", [0.0] * 3, [0.05] * 3)  # fixed scale (the paper setup's normalization)
        if c.com_mass_range is not None:
            add("com_mass", [sum(c.com_mass_range) / 2], [(c.com_mass_range[1] - c.com_mass_range[0]) / 2])
        self.link_bodies: list[str] = []
        if c.link_mass_range is not None:
            m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
            excl = (robot.com_body, *c.link_mass_exclude)
            self.link_bodies = [m.body(b).name for b in range(1, m.nbody)
                                if not any(re.fullmatch(p, m.body(b).name) for p in excl)]  # fmt: skip
            add("link_mass", [sum(c.link_mass_range) / 2] * len(self.link_bodies),
                [(c.link_mass_range[1] - c.link_mass_range[0]) / 2] * len(self.link_bodies))  # fmt: skip
        if c.solref_geoms is not None:
            add("solref", [sum(c.solref_timeconst) / 2, sum(c.solref_dampratio) / 2],
                [max((c.solref_timeconst[1] - c.solref_timeconst[0]) / 2, 1e-6),
                 max((c.solref_dampratio[1] - c.solref_dampratio[0]) / 2, 1e-6)])  # fmt: skip
        if c.pd_gain_scale:
            add("kp", [1.0] * J, [0.2] * J)
            add("kd", [1.0] * J, [0.5] * J)
        if c.armature_range is not None:
            add("armature", [sum(c.armature_range) / 2] * J, [(c.armature_range[1] - c.armature_range[0]) / 2] * J)
        if c.damping_range is not None:
            add("damping", [sum(c.damping_range) / 2] * J, [(c.damping_range[1] - c.damping_range[0]) / 2] * J)
        if s.imu is not None:
            add("imu_tilt", [0.0, 0.0], [max(s.imu.tilt, 1e-6)] * 2)
            add("gyro_bias", [0.0] * 3, [max(s.imu.gyro_bias, 1e-6)] * 3)
        if s.joint_vel_filter is not None:
            lo, hi = s.joint_vel_filter.tau_scale_range
            add("qd_tau_scale", [(lo + hi) / 2], [max((hi - lo) / 2, 1e-6)])
        self.dim = len(centers)
        self.center = torch.tensor(centers, device=device)
        self.scale = torch.tensor(scales, device=device)

    def normalize(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.center) / self.scale

    def get(self, xi: torch.Tensor, name: str) -> torch.Tensor:
        return xi[..., self.slices[name]]


class DomainRandomizer:
    """Samples and applies the per-env simulator parameters of ``robot.dr`` and writes them into xi.

    ``env`` is the :class:`instanttrack.env.TrackingSim` (it carries ``scene``, ``sim``, ``num_envs`` and
    ``device``, which mjlab's ``dr`` functions use).
    """

    def __init__(self, robot: RobotSpec, layout: XiLayout, env, eval_mode: bool):
        self.cfg, self.spec, self.layout, self.env = robot.dr, robot, layout, env
        self.eval_mode = eval_mode
        c, sim, ent = self.cfg, env.sim, env.robot
        dev = env.device
        self.J = robot.num_joints
        self.com_body_id = ent.indexing.body_ids[ent.find_bodies(robot.com_body)[0][0]]
        self.joint_offset_half = torch.tensor(_match(c.joint_offset, robot.joint_names), device=dev)
        geom_names = ent.geom_names
        self.friction_geoms = (ent.indexing.geom_ids if c.friction_geoms is None else
                               ent.indexing.geom_ids[[i for i, n in enumerate(geom_names) if re.fullmatch(c.friction_geoms, n)]])  # fmt: skip
        self.ground_geoms = torch.nonzero(torch.as_tensor(sim.mj_model.geom_bodyid == 0, device=dev)).flatten()
        if c.solref_geoms is not None:
            self.solref_geoms = ent.indexing.geom_ids[[i for i, n in enumerate(geom_names) if re.fullmatch(c.solref_geoms, n)]]
        self.dof_ids = ent.indexing.joint_v_adr

        # mjlab dr terms used per episode, with their resolved entity configs
        self._terms: list[tuple] = []
        if c.com_mass_range is not None:
            (t1, t2, t3) = robot.com_range
            cfg = SceneEntityCfg("robot", body_names=(robot.com_body,))
            alpha = (math.log(c.com_mass_range[0]) / 2, math.log(c.com_mass_range[1]) / 2)
            self._terms.append((dr.pseudo_inertia, cfg, dict(alpha_range=alpha, t1_range=t1, t2_range=t2, t3_range=t3)))
        if c.link_mass_range is not None:
            cfg = SceneEntityCfg("robot", body_names=tuple(f"^{re.escape(b)}$" for b in layout.link_bodies))
            alpha = (math.log(c.link_mass_range[0]) / 2, math.log(c.link_mass_range[1]) / 2)
            self._terms.append((dr.pseudo_inertia, cfg, dict(alpha_range=alpha)))
            self.link_body_ids = ent.indexing.body_ids[[ent.body_names.index(b) for b in layout.link_bodies]]
        for act_ids, kp, kd in c.pd_gain_scale:
            cfg = SceneEntityCfg("robot", actuator_ids=list(act_ids))
            self._terms.append((dr.pd_gains, cfg, dict(kp_range=kp, kd_range=kd, operation="scale")))
        if c.armature_range is not None:
            cfg = SceneEntityCfg("robot", joint_names=(".*",))
            self._terms.append((dr.joint_armature, cfg, dict(ranges=c.armature_range, operation="scale", shared_random=False)))
        if c.damping_range is not None:
            cfg = SceneEntityCfg("robot", joint_names=(".*",))
            self._terms.append((dr.joint_damping, cfg, dict(ranges=c.damping_range, operation="scale", shared_random=False)))
        fields = {"geom_friction"}
        if c.com_mass_range is None and not eval_mode:
            fields.add("body_ipos")
        if c.solref_geoms is not None:
            fields.add("geom_solref")
        for func, cfg, _ in self._terms:
            cfg.resolve(env.scene)
            fields.update(func.model_fields)
        self.recompute = max((func.recompute for func, _, _ in self._terms), default=RecomputeLevel.none)
        sim.expand_model_fields(tuple(sorted(fields)))

    # ------------------------------------------------------------------ once per env
    def startup(self, xi: torch.Tensor):
        c, sim, N, dev = self.cfg, self.env.sim, self.env.num_envs, self.env.device
        xi[:, self.layout.slices["friction"]] = 1.0 if self.eval_mode else torch.empty(N, 1, device=dev).uniform_(*c.friction_range)
        # ground (world body): lowest friction of the range, since MuJoCo takes the max of a pair
        sim.model.geom_friction[:, self.ground_geoms, 0] = c.friction_range[0] if not self.eval_mode else 1.0
        sim.model.geom_friction[:, self.friction_geoms, 0] = xi[:, self.layout.slices["friction"]]
        if c.com_mass_range is None and not self.eval_mode:
            com = torch.stack([torch.empty(N, device=dev).uniform_(lo, hi) for lo, hi in self.spec.com_range], -1)
            xi[:, self.layout.slices["com"]] = com
            default_com = sim.get_default_field("body_ipos")[self.com_body_id]
            sim.model.body_ipos[:, self.com_body_id] = default_com + com
        if "com_mass" in self.layout.slices:
            xi[:, self.layout.slices["com_mass"]] = 1.0
        for name in ("link_mass", "kp", "kd", "armature", "damping"):
            if name in self.layout.slices:
                xi[:, self.layout.slices[name]] = 1.0
        if "solref" in self.layout.slices:
            xi[:, self.layout.slices["solref"]] = sim.get_default_field("geom_solref")[self.solref_geoms[0]]
        if "qd_tau_scale" in self.layout.slices:
            xi[:, self.layout.slices["qd_tau_scale"]] = 1.0

    # ------------------------------------------------------------------ per episode
    def reset(self, xi: torch.Tensor, env_ids: torch.Tensor):
        if self.eval_mode:
            return
        c, sim, env, L = self.cfg, self.env.sim, self.env, self.layout
        n, dev = len(env_ids), env.device
        xi[env_ids, L.slices["joint_offset"]] = (2 * torch.rand(n, self.J, device=dev) - 1) * self.joint_offset_half
        if c.solref_geoms is not None:
            tc = torch.empty(n, 1, device=dev).uniform_(*c.solref_timeconst)
            dr_ = torch.empty(n, 1, device=dev).uniform_(*c.solref_dampratio)
            sim.model.geom_solref[env_ids[:, None], self.solref_geoms, 0] = tc
            sim.model.geom_solref[env_ids[:, None], self.solref_geoms, 1] = dr_
            xi[env_ids, L.slices["solref"]] = torch.cat((tc, dr_), -1)
        for func, cfg, params in self._terms:
            func(env, env_ids, asset_cfg=cfg, **params)
        if self.recompute > RecomputeLevel.none:
            sim.recompute_constants(self.recompute)
        # read the applied values back into xi
        if c.com_mass_range is not None:
            b = self.com_body_id
            xi[env_ids, L.slices["com_mass"]] = (sim.model.body_mass[env_ids, b] / sim.get_default_field("body_mass")[b]).unsqueeze(-1)
            xi[env_ids, L.slices["com"]] = sim.model.body_ipos[env_ids, b] - sim.get_default_field("body_ipos")[b]
        if c.link_mass_range is not None:
            b = self.link_body_ids
            xi[env_ids, L.slices["link_mass"]] = sim.model.body_mass[env_ids[:, None], b] / sim.get_default_field("body_mass")[b]
        if c.pd_gain_scale:
            kp = torch.ones(n, self.J, device=dev)
            kd = torch.ones(n, self.J, device=dev)
            for act in env.robot.actuators:
                kp[:, act.target_ids] = act.stiffness[env_ids] / act.default_stiffness[env_ids]
                kd[:, act.target_ids] = act.damping[env_ids] / act.default_damping[env_ids]
            xi[env_ids, L.slices["kp"]], xi[env_ids, L.slices["kd"]] = kp, kd
        if c.armature_range is not None:
            xi[env_ids, L.slices["armature"]] = sim.model.dof_armature[env_ids[:, None], self.dof_ids] / sim.get_default_field("dof_armature")[self.dof_ids]
        if c.damping_range is not None:
            default = sim.get_default_field("dof_damping")[self.dof_ids]
            xi[env_ids, L.slices["damping"]] = torch.where(default > 0, sim.model.dof_damping[env_ids[:, None], self.dof_ids] / default.clamp(min=1e-9), 1.0)
        # sensing terms (they shape what the policy reads, so the world model and critic get them too)
        s = self.spec.sensing
        if s.imu is not None:
            xi[env_ids, L.slices["imu_tilt"]] = torch.empty(n, 2, device=dev).uniform_(-s.imu.tilt, s.imu.tilt)
            xi[env_ids, L.slices["gyro_bias"]] = torch.empty(n, 3, device=dev).uniform_(-s.imu.gyro_bias, s.imu.gyro_bias)
        if s.joint_vel_filter is not None:
            xi[env_ids, L.slices["qd_tau_scale"]] = torch.empty(n, 1, device=dev).uniform_(*s.joint_vel_filter.tau_scale_range)

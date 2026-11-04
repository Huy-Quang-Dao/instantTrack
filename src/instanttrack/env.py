"""Motion-tracking simulator on mjlab (MuJoCo-Warp).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`instanttrack.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (SuperMimic paper Appendix D.1). Domain randomization
(:mod:`instanttrack.dr`) and the policy's sensing (:mod:`instanttrack.sensing`) follow ``RobotSpec``.
"""

from __future__ import annotations

import torch
import mujoco_warp as mjwarp
import warp as wp
from mjlab.entity.data import EntityData
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from .dr import DomainRandomizer, XiLayout
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import RobotSpec
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .sensing import ObsPipeline


# A python scalar written through a tensor index (x[ids] = 0.0) is copied host-to-device from pageable memory, which
# synchronizes the stream: 13 of the 14 host syncs per training step came from the reset path.
# The same writes with a device-side 0-dim value do not synchronize and give identical values.
def _dev_scalar(t: torch.Tensor, v):
    return torch.full((), v, dtype=t.dtype, device=t.device)


def _clear_state(self, env_ids=None) -> None:  # mjlab EntityData.clear_state without host syncs
    if self.is_actuated:
        env_ids = self._resolve_env_ids(env_ids)
        for t in (self.joint_pos_target, self.joint_vel_target, self.joint_effort_target, self.tendon_len_target,
                  self.tendon_vel_target, self.tendon_effort_target, self.site_effort_target):  # fmt: skip
            t[env_ids] = _dev_scalar(t, 0.0)


def _sim_reset(self, env_ids=None) -> None:  # mjlab Simulation.reset without host syncs
    with wp.ScopedDevice(self.wp_device):
        if env_ids is None:
            self._reset_mask.fill_(True)
        else:
            self._reset_mask.fill_(False)
            self._reset_mask[env_ids] = _dev_scalar(self._reset_mask._tensor, True)
        if self.use_cuda_graph and self.reset_graph is not None:
            wp.capture_launch(self.reset_graph)
        else:
            mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)


EntityData.clear_state, Simulation.reset = _clear_state, _sim_reset

DECIMATION = 4  # 4 physics substeps of 5 ms per 50 Hz control step
PHYSICS_DT = CONTROL_DT / DECIMATION


def make_scene_cfg(robot: RobotSpec, num_envs: int) -> SceneCfg:
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=0.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot.robot_cfg()},
    )


def make_sim(robot: RobotSpec) -> SimulationCfg:
    return SimulationCfg(
        njmax=robot.njmax,
        nconmax=robot.nconmax,
        mujoco=MujocoCfg(
            timestep=PHYSICS_DT,
            integrator="implicitfast",
            cone="pyramidal",
            solver="newton",
            iterations=100,
            ls_iterations=50,
            tolerance=1e-6,
            ccd_iterations=35,
        ),
    )


def _bin_ema_w(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, wt: torch.Tensor, old: float) -> torch.Tensor:
    """_bin_ema over the samples with weight wt = 1 (the others weight 0): fixed size, no host-side selection."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x * wt)
    cnt = torch.zeros_like(stat).index_add_(0, bins, wt)
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


def _bin_ema(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, old: float) -> torch.Tensor:
    """EMA update of per-bin statistics with possibly repeated bin indices: the samples of a bin are averaged, then
    one EMA step is taken for every bin that received at least one sample (no host sync)."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x)
    cnt = torch.zeros_like(stat).index_add_(0, bins, torch.ones_like(x))
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


class TrackingSim:
    """Batched motion-tracking simulator for the robot of ``robot``."""

    def __init__(self, sim_cfg: SimulationCfg, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP,
                 device: str, gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500,
                 reset_noise: bool = True, randomize: bool | None = None, fail_reset: float = 0.0, bin_fix: bool = False):  # fmt: skip
        """``randomize`` (default: not eval_mode) switches domain randomization, observation noise and
        delay, and pushes; ``eval_mode`` switches evaluation bookkeeping (no auto-reset, episode starts
        exactly on the reference)."""
        self.device = device
        self.scene = Scene(make_scene_cfg(robot, num_envs), device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == list(robot.joint_names), self.robot.joint_names
        self.N = num_envs
        self.J = robot.num_joints
        self.spec = robot
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.randomize = (not eval_mode) if randomize is None else randomize
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor(robot.action_scale_list(), device=dev)

        # ---- sim context xi (layout in instanttrack.dr; paper setup: friction, joint offsets, com offset) ----
        self.layout = XiLayout(robot, dev)
        assert self.layout.dim == mdp.xi_dim, (self.layout.dim, mdp.xi_dim)
        self.xi = torch.zeros(self.N, self.layout.dim, device=dev)
        self.randomizer = DomainRandomizer(robot, self.layout, self, eval_mode=not self.randomize)
        self.randomizer.startup(self.xi)

        # ---- sensing (stateful signal chain only if the robot defines one) ----
        self.pipe = ObsPipeline(robot, mdp, self.layout, dev, PHYSICS_DT)
        self.obs_fresh = torch.ones(self.N, dtype=torch.bool, device=dev)
        self._obs = self._obs_state = None
        if robot.sensing.imu is not None:
            m = self.sim.mj_model
            adr = lambda name: int(m.sensor_adr[m.sensor(f"robot/{name}").id])  # noqa: E731
            self._gyro_adr, self._accel_adr = adr(robot.sensing.imu.gyro_sensor), adr(robot.sensing.imu.accel_sensor)
        self.push_timer = torch.zeros(self.N, device=dev)
        self.push_enabled = robot.dr.push_interval_s is not None and self.randomize
        if self.push_enabled:
            self.push_timer.uniform_(*robot.dr.push_interval_s)

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, self.J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        # InstantTrack: share of resets sent one bin before recent failure points (per-bin termination rate)
        self.fail_reset = fail_reset
        # index_put_ with duplicate bin indices keeps one arbitrary episode per bin and step (undefined
        # behaviour); bin_fix averages the episodes of a bin first (as supermimic_vibed's update_bin_returns)
        self.bin_fix = bin_fix
        # --eprobe_share: externally set reset distribution (train.py: the windows before the deployed
        # policy's evaluation-condition failures), mixed in with share focus_share
        self.focus_q, self.focus_share = None, 0.0
        self.focus_static, self.focus_share_t = False, None  # use_static_focus()
        self.bin_fail = torch.zeros(motion.num_bins, device=dev)
        prev = torch.arange(motion.num_bins, device=dev) - 1
        self.bin_prev = torch.where((prev >= 0) & (motion.bin_clip[prev.clamp(min=0)] == motion.bin_clip), prev, prev + 1)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_envs(self) -> int:
        return self.N

    # ------------------------------------------------------------------ state
    def state(self):
        # root state from the free joint (qpos/qvel are current after a step; derived xpos lag a substep)
        idx = self.robot.indexing
        root_q = self.sim.data.qpos[:, idx.free_joint_q_adr]
        root_v = self.sim.data.qvel[:, idx.free_joint_v_adr]
        p = root_q[:, :3] - self.scene.env_origins
        R = quat_to_mat(root_q[:, 3:7])  # MuJoCo uses wxyz
        # free joint velocity: world-frame linear velocity of the root origin, angular velocity in the root frame
        w = (R @ root_v[:, 3:6].unsqueeze(-1)).squeeze(-1)
        d = self.robot.data
        return (p, R, root_v[:, :3].clone(), w, d.joint_pos.clone(), d.joint_vel.clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int, gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``gate`` (reset_masked): a 0-dim bool; the running average bin_w_avg only advances where it is true (the
        eager reset samples only when an episode ended)."""
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        if gate is None:
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        else:
            self.bin_w_avg.copy_(torch.where(gate, self.bin_w_avg * 0.999 + 0.001 * soft, self.bin_w_avg))
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(0, self.bin_prev, self.bin_fail)
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * torch.where(fail.sum() > 0, fail, probs)
        if self.focus_static:  # reset_masked: device-side focus (share 0 leaves probs bitwise unchanged)
            probs = (1 - self.focus_share_t) * probs + self.focus_share_t * self.focus_q
        elif self.focus_q is not None and self.focus_share > 0:
            probs = (1 - self.focus_share) * probs + self.focus_share * self.focus_q
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m, c = self.motion, self.spec.dr
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
            if c.reset_joint_vel > 0:
                qd = qd + torch.empty_like(qd).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        # mjlab's reset clears solver warm-start data and actuator targets, so it runs before the writes
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        pose = torch.cat((p + self.scene.env_origins[env_ids], mat_to_quat(R)), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim(q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)  # index_fill_: no host sync (see _dev_scalar)
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        self.randomizer.reset(self.xi, env_ids)  # after scene.reset, which would restore actuator defaults
        self.obs_fresh.index_fill_(0, env_ids, True)

    def use_static_focus(self):
        """Keep the focus distribution and share in fixed device tensors (updated in place by set_focus), so a captured
        reset_masked follows them."""
        self.focus_static = True
        self.focus_q = torch.full((self.motion.num_bins,), 1.0 / self.motion.num_bins, device=self.device)
        self.focus_share_t = torch.zeros((), device=self.device)

    def set_focus(self, q: torch.Tensor, share: float):
        if self.focus_static:
            self.focus_q.copy_(q), self.focus_share_t.fill_(share)
        else:
            self.focus_q, self.focus_share = q, share

    def reset_masked(self, mask: torch.Tensor, frames: torch.Tensor | None = None):
        """reset() of the worlds where ``mask`` is true, as fixed-size masked updates of all worlds (no host-side ids, so
        it can be captured in a CUDA graph). Worlds outside the mask are left bitwise unchanged; the reset worlds get the
        same values as reset() would give them for the same frames and random draws (random draws are made for all
        worlds, so the random stream differs from reset()). Only the configuration used here is supported: no pushes,
        no sensing state, no per-episode mjlab randomization terms, no actuator delays."""
        assert not self.eval_mode and not self.pipe.stateful and not self.push_enabled
        assert not self.randomizer._terms and self.spec.dr.solref_geoms is None and self.spec.dr.com_mass_range is None
        N, dev, m, c = self.N, self.device, self.motion, self.spec.dr
        d = mask
        d1 = d.unsqueeze(-1)
        if frames is None:
            frames, bins = self.sample_frames(N, gate=d.any())
        else:
            bins = m.frame_bin[frames]
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(N, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
            if c.reset_joint_vel > 0:
                qd = qd + torch.empty_like(qd).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        # Simulation.reset (solver warm start, controls, ...) of the masked worlds, before the writes
        sim = self.sim
        sim._reset_mask.copy_(d)
        with wp.ScopedDevice(sim.wp_device):
            if sim.use_cuda_graph and sim.reset_graph is not None:
                wp.capture_launch(sim.reset_graph)
            else:
                mjwarp.reset_data(sim.wp_model, sim.wp_data, reset=sim._reset_mask_wp)
        # Scene.reset: the robot's actuator targets (no actuator delays, no sensors; the terrain has no state)
        ed = self.robot.data
        for t in (ed.joint_pos_target, ed.joint_vel_target, ed.joint_effort_target, ed.tendon_len_target,
                  ed.tendon_vel_target, ed.tendon_effort_target, ed.site_effort_target):  # fmt: skip
            t.masked_fill_(d.view(-1, *([1] * (t.dim() - 1))), 0.0)
        # the writes of reset(): root pose, root velocity (angular in the new root frame), joints, position targets
        idx = self.robot.indexing
        qa, va = idx.free_joint_q_adr, idx.free_joint_v_adr
        qpos, qvel = sim.data.qpos, sim.data.qvel
        pose = torch.cat((p + self.scene.env_origins, mat_to_quat(R)), -1)
        qpos[:, qa] = torch.where(d1, pose, qpos[:, qa])
        qvel[:, va] = torch.where(d1, torch.cat((v, quat_apply_inverse(pose[:, 3:7], w)), -1), qvel[:, va])
        qpos[:, idx.joint_q_adr] = torch.where(d1, q, qpos[:, idx.joint_q_adr])
        qvel[:, idx.joint_v_adr] = torch.where(d1, qd, qvel[:, idx.joint_v_adr])
        ed.joint_pos_target.copy_(torch.where(d1, q, ed.joint_pos_target))
        self.f.copy_(torch.where(d, frames, self.f))
        self.ep_step.masked_fill_(d, 0)
        self.a_prev.masked_fill_(d1, 0.0)
        self.start_bin.copy_(torch.where(d, bins, self.start_bin))
        self.ep_disc_ret.masked_fill_(d, 0.0)
        self.ep_ret.masked_fill_(d, 0.0)
        self.randomizer.reset_masked(self.xi, d)
        self.obs_fresh |= d

    def step_tail_masked(self, out: dict) -> dict:
        """step_tail() without host-side ids (capturable): bin statistics of the ended episodes and reset_masked. Instead
        of per-episode tensors, out gets n_done and the sums ep_len_sum / ep_ret_norm_sum over the ended episodes."""
        assert self.bin_fix, "reset_masked needs --bin_fix (the index_put_ update keeps an arbitrary episode per bin)"
        done, term, f = out["done"], out["term"], out["f"]
        dw = done.float()
        T = self.ep_step.float()
        boot = torch.where(term, 0.0, (self.gamma**T) * (self.ep_ret / T) / (1 - self.gamma))
        G = self.ep_disc_ret + boot
        self.bin_G.copy_(_bin_ema_w(self.bin_G, self.start_bin, G, dw, 0.8))
        if self.fail_reset > 0:
            fb = self.motion.frame_bin[f.clamp(max=self.motion.num_frames - 1)]
            bf = torch.where(done.any(), self.bin_fail * 0.999, self.bin_fail)
            self.bin_fail.copy_(_bin_ema_w(bf, fb, term.float(), dw, 0.9))
        out["n_done"] = dw.sum()
        out["ep_len_sum"] = (T * dw).sum()
        out["ep_ret_norm_sum"] = (self.ep_ret / (T * MAX_STEP_REWARD) * dw).sum()
        self.reset_masked(done)
        return out

    def refresh(self):
        """Recompute derived quantities (body poses) so the robot's mjlab data match ``state()`` after a reset."""
        self.sim.forward()

    # ------------------------------------------------------------------ observation
    def _imu_site(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.sim.data.sensordata
        return d[:, self._gyro_adr : self._gyro_adr + 3], d[:, self._accel_adr : self._accel_adr + 3]

    def observe(self) -> torch.Tensor:
        """Policy observation of the current state (training: noisy; eval: noise- and delay-free)."""
        S = self.state()
        if not self.pipe.stateful:
            o = self.mdp.obs(S, self.a_prev, self.xi, self.f)
            self._obs = self.mdp.add_noise(o) if self.randomize else o
            return self._obs
        if self._obs_state is None or self.obs_fresh.any():
            # just reset: filters start converged on the true values (accelerometer: gravity only)
            fresh = self.obs_fresh.unsqueeze(-1)
            Rs = S[1] @ self.pipe.site_rot if self.spec.sensing.imu is not None else S[1]
            gyro = (Rs.transpose(-1, -2) @ S[3].unsqueeze(-1)).squeeze(-1)
            accel = Rs[..., 2, :] * 9.81
            init = self.pipe.init_state(S, self.xi, gyro, accel)
            init["prev_raw"] = self.pipe.raw_obs(init, S, self.a_prev, self.xi, self.f)
            st = init if self._obs_state is None else {k: torch.where(fresh, init[k], x) for k, x in self._obs_state.items()}
        else:
            st = self._obs_state
        raw = self.pipe.raw_obs(st, S, self.a_prev, self.xi, self.f)
        self._obs, self._obs_state = self.pipe.observe(st, raw, noise=self.randomize, delay=self.randomize)
        self.obs_fresh[:] = False
        return self._obs

    def _push(self, done: torch.Tensor) -> torch.Tensor:
        """Root velocity kicks at random intervals (between steps, after s1 is recorded)."""
        self.push_timer -= CONTROL_DT
        due = (self.push_timer <= 0) & ~done
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            c = self.spec.dr
            _, _, v, w, _, _ = self.state()
            kick = torch.stack([torch.empty(len(ids), device=self.device).uniform_(-h, h) for h in c.push_velocity], -1)
            vel = torch.cat((v[ids], w[ids]), -1) + kick
            self.robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
            self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(*c.push_interval_s)
        return due

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        out = self.step_core(a)
        if self.eval_mode:
            self.failed_eval |= out["term"] & ~self.done_eval
            self.done_eval |= out["done"]
            return out
        return self.step_tail(out)

    def step_core(self, a: torch.Tensor) -> dict:
        """The step up to the episode ends: no host synchronization (pushes aside), so it can be captured in a CUDA
        graph (train.py --collect_graph). The simulator wrapper rebinds f, ep_step and a_prev."""
        if self.pipe.stateful and self._obs is None:
            self.observe()
        o, o_state = self._obs, self._obs_state
        self._obs = None
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1 : 1 + self.J] + self.alpha * a
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
            if self.pipe.stateful:
                gyro, accel = self._imu_site() if self.spec.sensing.imu is not None else (None, None)
                self._obs_state = self.pipe.filter_update(self._obs_state, self.robot.data.joint_vel, gyro, accel, self.xi,
                                                          PHYSICS_DT, noise=self.randomize)  # fmt: skip
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms, "cut": done,
        }  # fmt: skip
        if self.pipe.stateful:
            out["o"], out["obs_state"] = o, self.pipe.pack(o_state)
        if self.push_enabled:
            out["cut"] = done | self._push(done)
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        return out

    def step_tail(self, out: dict) -> dict:
        """Episode ends of a training step: bin statistics and resets of the ended episodes (host-side ids)."""
        done, term, f = out["done"], out["term"], out["f"]
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            if self.bin_fix:
                self.bin_G.copy_(_bin_ema(self.bin_G, b, G, 0.8))
            else:
                self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            if self.fail_reset > 0:
                # termination rate of the bin where each episode ended (decays toward 0 as bins are mastered)
                fb = self.motion.frame_bin[f[ids].clamp(max=self.motion.num_frames - 1)]
                self.bin_fail.mul_(0.999)  # bins no episode ends in any more fade out
                if self.bin_fix:
                    self.bin_fail.copy_(_bin_ema(self.bin_fail, fb, term[ids].float(), 0.9))
                else:
                    self.bin_fail.index_put_((fb,), 0.9 * self.bin_fail[fb] + 0.1 * term[ids].float())
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out

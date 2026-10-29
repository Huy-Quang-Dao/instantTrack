"""Motion-tracking simulator on mjlab (MuJoCo-Warp).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`instanttrack.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1). Domain randomization
(:mod:`instanttrack.dr`) and the policy's sensing (:mod:`instanttrack.sensing`) follow ``RobotSpec``.
"""

from __future__ import annotations

import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from .dr import DomainRandomizer, XiLayout
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import RobotSpec
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .sensing import ObsPipeline

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


class TrackingSim:
    """Batched motion-tracking simulator for the robot of ``robot``."""

    def __init__(self, sim_cfg: SimulationCfg, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP,
                 device: str, gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500,
                 reset_noise: bool = True, randomize: bool | None = None, fail_reset: float = 0.0):  # fmt: skip
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
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(0, self.bin_prev, self.bin_fail)
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * torch.where(fail.sum() > 0, fail, probs)
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
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.randomizer.reset(self.xi, env_ids)  # after scene.reset, which would restore actuator defaults
        self.obs_fresh[env_ids] = True

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
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            if self.fail_reset > 0:
                # termination rate of the bin where each episode ended (decays toward 0 as bins are mastered)
                fb = self.motion.frame_bin[f[ids].clamp(max=self.motion.num_frames - 1)]
                self.bin_fail.mul_(0.999)  # bins no episode ends in any more fade out
                self.bin_fail.index_put_((fb,), 0.9 * self.bin_fail[fb] + 0.1 * term[ids].float())
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out

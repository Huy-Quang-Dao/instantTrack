"""Export a policy checkpoint to ONNX with one reference motion baked in (BeyondMimic-style deployment layout).

The motion goes through the training preprocessing (resampled to 50 Hz, finite-difference velocities,
goal features of the next K frames), so any npz the robot's loader reads can be baked in.

Inputs:  obs_history (1, H, O)  raw observations, oldest first, layout of instanttrack.mdp.MDP.obs:
                                q - q0, qd, root angular velocity (root frame), projected gravity,
                                previous action, reference-relative root orientation (6D)
         time_step   (1, 1)     reference frame at 50 Hz from the start of the motion (float)
Outputs: actions (1, J)         joint targets are q0 + action_scale * actions
         joint_pos, joint_vel (1, J), body_pos_w, body_quat_w (wxyz), body_lin_vel_w, body_ang_vel_w
                                (1, B, .): the reference at time_step (tracked bodies, body 0 = root)
         next_anchor_quat_w (1, 4)  root orientation at time_step + 1: the reference the next
                                observation's relative orientation is taken against
PD gains, default pose, action scale and the observation layout are in the ONNX metadata.

Usage: python tools/export_onnx.py --robot g1 --ckpt logs/<run>/ckpt_017324.pt \
           --motion data/lafan1_g1/dance1_subject2.csv --out logs/<run>/dance1_subject2.onnx
"""

import argparse
import os
import re

import onnx
import torch
from torch import nn

import instanttrack.networks as networks
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import obs_dim
from instanttrack.motion import CONTROL_DT, MotionLib
from instanttrack.networks import Policy
from instanttrack.robot import RobotSpec, get_robot
from instanttrack.rotations import mat_to_quat

OUTPUTS = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "next_anchor_quat_w"]


class PolicyWithMotion(nn.Module):
    def __init__(self, pi: Policy, motion: MotionLib, K: int):
        super().__init__()
        self.pi = pi
        f = torch.arange(motion.num_frames)
        self.register_buffer("goal", motion.goal(f, K))
        self.register_buffer("q", motion.q)
        self.register_buffer("qd", motion.qd)
        self.register_buffer("body_p", motion.body_p)
        self.register_buffer("body_quat", mat_to_quat(motion.body_R))
        self.register_buffer("body_v", motion.body_v)
        self.register_buffer("body_w", motion.body_w)
        self.last = motion.num_frames - 1

    def forward(self, obs_history: torch.Tensor, time_step: torch.Tensor):
        f = time_step.round().long().clamp(0, self.last).reshape(-1)
        f1 = (f + 1).clamp(max=self.last)
        a = self.pi(obs_history, self.goal[f])
        return a, self.q[f], self.qd[f], self.body_p[f], self.body_quat[f], self.body_v[f], self.body_w[f], self.body_quat[f1, 0]


def pd_gains(robot: RobotSpec) -> tuple[list[float], list[float]]:
    acts = robot.robot_cfg().articulation.actuators
    kp, kd = [], []
    for n in robot.joint_names:
        act = next(a for a in acts if any(re.fullmatch(p, n) for p in a.target_names_expr))
        kp.append(act.stiffness), kd.append(act.damping)
    return kp, kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", type=str, default="g1")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--motion", type=str, required=True, help="one motion file in the robot's format")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    dev = "cpu"
    robot = get_robot(a.robot)
    ck = torch.load(a.ckpt, map_location=dev)
    K, Hh = ck["args"]["goal_frames"], ck["args"]["hist"]
    motion = MotionLib([a.motion], Kinematics(robot, dev), dev, robot.load_motion)
    networks.AMP_DTYPE = None  # fp32 graph (training ran the MLPs under bf16 autocast)
    pi = Policy(obs_dim(robot.num_joints), motion.goal_dim(K), robot.num_joints, hist=Hh)
    pi.load_state_dict(ck["pi"])
    model = PolicyWithMotion(pi, motion, K).eval()

    obs = torch.zeros(1, Hh, obs_dim(robot.num_joints))
    t = torch.zeros(1, 1)
    torch.onnx.export(model, (obs, t), a.out, input_names=["obs_history", "time_step"], output_names=OUTPUTS,
                      opset_version=18, dynamo=False)  # fmt: skip

    kp, kd = pd_gains(robot)
    fmt = lambda xs: ",".join(f"{float(x):.6g}" for x in xs)  # noqa: E731
    meta = {
        "run_path": os.path.abspath(a.ckpt),
        "motion": os.path.splitext(os.path.basename(a.motion))[0],
        "motion_num_frames": str(motion.num_frames),
        "joint_names": ",".join(robot.joint_names),
        "joint_stiffness": fmt(kp),
        "joint_damping": fmt(kd),
        "default_joint_pos": fmt(robot.default_joint_pos_list()),
        "action_scale": fmt(robot.action_scale_list()),
        "command_names": "motion",
        "observation_names": "joint_pos,joint_vel,base_ang_vel,projected_gravity,actions,motion_anchor_ori_b",
        "observation_history": str(Hh),
        "goal_frames": str(K),
        "control_dt": str(CONTROL_DT),
        "anchor_body_name": robot.tracked_bodies[0],
        "body_names": ",".join(robot.tracked_bodies),
        "anchor_quat_w_init": fmt(model.body_quat[0, 0]),
    }
    m = onnx.load(a.out)
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, a.out)
    print(f"wrote {a.out}: it {ck['iteration']}, motion {meta['motion']} ({motion.num_frames} frames, "
          f"{(motion.num_frames - 1) * CONTROL_DT:.1f} s), obs_history (1, {Hh}, {obs.shape[-1]})")  # fmt: skip

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipped the ONNX vs torch check")
        return
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    err = 0.0
    with torch.no_grad():
        for f in torch.randint(0, motion.num_frames, (64,)).tolist():
            o = pi.obs_norm.mean + pi.obs_norm.std() * torch.randn(1, Hh, obs.shape[-1])
            t = torch.tensor([[float(f)]])
            ref = model(o, t)
            out = sess.run(None, {"obs_history": o.numpy(), "time_step": t.numpy()})
            err = max(err, max((torch.from_numpy(x) - r).abs().max().item() for x, r in zip(out, ref)))
    print(f"ONNX vs torch (fp32), 64 random frames: max abs diff {err:.2e}")


if __name__ == "__main__":
    main()

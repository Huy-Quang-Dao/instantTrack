"""Diagnose how a clip fails in a recorded evaluation: which termination fires and the tracking error."""
import sys
import numpy as np
import mujoco

d = dict(np.load(sys.argv[1])); clip = sys.argv[2]
names = list(d["names"]); c = names.index(clip)
alive = d["alive"][:, c]; f = d["f"][:, c]
T = int(alive.sum())
cs = int(d["clip_start"][c]); L = int(d["clip_len"][c])
def R_of(q):  # wxyz
    m = np.zeros(9); mujoco.mju_quat2Mat(m, q.astype(np.float64)); return m.reshape(3, 3)
print(f"{clip}: clip length {L} frames ({L*0.02:.1f}s), terminated after {T} steps (t={T*0.02:.2f}s, {100*T/(L-1):.0f}% of clip)")
rows = []
for t in range(0, T + 1):
    fr = f[t]
    Rr = d["ref_root_R"][fr]; R = R_of(d["root_quat"][t, c])
    e_z = d["ref_root_p"][fr][2] - d["root_p"][t, c][2]
    e_tilt = Rr[2, 2] - R[2, 2]
    e_xy = np.linalg.norm(d["ref_root_p"][fr][:2] - d["root_p"][t, c][:2])
    e_q = np.abs(d["ref_q"][fr] - d["q"][t, c]).mean()
    rows.append((t, fr - cs, d["root_p"][t, c][2], d["ref_root_p"][fr][2], e_z, R[2, 2], Rr[2, 2], e_tilt, e_xy, e_q))
print(" step  frame  z_robot z_ref   e_z   up_z_robot up_z_ref e_tilt  e_xy   mean|dq|")
for r in rows[max(0, T - 60)::5] + [rows[-1]]:
    print(" %4d  %5d  %6.3f %6.3f %6.3f   %6.3f    %6.3f  %6.3f  %5.2f  %6.3f" % r)
last = rows[-1]
print("terminated by:", "height (|e_z|>0.25)" if abs(last[4]) > 0.25 else "", "tilt (|e_tilt|>0.8)" if abs(last[7]) > 0.8 else "")

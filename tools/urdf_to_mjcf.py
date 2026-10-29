"""Compile the G1 URDF to an MJCF with a free-floating pelvis (same kinematics and inertias)."""

import os
import re
import sys

import mujoco

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "g1_description")
src = os.path.join(ROOT, "g1_29dof_rev_1_0.urdf")
tmp = os.path.join(ROOT, "_g1_mj.urdf")
out = os.path.join(ROOT, "g1_29dof.xml")

text = open(src).read()
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
# keep every link as its own body so the tracked link frames exist, and keep visuals for rendering
text = re.sub(
    r"<mujoco>.*?</mujoco>",
    '<mujoco><compiler meshdir="." balanceinertia="true" discardvisual="false" fusestatic="false" strippath="false"/></mujoco>',
    text,
    flags=re.S,
)
open(tmp, "w").write(text)
model = mujoco.MjModel.from_xml_path(tmp)
mujoco.mj_saveLastXML(out, model)
os.remove(tmp)

xml = open(out).read()
# free-floating pelvis
xml = re.sub(r'(<body name="pelvis"[^>]*>)', r'\1\n      <freejoint name="floating_base_joint"/>', xml, count=1)
if '<body name="pelvis"' not in xml:
    # URDF root link becomes worldbody children; wrap them in a pelvis body
    sys.exit("pelvis body not found; inspect " + out)
open(out, "w").write(xml)
m2 = mujoco.MjModel.from_xml_path(out)
print("nq", m2.nq, "nv", m2.nv, "nbody", m2.nbody, "ngeom", m2.ngeom, "mass", m2.body_mass.sum())
print([m2.joint(i).name for i in range(m2.njnt)])

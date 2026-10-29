#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and the G1 description, then build the MJCF asset.
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
list() { curl -s "$API/$1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"; }
mkdir -p data/lafan1_g1 data/g1_description/meshes
for f in $(list g1); do curl -sfL -o "data/lafan1_g1/$(basename "$f")" "$HF/resolve/main/$f"; done
curl -sfL -o data/g1_description/g1_29dof_rev_1_0.urdf "$HF/resolve/main/robot_description/g1/g1_29dof_rev_1_0.urdf"
for f in $(list robot_description/g1/meshes); do curl -sfL -o "data/g1_description/meshes/$(basename "$f")" "$HF/resolve/main/$f"; done
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
uv run python tools/urdf_to_mjcf.py

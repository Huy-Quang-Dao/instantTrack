"""Robot description shared by the motion library, FK, MDP and simulator.

Each robot lives in its own package ``instanttrack/<name>/`` whose ``<name>_constants.py`` defines
``SPEC``: the MJCF, joint/body names, actuators (an mjlab ``EntityCfg``) and the motion format.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Callable

import numpy as np
from mjlab.entity import EntityCfg

from .dr import DomainRandCfg
from .sensing import SensingCfg

# (root position (T,3), root quaternion wxyz (T,4), joint positions (T,J) in joint_names order, fps)
MotionArrays = tuple[np.ndarray, np.ndarray, np.ndarray, float]


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf_path: str
    """Robot MJCF; the simulated entity and the torch FK are both built from it."""
    joint_names: tuple[str, ...]
    """Actuated joints in motion-file / action order (must equal the MJCF joint order)."""
    tracked_bodies: tuple[str, ...]
    """Bodies in the tracking reward; the first one is the root (anchor) body."""
    default_joint_pos: dict[str, float]
    """Regex -> nominal joint position; actions are offsets from it."""
    action_scale: dict[str, float]
    """Joint name -> action scale (rad per unit action)."""
    com_body: str
    """Body whose center of mass is randomized (part of the sim context xi)."""
    com_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    robot_cfg: Callable[[], EntityCfg]
    motion_dir: str
    motion_glob: str
    load_motion: Callable[[str], MotionArrays]
    dr: DomainRandCfg = field(default_factory=DomainRandCfg)
    """Simulator randomization (default: the paper's G1 setup)."""
    sensing: SensingCfg = field(default_factory=SensingCfg)
    """How the policy observation is sensed (default: true state plus Table A.1 noise)."""
    njmax: int = 500
    nconmax: int = 128
    """MuJoCo-Warp constraint / contact capacity per world."""

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def default_joint_pos_list(self) -> list[float]:
        out = []
        for name in self.joint_names:
            val = 0.0
            for pat, v in self.default_joint_pos.items():
                if re.fullmatch(pat, name):
                    val = v
            out.append(val)
        return out

    def action_scale_list(self) -> list[float]:
        return [self.action_scale[n] for n in self.joint_names]

    @property
    def has_action_delay(self) -> bool:
        """Whether any actuator delays its command (then s_{t+1} also depends on a_{t-1})."""
        art = self.robot_cfg().articulation
        return art is not None and any(act.delay_max_lag > 0 for act in art.actuators)


def get_robot(name: str) -> RobotSpec:
    """Load ``instanttrack.<name>.<name>_constants.SPEC``."""
    return importlib.import_module(f"instanttrack.{name}.{name}_constants").SPEC

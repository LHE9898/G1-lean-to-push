# Copyright (c) 2022-2025, unitree_rl_lab
"""다리(12DoF) + 허리(3DoF) + 왼팔(7DoF) residual joint-position action term — State 2 전용.

State 2 Action Space:
  - Left  leg  :  6-DoF
  - Right leg  :  6-DoF
  - Waist      :  3-DoF  ← RL이 pitch 10° 목표로 스스로 학습
  - Left arm   :  7-DoF  ← RL이 테이블 지지 자세를 스스로 학습
  ─────────────────────────
  합계          : 22-DoF   →  a_t ∈ ℝ²²
  q_cmd = q_nominal + Δq,   Δq ∈ [-δ, δ]

오른팔(7DoF)은 action에서 완전 제외.
  → controllers.py 의 RightArmDisturbanceController 가 별도 구동.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── 제어 관절 목록 (순서 고정) ────────────────────────────────────────────────
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint",  "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

# 22-DoF 순서: 다리 → 허리 → 왼팔
CONTROLLED_JOINT_NAMES = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + LEFT_ARM_JOINT_NAMES

NUM_LEG_JOINTS      = len(LEG_JOINT_NAMES)        # 12
NUM_WAIST_JOINTS    = len(WAIST_JOINT_NAMES)      #  3
NUM_LEFT_ARM_JOINTS = len(LEFT_ARM_JOINT_NAMES)   #  7
NUM_CTRL_JOINTS     = len(CONTROLLED_JOINT_NAMES) # 22

# nominal 벡터 내 슬라이스
_LEG_SLICE  = slice(0,  12)
_WAIST_SLICE = slice(12, 15)
_LARM_SLICE  = slice(15, 22)


class LegWaistLeftArmResidualAction(ActionTerm):
    """RL 출력 Δq (22-dim) 를 nominal 자세에 더해 관절 위치 목표로 전달.

    Notes
    -----
    * _ctrl_ids 는 super().__init__() 이전에 설정해야 함.
      IsaacLab 이 __init__ 내부에서 action_dim property 를 호출하기 때문.
    * 오른팔은 건드리지 않음 — controllers.py 담당.
    """
    cfg: "LegWaistLeftArmResidualActionCfg"

    def __init__(self, cfg: "LegWaistLeftArmResidualActionCfg", env: ManagerBasedRLEnv):
        # ① super() 전에 joint index 확정
        self._asset    = env.scene[cfg.asset_name]
        all_names      = self._asset.joint_names
        self._ctrl_ids = torch.tensor(
            [all_names.index(n) for n in CONTROLLED_JOINT_NAMES],
            dtype=torch.long, device=env.device,
        )

        # ② super() 호출
        super().__init__(cfg, env)

        # ③ Nominal 자세 (N, 22) ─────────────────────────────────────────────
        nominal = torch.zeros(NUM_CTRL_JOINTS, device=env.device)

        # 다리: 무릎 살짝 구부림 / 발목 중립
        nominal[3]  =  0.30   # left_knee
        nominal[4]  = -0.20   # left_ankle_pitch
        nominal[9]  =  0.30   # right_knee
        nominal[10] = -0.20   # right_ankle_pitch

        # 허리: upright 0° — RL 이 스스로 pitch 10° 로 이동하도록 학습
        # 왼팔: 내려뜨린 자세 0° — RL 이 스스로 테이블을 짚도록 학습

        self._nominal = nominal.unsqueeze(0).expand(env.num_envs, -1).clone()

        # reward / observation 함수 참조용
        env.leg_nominal_q      = self._nominal[:, _LEG_SLICE]
        env.waist_nominal_q    = self._nominal[:, _WAIST_SLICE]
        env.left_arm_nominal_q = self._nominal[:, _LARM_SLICE]

        # ④ Motor delay buffer (N, delay+1, 22)
        d = cfg.motor_delay_steps
        self._delay_buf         = torch.zeros(env.num_envs, d + 1, NUM_CTRL_JOINTS, device=env.device)
        self._limit             = cfg.delta_q_limit
        self._raw_actions       = torch.zeros(env.num_envs, NUM_CTRL_JOINTS, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    # ── ActionTerm interface ──────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return NUM_CTRL_JOINTS  # 22

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clip + motor delay 적용."""
        self._raw_actions = actions.clone()
        delta_q = actions.clamp(-self._limit, self._limit)
        self._delay_buf = torch.roll(self._delay_buf, 1, dims=1)
        self._delay_buf[:, 0, :] = delta_q
        self._processed_actions  = self._delay_buf[:, -1, :]

    def apply_actions(self) -> None:
        """q_cmd = q_nominal + Δq 를 시뮬레이터에 기록."""
        q_target    = self._nominal + self._processed_actions       # (N, 22)
        full_target = self._asset.data.joint_pos.clone()            # (N, n_joints)
        full_target[:, self._ctrl_ids] = q_target
        self._asset.set_joint_position_target(full_target)


@configclass
class LegWaistLeftArmResidualActionCfg(ActionTermCfg):
    """LegWaistLeftArmResidualAction 설정."""
    class_type: type       = LegWaistLeftArmResidualAction
    asset_name: str        = "robot"
    delta_q_limit: float   = 0.30   # ±0.30 rad (왼팔 포함 → 범위 확대)
    motor_delay_steps: int = 2
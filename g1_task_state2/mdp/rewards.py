# Copyright (c) 2022-2025, unitree_rl_lab
"""Reward functions for G1 State2.

R = r_torso + r_contact + r_balance + r_slip + r_effort + r_auxiliary

허리·왼팔이 RL 제어로 변경됨에 따른 reward 구성:
  ① Torso Tracking   : torso 위치(x, z) + pitch 각도 목표 추종
  ② Contact 유지      : 왼손 테이블 contact 유지 보상 + 높이 유도 보상
  ③ Balance          : torso 각속도 + IMU 가속도 패널티
  ④ Slip Penalty      : 왼손 접선 속도 패널티
  ⑤ Effort           : 관절 속도 + action 크기 패널티
  ⑥ 보조 (안전)       : hip roll/yaw, leg nominal, dynamics tracking
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── State2 목표 torso 상태 (초기 upright CoM frame 기준) ──────────────────────
TARGET_X_D     =  0.20          # [m]   앞으로 +0.2 m
TARGET_Z_D     = -0.10          # [m]   아래로 -0.1 m
TARGET_PITCH_D =  math.radians(10.0)   # [rad] +10°


# ─────────────────────────────────────────────────────────────────────────────
# ① Torso Tracking
# ─────────────────────────────────────────────────────────────────────────────

def torso_pos_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    x_d: float = TARGET_X_D,
    z_d: float = TARGET_Z_D,
    w_x: float = 2.0,
    w_z: float = 2.0,
) -> torch.Tensor:
    """
    e_x = x_rel - x_d,   e_z = z_rel - z_d
    r = -w_x·e_x² - w_z·e_z²

    init_torso_pos_w 는 events.reset_robot_to_standing 에서 캐싱.
    """
    asset   = env.scene[asset_cfg.name]
    pos_w   = asset.data.root_pos_w
    init    = env.extras.get("init_torso_pos_w", pos_w.detach().clone())
    rel     = pos_w - init

    e_x = rel[:, 0] - x_d
    e_z = rel[:, 2] - z_d
    return -w_x * e_x.pow(2) - w_z * e_z.pow(2)


def torso_pitch_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    theta_d: float = TARGET_PITCH_D,
    w_theta: float = 1.0,
) -> torch.Tensor:
    """
    e_θ = pitch - θ_d
    r = -w_θ·e_θ²

    허리를 RL이 제어하므로 이 reward 로 pitch 목표를 유도.
    """
    _, pitch, _ = euler_xyz_from_quat(
        env.scene[asset_cfg.name].data.root_quat_w
    )
    return -w_theta * (pitch - theta_d).pow(2)


# ─────────────────────────────────────────────────────────────────────────────
# ② Contact 유지
# ─────────────────────────────────────────────────────────────────────────────

def left_hand_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    w_c: float = 1.0,
) -> torch.Tensor:
    """r_contact = w_c · c_lh  (왼손 테이블 contact)."""
    f = env.scene[sensor_cfg.name].data.net_forces_w_history
    c = (f[:, :, 0, 2].abs().max(dim=1).values > 1.0).float()
    return w_c * c


def arm_height_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_body_name: str = "left_wrist_yaw_link",
    table_height: float = 0.6,
) -> torch.Tensor:
    """왼손을 테이블 높이로 유도하는 shaped reward.

    r = exp(-4·|z_hand - z_table|)

    왼팔을 RL 이 제어하므로 contact 유도를 위해 필수.
    """
    asset = env.scene[asset_cfg.name]
    names = asset.body_names
    lz    = asset.data.body_pos_w[:, names.index(left_body_name), 2]
    tbl_z = env.scene.env_origins[:, 2] + table_height
    return torch.exp(-4.0 * (lz - tbl_z).abs())

TABLE_DISTANCE_X = 0.15   # 로봇 정면 → 테이블 앞면 [m]
TABLE_WIDTH      = 0.4
TABLE_DEPTH      = 0.4
TABLE_HEIGHT     = 0.7   # 테이블 상면 z [m]

def arm_reach_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_body_name: str = "left_wrist_yaw_link",
    table_height: float = 0.6,
) -> torch.Tensor:
    """왼손을 테이블 앞 방향으로 유도하는 shaped reward.

    r = exp(-2·||p_hand_xy - p_table_xy||²)

    왼팔 RL 제어 시 손이 테이블 표면 위에 오도록 x,y 위치 유도.
    """

    asset    = env.scene[asset_cfg.name]
    names    = asset.body_names
    hand_pos = asset.data.body_pos_w[:, names.index(left_body_name), :2]  # (N,2) xy

    tbl_x = env.scene.env_origins[:, 0] + TABLE_DISTANCE_X + TABLE_WIDTH * 0.5
    tbl_y = env.scene.env_origins[:, 1]
    tbl_xy = torch.stack([tbl_x, tbl_y], dim=-1)

    dist_sq = (hand_pos - tbl_xy).pow(2).sum(dim=-1)
    return torch.exp(-2.0 * dist_sq)


# ─────────────────────────────────────────────────────────────────────────────
# ③ Balance
# ─────────────────────────────────────────────────────────────────────────────

def torso_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """r_balance = -||ω_torso||²"""
    return -env.scene[asset_cfg.name].data.root_ang_vel_w.norm(dim=-1).pow(2)


def imu_acc_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """IMU 선가속도 패널티 (진동 억제)."""
    return -env.scene[sensor_cfg.name].data.lin_acc_b.norm(dim=-1).pow(2)


# ─────────────────────────────────────────────────────────────────────────────
# ④ Slip Penalty
# ─────────────────────────────────────────────────────────────────────────────

def left_hand_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_body_name: str = "left_wrist_yaw_link",
    w_s: float = 2.0,
) -> torch.Tensor:
    """
    v_t = v_hand - (v_hand · n̂)n̂    (n̂ = 테이블 상면 법선 = world-z)
    r_slip = -w_s · ||v_t||²

    contact 중일 때만 패널티 적용 (contact 없을 때는 slip 의미 없음).
    """
    asset    = env.scene[asset_cfg.name]
    names    = asset.body_names
    body_idx = names.index(left_body_name)
    v_hand   = asset.data.body_lin_vel_w[:, body_idx, :]   # (N, 3)

    n       = torch.zeros_like(v_hand)
    n[:, 2] = 1.0
    v_norm  = (v_hand * n).sum(dim=-1, keepdim=True) * n
    v_t     = v_hand - v_norm

    return -w_s * v_t.norm(dim=-1).pow(2)


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Effort Regularization
# ─────────────────────────────────────────────────────────────────────────────

def leg_nominal_pose_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """다리가 nominal 자세에서 멀어지면 패널티."""
    q_leg = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return -(q_leg - env.leg_nominal_q).pow(2).sum(dim=-1)


def leg_roll_yaw_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """hip roll/yaw 과도 사용 패널티."""
    return -env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].pow(2).sum(dim=-1)


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """전체 관절 속도 패널티."""
    return -env.scene[asset_cfg.name].data.joint_vel.pow(2).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# ⑦ 발바닥 위치 고정 (foot grounding)
# ─────────────────────────────────────────────────────────────────────────────

def foot_slide_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_foot_body_name:  str = "left_ankle_roll_link",
    right_foot_body_name: str = "right_ankle_roll_link",
    w_slide: float = 5.0,
) -> torch.Tensor:
    """발바닥 xy 위치가 초기 위치에서 이탈하면 패널티.

    허리를 앞으로 숙이는 동안 발이 ground 위에서 미끄러지지 않도록 고정.
    z 방향(수직)은 발목이 움직이면서 자연스럽게 변할 수 있으므로 제외.
    발목 관절(ankle_pitch, ankle_roll) 자체는 RL 이 자유롭게 제어 가능.

    r = -w_slide · (||Δx_lf||² + ||Δx_rf||²)
      Δx = foot_pos_xy - init_foot_pos_xy
    """
    asset = env.scene[asset_cfg.name]
    names = asset.body_names
    l_idx = names.index(left_foot_body_name)
    r_idx = names.index(right_foot_body_name)

    lf_xy = asset.data.body_pos_w[:, l_idx, :2]   # (N, 2)
    rf_xy = asset.data.body_pos_w[:, r_idx, :2]   # (N, 2)

    # 초기 발 위치 캐싱 (reset 시 events.py 에서 갱신)
    if "init_left_foot_pos_xy" not in env.extras:
        env.extras["init_left_foot_pos_xy"]  = lf_xy.detach().clone()
        env.extras["init_right_foot_pos_xy"] = rf_xy.detach().clone()

    init_lf = env.extras["init_left_foot_pos_xy"]   # (N, 2)
    init_rf = env.extras["init_right_foot_pos_xy"]  # (N, 2)

    slide_l = (lf_xy - init_lf).pow(2).sum(dim=-1)  # (N,)
    slide_r = (rf_xy - init_rf).pow(2).sum(dim=-1)  # (N,)

    return -w_slide * (slide_l + slide_r)


def foot_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_foot_body_name:  str = "left_ankle_roll_link",
    right_foot_body_name: str = "right_ankle_roll_link",
    w_vel: float = 2.0,
) -> torch.Tensor:
    """발바닥 xy 방향 속도 패널티 (미끄러짐 억제 보조).

    발이 이미 미끄러지기 전에 속도 자체를 억제.
    z 방향(수직) 속도는 제외 — 발목 움직임으로 인한 수직 변화는 허용.

    r = -w_vel · (||v_lf_xy||² + ||v_rf_xy||²)
    """
    asset = env.scene[asset_cfg.name]
    names = asset.body_names
    l_idx = names.index(left_foot_body_name)
    r_idx = names.index(right_foot_body_name)

    lf_vel_xy = asset.data.body_lin_vel_w[:, l_idx, :2]   # (N, 2)
    rf_vel_xy = asset.data.body_lin_vel_w[:, r_idx, :2]   # (N, 2)

    return -w_vel * (
        lf_vel_xy.pow(2).sum(dim=-1) + rf_vel_xy.pow(2).sum(dim=-1)
    )

def free_floating_dynamics_tracking(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """목표 torso 궤적의 동역학 추종.

    V̇ = [a_imu; α_torso],   r = -||V̇ - V̇_d||²
    env.V_dot_desired 는 controllers.py 가 매 step 갱신.
    """
    dt      = env.step_dt
    a_imu   = env.scene[sensor_cfg.name].data.lin_acc_b
    omega   = env.scene[asset_cfg.name].data.root_ang_vel_w
    ang_acc = (omega - env.prev_omega) / (dt + 1e-6)
    V_dot   = torch.cat([a_imu, ang_acc], dim=-1)
    return -(V_dot - env.V_dot_desired).pow(2).sum(dim=-1)
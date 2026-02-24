# Copyright (c) 2022-2025, unitree_rl_lab
"""Observation functions for G1 State2.

Observation 벡터 구조 (총 90-dim):
  ┌─ 제어 관절 상태 ─────────────────────────────────────────── 50-dim ─┐
  │  q_leg      (12)  다리 관절 위치                                      │
  │  dq_leg     (12)  다리 관절 속도                                      │
  │  q_waist    ( 3)  허리 관절 위치   ← RL 제어                          │
  │  dq_waist   ( 3)  허리 관절 속도                                      │
  │  q_la       ( 7)  왼팔 관절 위치   ← RL 제어 (지지 팔)                │
  │  dq_la      ( 7)  왼팔 관절 속도                                      │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ Torso IMU ──────────────────────────────────────────────── 15-dim ─┐
  │  R_torso    ( 9)  torso 회전행렬 flatten                              │
  │  ω_torso    ( 3)  torso 각속도                                        │
  │  imu_acc    ( 3)  IMU 선가속도                                        │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ Contact & 보조 ─────────────────────────────────────────────8-dim ─┐
  │  c_lh       ( 1)  왼손 contact binary                                 │
  │  table_rel  ( 3)  torso → 테이블 상면 중심 벡터                        │
  │  phase_enc  ( 2)  sin/cos 위상 인코딩                                 │
  │  lean_phase ( 1)  현재 lean phase (0 or 1)                            │
  │  target_err ( 1)  torso pitch 오차 (스칼라 힌트)                       │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ 오른팔 disturbance observer ★ ─────────────────────────── 17-dim ─┐
  │  q_ra       ( 7)  오른팔 관절 위치  (action X, reward 직접 영향 X)     │
  │  dq_ra      ( 7)  오른팔 관절 속도                                     │
  │  v_rh       ( 3)  오른손 속도 (disturbance proxy)                     │
  └───────────────────────────────────────────────────────────────────┘
  합계: 50 + 15 + 8 + 17 = 90-dim
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────────────────────────────────────
# 다리
# ─────────────────────────────────────────────────────────────────────────────

def leg_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """다리 12DoF 관절 위치 → (N, 12)"""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def leg_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """다리 12DoF 관절 속도 → (N, 12)"""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 허리 (RL 제어)
# ─────────────────────────────────────────────────────────────────────────────

def waist_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """허리 3DoF 관절 위치 [yaw, roll, pitch] → (N, 3)"""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def waist_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """허리 3DoF 관절 속도 → (N, 3)"""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 왼팔 (RL 제어 — 지지 팔)
# ─────────────────────────────────────────────────────────────────────────────

def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """왼팔 7DoF 관절 위치 → (N, 7)"""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def left_arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """왼팔 7DoF 관절 속도 → (N, 7)"""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def left_arm_contact_flag(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """왼손 contact binary → (N, 1).  c_lh ∈ {0, 1}"""
    f = env.scene[sensor_cfg.name].data.net_forces_w_history  # (N, T, B, 3)
    return (f[:, :, 0, 2].abs().max(dim=1).values > 1.0).float().unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Torso IMU
# ─────────────────────────────────────────────────────────────────────────────

def torso_rotation_matrix(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso 3×3 회전행렬 flatten → (N, 9)."""
    q = env.scene[asset_cfg.name].data.root_quat_w   # (N,4) [w,x,y,z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
          2*(x*y + z*w),  1 - 2*(x*x + z*z),     2*(y*z - x*w),
          2*(x*z - y*w),    2*(y*z + x*w),      1 - 2*(x*x + y*y),
    ], dim=-1)


def torso_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso 각속도 (world frame) → (N, 3)."""
    return env.scene[asset_cfg.name].data.root_ang_vel_w


def imu_lin_acc(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """IMU 선가속도 (body frame) → (N, 3)."""
    return env.scene[sensor_cfg.name].data.lin_acc_b.clone()


# ─────────────────────────────────────────────────────────────────────────────
# 보조 관측량
# ─────────────────────────────────────────────────────────────────────────────
TABLE_DISTANCE_X = 0.15   # 로봇 정면 → 테이블 앞면 [m]
TABLE_WIDTH      = 0.4
TABLE_DEPTH      = 0.4
TABLE_HEIGHT     = 0.7   # 테이블 상면 z [m]

def table_relative_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso → 테이블 상면 중심 벡터 (world frame) → (N, 3).

    lean_env_cfg 의 TABLE_* 상수를 직접 참조하여 계산.
    RL 이 왼손을 뻗어야 하는 방향을 알 수 있도록 제공.
    """
    # from unitree_rl_lab.tasks.g1_task_state2.robots.g1.dof29.lean_env_cfg import (
    #     TABLE_DISTANCE_X, TABLE_WIDTH, TABLE_HEIGHT,
    # )
    asset = env.scene[asset_cfg.name]
    tbl   = env.scene.env_origins.clone()
    tbl[:, 0] += TABLE_DISTANCE_X + TABLE_WIDTH * 0.5
    tbl[:, 2] += TABLE_HEIGHT
    return tbl - asset.data.root_pos_w   # (N, 3)


def phase_encoding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """sin/cos 위상 인코딩 → (N, 2)."""
    T      = env.cfg.phase1_duration + env.cfg.phase2_duration
    t_norm = env.lean_phase_time / T
    return torch.stack([
        torch.sin(2.0 * math.pi * t_norm),
        torch.cos(2.0 * math.pi * t_norm),
    ], dim=-1)


def lean_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 lean phase (0=숙이는 중, 1=안정 유지 중) → (N, 1)."""
    if not hasattr(env, "lean_phase"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.lean_phase.float().unsqueeze(-1)


def torso_pitch_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """현재 torso pitch 와 목표 pitch 의 오차 → (N, 1).

    RL 이 허리를 얼마나 더 숙여야 하는지 직접적인 힌트.
    """
    from unitree_rl_lab.tasks.g1_task_state2.mdp.rewards import TARGET_PITCH_D
    _, pitch, _ = euler_xyz_from_quat(
        env.scene[asset_cfg.name].data.root_quat_w
    )
    return (pitch - TARGET_PITCH_D).unsqueeze(-1)   # (N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 오른팔 Disturbance Observer ★
# ─────────────────────────────────────────────────────────────────────────────

def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """오른팔 7DoF 관절 위치 → (N, 7).  disturbance observer."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def right_arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """오른팔 7DoF 관절 속도 → (N, 7).  disturbance observer."""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def right_hand_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """오른손 속도 (world frame) → (N, 3).  disturbance proxy.

    오른팔이 빠르게 움직일수록 큰 τ_dist 가 예상됨.
    asset_cfg: SceneEntityCfg(body_names=["right_wrist_yaw_link"])
    """
    asset    = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    return asset.data.body_lin_vel_w[:, body_idx, :]   # (N, 3)
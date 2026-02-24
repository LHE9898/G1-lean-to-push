# Copyright (c) 2022-2025, unitree_rl_lab
"""Termination functions for G1 State2.

성공 조건 (task_success):
  ① 왼손이 테이블에 닿은 순간을 first_contact_time 으로 기록
  ② first_contact_time 이후 contact 를 유지하면서
     torso 가 목표 자세(x_d, z_d, pitch_d) 오차 이내로 들어오면
     success_hold_time 카운터 시작
  ③ success_hold_time >= hold_duration (5초) 이면 성공 종료

실패 조건:
  - torso roll 초과 / 높이 미달 / hip yaw 초과 → 즉시 종료
  - contact 달성 후 손이 다시 떨어지면 success_hold_time 리셋
  - time_out : episode_length_s 이내에 성공 못하면 실패
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 성공 판정 허용 오차
_SUCCESS_POS_TOL   = 0.05    # [m]   x, z 오차 허용 범위
_SUCCESS_PITCH_TOL = math.radians(3.0)  # [rad] pitch 오차 허용 범위


# ─────────────────────────────────────────────────────────────────────────────
# 기본 종료 조건
# ─────────────────────────────────────────────────────────────────────────────

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """episode_length_s 이내에 성공하지 못하면 실패 종료."""
    return env.episode_length_buf >= env.max_episode_length - 1


def torso_roll_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    limit_deg: float = 40.0,
) -> torch.Tensor:
    """Torso roll 초과 → 넘어짐."""
    roll, _, _ = euler_xyz_from_quat(env.scene[asset_cfg.name].data.root_quat_w)
    return roll.abs() > math.radians(limit_deg)


def torso_too_low(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.50,
) -> torch.Tensor:
    """Torso CoM 높이 미달 → 넘어짐."""
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2] < min_height


def hip_yaw_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    limit_deg: float = 30.0,
) -> torch.Tensor:
    """Hip yaw 과도 → 몸통 회전 이상."""
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return q.abs().max(dim=-1).values > math.radians(limit_deg)


def foot_contact_loss(
    env: ManagerBasedRLEnv,
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """두 발 contact 상실 → 발이 들림."""
    lf = env.scene[left_foot_cfg.name].data.net_forces_w_history
    rf = env.scene[right_foot_cfg.name].data.net_forces_w_history
    lf_ok = lf[:, :, 0, :].norm(dim=-1).max(dim=1).values > threshold
    rf_ok = rf[:, :, 0, :].norm(dim=-1).max(dim=1).values > threshold
    return ~lf_ok | ~rf_ok


# ─────────────────────────────────────────────────────────────────────────────
# ★ 성공 판정
# ─────────────────────────────────────────────────────────────────────────────

def task_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    hold_duration: float = 5.0,
    contact_threshold: float = 1.0,
    pos_tol: float = _SUCCESS_POS_TOL,
    pitch_tol: float = _SUCCESS_PITCH_TOL,
) -> torch.Tensor:
    """
    성공 판정 로직:

    [Phase A] 왼손이 테이블에 처음 닿기 전
      - first_contact_time = -1 (미달성)
      - success_hold_time = 0

    [Phase B] 왼손이 테이블에 닿은 순간
      - first_contact_time = 현재 episode_time 으로 기록
      - torso 가 목표 자세 오차 이내에 있으면 success_hold_time 카운트 시작

    [Phase C] hold_duration (5초) 유지
      - contact 유지 AND torso 목표 오차 이내 → 카운터 증가
      - 둘 중 하나라도 깨지면 카운터 리셋
      - 카운터 >= hold_duration → 성공 종료 (time_out=False)

    로그에서 확인:
      env.extras["success_hold_time"]  : 현재 유지 시간 (N,)
      env.extras["first_contact_time"] : 최초 contact 달성 시간 (N,)  -1=미달성
      env.extras["task_success_count"] : 누적 성공 환경 수 (scalar)
    """
    from unitree_rl_lab.tasks.g1_task_state2.mdp.rewards import (
        TARGET_X_D, TARGET_Z_D, TARGET_PITCH_D,
    )

    N   = env.num_envs
    dev = env.device
    dt  = env.step_dt

    # ── 버퍼 초기화 (최초 1회) ────────────────────────────────────────────────
    if "success_hold_time" not in env.extras:
        env.extras["success_hold_time"]  = torch.zeros(N, device=dev)
        env.extras["first_contact_time"] = torch.full((N,), -1.0, device=dev)
        env.extras["task_success_count"] = torch.tensor(0, device=dev)

    hold_time    = env.extras["success_hold_time"]     # (N,)
    first_ct     = env.extras["first_contact_time"]    # (N,)
    episode_time = env.episode_length_buf * dt          # (N,)

    # ── 왼손 contact 판정 ─────────────────────────────────────────────────────
    f = env.scene[sensor_cfg.name].data.net_forces_w_history  # (N, T, B, 3)
    in_contact = (
        f[:, :, 0, 2].abs().max(dim=1).values > contact_threshold
    )  # (N,) bool

    # ── 최초 contact 시간 기록 ────────────────────────────────────────────────
    first_touch = in_contact & (first_ct < 0.0)
    first_ct    = torch.where(first_touch, episode_time, first_ct)
    env.extras["first_contact_time"] = first_ct

    # ── torso 목표 자세 오차 계산 ─────────────────────────────────────────────
    asset  = env.scene[asset_cfg.name]
    pos_w  = asset.data.root_pos_w
    init   = env.extras.get("init_torso_pos_w", pos_w.detach().clone())
    rel    = pos_w - init                                   # (N, 3)

    e_x   = (rel[:, 0] - TARGET_X_D).abs()                 # (N,)
    e_z   = (rel[:, 2] - TARGET_Z_D).abs()                 # (N,)
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    e_p   = (pitch - TARGET_PITCH_D).abs()                  # (N,)

    torso_ok = (e_x < pos_tol) & (e_z < pos_tol) & (e_p < pitch_tol)  # (N,)

    # ── 성공 카운터 갱신 ──────────────────────────────────────────────────────
    # contact 달성 이후에만 카운터 동작
    after_contact = first_ct >= 0.0                         # (N,)
    counting      = after_contact & in_contact & torso_ok

    hold_time = torch.where(
        counting,
        hold_time + dt,
        torch.zeros_like(hold_time),   # 조건 깨지면 리셋
    )
    env.extras["success_hold_time"] = hold_time

    # ── 성공 판정 ─────────────────────────────────────────────────────────────
    success = hold_time >= hold_duration                    # (N,) bool

    # 누적 성공 수 로그
    env.extras["task_success_count"] = (
        env.extras["task_success_count"] + success.sum()
    )

    return success

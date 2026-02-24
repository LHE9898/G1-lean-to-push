# Copyright (c) 2022-2025, unitree_rl_lab
"""State 2 Model-based 상체 controller — 오른팔 disturbance 전용.

허리 / 왼팔은 RL policy 가 직접 제어하므로 여기서 건드리지 않음.
이 파일은 오직 오른팔(7DoF) disturbance 만 담당한다.

  오른팔 Curriculum:
    Phase 1  [0,       5M steps) : 고정  A = 0.0 rad
    Phase 2  [5M,    15M steps)  : sin   q_ra(t) = q_nominal + A·sin(ωt + φ)
                                   A 선형 증가 0.0 → 0.15 rad
    Phase 3  [15M,       ∞     ) : large A = 0.30 rad

  오른팔이 움직이면 torso 에 관성 토크가 발생:
    τ_dist = M_ra(q)·q̈_ra + C_ra(q,q̇)·q̇_ra
  시뮬레이터가 내부에서 자동으로 처리하므로 별도 계산 불필요.
  Policy 는 observation 의 오른팔 state 를 통해
    q_right_arm → τ_dist → torso deviation 관계를 학습.

호출 순서:
  upper_body_setup(env)         — G1State2LeanEnv.load_managers() 직후 1회
  upper_body_control_step(env)  — G1State2LeanEnv._pre_physics_step() 마다
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── 오른팔 관절 이름 ──────────────────────────────────────────────────────────
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
NUM_RIGHT_ARM = len(RIGHT_ARM_JOINT_NAMES)  # 7

# 오른팔 nominal (자연스러운 옆으로 내린 자세)
_RIGHT_SHOULDER_PITCH_NOM_DEG = 20.0   # 약간 앞으로 들린 자세
_DISTURBANCE_OMEGA            = 1.0    # sin 주파수 [rad/s]

# Curriculum step threshold
_PHASE1_STEPS = 5_000_000
_PHASE2_STEPS = 15_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def upper_body_setup(env: ManagerBasedRLEnv) -> None:
    """오른팔 관절 인덱스 확정 + per-env 버퍼 초기화.

    G1State2LeanEnv.load_managers() 직후 반드시 호출.
    허리 / 왼팔 인덱스는 여기서 설정하지 않음 (actions.py 가 담당).
    """
    robot     = env.scene["robot"]
    all_names = robot.joint_names

    env._right_arm_idx = torch.tensor(
        [all_names.index(n) for n in RIGHT_ARM_JOINT_NAMES],
        dtype=torch.long, device=env.device,
    )

    N, dev = env.num_envs, env.device

    # 오른팔 nominal position
    env.right_arm_nominal = _build_right_arm_nominal(N, dev)

    # 오른팔 sin disturbance 위상 오프셋 (환경마다 다른 패턴)
    env.disturbance_phase_offsets = (
        torch.rand(N, NUM_RIGHT_ARM, device=dev) * 2.0 * math.pi
    )

    # lean_phase_time: phase_encoding obs 에서 사용 (미리 초기화)
    if not hasattr(env, "lean_phase_time"):
        env.lean_phase_time = torch.zeros(N, device=dev)

    # dynamics tracking reward 에서 사용하는 버퍼
    env.prev_omega    = torch.zeros(N, 3, device=dev)
    env.V_dot_desired = torch.zeros(N, 6, device=dev)


def upper_body_control_step(env: ManagerBasedRLEnv) -> None:
    """매 physics step 마다 오른팔 disturbance target 설정.

    허리 / 왼팔은 actions.py 의 position target 으로 이미 제어되므로
    여기서는 오른팔 effort target 만 계산한다.
    """
    robot = env.scene["robot"]

    full_tau = torch.zeros(env.num_envs, robot.num_joints, device=env.device)

    # ── 오른팔 disturbance (PD → effort) ─────────────────────────────────────
    amplitude = _get_disturbance_amplitude(env)
    ra_tgt = _right_arm_disturbance_target(
        t            = env.lean_phase_time,
        amplitude    = amplitude,
        nominal      = env.right_arm_nominal,
        phase_offsets= env.disturbance_phase_offsets,
    )
    q_ra  = robot.data.joint_pos[:, env._right_arm_idx]   # (N, 7)
    dq_ra = robot.data.joint_vel[:, env._right_arm_idx]   # (N, 7)
    full_tau[:, env._right_arm_idx] = (
        env.cfg.kp_arm * (ra_tgt - q_ra) - env.cfg.kd_arm * dq_ra
    )

    robot.set_joint_effort_target(full_tau)

    # ── 보조 버퍼 갱신 (dynamics tracking reward 용) ─────────────────────────
    env.prev_omega     = robot.data.root_ang_vel_w.clone()
    env.V_dot_desired  = _compute_V_dot_d(
        env.lean_phase_time, env.cfg.phase1_duration,
        env.num_envs, env.device
    )
    env.lean_phase_time = env.lean_phase_time + env.step_dt


def reset_upper_body_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """에피소드 리셋 시 controller 내부 버퍼 초기화.
    events.py 의 reset_robot_to_standing 에서 호출.
    """
    env.lean_phase_time[env_ids]  = 0.0
    env.prev_omega[env_ids]       = 0.0
    env.V_dot_desired[env_ids]    = 0.0
    # 에피소드마다 다른 disturbance 위상
    n = len(env_ids)
    env.disturbance_phase_offsets[env_ids] = (
        torch.rand(n, NUM_RIGHT_ARM, device=env.device) * 2.0 * math.pi
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_right_arm_nominal(N: int, dev) -> torch.Tensor:
    nom = torch.zeros(N, NUM_RIGHT_ARM, device=dev)
    nom[:, 0] = math.radians(_RIGHT_SHOULDER_PITCH_NOM_DEG)
    return nom


def _get_disturbance_amplitude(env: ManagerBasedRLEnv) -> float:
    """global step 기반 curriculum amplitude."""
    step = env.common_step_counter
    if step < _PHASE1_STEPS:
        return 0.0
    elif step < _PHASE2_STEPS:
        progress = (step - _PHASE1_STEPS) / (_PHASE2_STEPS - _PHASE1_STEPS)
        return 0.15 * progress
    else:
        return 0.30


def _right_arm_disturbance_target(
    t: torch.Tensor,
    amplitude: float,
    nominal: torch.Tensor,       # (N, 7)
    phase_offsets: torch.Tensor, # (N, 7)
) -> torch.Tensor:
    """
    Phase 1 (amplitude=0): q_ra = q_nominal
    Phase 2+             : q_ra = q_nominal + A·sin(ωt + φ)
    """
    if amplitude == 0.0:
        return nominal.clone()
    t_exp  = t.unsqueeze(-1)                                         # (N, 1)
    sin_term = amplitude * torch.sin(
        _DISTURBANCE_OMEGA * t_exp + phase_offsets                   # (N, 7)
    )
    return nominal + sin_term


def _compute_V_dot_d(
    t: torch.Tensor, T1: float, N: int, dev
) -> torch.Tensor:
    """목표 torso 가속도·각가속도 (dynamics tracking reward 용)."""
    from unitree_rl_lab.tasks.g1_task_state2.mdp.rewards import TARGET_PITCH_D
    tc   = t.clamp(0.0, T1)
    alph = TARGET_PITCH_D * (0.5 * math.pi / T1) ** 2 * torch.cos(math.pi * tc / T1)
    ax   = 0.04 * (math.pi / T1) ** 2 * torch.cos(2.0 * math.pi * tc / T1)
    V    = torch.zeros(N, 6, device=dev)
    V[:, 0] = ax
    V[:, 4] = alph
    return V
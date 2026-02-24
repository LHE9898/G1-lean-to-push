# Copyright (c) 2022-2025, unitree_rl_lab
"""Event functions for G1 State2."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from .controllers import reset_upper_body_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_to_standing(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """기본 직립 자세로 리셋 + controller / 성공 판정 버퍼 초기화."""
    asset      = env.scene[asset_cfg.name]
    joint_pos  = asset.data.default_joint_pos[env_ids].clone()
    joint_vel  = torch.zeros_like(joint_pos)
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 오른팔 disturbance controller 버퍼 리셋
    reset_upper_body_buffers(env, env_ids)

    # torso tracking reward 용 초기 위치 캐싱
    if "init_torso_pos_w" not in env.extras:
        env.extras["init_torso_pos_w"] = torch.zeros(
            env.num_envs, 3, device=env.device
        )
    env.extras["init_torso_pos_w"][env_ids] = (
        env.scene[asset_cfg.name].data.root_pos_w[env_ids].detach().clone()
    )

    # ★ 성공 판정 버퍼 리셋 (에피소드마다 초기화)
    for key in ("success_hold_time", "first_contact_time"):
        if key in env.extras:
            if key == "first_contact_time":
                env.extras[key][env_ids] = -1.0   # -1 = 아직 contact 없음
            else:
                env.extras[key][env_ids] = 0.0

    # ★ 발 초기 위치 캐싱 (foot_slide_penalty 용)
    robot = env.scene[asset_cfg.name]
    names = robot.body_names
    l_idx = names.index("left_ankle_roll_link")
    r_idx = names.index("right_ankle_roll_link")

    if "init_left_foot_pos_xy" not in env.extras:
        env.extras["init_left_foot_pos_xy"]  = torch.zeros(
            env.num_envs, 2, device=env.device
        )
        env.extras["init_right_foot_pos_xy"] = torch.zeros(
            env.num_envs, 2, device=env.device
        )

    env.extras["init_left_foot_pos_xy"][env_ids]  = (
        robot.data.body_pos_w[env_ids, l_idx, :2].detach().clone()
    )
    env.extras["init_right_foot_pos_xy"][env_ids] = (
        robot.data.body_pos_w[env_ids, r_idx, :2].detach().clone()
    )


def randomize_robot_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range:  tuple = (0.5, 1.1),
    dynamic_friction_range: tuple = (0.4, 0.9),
) -> None:
    """마찰 계수 무작위화 stub."""
    pass  # TODO: mdp.randomize_rigid_body_material


def randomize_robot_mass(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple = (-2.0, 2.0),
) -> None:
    """질량 무작위화 stub."""
    pass  # TODO: mdp.randomize_rigid_body_mass


def randomize_table_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    table_x_range: tuple = (-0.05, 0.05),
    table_y_range: tuple = (-0.05, 0.05),
) -> None:
    """테이블 위치 소량 무작위화."""
    table      = env.scene["table"]
    root_state = table.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    n = len(env_ids)
    root_state[:, 0] += torch.empty(n, device=env.device).uniform_(*table_x_range)
    root_state[:, 1] += torch.empty(n, device=env.device).uniform_(*table_y_range)
    table.write_root_state_to_sim(root_state, env_ids=env_ids)
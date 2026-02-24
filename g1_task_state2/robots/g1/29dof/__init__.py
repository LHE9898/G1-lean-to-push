# Copyright (c) 2022-2025, unitree_rl_lab
"""G1 29DOF State2 태스크 등록.

Isaac Lab gym 환경 등록 패턴 (velocity_env_cfg.py / g1_lean_push 참고):
  gym.register(id=..., entry_point=..., kwargs={"env_cfg_entry_point": ...})
"""
import gymnasium as gym

##############################################################################
# 학습용 환경
##############################################################################
gym.register(
    id="Unitree-G1-State2-Lean-v0",
    entry_point=f"{__name__}.lean_env_cfg:G1State2LeanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lean_env_cfg:G1State2LeanEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.g1_task_state2.agents.rsl_rl_ppo_cfg:G1State2LeanPPORunnerCfg",
    },
)

##############################################################################
# 평가 / 시각화용 환경
##############################################################################
gym.register(
    id="Unitree-G1-State2-Lean-Play-v0",
    entry_point=f"{__name__}.lean_env_cfg:G1State2LeanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lean_env_cfg:G1State2LeanEnvCfgPLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.`tasks.g1_task_state2.agents.rsl_rl_ppo_cfg:G1State2LeanPPORunnerCfg",
    },
)

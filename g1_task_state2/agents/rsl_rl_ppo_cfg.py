# Copyright (c) 2022-2025, unitree_rl_lab
"""PPO runner config for G1 State2: Disturbance-aware Multi-Contact Lean Stabilization.

g1_lean_push 의 G1TableLeanPPORunnerCfg 패턴을 그대로 따르며,
State2 태스크에 맞게 experiment_name / iteration 수 조정.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1State2LeanPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int        = 24
    max_iterations: int           = 5000        # State2 는 disturbance 적응 포함 → 더 길게
    save_interval: int            = 200
    experiment_name: str          = "g1_state2_lean"
    empirical_normalization: bool = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# Copyright (c) 2022-2025, unitree_rl_lab
"""G1 State2: Disturbance-aware Multi-Contact Lean Stabilization.

제어 구조 확정:
  RL Policy (22-DoF)   : 다리(12) + 허리(3) + 왼팔(7)
  Model-based Controller: 오른팔(7) disturbance 전용

목표:
  torso 를 초기 upright CoM 기준으로
    x_d = +0.2 m (앞으로),  z_d = -0.1 m (아래로),  θ_pitch = +10°
  로 이동한 뒤, 왼손을 테이블에 짚어 3-point contact 로 균형을 유지.

Observation: 90-dim (disturbance-aware)
Action     : 22-dim residual (legs 12 + waist 3 + left_arm 7)
"""
from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.g1_task_state2 import mdp
from unitree_rl_lab.tasks.g1_task_state2.mdp.controllers import (
    upper_body_setup,
    upper_body_control_step,
)

# ── 기하 상수 ──────────────────────────────────────────────────────────────────
TABLE_DISTANCE_X = 0.15   # 로봇 정면 → 테이블 앞면 [m]
TABLE_WIDTH      = 0.4
TABLE_DEPTH      = 0.4
TABLE_HEIGHT     = 0.7   # 테이블 상면 z [m]

# ── Joint 그룹 ────────────────────────────────────────────────────────────────
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint",  "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_roll_joint",      "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HIP_ROLL_YAW_NAMES  = [
    "left_hip_roll_joint",  "left_hip_yaw_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint",
]
HIP_YAW_JOINT_NAMES = ["left_hip_yaw_joint", "right_hip_yaw_joint"]


##############################################################################
# Custom Env
##############################################################################

class G1State2LeanEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + 오른팔 disturbance controller hook.

    허리 / 왼팔은 RL policy 가 직접 제어.
    오른팔만 model-based disturbance controller 로 구동.
    """

    def load_managers(self) -> None:
        # phase_encoding obs 가 _prepare_terms() 내에서 호출되므로 사전 초기화
        self.lean_phase_time = torch.zeros(self.num_envs, device=self.device)
        self.lean_phase      = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        super().load_managers()
        upper_body_setup(self)   # 오른팔 인덱스 확정 + 버퍼 초기화

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        upper_body_control_step(self)   # 오른팔 disturbance effort 기록
        super()._pre_physics_step(action)


##############################################################################
# Scene
##############################################################################

@configclass
class G1State2SceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 작업 테이블 (0.4×0.4×0.6 m, kinematic)
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_WIDTH, TABLE_DEPTH, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_DISTANCE_X + TABLE_WIDTH * 0.5, 0.0, TABLE_HEIGHT * 0.5),
        ),
    )

    # IMU
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        gravity_bias=(0.0, 0.0, 0.0),
    )

    # 왼손목 contact (3-point 지지 + slip 감지) — 테이블만 필터링
    left_wrist_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )

    # 오른손목 contact (관측용만 — reward 직접 영향 X)
    right_wrist_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )

    # 발바닥 contact (foot_contact_loss termination 용)
    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=3,
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=3,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


##############################################################################
# MDP Configs
##############################################################################

@configclass
class ActionsCfg:
    """22-DoF residual: 다리(12) + 허리(3) + 왼팔(7)."""
    leg_waist_left_arm: mdp.LegWaistLeftArmResidualActionCfg = (
        mdp.LegWaistLeftArmResidualActionCfg(
            asset_name="robot",
            delta_q_limit=0.30,
            motor_delay_steps=2,
        )
    )


@configclass
class ObservationsCfg:
    """
    총 90-dim:
      q_leg(12) + dq_leg(12)
      q_waist(3) + dq_waist(3)
      q_la(7) + dq_la(7) + c_lh(1)
      R_torso(9) + ω_torso(3) + imu_acc(3)
      c_lh(1) + table_rel(3) + phase_enc(2) + lean_phase(1) + pitch_err(1)
      q_ra(7) + dq_ra(7) + v_rh(3)   ← disturbance observer ★
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ── 다리 (12+12) ──────────────────────────────────────────────────────
        leg_joint_pos = ObsTerm(
            func=mdp.leg_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.004),
        )
        leg_joint_vel = ObsTerm(
            func=mdp.leg_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.02),
        )
        # ── 허리 (3+3) RL 제어 ────────────────────────────────────────────────
        waist_joint_pos = ObsTerm(
            func=mdp.waist_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.002),
        )
        waist_joint_vel = ObsTerm(
            func=mdp.waist_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.01),
        )
        # ── 왼팔 (7+7+1) RL 제어 ─────────────────────────────────────────────
        left_arm_joint_pos = ObsTerm(
            func=mdp.left_arm_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.004),
        )
        left_arm_joint_vel = ObsTerm(
            func=mdp.left_arm_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.02),
        )
        left_contact = ObsTerm(
            func=mdp.left_arm_contact_flag,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_contact")},
        )
        # ── Torso IMU (9+3+3) ─────────────────────────────────────────────────
        torso_rot_mat = ObsTerm(
            func=mdp.torso_rotation_matrix,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        torso_ang_vel = ObsTerm(
            func=mdp.torso_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoise(mean=0.0, std=0.01),
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=GaussianNoise(mean=0.0, std=0.04),
        )
        # ── 보조 (3+2+1+1) ────────────────────────────────────────────────────
        table_rel_pos = ObsTerm(
            func=mdp.table_relative_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        phase_enc = ObsTerm(func=mdp.phase_encoding)
        lean_phase = ObsTerm(func=mdp.lean_phase_obs)
        pitch_error = ObsTerm(
            func=mdp.torso_pitch_error,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # ── 오른팔 disturbance observer (7+7+3) ★ ────────────────────────────
        right_arm_joint_pos = ObsTerm(
            func=mdp.right_arm_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.004),
        )
        right_arm_joint_vel = ObsTerm(
            func=mdp.right_arm_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
            noise=GaussianNoise(mean=0.0, std=0.02),
        )
        right_hand_velocity = ObsTerm(
            func=mdp.right_hand_velocity,
            params={"asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_wrist_yaw_link"]
            )},
        )

        def __post_init__(self):
            self.enable_corruption  = True
            self.concatenate_terms  = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_standing,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_robot_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_robot_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_table = EventTerm(
        func=mdp.randomize_table_position,
        mode="reset",
    )


@configclass
class RewardsCfg:
    # ① Torso Tracking ────────────────────────────────────────────────────────
    # weight 를 양수로: 함수 내부에서 이미 음수 반환하므로 부호 통일
    torso_pos = RewTerm(
        func=mdp.torso_pos_tracking,
        weight=1.0,          # 로그에서 거의 0 → 초기엔 낮게 유지
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_d": 0.20, "z_d": -0.10,
            "w_x": 2.0,  "w_z": 2.0,
        },
    )
    torso_pitch = RewTerm(
        func=mdp.torso_pitch_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "theta_d": math.radians(10.0),
            "w_theta": 1.0,
        },
    )
    # ② Contact 유도 (가장 중요 — 학습 초기 핵심 신호) ────────────────────────
    # arm_height / arm_reach 를 높게 설정해 손을 테이블 위로 유도
    arm_height = RewTerm(
        func=mdp.arm_height_tracking,
        weight=5.0,          # ★ 손 높이 유도: 가장 강한 신호
        params={
            "asset_cfg":      SceneEntityCfg("robot"),
            "left_body_name": "left_wrist_yaw_link",
            "table_height":   TABLE_HEIGHT,
        },
    )
    arm_reach = RewTerm(
        func=mdp.arm_reach_reward,
        weight=4.0,          # ★ 손 x,y 위치 유도
        params={
            "asset_cfg":      SceneEntityCfg("robot"),
            "left_body_name": "left_wrist_yaw_link",
            "table_height":   TABLE_HEIGHT,
        },
    )
    left_hand_contact = RewTerm(
        func=mdp.left_hand_contact_reward,
        weight=10.0,         # ★ contact 달성 시 큰 보상
        params={"sensor_cfg": SceneEntityCfg("left_wrist_contact"), "w_c": 1.0},
    )
    # ③ Balance ───────────────────────────────────────────────────────────────
    torso_ang_vel = RewTerm(
        func=mdp.torso_ang_vel_penalty,
        weight=-0.1,         # 낮게 — 초기 학습 방해하지 않도록
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    imu_acc = RewTerm(
        func=mdp.imu_acc_penalty,
        weight=-0.01,        # ★ 크게 낮춤 (로그에서 2.45로 지배적이었음)
        params={"sensor_cfg": SceneEntityCfg("imu")},
    )
    # ④ Slip Penalty — contact 없을 때는 의미 없으므로 초기엔 낮게 ─────────────
    left_hand_slip = RewTerm(
        func=mdp.left_hand_slip_penalty,
        weight=-1.0,
        params={
            "asset_cfg":      SceneEntityCfg("robot"),
            "left_body_name": "left_wrist_yaw_link",
            "w_s":            1.0,
        },
    )
    # ⑤ Effort ────────────────────────────────────────────────────────────────
    leg_nominal = RewTerm(
        func=mdp.leg_nominal_pose_penalty,
        weight=-0.5,         # 낮게 — 왼팔이 움직여야 하므로 너무 강하면 방해
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    leg_roll_yaw = RewTerm(
        func=mdp.leg_roll_yaw_penalty,
        weight=-1.0,         # 이전 -6.0 → 낮춤
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_ROLL_YAW_NAMES)},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_penalty,
        weight=-0.005,       # 낮게 — 초기 탐색 허용
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # ⑦ 발바닥 위치 고정 ──────────────────────────────────────────────────────
    # 허리를 숙이는 동안 발이 미끄러지지 않도록 고정
    # 발목(ankle_pitch, ankle_roll)은 RL 이 자유롭게 제어 가능 — 이 reward 와 무관
    foot_slide = RewTerm(
        func=mdp.foot_slide_penalty,
        weight=-10.0,        # 강하게 — 발 미끄러짐은 절대 허용하지 않음
        params={
            "asset_cfg":            SceneEntityCfg("robot"),
            "left_foot_body_name":  "left_ankle_roll_link",
            "right_foot_body_name": "right_ankle_roll_link",
            "w_slide":              5.0,
        },
    )
    foot_vel = RewTerm(
        func=mdp.foot_vel_penalty,
        weight=-5.0,         # 미끄러지기 전에 속도 자체를 억제
        params={
            "asset_cfg":            SceneEntityCfg("robot"),
            "left_foot_body_name":  "left_ankle_roll_link",
            "right_foot_body_name": "right_ankle_roll_link",
            "w_vel":                2.0,
        },
    )


@configclass
class TerminationsCfg:
    # ── 시간 초과 (성공 못하면 실패) ─────────────────────────────────────────
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # ── 실패 조건 ────────────────────────────────────────────────────────────
    roll_exceeded = DoneTerm(
        func=mdp.torso_roll_exceeded,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_deg": 40.0},
    )
    too_low = DoneTerm(
        func=mdp.torso_too_low,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.50},
    )
    hip_yaw = DoneTerm(
        func=mdp.hip_yaw_exceeded,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_JOINT_NAMES),
            "limit_deg": 30.0,
        },
    )

    # ── ★ 성공 조건: contact 후 5초 균형 유지 ─────────────────────────────────
    # time_out=False → 성공은 실패가 아니라 에피소드 완료로 기록됨
    task_success = DoneTerm(
        func=mdp.task_success,
        time_out=False,
        params={
            "asset_cfg":         SceneEntityCfg("robot"),
            "sensor_cfg":        SceneEntityCfg("left_wrist_contact"),
            "hold_duration":     5.0,    # contact 유지 시간 [s]
            "contact_threshold": 1.0,    # 접촉 판정 힘 임계값 [N]
            "pos_tol":           0.05,   # torso 위치 허용 오차 [m]
            "pitch_tol":         math.radians(3.0),  # torso pitch 허용 오차
        },
    )

    # ── 발 contact 손실 (학습 안정 후 활성화 권장) ───────────────────────────
    # foot_contact_loss = DoneTerm(
    #     func=mdp.foot_contact_loss,
    #     params={
    #         "left_foot_cfg":  SceneEntityCfg("left_foot_contact"),
    #         "right_foot_cfg": SceneEntityCfg("right_foot_contact"),
    #     },
    # )


##############################################################################
# Main Env Config
##############################################################################

@configclass
class G1State2LeanEnvCfg(ManagerBasedRLEnvCfg):
    scene:        G1State2SceneCfg = G1State2SceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    events:       EventCfg        = EventCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 오른팔 disturbance controller PD gain
    kp_arm: float = 80.0
    kd_arm: float = 8.0

    # 위상 시간 (phase_encoding / V_dot_desired 계산용)
    phase1_duration: float = 3.0   # torso 이동 기준 시간 [s]
    phase2_duration: float = 5.0   # 3-point 유지 시간 [s]

    def __post_init__(self):
        self.decimation         = 4
        self.episode_length_s   = 15.0  # 10 → 15s: 초기 탐색 시간 확보
        self.sim.dt             = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.eye    = (2.5, -2.0, 2.0)
        self.viewer.lookat = (0.5,  0.0, 0.8)

        self.scene.left_wrist_contact.update_period  = self.sim.dt
        self.scene.right_wrist_contact.update_period = self.sim.dt
        self.scene.left_foot_contact.update_period   = self.sim.dt
        self.scene.right_foot_contact.update_period  = self.sim.dt
        self.scene.imu.update_period                 = self.sim.dt


@configclass
class G1State2LeanEnvCfg_PLAY(G1State2LeanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.randomize_friction = None
        self.events.randomize_mass     = None
        self.events.randomize_table    = None
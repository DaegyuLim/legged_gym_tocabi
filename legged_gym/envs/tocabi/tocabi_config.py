# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TocabiRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096 # 4096
        num_observations = 169 # 3+3+3+4+n+n+n+120
        num_actions = 12 # 12

    
    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 80 # number of terrain cols (types) default:20

    class commands( LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.3, 0.5] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-1., 1.]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'L_HipYaw_Joint': 0.0,
            'L_HipRoll_Joint': 0.0,
            'L_HipPitch_Joint': -0.20,
            'L_Knee_Joint': 0.6,
            'L_AnklePitch_Joint': -0.40,
            'L_AnkleRoll_Joint': 0.0,

            'R_HipYaw_Joint': 0.0,
            'R_HipRoll_Joint': 0.0,
            'R_HipPitch_Joint': -0.20,
            'R_Knee_Joint': 0.6,
            'R_AnklePitch_Joint': -0.40,
            'R_AnkleRoll_Joint': 0.0,

            # 'Waist1_Joint': 0.0,
            # 'Waist2_Joint': 0.0,
            # 'Upperbody_Joint': 0.0,
            # 'Neck_Joint': 0.0,
            # 'Head_Joint': 0.0,

            # 'L_Shoulder1_Joint': 0.57,
            # 'L_Shoulder2_Joint': 0.0,
            # 'L_Shoulder3_Joint': 0.0,
            # 'L_Armlink_Joint': 0.0,
            # 'L_Elbow_Joint': 0.0,
            # 'L_Forearm_Joint': 0.0,
            # 'L_Wrist1_Joint': 0.0,
            # 'L_Wrist2_Joint': 0.0,

            # 'R_Shoulder1_Joint': -1.57,
            # 'R_Shoulder2_Joint': 0.0,
            # 'R_Shoulder3_Joint': 0.0,
            # 'R_Armlink_Joint': 0.0,
            # 'R_Elbow_Joint': 0.0,
            # 'R_Forearm_Joint': 0.0,
            # 'R_Wrist1_Joint': 0.0,
            # 'R_Wrist2_Joint': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'Joint': 2000.0}
        # {   'HipYaw_Joint': 400.0, 'HipRoll_Joint': 400.0, 'HipPitch_Joint': 400.0,
        #                 'Knee_Joint': 400., 'AnklePitch_Joint': 400., 'AnkleRoll_Joint': 400.}  # [N*m/rad]
        damping = {'Joint': 50.0}
        # {   'HipYaw_Joint': 50.0, 'HipRoll_Joint': 50.0, 'HipPitch_Joint': 50.0,
        #                 'Knee_Joint': 50.0, 'AnklePitch_Joint': 50.0, 'AnkleRoll_Joint': 50.0}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT (200hz) per policy DT 
        decimation = 4  # 50hz
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tocabi/urdf/tocabi.urdf'
        name = "tocabi"
        foot_name = "AnkleRoll_Link"
        # penalize_contacts_on = ['Thigh_Link', 'Knee_Link']
        terminate_after_contacts_on = ['base']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # orientation = -0.5
            torques = -1.e-7
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            collision = -1.0
            dof_pos_limits = -1.
            # dof_vel_limits = -0.1
            no_fly = 0.25
            dsp = 0.00
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = 0.0
            action_rate = -0.01

        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 1000.0
        base_height_target = 1.0
        only_positive_rewards = False

    class noise:
        add_noise = True
        noise_level = 1.00 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

class TocabiRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'run_tocabi_2'
        experiment_name = 'rough_tocabi'
        num_steps_per_env = 24
        max_iterations = 1500

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  
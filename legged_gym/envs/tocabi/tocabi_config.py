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
        num_observations = 57

        '''
        self.contacts: [2]
        self.contact_forces: [6] xyz for feet
        self.base_z: [1]
        self.base_lin_vel:  torch.Size([4096, 3])
        self.base_ang_vel:  torch.Size([4096, 3])
        self.projected_gravity:  torch.Size([4096, 3])
        self.commands[:, :3]:  torch.Size([4096, 3])
        (self.dof_pos - self.default_dof_pos):  torch.Size([4096, 6])
        self.dof_vel:  torch.Size([4096, 6])
        self.actions:  torch.Size([4096, 6])

        2 + 6 + 1 + 3 + 3 + 3 + 3 + 12 + 12 + 12 = 57(num_observation)
        '''

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12 # robot actuation
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.001 # [m] Rui
        border_size = 25 # [m]
        curriculum = False # Rui
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.3
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete] # Rui
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0] # Rui
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces # Rui

    class commands( LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 10.0
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0] # min max [m/s] seems like less than or equal to 0.2 it sends 0 command
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'L_HipYaw_Joint': 0.0,
            'L_HipRoll_Joint': 0.0,
            'L_HipPitch_Joint': -0.24,
            'L_Knee_Joint': 0.6,
            'L_AnklePitch_Joint': -0.36,
            'L_AnkleRoll_Joint': 0.0,

            'R_HipYaw_Joint': 0.0,
            'R_HipRoll_Joint': 0.0,
            'R_HipPitch_Joint': -0.24,
            'R_Knee_Joint': 0.6,
            'R_AnklePitch_Joint': -0.36,
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
        control_type = 'T'
        # PD Drive parameters:
        stiffness = {'Joint': 2000.0}
        # {   'HipYaw_Joint': 400.0, 'HipRoll_Joint': 400.0, 'HipPitch_Joint': 400.0,
        #                 'Knee_Joint': 400., 'AnklePitch_Joint': 400., 'AnkleRoll_Joint': 400.}  # [N*m/rad]
        damping = {'Joint': 50.0}
        # {   'HipYaw_Joint': 50.0, 'HipRoll_Joint': 50.0, 'HipPitch_Joint': 50.0,
        #                 'Knee_Joint': 50.0, 'AnklePitch_Joint': 50.0, 'AnkleRoll_Joint': 50.0}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 100.0
        # decimation: Number of control action updates @ sim DT (200hz) per policy DT 
        decimation = 2  # 100hz
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tocabi/urdf/tocabi.urdf'
        name = "tocabi"
        foot_name = "AnkleRoll_Link"
        # penalize_contacts_on = ['Thigh_Link', 'Knee_Link']
        terminate_after_contacts_on = ['base']

        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        # fix_base_link = True
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 20.
        max_linear_velocity = 20.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.3

        ext_force_robots = False
        ext_force_vector_6d = [0, -20, 0, 0, 0, 0]
        ext_force_start_time = 5.0
        ext_force_duration = 0.2

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -500.
            # traking
            tracking_lin_vel = 10
            tracking_ang_vel = 10.

            # regulation in task space
            lin_vel_z = -0.
            ang_vel_xy = -0.0
            
            # regulation in joint space
            energy = 0.0 # 0.01
            torques = -1.e-6
            dof_vel = -0.0
            dof_acc = -0
            action_rate = -2e-4 # -0.000001

            # walking specific rewards
            feet_air_time = 100.
            collision = -0.
            feet_stumble = -0.0 
            stand_still = 3.0
            no_fly = 5.0
            
            
            # joint limits
            torque_limits = -0.1
            dof_vel_limits = -1
            dof_pos_limits = -10.            
            
            # DRS
            orientation = 0.0 # Rui
            base_height = 0.0
            joint_regularization = 0.0

            # PBRS rewards
            ori_pb = 30.0
            baseHeight_pb = 5.0
            jointReg_pb = 5.0
            energy_pb = 0.01
            action_rate_pb = 0.0
            feet_contact_forces_pb = 10.0
            feet_ori_pb = 1.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
        orientation_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        energy_sigma = 1e5
        action_rate_sigma = 1
        force_sigma = 1e3

        soft_dof_pos_limit = 0.98 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.5

        base_height_target = 0.95 # 0.93 for model-base
        max_contact_force = 1500. # forces above this value are penalized
        

    class normalization:
        class obs_scales:
            lin_vel = 1.0 # Rui
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 1e6
        clip_actions = 500.

    class noise:
        add_noise = False
        noise_level = 0.05 # scales other values
        class noise_scales:
            dof_pos = 0.005
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.02
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [-10., 0, 0]  # [m]
        # pos = [10, -1, 6]  # [m]
        # lookat = [-10., 0, 0.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 20
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class TocabiRoughCfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'tanh' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches (4)
        learning_rate = 2.e-5 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration (24)
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'tocabi_test'
        run_name = 'tocabi_test'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt



  
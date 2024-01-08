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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Bolt12(LeggedRobot):

    def _custom_init(self, cfg):
        self.control_tick = torch.zeros(
            self.num_envs, 1, dtype=torch.int,
            device=self.device, requires_grad=False)
 
        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
       

    def compute_observations(self):
        """ Computes observations
        """
        self.base_z = self.root_states[:, 2].unsqueeze(1)
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        
        self.obs_buf = torch.cat((
            self.contacts,
            self.base_z,  
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ),dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _custom_reset(self, env_ids):
        self.control_tick[env_ids, 0] = 0

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        single_contact = torch.sum(1.*contacts, dim=1)==1
        single_contact *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        return 1.*single_contact
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.001
        single_contact = torch.sum(1.*contacts, dim=1)==1
        contact_filt = torch.logical_or(contacts, self.last_contacts) 
        self.last_contacts = contacts
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum( torch.clip(self.feet_air_time - 0.3, min=0.0, max=0.7) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        # rew_airTime *= single_contact #no reward for flying or double support
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        joint_error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-joint_error/self.cfg.rewards.tracking_sigma) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_torques(self):
        # Penalize torques
        return torch.mean(torch.square(self.torques), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.mean(torch.square( (self.last_actions - self.actions)/self.dt ), dim=1)
 
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     action_rate = torch.sum(torch.square( (self.last_actions - self.actions) ), dim=1)
    #     # print("action_rate: ", action_rate[0])
    #     return torch.exp(-action_rate/self.cfg.rewards.action_rate_sigma)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # print("orientation_error: ", orientation_error[0])
        return torch.exp(-orientation_error/self.cfg.rewards.orientation_sigma)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_energy(self):
        #veltorque, value  scale or sum
              
        if self.cfg.rewards.only_positive_rewards:
            positive_energy=(self.torques*self.dof_vel).clip(min=0.)
            positive_energy_square = torch.mean(torch.square(positive_energy), dim=1)
            return torch.exp(-positive_energy_square/self.cfg.rewards.energy_sigma)
        else:
            energy = (self.torques*self.dof_vel)
            energy_square = torch.mean(torch.square(energy), dim=1)
            # print("energy_square: ", energy_square[0])
            return torch.exp(-energy_square/self.cfg.rewards.energy_sigma)
        
        ### sum over all columns of the square values for energy (using joint velocities)

    def _reward_base_height(self):
        # Reward tracking specified base height
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error = error.flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 6]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        error += self.sqrdexp(
            ( (self.dof_pos[:, 1] - self.default_dof_pos[:, 1] ) - (self.dof_pos[:, 7] - self.default_dof_pos[:, 7]) )
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        error += self.sqrdexp(
            (self.dof_pos[:, 2] + self.dof_pos[:, 8])
            / self.cfg.normalization.obs_scales.dof_pos)
        # print("self.dof_pos[0, 6]: ", self.dof_pos[0, 1], "// self.dof_pos[0, 6]: ", self.dof_pos[0, 6])
        return error/4

    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_energyPrev = self._reward_energy()
        self.rwd_actionRatePrev = self._reward_action_rate()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.pre_physics_step()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.cfg.domain_rand.ext_force_robots:
                for env_idx in range(self.num_envs):
                    if (self.control_tick[env_idx, 0] >self.cfg.domain_rand.ext_force_start_time/self.dt )&(self.control_tick[env_idx, 0] <= (self.cfg.domain_rand.ext_force_start_time+self.cfg.domain_rand.ext_force_duration)/self.dt ):
                        self.ext_forces[env_idx, 0, 0:3] = torch.tensor(self.cfg.domain_rand.ext_force_vector_6d[0:3], device=self.device, requires_grad=False)    #index: root, body, force axis(6)
                        self.ext_torques[env_idx, 0, 0:3] = torch.tensor(self.cfg.domain_rand.ext_force_vector_6d[3:6], device=self.device, requires_grad=False)
                        
                        # print("self.ext_forces[env_idx, 0, 1]: ", self.ext_forces[env_idx, 0, 1])
                    else:
                        self.ext_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                        self.ext_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ext_forces), gymtorch.unwrap_tensor(self.ext_torques), gymapi.ENV_SPACE)

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.control_tick = self.control_tick + 1
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.70 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_curriculum)
        
        if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.50 * self.reward_scales["tracking_ang_vel"]:
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.05, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.05, 0., self.cfg.commands.max_curriculum)


    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt

    def _reward_energy_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_energy() - self.rwd_energyPrev)
        return delta_phi / self.dt
    
    def _reward_action_rate_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_action_rate() - self.rwd_actionRatePrev)
        return delta_phi / self.dt

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # orientation x
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        # orientation y
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        # base height z
        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.15, dim=1)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf


    # ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

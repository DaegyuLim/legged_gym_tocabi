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

class Tocabi(LeggedRobot):
    def _reward_no_fly(self):
        # contacts = torch.norm(self.contact_forces[:, self.feet_indices, :3],  dim=2) > 1.0
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        # print(self.feet_indices)
        # print(self.contact_forces[:, self.feet_indices, 2])
        single_contact = torch.sum(1.*contacts, dim=1)==1
        # single_contact *= (torch.norm(self.commands[:, :2], dim=1) > 0.05) # no reward for zero command
        # single_contact *= (torch.norm((self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1) < 0.04) # no reward for large tracking error
        # single_contact *= (torch.norm((self.commands[:, 2] - self.base_ang_vel[:, 2]), dim=1) < 0.04) # no reward for large tracking error
        return 1.*single_contact

    def _reward_dsp(self):
        # contacts = torch.norm(self.contact_forces[:, self.feet_indices, :3],  dim=2) > 1.0
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        # print(self.feet_indices)
        # print(self.contact_forces[:, self.feet_indices, 2])
        double_contact = torch.sum(1.*contacts, dim=1)==2
        double_contact *= (torch.norm(self.commands[:, :2], dim=1) <= 0.05) # reward for zero command
        return 1.*double_contact

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.03)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = (torch.sum((self.feet_air_time - 0.5)* first_contact, dim=1)).clip(min=-0.5, max=0.5) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.05 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
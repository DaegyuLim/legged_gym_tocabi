from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Bolt6(LeggedRobot):
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        
        
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    
    def _reward_energy(self):
        #veltorque, value  scale or sum
        energy=(self.torques*self.dof_vel)
        positive_energy=(self.torques*self.dof_vel).clip(min=0.)
        if self.cfg.rewards.positive_energy_reward:
            return torch.sum(torch.square(positive_energy), dim=1)
        else:
            return torch.sum(torch.square(energy), dim=1)
        ### sum over all columns of the square values for energy (using joint velocities)
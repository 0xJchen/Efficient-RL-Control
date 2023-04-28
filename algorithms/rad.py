import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from copy import deepcopy

import utils
from encoder import make_encoder
import data_augs as rad 
from algorithms.sac import SAC


class RAD(SAC):
    """Augment obs in SAC"""
    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(self.augs_funcs)
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.pcr.encoder, self.pcr.encoder_target,
                self.encoder_tau
            )
            

        

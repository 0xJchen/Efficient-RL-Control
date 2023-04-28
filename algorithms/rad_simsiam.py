import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from copy import deepcopy

import utils, utilsmod
from encoder import make_encoder
import data_augs as rad 
from algorithms.rad import RAD
from utils import cos_sim, cos_sim_siam

class RAD_SIMSIAM(RAD):
    def update_aux(self, obses_q, obses_k, obses_next, actions, L, step):
        self.pcr.train(True)

        ploss, closs = 0, 0

        z_o = self.pcr.encoder(obses_q)
        z_c = self.pcr.encoder(obses_k)  # for consistency

        for i in range(self.pred_step):
            z_p = self.pcr.encoder(obses_next[:, i, :, :, :])

            z_o = self.pcr.transition(z_o, actions[:, i, :])
            z_c = self.pcr.transition(z_c, actions[:, i, :])

            proj_z_o = self.pcr.projector(z_o)
            proj_z_c = self.pcr.projector(z_c)
            proj_z_p = self.pcr.projector(z_p)

            pred_z_o_pre = self.pcr.predictor_pre(proj_z_o)
            pred_z_o_con = self.pcr.predictor_con(proj_z_o)
            pred_z_p = self.pcr.predictor_pre(proj_z_p)
            pred_z_c = self.pcr.predictor_con(proj_z_c)

            ploss += cos_sim_siam(pred_z_o_pre, proj_z_p)/2 + cos_sim_siam(pred_z_p, proj_z_o)/2
            closs += cos_sim_siam(pred_z_o_con, proj_z_c)/2 + cos_sim_siam(pred_z_c, proj_z_o)/2

        self.pcr_optimizer.zero_grad()
        loss = (ploss + closs) / self.pred_step
        loss.backward()
        self.pcr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_pcr/ploss', ploss, step)
            L.log('train_pcr/closs', closs, step)

    def update(self, replay_buffer, L, step):
        obs_q, obs_k, action, reward, next_obs, not_done = replay_buffer.sample_traj_test(self.augs_funcs, self.pred_step)
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            self.update_aux(obs_q, obs_k, next_obs, action, L, step)

        self.update_critic(obs_q, action[:,0,:], reward, next_obs[:,0,:,:,:], not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs_q, L, step)

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


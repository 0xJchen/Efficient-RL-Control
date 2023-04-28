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
from algorithms.rad_byol import RAD_BYOL
from utils import cos_sim

class RAD_BYOL_DEMA_CURL(RAD_BYOL):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.5,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        soda_tau=0.005,
        data_augs = '',
        pred_step=1,
        weight=1,
        view=1
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.encoder_feature_dim = encoder_feature_dim
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.data_augs = data_augs
        self.soda_tau = soda_tau
        self.pred_step = pred_step
        self.weight = weight
        self.view = view
        self.augs_funcs = {}

        aug_to_func = {
                'crop':rad.random_crop,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'translate':rad.random_translate,
                'no_aug':rad.no_aug,
            }

        for aug_name in self.data_augs.split('-'):
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

        self.actor = utilsmod.Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = utilsmod.Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = utilsmod.Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # add predictor and dynamic
        self.pcr = utilsmod.CURL(self.critic, self.critic_target, encoder_feature_dim, action_shape[0]).to(device)
    
        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.cnn = self.critic.encoder.cnn
        #self.actor.encoder.projector = self.critic.encoder.projector

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'pixel':
            # optimizer for critic encoder for reconstruction loss
            self.pcr_optimizer = torch.optim.Adam(
                self.pcr.parameters(), lr=encoder_lr, betas=(critic_beta, 0.999)
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def update_aux(self, obses_q, obses_k, obses_next, actions, L, step):
        self.pcr.train(True)

        ploss, closs = 0, 0

        if self.weight != 0:
            z_o = self.pcr.encoder(obses_q)
            z_c = self.pcr.encoder_target(obses_k)  # for consistency

            for i in range(self.pred_step):
                z_p = self.pcr.encoder_target(obses_next[:, i, :, :, :]).detach()

                z_o = self.pcr.transition(z_o, actions[:, i, :])
                z_c = self.pcr.transition_target(z_c, actions[:, i, :]).detach()

                p_logits = self.pcr.compute_logits(z_o, z_p)
                c_logits = self.pcr.compute_logits(z_o, z_c)
                labels = torch.arange(p_logits.shape[0]).long().to(self.device)

                ploss += self.cross_entropy_loss(p_logits, labels)
                closs += self.cross_entropy_loss(c_logits, labels)

            self.pcr_optimizer.zero_grad()
            loss = (ploss + self.weight * closs) / self.pred_step
            loss.backward()
            self.pcr_optimizer.step()

            if step % self.log_interval == 0:
                L.log('train_pcr/ploss', ploss, step)
                L.log('train_pcr/closs', closs, step)
        else:
            z_o = self.pcr.encoder(obses_q)

            for i in range(self.pred_step):
                z_p = self.pcr.encoder_target(obses_next[:, i, :, :, :]).detach()

                z_o = self.pcr.transition(z_o, actions[:, i, :])

                p_logits = self.pcr.compute_logits(z_o, z_p)
                labels = torch.arange(p_logits.shape[0]).long().to(self.device)

                ploss += self.cross_entropy_loss(p_logits, labels)

            self.pcr_optimizer.zero_grad()
            loss = ploss / self.pred_step
            loss.backward()
            self.pcr_optimizer.step()

            if step % self.log_interval == 0:
                L.log('train_pcr/ploss', ploss, step)

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
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
            utils.soft_update_params(
                self.pcr.transition, self.pcr.transition_target,
                self.encoder_tau
            )

        obs_q, obs_k, action, reward, next_obs, not_done = replay_buffer.sample_traj(self.augs_funcs, self.pred_step)

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            self.update_aux(obs_q, obs_k, next_obs, action, L, step)
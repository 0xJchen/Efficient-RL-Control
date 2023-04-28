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

class RAD_BYOL_NOINV(RAD_BYOL):
    def update_aux(self, obses_q, obses_k, obses_next, actions, L, step):
        self.pcr.train(True)

        ploss, closs = 0, 0

        z_o = self.pcr.encoder(obses_q)

        for i in range(self.pred_step):
            z_t = self.pcr.encoder_target(obses_next[:, i, :, :, :])
            z_o = self.pcr.transition(z_o, actions[:, i, :])

            proj_z_o = self.pcr.projector(z_o)
            proj_z_t = self.pcr.projector_target(z_t).detach()

            pred_z_o_pre = self.pcr.predictor_pre(proj_z_o)

            ploss += utils.cos_sim(pred_z_o_pre, proj_z_t)

        self.pcr_optimizer.zero_grad()
        loss = ploss/self.pred_step
        loss.backward()
        self.pcr_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train_pcr/ploss', ploss, step)
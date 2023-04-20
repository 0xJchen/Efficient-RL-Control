import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from copy import deepcopy

import utils
from utils import _get_out_shape, weight_init
from encoder import make_encoder
import data_augs as rad 

LOG_FREQ = 10000

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False,
        ):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        # for k, v in self.outputs.items():
        #     L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        # for k, v in self.outputs.items():
        #     L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


# Forward dynamic model with two predictors
class Transition(nn.Module):
	def __init__(self, feature_dim, action_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(feature_dim+action_dim, int(feature_dim/2)),
            nn.LayerNorm(int(feature_dim/2)),
			nn.ReLU(),
			nn.Linear(int(feature_dim/2), feature_dim)
		)
		self.apply(weight_init)
		# self.W_acc = nn.Parameter(torch.rand(feature_dim, feature_dim))

        # # MLP: two layers
		# self.predictor_acc = PredMLP(feature_dim, feature_dim, feature_dim)
		# self.predictor_inv = PredMLP(feature_dim, feature_dim, feature_dim)

	def forward(self, z, a):
		za = torch.cat([z, a], dim=1).cuda()
		return self.net(za)


class Projector(nn.Module):
    def __init__(self, projection_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.apply(weight_init)
    
    def forward(self, x):
        return self.mlp(x)


class Predictor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.nn = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x):
        return self.nn(x)


class Predictor_Two_Layers(nn.Module):
    def __init__(self, projection_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.apply(weight_init)
    
    def forward(self, x):
        return self.mlp(x)


class MLPhead(nn.Module):
    '''Compressor'''
    def __init__(self, cnn_dim, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cnn_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)


class PCR(nn.Module):
    def __init__(self, critic, critic_target, feature_dim, action_dim):
        super().__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.transition = Transition(feature_dim, action_dim)
        self.projector = Projector(feature_dim, 256, feature_dim)
        self.projector_target = Projector(feature_dim, 256, feature_dim)
        self.projector_target.load_state_dict(self.projector.state_dict())
        self.predictor_pre = Predictor(feature_dim) # Predictive loss
        self.predictor_con = Predictor(feature_dim) # Consistent loss


class PCR_DEMA(nn.Module):
    def __init__(self, critic, critic_target, feature_dim, action_dim):
        super().__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.transition = Transition(feature_dim, action_dim)
        self.transition_target = Transition(feature_dim, action_dim)
        self.transition_target.load_state_dict(self.transition.state_dict())
        self.projector = Projector(feature_dim, 256, feature_dim)
        self.projector_target = Projector(feature_dim, 256, feature_dim)
        self.projector_target.load_state_dict(self.projector.state_dict())
        self.predictor_pre = Predictor(feature_dim) # Predictive loss
        self.predictor_con = Predictor(feature_dim) # Consistent loss


class PCR_DEMA_TWO_PRED(nn.Module):
    def __init__(self, critic, critic_target, feature_dim, action_dim):
        super().__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.transition = Transition(feature_dim, action_dim)
        self.transition_target = Transition(feature_dim, action_dim)
        self.transition_target.load_state_dict(self.transition.state_dict())
        self.projector = Projector(feature_dim, 256, feature_dim)
        self.projector_target = Projector(feature_dim, 256, feature_dim)
        self.projector_target.load_state_dict(self.projector.state_dict())
        self.predictor_pre = Predictor_Two_Layers(feature_dim, 256, feature_dim) # Predictive loss
        self.predictor_con = Predictor_Two_Layers(feature_dim, 256, feature_dim) # Consistent loss


class PCR_DEC(nn.Module):
    def __init__(self, critic, critic_target, feature_dim, action_dim):
        super().__init__()
        self.encoder = critic.encoder.cnn
        self.encoder_target = critic_target.encoder.cnn

        self.mlphead = MLPhead(critic.encoder.cnn_out_dim[0], feature_dim)
        self.mlphead_target = MLPhead(critic.encoder.cnn_out_dim[0], feature_dim)
        self.mlphead_target.load_state_dict(self.mlphead.state_dict())

        self.transition = Transition(feature_dim, action_dim)
        self.projector = Projector(feature_dim, 256, feature_dim)
        self.projector_target = Projector(feature_dim, 256, feature_dim)
        self.projector_target.load_state_dict(self.projector.state_dict())
        
        self.predictor_pre = Predictor(feature_dim) # Predictive loss
        self.predictor_con = Predictor(feature_dim) # Consistent loss
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.mlphead(x)

        return x

    def encode_target(self, x):
        x = self.encoder_target(x)
        x = self.mlphead_target(x)
        
        return x



class ATC(nn.Module):
    def __init__(self, critic, critic_target, feature_dim, action_dim):
        super().__init__()
        self.encoder = critic.encoder.cnn
        self.encoder_target = critic_target.encoder.cnn

        self.mlphead = MLPhead(critic.encoder.cnn_out_dim[0], feature_dim)
        self.mlphead_target = MLPhead(critic.encoder.cnn_out_dim[0], feature_dim)
        self.mlphead_target.load_state_dict(self.mlphead.state_dict())

        self.predictor = Projector(feature_dim, 256, feature_dim)

        self.W = Predictor(feature_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = self.mlphead(x)

        return x
    
    def encode_target(self, x):
        x = self.encoder_target(x)
        x = self.mlphead_target(x)
        
        return x


class CURL(nn.Module):
    """
    CURL
    """ 
    def __init__(self, critic, critic_target, feature_dim, action_dim):
        super(CURL, self).__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder 
        self.transition = Transition(feature_dim, action_dim)
        self.transition_target = Transition(feature_dim, action_dim)
        self.transition_target.load_state_dict(self.transition.state_dict())
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
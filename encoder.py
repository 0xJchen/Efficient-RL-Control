import torch
import torch.nn as nn
from utils import _get_out_shape, weight_init

class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)
        

# class PixelEncoder(nn.Module):
#     """Convolutional encoder of pixels observations."""
#     def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, output_logits=False):
#         super().__init__()
#         assert len(obs_shape) == 3
#         self.num_layers = num_layers
#         self.num_filters = num_filters
#         self.feature_dim = feature_dim

#         self.cnn = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
#         for i in range(num_layers-1):
#             self.cnn.append(nn.ReLU())
#             self.cnn.append(nn.Conv2d(num_filters, num_filters, 3, stride=1)),
#         self.cnn.append(Flatten())
#         self.cnn = nn.Sequential(*self.cnn)
#         self.cnn_out_dim = _get_out_shape(obs_shape, self.cnn)

#         self.projector = nn.Sequential(
#             nn.Linear(self.cnn_out_dim[0], feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.Tanh()
#         )

#         self.apply(weight_init)

#     def forward(self, x, detach=False):
#         x = self.cnn(x)
#         if detach:
#             x = x.detach()
#         return self.projector(x)
        

#     def reparameterize(self, mu, logstd):
#         std = torch.exp(logstd)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def log(self, L, step, log_freq):
#         if step % log_freq != 0:
#             return

#         L.log_param('train_encoder/cnn', self.cnn, step)
#         L.log_param('train_encoder/proj', self.projector, step)


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, output_logits=False, hidden_dim=1024):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.feature_dim = feature_dim

        self.cnn = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        for i in range(num_layers-1):
            self.cnn.append(nn.ReLU())
            self.cnn.append(nn.Conv2d(num_filters, num_filters, 3, stride=1)),
        self.cnn.append(Flatten())
        self.cnn = nn.Sequential(*self.cnn)
        self.cnn_out_dim = _get_out_shape(obs_shape, self.cnn)

        self.projector = nn.Sequential(
            nn.Linear(self.cnn_out_dim[0], feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self.apply(weight_init)

    def forward(self, x, detach=False):
        x = self.cnn(x)
        if detach:
            x = x.detach()

        x = self.projector(x)
        return x
        

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        L.log_param('train_encoder/cnn', self.cnn, step)
        L.log_param('train_encoder/proj', self.projector, step)



class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )

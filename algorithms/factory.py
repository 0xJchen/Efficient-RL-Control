from algorithms.sac import SAC
from algorithms.rad import RAD

from algorithms.sac_byol import SAC_BYOL

from algorithms.rad_byol import RAD_BYOL
from algorithms.rad_byol_noinv import RAD_BYOL_NOINV
from algorithms.rad_byol_norm import RAD_BYOL_NORM
from algorithms.rad_byol_mv import RAD_BYOL_MV
from algorithms.rad_byol_mv_test import RAD_BYOL_MV_TEST
from algorithms.rad_byol_test import RAD_BYOL_TEST
from algorithms.rad_byol_conv import RAD_BYOL_CONV
from algorithms.rad_byol_dema import RAD_BYOL_DEMA
from algorithms.rad_byol_dema_2pred import RAD_BYOL_DEMA_2PRED
from algorithms.rad_byol_dema_curl import RAD_BYOL_DEMA_CURL
from algorithms.rad_byol_dema_1pred import RAD_BYOL_DEMA_1PRED
from algorithms.rad_byol_dema_0pred import RAD_BYOL_DEMA_0PRED
from algorithms.rad_byol_dema_noproj import RAD_BYOL_DEMA_NOPROJ

from algorithms.rad_simsiam import RAD_SIMSIAM
from algorithms.rad_simsiam_noinv import RAD_SIMSIAM_NOINV

from algorithms.drq_reg import DRQ_REG
from algorithms.drq_reg01 import DRQ_REG01

algorithm = {
	'SAC': SAC,
    'RAD': RAD,

    'SAC_BYOL': SAC_BYOL,

    'RAD_BYOL': RAD_BYOL,
    'RAD_BYOL_NOINV': RAD_BYOL_NOINV,
    'RAD_BYOL_NORM': RAD_BYOL_NORM,
    'RAD_BYOL_MV': RAD_BYOL_MV,
    'RAD_BYOL_MV_TEST': RAD_BYOL_MV_TEST,
    'RAD_BYOL_TEST': RAD_BYOL_TEST,
    'RAD_BYOL_CONV': RAD_BYOL_CONV,
    'RAD_BYOL_DEMA': RAD_BYOL_DEMA,
    'RAD_BYOL_DEMA_2PRED': RAD_BYOL_DEMA_2PRED,
    'RAD_BYOL_DEMA_CURL': RAD_BYOL_DEMA_CURL,
    'RAD_BYOL_DEMA_1PRED': RAD_BYOL_DEMA_1PRED,
    'RAD_BYOL_DEMA_0PRED': RAD_BYOL_DEMA_0PRED,
    'RAD_BYOL_DEMA_NOPROJ': RAD_BYOL_DEMA_NOPROJ,

    'RAD_SIMSIAM': RAD_SIMSIAM,
    'RAD_SIMSIAM_NOINV': RAD_SIMSIAM_NOINV,

    'DRQ_REG': DRQ_REG,
    'DRQ_REG01': DRQ_REG01,
}

def make_agent(obs_shape, action_shape, args, device):
    if args.agent not in algorithm.keys():
        assert 'agent is not supported: %s' % args.agent
    else:
        return algorithm[args.agent](
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            pred_step=args.pred_step,
            weight=args.weight,
            view=args.view
        )
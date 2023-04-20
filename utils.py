import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84, 
                 pre_image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size # for translation
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        #@ksl
        self.eoo = np.empty((capacity, 1), dtype=np.float32)


    def add(self, obs, action, reward, next_obs, done, eoo):
       
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        #@ksl
        np.copyto(self.eoo[self.idx], eoo)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_traj(self, aug_funcs, k):
        batch_size = 128
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        end_ranges = end_idxs - k

        traj_idxs = []

        n_slots = len(end_idxs)

        beg_idxs = []

        for _ in range(batch_size):
            slot_idx = np.random.choice(n_slots)
            beg = np.random.choice(range(beg_ranges[slot_idx], end_ranges[slot_idx]))
            traj_idxs.append([beg + i for i in range(k)])
            beg_idxs.append(beg)

        actions = np.array([self.actions[traj_idxs[i]] for i in range(batch_size)])
        #print("in sample, action=: ",actions.shape)
        #@wjc
        obses_q = self.obses[beg_idxs].copy()
        obses_k = self.obses[beg_idxs].copy()
        next_obses = np.array([
            self.next_obses[traj_idxs[i]] for i in range(batch_size)
        ]).copy()


        batch_,k_,stack_,h_,w_=next_obses.shape


        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    #===============================================
                    obses_q = func(obses_q)
                    obses_k = func(obses_k)
                    # next_obses = func(next_obses)
                    # assert False#test only translate
                    #===============================================
                    tmp_og_next_obses = func(next_obses[:,0,:])
                    td_next_obses=np.expand_dims(tmp_og_next_obses,1)
                    for td_idx in range(1, k):
                        tmp_og_next_obses = np.expand_dims(func(next_obses[:,td_idx,:]),1)
                        td_next_obses = np.concatenate( (td_next_obses,tmp_og_next_obses) ,1)   

                elif 'translate' in aug:
                    og_obses_q = center_crop_images(obses_q, self.pre_image_size)
                    og_obses_k = center_crop_images(obses_k, self.pre_image_size)
                    # og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses_q, rndm_idxs = func(og_obses_q, self.image_size, return_random_idxs=True)
                    obses_k = func(og_obses_k, self.image_size)
                    # next_obses = func(og_next_obses, self.image_size, **rndm_idxs)
                    tmp_og_next_obses = center_crop_images(next_obses[:,0,:], self.pre_image_size)
                    td_next_obses = func(tmp_og_next_obses, self.image_size, **rndm_idxs)
                    td_next_obses=np.expand_dims(td_next_obses,1)
         #           print("1st,",td_next_obses.shape)
                    for td_idx in range(1, k):
                        tmp_og_next_obses = center_crop_images(next_obses[:,td_idx,:], self.pre_image_size)
                        tmp_og_next_obses = np.expand_dims(func(tmp_og_next_obses,self.image_size, **rndm_idxs),1)
          #              print("tmp,",tmp_og_next_obses.shape)
                        td_next_obses = np.concatenate( (td_next_obses,tmp_og_next_obses) ,1)
        assert len(td_next_obses.shape) ==5
        obses_q = obses_q / 255.
        obses_k = obses_k / 255.
        next_obses = td_next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                assert False
                obses_q = func(obses_q)
                obses_k = func(obses_k)
                next_obses = func(next_obses)
        # next_obses=next_obses.reshape([batch_,k_,stack_,h_,w_])

        #@wjc
        rewards = torch.as_tensor(self.rewards[beg_idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[beg_idxs], device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        obses_q = torch.tensor(obses_q, device=self.device).float()
        obses_k = torch.tensor(obses_k, device=self.device).float()
        next_obses = torch.tensor(next_obses, device=self.device).float()
        # rewards = torch.tensor(rewards, device=self.device)

        #===================================================================================
        #@ksl
        #return obses, actions, obses_next, rewards
        #===================================================================================

        #obses_copy: [batch_size]: o_t
        #next_k_obses_copy: [batch_size, k], o_{t+1},...o_{t+k}
        #obses: [batch_size]: aug o_t
        #next_obses: [batch_size, k],aug {o_{t+1},...o_{t+k}}
        #return obses_copy, actions, rewards, next_k_obses_copy, not_dones, obses, next_obses
        
        # return obses_q, obses_k, actions, next_obses#orginal sample_traj
        return obses_q, obses_k, actions, rewards, next_obses, not_dones

    def sample_traj_multi_view(self, aug_funcs, k, num_views):
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        end_ranges = end_idxs - k

        traj_idxs = []

        n_slots = len(end_idxs)

        beg_idxs = []

        for _ in range(self.batch_size):
            slot_idx = np.random.choice(n_slots)
            beg = np.random.choice(range(beg_ranges[slot_idx], end_ranges[slot_idx]))
            traj_idxs.append([beg + i for i in range(k)])
            beg_idxs.append(beg)

        actions = np.array([self.actions[traj_idxs[i]] for i in range(self.batch_size)])
        #print("in sample, action=: ",actions.shape)
        #@wjc
        obses_q = self.obses[beg_idxs].copy()
        obses_k = self.obses[beg_idxs].copy()
        next_obses = np.array([
            self.next_obses[traj_idxs[i]] for i in range(self.batch_size)
        ])


        batch_,k_,stack_,h_,w_=next_obses.shape


        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    #===============================================
                    obses_q = func(obses_q)
                    obses_k = func(obses_k)
                    for view_idx in range(num_views):

                        tmp_view = func(obses_q)
                        tmp_view = np.expand_dims(func(tmp_view),0)
                        if view_idx ==0:
                            obses_q_views = tmp_view  
                        else:
                            obses_q_views=np.concatenate((obses_q_views, tmp_view),0)# [views, obses_shape]

                    tmp_og_next_obses = func(next_obses[:,0,:])
                    td_next_obses=np.expand_dims(tmp_og_next_obses,1)
                    for td_idx in range(1, k):
                        tmp_og_next_obses = np.expand_dims(func(next_obses[:,td_idx,:]),1)
                        td_next_obses = np.concatenate( (td_next_obses,tmp_og_next_obses) ,1)   

                elif 'translate' in aug:
                    # og_obses_q = center_crop_images(obses_q, self.pre_image_size)
                    og_obses_k = center_crop_images(obses_k, self.pre_image_size)
                    obses_k, rndm_idxs = func(og_obses_k, self.image_size, return_random_idxs=True)
                    # obses_q = func(og_obses_q, self.image_size)

                    for view_idx in range(num_views):
                        tmp_view = center_crop_images(obses_q, self.pre_image_size)
                        tmp_view = np.expand_dims(func(tmp_view, self.image_size),0)
                        if view_idx ==0:
                            obses_q_views = tmp_view  
                        else:
                            obses_q_views=np.concatenate((obses_q_views, tmp_view),0)# [views, obses_shape]

                    tmp_og_next_obses = center_crop_images(next_obses[:,0,:], self.pre_image_size)
                    td_next_obses = func(tmp_og_next_obses, self.image_size, **rndm_idxs)
                    td_next_obses=np.expand_dims(td_next_obses,1)

                    for td_idx in range(1, k):
                        tmp_og_next_obses = center_crop_images(next_obses[:,td_idx,:], self.pre_image_size)
                        tmp_og_next_obses = np.expand_dims(func(tmp_og_next_obses,self.image_size, **rndm_idxs),1)

                        td_next_obses = np.concatenate( (td_next_obses,tmp_og_next_obses) ,1)
        assert len(td_next_obses.shape) ==5
        obses_q_views = obses_q_views / 255.
        obses_k = obses_k / 255.
        next_obses = td_next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                assert False
                obses_q = func(obses_q)
                obses_k = func(obses_k)
                next_obses = func(next_obses)
        # next_obses=next_obses.reshape([batch_,k_,stack_,h_,w_])

        #@wjc
        rewards = torch.as_tensor(self.rewards[beg_idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[beg_idxs], device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        obses_q_views = torch.tensor(obses_q_views, device=self.device).float()
        obses_k = torch.tensor(obses_k, device=self.device).float()
        next_obses = torch.tensor(next_obses, device=self.device).float()
        # rewards = torch.tensor(rewards, device=self.device)

        #===================================================================================
        #@ksl
        #return obses, actions, obses_next, rewards
        #===================================================================================

        #obses_copy: [batch_size]: o_t
        #next_k_obses_copy: [batch_size, k], o_{t+1},...o_{t+k}
        #obses: [batch_size]: aug o_t
        #next_obses: [batch_size, k],aug {o_{t+1},...o_{t+k}}
        #return obses_copy, actions, rewards, next_k_obses_copy, not_dones, obses, next_obses
        
        # return obses_q, obses_k, actions, next_obses#orginal sample_traj
        #print("in test:obses_q={},obses_k={},a={},r={},next_o={} ".format(obses_q_views.shape, obses_k.shape, actions.shape,rewards.shape,next_obses.shape))
        #in test:obses_q=torch.Size([2, 512, 9, 92, 92]),obses_k=torch.Size([512, 9, 92, 92]),a=torch.Size([512, 1, 2]),r=torch.Size([512, 1]),next_o=torch.Size([512, 1, 9, 92, 92])
        return obses_q_views, obses_k, actions, rewards, next_obses, not_dones

    def sample_double_batch(self, aug_funcs):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_copy=obses.copy()
        next_obses_copy=next_obses.copy()

        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug: 
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)                     

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)
        #here comes non-augmented part

        # here we need center augmentation
        obses_copy = center_crop_images(obses_copy, 100)
        obses_copy = center_translates(obses_copy, 108)
        next_obses_copy = center_crop_images(next_obses_copy, 100)
        next_obses_copy = center_translates(next_obses_copy, 108)

        obses_copy = torch.as_tensor(obses_copy, device=self.device).float()
        next_obses_copy = torch.as_tensor(next_obses_copy, device=self.device).float()

        obses_copy = obses_copy / 255.
        next_obses_copy = next_obses_copy / 255.        
    
        return  obses_copy, actions, rewards, next_obses_copy, not_dones, obses, next_obses


    def sample_double_crop(self, aug_funcs):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=128   # default 128
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_copy=obses.copy()
        next_obses_copy=next_obses.copy()

        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug: 
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)                     

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)
        #here comes non-augmented part

        # here we need center augmentation
        obses_copy = center_crop_images(obses_copy, 84)
        next_obses_copy = center_crop_images(next_obses_copy, 84)

        obses_copy = torch.as_tensor(obses_copy, device=self.device).float()
        next_obses_copy = torch.as_tensor(next_obses_copy, device=self.device).float()

        obses_copy = obses_copy / 255.
        next_obses_copy = next_obses_copy / 255.        
    
        return  obses_copy, actions, rewards, next_obses_copy, not_dones, obses, next_obses

#@ht
    def sample_double_batch_aug(self, aug_funcs):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_copy=obses.copy()
        next_obses_copy=next_obses.copy()

        # First one
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug: 
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)                     

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        # Second one: original 
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses_copy = func(obses_copy)
                    next_obses_copy = func(next_obses_copy)
                elif 'translate' in aug: 
                    og_obses_copy = center_crop_images(obses_copy, self.pre_image_size)
                    og_next_obses_copy = center_crop_images(next_obses_copy, self.pre_image_size)
                    obses_copy, rndm_idxs_copy = func(og_obses_copy, self.image_size, return_random_idxs=True)
                    next_obses_copy = func(og_next_obses_copy, self.image_size, **rndm_idxs_copy)  

        obses_copy = torch.as_tensor(obses_copy, device=self.device).float()
        next_obses_copy = torch.as_tensor(next_obses_copy, device=self.device).float()
        obses_copy = obses_copy / 255.
        next_obses_copy = next_obses_copy / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses_copy = func(obses_copy)
                next_obses_copy = func(next_obses_copy)      
    
        return  obses_copy, actions, rewards, next_obses_copy, not_dones, obses, next_obses

#@wjc test with original rad
    def sample_rad(self,aug_funcs):
        
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler


        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug: 
                    og_obses = center_crop_images(obses, self.pre_image_size)
                    og_next_obses = center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)                     

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


def center_translate(image, size):
    c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, h1:h1 + h, w1:w1 + w] = image
    return outs

def center_translates(image, size):
    n, c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, :, h1:h1 + h, w1:w1 + w] = image
    return outs

# compute output dim of cnn
def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape

# initiate weight of nn
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def cos_sim(a, b):
    a = F.normalize(a, dim=-1, p=2)
    b = F.normalize(b, dim=-1, p=2)

    return -(a * b).sum(-1).mean()

def cos_sim_siam(a, b):
    b = b.detach()

    a = F.normalize(a, dim=-1, p=2)
    b = F.normalize(b, dim=-1, p=2)

    return -(a * b).sum(-1).mean()

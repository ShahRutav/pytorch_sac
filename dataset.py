import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, os

class Dataset(object):
    __doc__ = 'Buffer to store environment transitions.'

    def __init__(self, obs_shape, state_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.states = np.empty((capacity, *state_shape), dtype=(np.float32))
        self.actions_expert = np.empty((capacity, *action_shape), dtype=(np.float32))
        self.rewards = np.empty((capacity, 1), dtype=(np.float32))
        self.dones = np.empty((capacity, 1), dtype=bool)
        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.idx

    def add(self, obs, state, action_expert, reward, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions_expert[self.idx], action_expert)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, (self.capacity if self.full else self.idx), size=batch_size)
        obses = torch.as_tensor((self.obses[idxs]), device=(self.device)).float()
        states = torch.as_tensor((self.states[idxs]), device=(self.device)).float()
        actions_expert = torch.as_tensor((self.actions_expert[idxs]), device=(self.device))
        reward = torch.as_tensor((self.rewards[idxs]), device=(self.device))
        done = torch.as_tensor((self.dones[idxs]), device=(self.device))
        return (obses, states, actions_expert, reward, done)

    def save(self, save_dir, prefix=''):
        if self.idx == self.last_save:
            if not self.full:
                print('Returning without saving.')
                return
        start_ind = 0
        end_ind = self.capacity if self.full else self.idx
        path = os.path.join(save_dir, prefix + '%d_%d.pt' % (start_ind, end_ind))
        payload = [
        self.obses[start_ind:end_ind],
        self.states[start_ind:end_ind],
        self.actions_expert[start_ind:end_ind],
        self.rewards[start_ind:end_ind],
        self.dones[start_ind:end_ind]]
        self.last_save = self.idx
        torch.save(payload, path)

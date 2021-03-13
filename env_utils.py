import dmc2gym, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, copy, math, os, sys, time, pickle as pkl, hydra, dmc2gym, utils
from PIL import Image
from utils import ObservationWrapper
import time

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)

    return np.concatenate(obs_pieces, axis=0)


def get_env_state_dim(conf):
    if conf.env == 'cheetah_run':
        return 17
    elif conf.env == 'reacher_easy':
        return 6
    elif conf.env == 'ball_in_cup_catch':
        return 8
    elif conf.env == 'cartpole_swingup':
        return 5
    elif conf.env == 'walker_walk':
        return 24
    else :
        print('Please enter a valid env. Current : ', conf.env)
        exit()


def get_env_state(env, cfg):
    if cfg.env == 'cheetah_run':
        state = env.physics.get_state().copy()
        state = state[1:]
    elif cfg.env == 'cartpole_swingup':
        state = np.concatenate((env.physics.bounded_position().copy(), env.physics.velocity().copy()), axis=0)
        state = state
    elif cfg.env == 'reacher_easy':
        state = np.concatenate((env.physics.position(), env.physics.finger_to_target().copy(), env.physics.velocity()), axis=0)
        state = state
    elif cfg.env == 'ball_in_cup_catch':
        state = np.concatenate((env.physics.position(), env.physics.velocity()), axis=0)
        state = state
    elif cfg.env == 'walker_walk':
        state = np.concatenate((env.physics.orientations(), env.physics.torso_height(), env.physics.velocity()), axis=0)
        state = state
    else:
        print('Please enter a valid env. Current : ', cfg.env)
        exit()
    return state

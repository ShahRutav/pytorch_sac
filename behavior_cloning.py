import dmc2gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import hydra
import dmc2gym
import utils
from PIL import Image
from utils import ObservationWrapper
import time
from env_utils import * 
from dataset import *
from pathlib import Path
home = str(Path.home())

def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=cfg.from_pixels,
                       height=cfg.height,
                       width=cfg.width,
                       camera_id=cfg.camera_id,
                       frame_skip=cfg.frame_skip,
                       channels_first=False,
                       )

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    if cfg.from_pixels == True:
    	env = ObservationWrapper(env, "resnet34")	

    return env

def update_actor_bc(agent, obses, actions_expert, loss):
	dist = agent.actor(obses)
	actions = dist.rsample()

	actor_loss = loss(actions, actions_expert) 
	#print("Actor loss : ", actor_loss)

	agent.actor_optimizer.zero_grad()
	actor_loss.backward()
	agent.actor_optimizer.step()
	return actor_loss

def evaluate(env, agent, cfg, attach_state):
        average_episode_reward = 0
        for episode in range(cfg.num_eval_episodes):
            obs = env.reset()
            agent.reset()
            #self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(agent):
                    if attach_state :
                    	obs = np.concatenate((obs, get_env_state(env, cfg) ), axis=0)
                    action = agent.act(obs, sample=False)
                obs, reward, done, _ = env.step(action)
                #video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            #video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= cfg.num_eval_episodes
        return average_episode_reward



@hydra.main(config_path='./expert/cartpole_swingup_state/config.yaml', strict=True)
def main(cfg):
	expert_path="/home/bt1/18CS10050/pytorch_sac/expert/cartpole_swingup_state"
	actor_path=expert_path + "/actor.pt"
	attach_state = False
	
	cfg.from_pixels=False
	env = utils.make_env(cfg) # Make env based on cfg.
	#if cfg.frame_stack = True:
	#	self.env = utils.FrameStack(self.env, k=3)
	cfg.agent.params.obs_dim = env.observation_space.shape[0]
	if attach_state :
		cfg.agent.params.obs_dim += get_env_state_dim(cfg)
	cfg.agent.params.action_dim = env.action_space.shape[0]
	cfg.agent.params.action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]
	agent = hydra.utils.instantiate(cfg.agent)
	print("Observation Dimension : ", cfg.agent.params.obs_dim)
	

	from omegaconf import OmegaConf
	conf = OmegaConf.load(expert_path + '/config.yaml')
	assert conf.env == cfg.env
	conf.agent.params.action_dim = env.action_space.shape[0]
	conf.agent.params.action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]
	conf.agent.params.obs_dim = get_env_state_dim(conf)

	agent_expert = hydra.utils.instantiate(conf.agent)
	agent_expert.actor.load_state_dict(
            torch.load(actor_path)
        )
	#video_recorder = VideoRecorder(None)

	#print("DATASET CAPACITY : 1000000")
	start_ind = 0
	end_ind = 1000000
	load_start_time = time.time()
	data = torch.load(home + "/pytorch_sac/Data/" + cfg.env + str(start_ind) +  "_" + str(end_ind) +".pt")
	print(data[0].shape)
	print(data[1].shape)
	print(data[2].shape)
	print(data[3].shape)
	print("Time to load the data  : ", time.time()-load_start_time)
	dataset = Dataset((data[0].shape[1],), (data[1].shape[1],),
				env.action_space.shape, 
				int(end_ind - start_ind),
				torch.device("cuda"))
	
	buffer_insert_start_time = time.time()
	for i in range(data[0].shape[0]):
		obs = data[0][i]
		state = data[1][i]	
		action_expert = data[2][i]
		reward = data[3][i]
		done = data[4][i]
		
		dataset.add(obs, state, action_expert, reward, done)
	print("Time taken to add into buffer : ", time.time() - buffer_insert_start_time)

	bc_update_steps = 100
	bc_batch_size = 512
	bc_eval_freq = 10
	loss_fn = nn.MSELoss() 
	bc_steps = 0
	start_bc_time = time.time()
	for i in range(bc_update_steps):
		obses, state, actions_expert, _, _ = dataset.sample(bc_batch_size)
		if attach_state :
			obses = np.concatenate((obses, state), axis = 1)
			print(obses.shape)
			exit()
		if not cfg.from_pixels :
			obses = state 
		loss = update_actor_bc(agent, obses, actions_expert, loss_fn)
		bc_steps += 1
		if bc_steps % bc_eval_freq == 0:
			average_ep_reward = evaluate(env, agent, cfg, attach_state)
			print("Step : ", bc_steps, " Loss : ", loss.data, " Time taken : ", time.time() - start_bc_time)
			print("Average Episode Reward : ", average_ep_reward)
	print("Total time taken : ", time.time() - load_start_time)
			
		
	

if __name__ == '__main__':
    main()

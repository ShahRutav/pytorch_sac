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


def update_actor_bc(agent, obses, actions_expert, loss):
	dist = agent.actor(obses)
	actions = dist.rsample()

	actor_loss = loss(actions, actions_expert) 
	#print("Actor loss : ", actor_loss)

	agent.actor_optimizer.zero_grad()
	actor_loss.backward()
	agent.actor_optimizer.step()

def evaluate(env, agent, cfg):
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


@hydra.main(config_path='./expert/cartpole_swingup_state/config.yaml', strict=False)
def main(cfg):
	from omegaconf import OmegaConf
	attach_state = cfg.attach_state
	from_pixels = cfg.from_pixels
	encoder_type = cfg.encoder_type
	if cfg.user_config:
		print("+++++++++++++++++ Using user specified config")
		cfg = OmegaConf.load(cfg.user_config)
		cfg.attach_state = attach_state
		cfg.from_pixels = from_pixels
		cfg.encoder_type = encoder_type
		
	print("+++++++++++++++++ Configuration : \n", cfg)
	expert_path = home + "/pytorch_sac/expert/" + cfg.env  +  "_state"
	print("+++++++++++++++++ Expert Path : ", expert_path)
	actor_path = expert_path + "/actor.pt"
	
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

	data = Dataset((cfg.agent.params.obs_dim,), (conf.agent.params.obs_dim,),
				env.action_space.shape, 
				1000000,
				torch.device("cuda"))
	collect_steps = 1000000
	print("DATASET CAPACITY : 1000000, Collecting Steps : ", collect_steps)
	loss = nn.MSELoss() 
	#collect_ep = 4000
	
	step = 0
	ep = 0
	start_time = time.time()

	while(step < collect_steps):	
		obs = env.reset()
		state = get_env_state(env, cfg)

		action_expert = None
		done = False
		episode_step = 0
		episode_reward = 0
		ep_start_time = time.time()
		while not done:
			with utils.eval_mode(agent_expert):
				action_expert = agent_expert.act(state, sample=False)
			next_obs, reward, done, extra = env.step(action_expert)
			next_state = get_env_state(env, cfg)	
			#print("XXXXXX\n", obs.shape,"\n", state.shape)
			data.add(obs, state, action_expert, reward, done)			

			step += 1
			episode_step += 1
			episode_reward += reward
			done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
			
			obs = next_obs
			state = next_state
		ep += 1
		if ep%100 == 0:
			print("Episode : ", ep, " Episode Reward : ", episode_reward, " Time taken by one episode : ", time.time() - ep_start_time)
	
	print("Total Time taken : ", time.time() - start_time)
	data.save(home + "/pytorch_sac/Data", prefix = cfg.env + "_" + cfg.encoder_type)

if __name__ == '__main__':
    main()

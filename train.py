#!/usr/bin/env python3
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
from omegaconf import OmegaConf

from vrl.algos.pytorch_sac.video import VideoRecorder
from vrl.algos.pytorch_sac.logger import Logger
from vrl.algos.pytorch_sac.replay_buffer import ReplayBuffer
from vrl.algos.pytorch_sac import utils
from vrl.utils.utils import make_dir, make_env, make_encoder
import vrl.envs

import dmc2gym
import hydra


class SACWorkspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.log_dir = os.path.join(self.work_dir, 'logs')
        make_dir(self.log_dir)
        self.logger = Logger(self.log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        encoder = make_encoder(**cfg.encoder_args) 
        transform = encoder.get_transform
        latent_dim = encoder.latent_dim

        self.env = make_env(**cfg.env_args, encoder=encoder, transform=transform, latent_dim=latent_dim)
        self.eval_episode_freq  = int(cfg.eval_frequency/self.env.horizon)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        if cfg.save_model or cfg.save_actor: 
            self.model_dir = self.work_dir + "/models"
            make_dir(self.model_dir)
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/eval_score', average_episode_reward,
                        self.step)
        self.logger.log('eval/norm_score', self.env.get_normalized_score(average_episode_reward),
                        self.step)
        self.logger.log('eval/total_num_samples', self.step,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and episode % self.eval_episode_freq == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    if self.cfg.save_model:
                        self.agent.save_sac(self.model_dir + f"/sac_ep{episode:06d}.pickle")
                    if self.cfg.save_actor : 
                        self.agent.save_actor(self.model_dir + f"/actor_ep{episode:06d}.pickle")

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config', config_name="train.yaml")
def main(cfg):
    workspace = SACWorkspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

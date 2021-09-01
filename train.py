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
from vrl.algos.pytorch_sac import tensor_utils
from omegaconf import OmegaConf

from vrl.algos.pytorch_sac.video import VideoRecorder
from vrl.algos.pytorch_sac.logger import Logger
from vrl.algos.pytorch_sac.replay_buffer import ReplayBuffer
from vrl.algos.pytorch_sac import utils
from vrl.utils.utils import make_dir, make_env, make_encoder
from vrl.algos.pytorch_sac.agent.sac import SACAgent
import vrl.envs

#import dmc2gym
import hydra


class SACWorkspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.log_dir = os.path.join(self.work_dir, 'logs')
        make_dir(self.log_dir)
        self.logger = Logger(self.log_dir,
                             save_tb=cfg.algos.log_save_tb,
                             log_frequency=cfg.algos.log_frequency,
                             agent=cfg.algos.algo)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.algos.device)
        encoder = make_encoder(**cfg.encoder_args)
        transform = encoder.get_transform
        latent_dim = encoder.latent_dim

        self.env = make_env(**cfg.env_args, encoder=encoder, transform=transform, latent_dim=latent_dim)
        self.eval_episode_freq  = int(cfg.algos.eval_frequency/self.env.horizon)

        cfg.algos.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.algos.agent.action_dim = self.env.action_space.shape[0]
        cfg.algos.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = SACAgent(**cfg.algos.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.algos.replay_buffer_capacity),
                                          self.device)

        if cfg.algos.save_model or cfg.algos.save_actor:
            self.model_dir = self.work_dir + "/models"
            make_dir(self.model_dir)
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.algos.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        paths = []
        for episode in range(self.cfg.algos.num_eval_episodes):
            observations=[]
            actions=[]
            rewards=[]
            agent_infos = []
            env_infos = []

            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                observations.append(obs)
                obs, reward, done, env_info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

                actions.append(action)
                rewards.append(reward)
                env_infos.append(env_info)

            path = dict(
                    observations=np.array(observations),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                    terminated=done
                )
            paths.append(path)

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.algos.num_eval_episodes

        try :
            success_percentage = self.env.env.evaluate_success(paths)
            self.logger.log('eval/eval_success', success_percentage,
                            self.step)
        except Exception as e:
            print("ERROR: ", str(e))
            pass
        self.logger.log('eval/eval_score', average_episode_reward,
                        self.step)
        self.logger.log('eval/norm_score', self.env.get_normalized_score(average_episode_reward),
                        self.step)
        self.logger.log('eval/total_num_samples', self.step,
                        self.step)
        self.logger.dump(self.step)
        return average_episode_reward

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        best_eval_score = -1e6
        while self.step < self.cfg.algos.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.algos.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and episode % self.eval_episode_freq == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    curr_eval_score = self.evaluate()
                    if self.cfg.algos.save_model:
                        self.agent.save_sac(self.model_dir + f"/sac_ep{episode:06d}.pickle")
                        if curr_eval_score > best_eval_score:
                            print("+++++ Updating best model")
                            self.agent.save_sac(self.model_dir + f"/sac_best.pickle")
                    if self.cfg.algos.save_actor :
                        self.agent.save_actor(self.model_dir + f"/actor_ep{episode:06d}.pickle")
                        if curr_eval_score > best_eval_score:
                            print("+++++ Updating best actor")
                            self.agent.save_sac(self.model_dir + f"/actor_best.pickle")
                    best_eval_score = max(curr_eval_score, best_eval_score)

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
            if self.step < self.cfg.algos.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.algos.num_seed_steps:
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

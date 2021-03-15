#!/bin/sh
#$ -cwd
#PBS -N testpy
#PBS -l walltime=00:05:00
#PBS -q workq
#PBS -V

export MUJOCO_GL="osmesa"
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt

export PATH="$PATH:$HOME/rpm/usr/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/rpm/usr/lib:$HOME/rpm/usr/lib64"
export LDFLAGS="-L$HOME/rpm/usr/lib -L$HOME/rpm/usr/lib64"
export CPATH="$CPATH:$HOME/rpm/usr/include"

export PATH="/home/bt1/18CS10050/anaconda3/bin:$PATH"
export PATH="/home/bt1/18CS10050/anaconda3/envs/pytorch_sac/bin:$PATH"

#CUDA_VISIBLE_DEVICES=0,1 python record_dataset.py env=reacher_easy frame_skip=4 > recording_cartpole.log  
#CUDA_VISIBLE_DEVICES=0,1 python behavior_cloning.py user_config=$HOME/pytorch_sac/expert/reacher_easy_state/config.yaml env=reacher_easy frame_skip=4 attach_state=False from_pixels=False > reacher_easy_state_bc_bs1024.log  
#CUDA_VISIBLE_DEVICES=0,1 python behavior_cloning.py user_config=$HOME/pytorch_sac/expert/reacher_easy_state/config.yaml env=reacher_easy frame_skip=4 attach_state=False from_pixels=True > reacher_easy_obs_bc_bs1024.log
#CUDA_VISIBLE_DEVICES=0,1 python behavior_cloning.py user_config=$HOME/pytorch_sac/expert/reacher_easy_state/config.yaml env=reacher_easy frame_skip=4 attach_state=True from_pixels=True > reacher_easy_obs_state_bc_bs1024.log 

CUDA_VISIBLE_DEVICES=0,1 python record_vae_dataset.py user_config=$HOME/pytorch_sac/expert/reacher_easy_state/config.yaml env=reacher_easy frame_skip=4 from_pixels=True height=64 width=64

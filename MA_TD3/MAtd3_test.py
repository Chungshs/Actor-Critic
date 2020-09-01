# 필요 Library import
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from MAtd3 import TD3Agent
from MA_utils import mini_batch_train
import gym
import json
from argparse import ArgumentParser
from common.params_manager import ParamsManager

#Argument parser 
args = ArgumentParser("td3_agent")
args.add_argument("--env",
                  help="Name of the environment",
                  default=None,
                  type=str,
                  metavar="ENV"
                  )

args = args.parse_args()


# Hyperparameters and training setups
params_manager = ParamsManager('./config.json')

if args.env is not None:
    params_manager.get_env_params()['name']=args.env
    params_manager.export_json('./config.json',params_manager.params)

# "HalfCheetah-v2", "Hopper-v2", "Ant-v2"
env_name = params_manager.get_env_params()['name']

# 초기 Hyper Parmeter 설정
env = gym.make(env_name)
batch_size = params_manager.get_agent_params()['BATCH_SIZE']
gamma = params_manager.get_agent_params()['DISCOUNT_FACTOR']
tau = params_manager.get_agent_params()['SMOOTH_UPDATE_RATE']
noise_std = params_manager.get_agent_params()['NOISE_STD']
noise_bound = params_manager.get_agent_params()['NOISE_BOUND']
delay_step = params_manager.get_agent_params()['DELAY_STEP']
buffer_maxlen = int(params_manager.get_agent_params()['MEMORY_SIZE'])
critic_lr = params_manager.get_agent_params()['Q_LEARN_RATE']
actor_lr = params_manager.get_agent_params()['PI_LEARN_RATE']
max_episodes = params_manager.get_agent_params()['MAX_EPISODES']
max_steps = params_manager.get_agent_params()['MAX_STEPS']

# Class 생성후 Train
agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, noise_std, noise_bound, critic_lr, actor_lr)
episode_rewards = mini_batch_train(env, agent, env_name, max_episodes, max_steps, batch_size)

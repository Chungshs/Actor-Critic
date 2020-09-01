import numpy as np
import math
import gym
import torch
from datetime import date
import os
import shutil
import time
import random
file_name = "test.csv"
last_file_name = '/previous_action_state'
Algorithm_name = "./SACv2"
global prob_coeff
prob_coeff = 0.5

def create_file(env_name: str)->str:
    # 저장 File 생성
    ENV_NAME = env_name
    today = date.today()
    folder_name = ENV_NAME + '/' + str(today) + last_file_name
    WORKSPACE = Algorithm_name + '/' + folder_name + '/'
    fname = WORKSPACE + file_name
    if not os.path.exists(WORKSPACE):
        if not os.path.exists(Algorithm_name):
            os.mkdir(Algorithm_name)
        if not os.path.exists(Algorithm_name+ '/' +  ENV_NAME):
            os.mkdir(Algorithm_name + '/' + ENV_NAME)
        if not os.path.exists(Algorithm_name+ '/' + ENV_NAME + '/' + str(today)):
            os.mkdir(Algorithm_name + '/' + ENV_NAME + '/' + str(today))
        os.mkdir(WORKSPACE)
    else:
        shutil.rmtree(Algorithm_name+ '/' +folder_name)
        os.mkdir(WORKSPACE)    
    file = open(fname,"w")
    file.write("==========This is the beginning==========\n")
    file.write("File format is :\n")
    file.write("Timesteps, Steps, Episode, Reward\n")
    file.close()
    return WORKSPACE

# Appending data to the created file over times
def write_to_file(workspace: str, timestamp: float, env_steps:int, episode: int, score: float):
    # Open file and append data
    WORKSPACE = workspace
    fname = WORKSPACE + file_name
    file = open(fname,"a+")
    file.write(str(timestamp) + "," + str(env_steps) + "," + str(episode) + "," + str(score)+ "\n")
    file.close()


def mini_batch_train(env: object, agent: object, env_name: str, max_episodes: int, max_steps: int, batch_size: int)-> float:
    global prob_coeff
    # Initialize variable
    episode_rewards = []   
    total_steps = 0
    WORKSPACE = create_file(env_name)
    start_time = time.time()
    done_array = []
    action_dim = env.action_space.shape[0]   
    # Do the episode
    for episode in range(max_episodes):
        # Initialized variable
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        reward = 0
        # 죽기전까지 계속 Step 진행
        while not done :
            action_=0
            # 초기 total_step 동안 Action Random sampling
            action, _, _ = agent.get_action(state, True)

            # Apply action to environment
            next_state, reward, done, _ = env.step(action)

            # push to the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # To make next step, update the variable         
            episode_reward += reward
            step += 1
            total_steps +=1            

            # Update when buffer is over batch_size
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)               

            # Evaluate whenever total_steps is 1000의 배수
            if total_steps % 1000 == 0:
                evaluate(env_name, agent, WORKSPACE, episode, total_steps, start_time)
                if total_steps <= 1e6:
                    agent.Actor_save(WORKSPACE)

            # 죽거나 Final step 도달시 게임종료 및 episode 결과                
            if done or step == 999 :          
                episode_rewards.append(episode_reward)
                print("episode :", episode, "total_steps " + str(total_steps) + ": " + str(episode_reward))
                break
            state = next_state

    return episode_rewards


# This one is for evaluate don't need to change and is the same with the train code only without updating
def evaluate(env_name: str, agent: object, WORKSPACE: str, episode: int, total_steps: int, start_time: float):
    global prob_coeff
    eval_env = gym.make(env_name)
    eval_episode_rewards = []
    for eval_episode in range(5):
        eval_state = eval_env.reset()
        eval_episode_reward = 0
        eval_step = 0
        eval_done = False
        eval_action_ =0
        action_dim = eval_env.action_space.shape[0]        

        while not eval_done :
            eval_action, _, _ = agent.get_action(eval_state, True)
            eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)                         
            eval_episode_reward += eval_reward
            eval_step += 1

            if eval_done:

                eval_episode_rewards.append(eval_episode_reward)
                if eval_episode == 4:
                    print("episode :", episode,"total_steps " + str(total_steps) + "average reward:" + str(np.mean(eval_episode_rewards)) + \
                     "min_reward :" + str(np.min(eval_episode_rewards)) + "max_reward :" + str(np.max(eval_episode_rewards)))
                    print(eval_episode_rewards)
                    timestamp = time.time()-start_time
                    write_to_file(WORKSPACE, timestamp, total_steps, episode, np.mean(eval_episode_rewards))
                    break

            eval_state = eval_next_state


import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import Critic, Actor
from common.replay_buffers import BasicBuffer
from typing import Tuple

class TD3Agent:
    """
    Each joint will be the agent. Thus we will have one action (Agent) value on each joint.
    """
    def __init__(self, env: object, gamma: float, tau: float, buffer_maxlen: int,
     delay_step: int, noise_std: float, noise_bound: float, critic_lr: float, actor_lr:float):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Environment로 부터 State(observation), Action space 설정
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # hyperparameters    
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.update_step = 0 
        self.delay_step = delay_step
        self.buffer_maxlen = buffer_maxlen
        self.critic1 = []
        self.critic2 = []
        self.critic_target1 = []
        self.critic_target2 = []
        self.actor = []
        self.actor_target = []
        self.critic_optimizer1 = []
        self.critic_optimizer2 = []
        self.actor_optimizer = []

        # initialize actor and critic networks depends on the action_dims(because it's MA)
        for _ in range(self.action_dim):
            self.critic1.append(Critic(self.obs_dim, self.action_dim).to(self.device))
            self.critic2.append(Critic(self.obs_dim, self.action_dim).to(self.device))
            self.critic_target1.append(Critic(self.obs_dim, self.action_dim).to(self.device))
            self.critic_target2.append(Critic(self.obs_dim, self.action_dim).to(self.device))

        for _ in range(self.action_dim):
            self.actor.append(Actor(self.obs_dim, self.action_dim).to(self.device))
            self.actor_target.append(Actor(self.obs_dim, self.action_dim).to(self.device))

        # Copy critic target parameters
        for i in range(self.action_dim):
            for target_param, param in zip(self.critic_target1[i].parameters(), self.critic1[i].parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.critic_target2[i].parameters(), self.critic2[i].parameters()):
                target_param.data.copy_(param.data)                

        # initialize optimizers        
        for i in range(self.action_dim):        
            self.critic_optimizer1.append(optim.Adam(self.critic1[i].parameters(), lr=critic_lr))
            self.critic_optimizer2.append(optim.Adam(self.critic2[i].parameters(), lr=critic_lr))
            self.actor_optimizer.append(optim.Adam(self.actor[i].parameters(), lr=actor_lr))

        self.replay_buffer = BasicBuffer(10000)        
        self.replay_buffer_base = BasicBuffer(self.buffer_maxlen)

    def get_action(self, obs: np.ndarray)-> Tuple[list, list]:
        # Action 을 얻기 위해 state를 받는다.
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_list = []
        action_list_ = []
        # 그후, 각 node별로 NN으로 부터 Inference를 하고 이를 List에 append한다. 이때 acion은 학습용으로 with Noise, action_은 Test용으로 without Noise.
        for i in range(self.action_dim):
            action_list.append((self.actor[i].forward(state[0,i])).cpu().detach() + (self.generate_action_space_noise(0.4)).cpu().detach()) 
            action_list_.append((self.actor[i].forward(state[0,i])).cpu().detach())
        
        action = action_list 
        action_ = action_list_        

        return action, action_
    

    def update(self, batch_size: int, step_env: int):

        # Replay Buffer로 부터 batch Sample
        state_batch, action_batch, reward_batch, next_state_batch, dones = self.replay_buffer.sample(batch_size)
        
        # Batch_sample Data variable 초기화
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)

        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        # 각 Node별 Update진행
        for i in range(self.action_dim):
            # null Action 확인 후 관련 Node 모두 0으로 초기화
            action_null = np.where(action_batch[:,i] == 0 )   
            state_batch_null = state_batch[:,i]
            state_batch_null[action_null] = 0
            state_batch_null = torch.FloatTensor(state_batch_null).to(self.device)

            action_batch_null = action_batch[:,i]
            action_batch_null[action_null] = 0            
            action_batch_null = torch.FloatTensor(action_batch_null).to(self.device)            

            next_state_batch_null = next_state_batch[:,i]
            next_state_batch_null[action_null] = 0   
            next_state_batch_null = torch.FloatTensor(next_state_batch_null).to(self.device)                        

            reward_batch_null = reward_batch
            reward_batch_null[action_null] = 0   
            reward_batch_null = torch.FloatTensor(reward_batch_null).to(self.device)   

            # Add Noise for next action
            action_space_noise = self.generate_action_space_noise(0.2)

            next_actions = self.actor[i].forward(next_state_batch_null) + action_space_noise

            # To make expected Q-value(s_t+1)
            next_Q1 = self.critic_target1[i].forward(next_state_batch_null, next_actions)
            next_Q2 = self.critic_target2[i].forward(next_state_batch_null, next_actions)
            expected_Q = reward_batch_null + (1 - dones) * self.gamma * torch.min(next_Q1, next_Q2)
            expected_Q = expected_Q.cpu().detach().numpy()
            expected_Q[action_null] = 0
            expected_Q = torch.FloatTensor(expected_Q).to(self.device)
   
            # To remove the effect of null node, Masking array 생성
            masking_torch = np.ones([100,1])
            masking_torch[action_null] = 0
            masking_torch = torch.FloatTensor(masking_torch).to(self.device)


            # 학습 위해 Critic value inference
            curr_Q1 = self.critic1[i].forward(state_batch_null, action_batch_null.reshape(-1,1))                        
            curr_Q1 *= masking_torch.detach()
            curr_Q2 = self.critic2[i].forward(state_batch_null, action_batch_null.reshape(-1,1))
            curr_Q2 *= masking_torch.detach()
            # Critic value inference and Critic value(S_+1) (Q(s_t) -(r+Q(s_t+1) )^2
            critic1_loss = F.mse_loss(curr_Q1, expected_Q.detach())
            critic2_loss = F.mse_loss(curr_Q2, expected_Q.detach())

            # Do the optimizer
            self.critic_optimizer1[i].zero_grad()
            critic1_loss.backward()
            self.critic_optimizer1[i].step()

            self.critic_optimizer2[i].zero_grad()
            critic2_loss.backward()
            self.critic_optimizer2[i].step()

            # delyaed update for actor & target networks  
            if(self.update_step % self.delay_step == 0):
                # actor
                new_actions = self.actor[i](state_batch_null)
                policy_gradient = -self.critic1[i](state_batch_null, new_actions)
                policy_gradient *= masking_torch.detach()
                policy_gradient = policy_gradient.mean()
                self.actor_optimizer[i].zero_grad()            
                policy_gradient.backward()
                self.actor_optimizer[i].step()

                # target networks
                self.update_targets(i)

        self.update_step += 1

    def generate_action_space_noise(self, noise_std: float)-> torch.Tensor:
        noise = torch.normal(torch.zeros(1), noise_std).clamp(-self.noise_bound, self.noise_bound).to(self.device)
        return noise

    def update_targets(self, i: int):
        for target_param, param in zip(self.critic_target1[i].parameters(), self.critic1[i].parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target2[i].parameters(), self.critic2[i].parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target[i].parameters(), self.actor[i].parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
    def Actor_save(self):
        # save 각 node별 모델 저장
        print("Save the torch model")        
        for i in range(self.action_dim):
            savePath = "./actor_model5_Hop_"+str(i)+ ".pth"
            torch.save(self.actor[i].state_dict(), savePath)

    def Actor_load(self):
        # save 각 node별 모델 로드
        print("load the torch model")
        for i in range(self.action_dim):
            savePath = "./actor_model_wlk"+str(i)+ ".pth" # Best
            self.actor[i] = Actor(self.obs_dim, self.action_dim).to(self.device)
            self.actor[i].load_state_dict(torch.load(savePath)) 
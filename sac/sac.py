import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from models import SoftQNetwork, PolicyNetwork
from common.replay_buffers import BasicBuffer
from typing import Tuple


class SACAgent():
    def __init__(self, env: object, gamma: float, tau: float, buffer_maxlen: int,
     critic_lr: float, actor_lr:float, reward_scale: int):

        # Selecting the device to use, wheter CUDA (GPU) if available or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Creating the Gym environments for training and evaluation
        self.env = env
        # Get max and min values of the action of this environment
        self.action_range = [self.env.action_space.low, self.env.action_space.high]
        # Get dimension of of the state and the action
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.buffer_maxlen = buffer_maxlen
        self.reward_scale = reward_scale

        # Scaling and bias factor for the actions -> We need scaling of the actions because each environment has different min and max values of actions
        self.scale = (self.action_range[1] - self.action_range[0]) / 2.0
        self.bias = (self.action_range[1] + self.action_range[0]) / 2.0

        # initialize networks
        self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)


        # copy weight parameters to the target Q networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)
        

        # initialize optimizers 
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=self.critic_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.actor_lr)
        
        # Create a replay buffer
        self.replay_buffer = BasicBuffer(self.buffer_maxlen)


    def update(self, batch_size: int):
        # Sampling experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert numpy arrays of experience tuples into pytorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = self.reward_scale * torch.FloatTensor(rewards).to(self.device) # in SAC we do reward scaling for the sampled rewards
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)


        # Critic update (computing the loss) 
        # Please refer to equation (6) in the paper for details
        # Sample actions for the next states (s_t+1) using the current policy
        next_actions, next_log_pi,_,_ = self.policy.sample(next_states,self.scale)
        next_actions = self.rescale_action(next_actions)
        
        # Compute Q(s_t+1,a_t+1) by giving the states and actions to the Q network and choose the minimum from 2 target Q networks
        next_q1 = self.target_q_net1(next_states, next_actions) 
        next_q2 = self.target_q_net2(next_states, next_actions)
        min_q = torch.min(next_q1,next_q2) # find minimum between next_q1 and next_q2

        # Compute the next Q_target (Q(s_t,a_t)-alpha(next_log_pi))
        next_q_target = (min_q - next_log_pi)

       # Compute the Q(s_t,a_t) using s_t and a_t from the replay buffer
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)

        # Find expected Q, i.e., r(t) + gamma*next_q_target
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # Compute loss between Q network and expected Q
        q1_loss = F.mse_loss(curr_q1, expected_q.detach()) 
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())
        
        # Backpropagate the losses and update Q network parameters
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy update (computing the loss)
        # Sample new actions for the current states (s_t) using the current policy
        new_actions, log_pi,_,_ = self.policy.sample(states,self.scale)
        new_actions = self.rescale_action(new_actions)

        # Compute Q(s_t,a_t) and choose the minimum from 2 Q networks
        new_q1 = self.q_net1.forward(states, new_actions)
        new_q2 = self.q_net2.forward(states, new_actions)
        min_q = torch.min(new_q1,new_q2)

        # Compute the next policy loss, i.e., alpha*log_pi - Q(s_t,a_t) eq. (7)
        policy_loss = (log_pi - min_q).mean()

        # Backpropagate the losses and update policy network parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Updating target networks with soft update using update rate tau
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


    def get_action(self, state: np.ndarray, stochastic: bool)-> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        # state: the state input to the pi network
        # stochastic: boolean (True -> use noisy action, False -> use noiseless (deterministic action))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get mean and sigma from the policy network 
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()

        # Stochastic mode is used for training, non-stochastic mode is used for evaluation
        if stochastic:
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).numpy()
        else:
            normal = Normal(mean, 0)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).numpy()
        
        # return a rescaled action, and also the mean and standar deviation of the action
        # we use a rescaled action since the output of the policy network is [-1,1] and the mujoco environments could be ranging from [-n,n] where n is an arbitrary real value
        return self.rescale_action(action),mean,std
    
    def rescale_action(self, action: np.ndarray)->np.ndarray:
        # we use a rescaled action since the output of the policy network is [-1,1] and the mujoco environments could be ranging from [-n,n] where n is an arbitrary real value
        # scale -> scalar multiplication
        # bias -> scalar offset
        return action * self.scale[0] + self.bias[0]

    def Actor_save(self, WORKSPACE: str):
        # save 각 node별 모델 저장
        print("Save the torch model")        
        savePath = WORKSPACE + "./policy_model5_Hop_.pth"
        torch.save(self.policy.state_dict(), savePath)

    def Actor_load(self, WORKSPACE: str):
        # save 각 node별 모델 로드
        print("load the torch model")
        savePath = WORKSPACE + "./policy_model5_Hop_.pth" # Best
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy.load_state_dict(torch.load(savePath)) 
        




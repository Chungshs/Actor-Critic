import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from models import Critic, Actor
from common.replay_buffers import BasicBuffer

class DDPGAgent():
    def __init__(self, env: object, gamma: float, tau: float, buffer_maxlen: int,
     noise_std: float, noise_bound: float, critic_lr: float, actor_lr:float):

        # Selecting the device to use, wheter CUDA (GPU) if available or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Creating the Gym environments for training and evaluation
        self.env = env
        # Get max and min values of the action of this environment
        self.action_range = [self.env.action_space.low, self.env.action_space.high]
        # Get dimension of of the state and the state
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.buffer_maxlen = buffer_maxlen
        self.noise_std = noise_std
        self.noise_bound = noise_bound

        # Scaling and bias factor for the actions -> We need scaling of the actions because each environment has different min and max values of actions
        self.scale = (self.action_range[1] - self.action_range[0]) / 2.0
        self.bias = (self.action_range[1] + self.action_range[0]) / 2.0

        # initialize networks
        self.critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.target_critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.target_actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        torch.save(self.actor,'tes')

        # copy weight parameters to the target Critic network and actor network
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param)


        # initialize optimizers 
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Create a replay buffer
        self.replay_buffer = BasicBuffer(self.buffer_maxlen)


    def update(self, batch_size: int): 
        # Sampling experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert numpy arrays of experience tuples into pytorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)


        # Critic update (computing the loss)
        # Please refer to equation (6) in the paper for details
        # Sample actions for the next states (s_t+1) using the target actor
        next_actions= self.target_actor.forward(next_states)
        next_actions = self.rescale_action(next_actions)

        # Adding gaussian noise to the actions
        noise = self.get_noise(next_actions, self.noise_std + 0.1, -self.noise_bound, self.noise_bound)
        noisy_next_actions = next_actions + noise

        # Compute Q(s_t+1,a_t+1) 
        next_q1 = self.target_critic1(next_states, noisy_next_actions)

        # Find expected Q, i.e., r(t) + gamma*next_q
        expected_q = rewards + (1-dones) * self.gamma * next_q1

        # Compute loss between Q network and expected Q
        curr_q1 = self.critic1.forward(states, actions)
        critic1_loss = F.mse_loss(curr_q1, expected_q.detach()) 

        # Backpropagate the losses and update Q network parameters
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # actor update (computing the loss)
        # Sample new actions for the current states (s_t) using the current actor
        new_actions= self.actor.forward(states)

        # Compute Q(s_t,a_t) 
        new_q1 = self.critic1.forward(states, new_actions)

        # Compute the actor loss, i.e., -Q
        actor_loss = -new_q1.mean() 

        # Backpropagate the losses and update actor network parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # Update the target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


    def get_noise(self, action: torch.Tensor, sigma: float, bottom: float, top: float)-> torch.Tensor:
        # sigma: standard deviation of the noise
        # bottom,top: minimum and maximum values for the given noiuse
        return torch.normal(torch.zeros(action.size()), sigma).clamp(bottom, top).to(self.device)

    def get_action(self, state: np.ndarray, stochastic: bool)->np.ndarray:
        # state: the state input to the pi network
        # stochastic: boolean (True -> use noisy action, False -> use noiseless,deterministic action)

        # Convert state numpy to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.forward(state) 
        
        if stochastic:
            # Add gaussian noise to the rescaled action
            action = self.rescale_action(action) + self.get_noise(action, self.noise_std, -self.noise_bound, self.noise_bound)
        else:
            action = self.rescale_action(action)

        # Convert action tensor to numpy
        action = action.squeeze(0).cpu().detach().numpy()
        return action
    
    def rescale_action(self, action: torch.Tensor)->torch.Tensor:
        # we use a rescaled action since the output of the actor network is [-1,1] and the mujoco environments could be ranging from [-n,n] where n is an arbitrary real value
        # scale -> scalar multiplication
        # bias -> scalar offset
        return action * self.scale[0] + self.bias[0]

    def Actor_save(self, WORKSPACE: str):
        # save 각 node별 모델 저장
        print("Save the torch model")        
        savePath = WORKSPACE + "./actor_model5_Hop_.pth"
        torch.save(self.actor.state_dict(), savePath)

    def Actor_load(self, WORKSPACE: str):
        # save 각 node별 모델 로드
        print("load the torch model")
        savePath = WORKSPACE + "./actor_model5_Hop_.pth" # Best
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor.load_state_dict(torch.load(savePath)) 
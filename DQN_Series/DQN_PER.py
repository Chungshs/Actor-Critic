import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from segment_tree import MinSegmentTree, SumSegmentTree

""" This is classic version of the DQN for Gym Environment """
""" 2020.01.14 mady by chung								 """
""" Using tensorflow 									 """
""" Rainbow is all you is refered                              """

class Replay_Buffer:
    """ Clasiic version """

    def __init__(self, obs_dim: int, memory_size: int, batch_size: int):
        self.obs_in_buf 	    	 = np.zeros([memory_size, obs_dim], dtype=np.float32)
        self.next_obs_in_buf	 = np.zeros([memory_size, obs_dim], dtype=np.float32)
        self.act_in_buf 	    	 = np.zeros([memory_size], dtype=np.float32)
        self.rew_in_buf		 = np.zeros([memory_size], dtype=np.float32)
        self.done_in_buf		 = np.zeros([memory_size], dtype=np.float32)
        self.max_size, self.batch_size = memory_size, batch_size
        self.count, self.size = 0, 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, 
        next_obs: np.ndarray, done: bool):
        self.obs_in_buf[self.count] 	   = obs
        self.next_obs_in_buf[self.count] = next_obs
        self.act_in_buf[self.count]	   = act
        self.rew_in_buf[self.count] 	   = rew
        self.done_in_buf[self.count] 	   = done
        self.count = (self.count + 1) % self.max_size
        self.size  =  min(self.count , self.max_size)

    def sampling(self) -> Dict[str, np.ndarray]:
    	   idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
    	   return dict( obs 	= self.obs_in_buf[idxs],
                       act 	= self.act_in_buf[idxs],
                       rew 	= self.rew_in_buf[idxs],
                       done = self.done_in_buf[idxs],
                       next_obs = self.next_obs_in_buf[idxs])

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(Replay_Buffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_in_buf[indices]
        next_obs = self.next_obs_in_buf[indices]
        acts = self.act_in_buf[indices]
        rews = self.rew_in_buf[indices]
        done = self.done_in_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            act=acts,
            rew=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight


def Network(states):
    # Q Network
    W1 = tf.get_variable("W1", [in_dim, first_layer],
                       initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [first_layer],
                       initializer=tf.constant_initializer(0)) 
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1) 

    W2 = tf.get_variable("W2", [first_layer, second_layer],
                       initializer=tf.contrib.layers.xavier_initializer()) 
    b2 = tf.get_variable("b2", [second_layer],
                       initializer=tf.constant_initializer(0)) 
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.get_variable("W3", [second_layer, out_dim], 
                       initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [out_dim],
                       initializer=tf.constant_initializer(0))
    v = tf.matmul(h2, W3) + b3 

    return v


class DQN_Agent:

    def __init__(
     self, optimizer, session, Network, env:gym.Env, learning_rate : int, in_dim : int, out_dim : int, memory_size: int, batch_size: int, target_update: int, epsilon_decay: float,
     max_epsilon: float = 1.0, min_epsilon: float = 0.1, gamma: float = 0.99, alpha: float = 0.2, beta: float = 0.6, prior_eps: float = 1e-6):

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env  = env
        self.learning_rate = learning_rate

        self.memory = PrioritizedReplayBuffer( obs_dim, memory_size, batch_size, alpha )

        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.target_params = None
        self.mask = None

        self.alpha = alpha
        self.beta  =  beta
        self.prior_eps = prior_eps

        self.target_Q_Value = None
        self.estimated_Q_Value = None
        self.loss = None
        self.train_op = None



        self.sess = session
        self.optimizer = optimizer
                
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):    
             self.dqn = Network
             self.states = tf.placeholder(tf.float32, shape = [None, self.in_dim], name="states")       	
             self.estimated_Q_Value = self.dqn(self.states)

             self.mask = tf.placeholder(tf.float32, shape = [None, 2], name="mask")       	             
             self.replay_estimated_Q_Value = tf.multiply(self.dqn(self.states), self.mask)
             self.replay_estimated_Q_Value = tf.reduce_sum(self.replay_estimated_Q_Value, axis=1)

        with tf.variable_scope("target", reuse=False):    
             self.dqn_target = Network
             self.target_states = tf.placeholder(tf.float32, shape = [None, self.in_dim], name="next_states")       	             
             self.reward = tf.placeholder(tf.float32, shape = [None, ], name="reward")                
             self.target_Q_Value =  self.gamma * tf.reduce_max(self.dqn_target(self.target_states), axis=1) + self.reward
             self.target_Q_Value = tf.stop_gradient(self.target_Q_Value)

        self.loss = pow(self.target_Q_Value - self.replay_estimated_Q_Value,2)/2
        self.train_op = self.optimizer.minimize(self.loss)

        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists)) 
        self.sess.run(tf.assert_variables_initialized())


        self.transition = list()
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:

        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.sess.run(self.estimated_Q_Value, {
            self.states : state.astype(np.float32).reshape(1,4)
            })
            selected_action = selected_action.argmax()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray, score:np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
    	   next_state, reward, done, _ = self.env.step(action)
    	   tmp_score = score + reward
    	   reward = reward if not done or tmp_score == 500 else -100
    	   state = next_state    	   
    	   if not self.is_test:
    	   	self.transition += [reward, next_state, done]
    	   	self.memory.store(*self.transition)

    	   return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        mask = samples["act"].astype(np.float32)
        tmp_mask = np.zeros([self.batch_size,2])
        for i in range(len(mask)):
               if mask[i] == 0:
                   tmp_mask[i,np.int32(mask[i])] = 1
               else:
               	tmp_mask[i,np.int32(mask[i])] = 1
        feed_dict = {
        self.states: samples["obs"].astype(np.float32),
        self.target_states : samples["next_obs"].astype(np.float32),
        self.reward : samples["rew"].astype(np.float32),
        self.mask : tmp_mask.astype(np.float32)
        }
        loss, test = self.sess.run([self.loss,self.train_op], feed_dict)
        new_priorities = loss + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)        

        return np.sum(loss)

    def train(self, num_frames: int, plotting_interval: int = 10000):
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score  = 0

        for frame_idx in range(1, num_frames + 1):
            if (frame_idx%10000) == 0:
              self.learning_rate = self.learning_rate/2
              self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            action = self.select_action(state)
            next_state, reward, done = self.step(action, score)
            state = next_state
            score += reward

            if done:
              print(score)
              state = self.env.reset()
              scores.append(score)
              score = 0

            if len(self.memory) >= self.batch_size:
              loss = self.update_model()

              losses.append(loss/self.batch_size)
              update_cnt += 1


              self.epsilon = max( self.min_epsilon, self.epsilon - (
                   self.max_epsilon - self.min_epsilon) * self.epsilon_decay)

              epsilons.append(self.epsilon)                

              if update_cnt % self.target_update == 0:
                 self._target_hard_update()


            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)
                
        self.env.close()
                
    def test(self) -> List[np.ndarray]:

        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action, score)
            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames

    def _target_hard_update(self):

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
        target_init_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess.run(target_init_op)          
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):

        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()


# environment
env_name = "CartPole-v1"
env = gym.make(env_name)

# Network Parameter
first_layer = 256 #First_layer 수
second_layer = 256 #Second_layer 수
learning_rate = 0.0001

sess      = tf.Session() #  tensorflow Session 줄임말
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # learning_rate = 0.0001, train optimizer 

# Hyper Parameters
num_frames = 10000
memory_size = 2000
batch_size = 128
target_update = 50
epsilon_decay = 1 / 2000

# Input-Output Parameter
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

agent = DQN_Agent(optimizer, sess, Network, env, learning_rate, in_dim, out_dim, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)
frames = agent.test()






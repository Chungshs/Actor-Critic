import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
     self, optimizer, session, Network, env:gym.Env, learning_rate: int , in_dim : int, out_dim : int, memory_size: int, batch_size: int, target_update: int, epsilon_decay: float,
     max_epsilon: float = 1.0, min_epsilon: float = 0.1, gamma: float = 0.99):

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env  = env
        self.learning_rate = learning_rate
        self.memory = Replay_Buffer(obs_dim, memory_size, batch_size)
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
        self.doulbe_mask = None
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
             self.replay_estimated_Q_Value = tf.reduce_max(self.replay_estimated_Q_Value, axis=1)

        with tf.variable_scope("target", reuse=False):    
             self.dqn_target = Network
             self.target_states = tf.placeholder(tf.float32, shape = [None, self.in_dim], name="next_states")       	             
             self.doulbe_mask = tf.placeholder(tf.float32, shape = [None, 2], name="double_mask")
             self.reward = tf.placeholder(tf.float32, shape = [None, ], name="reward")                
             self.target_Q_Value =  tf.reduce_max( self.gamma * tf.multiply(self.dqn_target(self.target_states), self.doulbe_mask),axis = 1) + self.reward
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
        samples = self.memory.sampling()
        mask = samples["act"].astype(np.float32)
        tmp_mask = np.zeros([self.batch_size,2])
        doulbe_mask = np.zeros([self.batch_size,2])
        for i in range(len(mask)):
               if mask[i] == 0:
                tmp_mask[i,np.int32(mask[i])] = 1
               else:
               	tmp_mask[i,np.int32(mask[i])] = 1
        # print(samples["next_obs"].shape)
        tmp_doubl_mask = self.sess.run(self.estimated_Q_Value, feed_dict={self.states : samples["next_obs"].astype(np.float32)})
        # print(tmp_doubl_mask)
        tmp_doubl_mask = np.argmax(tmp_doubl_mask,axis=1)
        # print(tmp_doubl_mask)
        for i in range(self.batch_size):
               if tmp_doubl_mask[i] == 0:
                doulbe_mask[i,np.int32(tmp_doubl_mask[i])] = 1
               else:
                doulbe_mask[i,np.int32(tmp_doubl_mask[i])] = 1
        # print(doulbe_mask)
        feed_dict = {
        self.states: samples["obs"].astype(np.float32),
        self.target_states : samples["next_obs"].astype(np.float32),
        self.reward : samples["rew"].astype(np.float32),
        self.mask : tmp_mask.astype(np.float32),
        self.doulbe_mask : doulbe_mask.astype(np.float32)
        }
        loss, _ = self.sess.run([self.loss,self.train_op], feed_dict)
        
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
first_layer = 512 #First_layer 수
second_layer = 512 #Second_layer 수
learning_rate = 0.0001
sess      = tf.Session() #  tensorflow Session 줄임말
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # learning_rate = 0.0001, train optimizer 



# Hyper Parameters
num_frames = 50000
memory_size = 10000
batch_size = 128
target_update = 50
epsilon_decay = 1 / 2000

# Input-Output Parameter
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

agent = DQN_Agent(optimizer, sess, Network, env, learning_rate, in_dim, out_dim, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)
frames = agent.test()






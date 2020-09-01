import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

""" This is classic version of the DQN for Gym Environment """
""" 2020.01.07 mady by chung								 """
""" Using tensorflow 									 """
""" Rainbow is all you is refered                              """

# replay buffer
class replay_buffer:
	def __init__( self, memory_size : int, batch_size : int, input_dims : int, output_dims : int ):

		self.obs_in_buffer  = np.zeros([memory_size, input_dims])
		self.act_in_buffer  = np.zeros([memory_size])
		self.rew_in_buffer  = np.zeros([memory_size])
		self.nxt_obs_in_buffer = np.zeros([memory_size, input_dims])
		self.done_in_buffer = np.zeros([memory_size])
		
		# Parameters
		self.max_size = 3000
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.count, self.max = 0,0

	def store( self, obs : np.ndarray, act : np.ndarray, rew : float, nxt_obs : np.ndarray, done : bool):
		
		self.obs_in_buffer[self.count] = obs
		self.act_in_buffer[self.count] = act
		self.rew_in_buffer[self.count] = rew
		self.nxt_obs_in_buffer[self.count] = nxt_obs
		self.done_in_buffer[self.count] = done

		self.count = (self.count + 1) % self.max_size
		self.max = min(self.count, self.max_size)

	def sampling(self):
		idx = np.random.choice(self.max, self.batch_size, replace = False)
		return dict(
			obs = self.obs_in_buffer[idx],
			act = self.act_in_buffer[idx],
			rew = self.rew_in_buffer[idx],
			nxt_obs = self.nxt_obs_in_buffer[idx],
			done = self.done_in_buffer[idx]
			)


# network
def Network(states, env : gym.Env):
	first_layer = 24
	second_layer = 24
	input_dims = env.observation_space.shape[0]
	output_dims = env.action_space.n
	w1 = tf.get_variable( "w1",  [input_dims, first_layer],
	 initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable( "b1",  [first_layer],
		initializer = tf.constant_initializer(0))
	h1 = tf.nn.relu( tf.matmul(tf.cast(states, tf.float32),w1) + b1 )

	w2 = tf.get_variable( "w2",  [first_layer, second_layer],
	 initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable( "b2",  [second_layer],
		initializer = tf.constant_initializer(0))
	h2 = tf.nn.relu( tf.matmul(h1,w2) + b2 )

	w3 = tf.get_variable( "w3",  [second_layer, output_dims],
	 initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable( "b3", [output_dims],
		initializer = tf.constant_initializer(0))			
	h3 = tf.matmul(h2,w3) + b3

	w3_1 = tf.get_variable( "w3_1",  [second_layer, 1],
	 initializer = tf.contrib.layers.xavier_initializer())
	b3_1 = tf.get_variable( "b3_1", [1],
		initializer = tf.constant_initializer(0))			
	h3_1 = tf.matmul(h2,w3_1) + b3_1	

	return h3 + h3_1

# agent

class DQN_Agent:

	def __init__( self, sess, optimizier, replay_buffer, network, env : gym.Env, batch_size : int , memory_size : int , learningrate : float, gamma : float):
		self.sess = sess
		self.optimizier = optimizier
		self.memory = replay_buffer
		self.env = env
		input_dims = self.env.observation_space.shape[0]
		output_dims = self.env.action_space.n
		self.gamma = gamma
		self.batch_size = batch_size
		self.memory_size = memory_size
		self.estimate_Q = []
		self.epsilone = 0.9
		self.epsilone_decay = 0.99

		self.states = tf.placeholder(tf.float32, shape = [None, input_dims], name = "states")
		self.action = tf.placeholder(tf.int32, shape = [None], name = "action")		
		self.next_states = tf.placeholder(tf.float32, shape = [None, input_dims], name = "next_states")
		self.reward = tf.placeholder(tf.float32, shape = [None,], name = "reward")		
		
		with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
			self.dqn = Network
			self.estimated_action = self.dqn(self.states, self.env)
			for i in range(self.batch_size):
				self.estimate_Q.append(self.estimated_action[i][self.action[i]])

		with tf.variable_scope("target", reuse=tf.AUTO_REUSE):
			self.dqn_target = Network
			self.next_estimate_Q = self.dqn_target(self.next_states, self.env)
			self.next_estimate_Q = tf.stop_gradient(self.next_estimate_Q)

		with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
			self.loss = pow(self.reward + self.gamma*(tf.reduce_max(self.next_estimate_Q, axis=1)) - self.estimate_Q,2)/2
			self.train_op = self.optimizier.minimize(self.loss)

		var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

		self.sess.run(tf.variables_initializer(var_lists)) 
		self.sess.run(tf.assert_variables_initialized())

	def selected_action(self, states):
		a = np.random.rand(1)
		if a < self.epsilone:
			action = np.random.randint(2)

		else: 
			states = np.reshape(states, [1,4])
			action = self.sess.run(self.estimated_action, feed_dict={ self.states : states})
			action = np.argmax(action[0])

		self.epsilone = self.epsilone * self.epsilone_decay
		self.epsilone = max(self.epsilone, 0.1)
		return action

	def update_network(self):
		batch = memory.sampling()

		feed_dict = {
		self.states : batch["obs"],
		self.action : batch["act"],
		self.reward : batch["rew"],
		self.next_states : batch["nxt_obs"] 
		}

		self.sess.run( self.train_op, feed_dict )

	def target_update(self):

		model_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
		target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target")
		target_op = [tf.assign( t, m ) for t, m in zip(target_param, model_param)]
		self.sess.run(target_op)

	def test(self):

		obs = self.env.reset()
		done = False
		score = 0

		frames = []
		while not done:
			frames.append(self.env.render(mode="rgb_array"))
			action = self.selected_action(obs)
			next_state, reward, done, _ = self.env.step(action)
			obs = next_state
			score += reward

		print("score: ", score)
		self.env.close()

		return frames

env_name = "CartPole-v1"
env = gym.make(env_name)
memory = replay_buffer( memory_size = 3000, batch_size = 32, input_dims = 4 , output_dims = 1 )
episodes = 2000
Sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
agent = DQN_Agent(Sess, optimizer, memory, Network, env, batch_size = 32 , memory_size = 3000 , learningrate = 0.01, gamma = 0.99)
scores = []
for i in range(episodes):
	done = False
	obs = env.reset()
	score = 0
	while not done:
		action = agent.selected_action(obs)
		next_states, reward, done, info = env.step(action)
		if memory.count > 32:
			agent.update_network()
		reward = reward if score == 499 or not done else -100
		score = score + reward		
		agent.memory.store(obs, action, reward, next_states, done)
		obs = next_states
	scores.append(score)
	print(" episode : " , i, "score : ", score)
	agent.target_update()
	if len(scores) > 9:		
		if np.mean(scores[-10:]) > 450:
			break

frame = agent.test()



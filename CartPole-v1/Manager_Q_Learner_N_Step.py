import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import time

import os

import threading

import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'

import theano
theano.config.openmp = True


# Append path to ffmpeg binary
os.environ['PATH'] += os.pathsep + '/usr/local/bin'

import Q_Learner_N_Step

global_count = 1

class Q_worker_thread(threading.Thread):

	def __init__(self, threadID, update_network, target_network, target_lock, target_num):
		threading.Thread.__init__(self)
		self.threadId = threadID
		self.thread_network = update_network
		self.target_network = target_network
		self._stop = threading.Event()
		self.target_num = target_num
		self.target_lock = target_lock

	def run(self):
		env = gym.make('CartPole-v1')
		tick = 0
		while not self._stop.is_set():
			curr_obs = env.reset()
			done = False
			cum_reward = 0
			while not done:
				tick += 1
				action_space = env.action_space
				action, exploration = self.thread_network.propose_action(curr_obs, action_space)
				next_obs, reward, done, info = env.step(action)
				cum_reward += reward
				if done:
					#print('threadId: ', self.threadId, 'cum_reward = ', cum_reward)
					if cum_reward != 500:
						reward = -1
				update_network.add_experience(curr_obs, action, reward, next_obs, done)
				if tick % self.target_num == 0: # Update target network
					print('Thread Id:', self.threadId, 'Time to update the target network')
					self.target_lock.acquire()
					self.target_network.copy_params(self.thread_network)
					global global_count
					global_count += 1
					print('Global count: ', global_count)
					print('Release target network')
					self.target_lock.release()
				#print('threadID:', self.threadId, ' cum reward: ', cum_reward)

	def terminate(self):
		self._stop.set()

if __name__ == '__main__':
	num_workers = 3
	env = gym.make('CartPole-v1')
	#env = wrappers.Monitor(env, './CartPole-v1-exp-Q-Learner', force=True)
	target_network = Q_Learner_N_Step.Network(
		dense_layers=[128,64,32],
		num_features=4,
		num_actions=2,
		exploration=0.0,
		max_steps=1
	)
	# Make async Q networks
	thread_workers = list()
	target_lock = threading.Lock()
	for i in range(0, num_workers):
		update_network = Q_Learner_N_Step.Network(
			dense_layers=[128,64,32],
			num_features=4,
			num_actions=2,
			exploration=np.random.random_sample()*0.6 + 0.4,
			max_steps=32,
		)
		thread_workers.append(
			Q_worker_thread(i, update_network, target_network, target_lock, 150)
		)

	for thread in thread_workers:
		thread.start()

	plt.ion()
	plt.figure(1)
	plt.subplot(111)
	plt.title('Target Cumulative Reward Training')

	goal_reached = False

	i_episode = 0
	while not goal_reached:
		cum_reward = 0
		target_lock.acquire()
		#print('Main Thread checking global')
		if global_count % 25 == 0:
			done = False
			curr_obs = env.reset()
			while not done:
				env.render()
				action = target_network.final_action(curr_obs)
				next_obs, reward, done, info = env.step(action)
				cum_reward += reward
				curr_obs = next_obs

			i_episode += 1

			plt.subplot(111)
			plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
			plt.pause(0.01)

			if cum_reward >= 480:
				goal_reached = True
		target_lock.release()
	plt.waitforbuttonpress()
	# Goal has been reached..
	# Stop the threads
	for thread in thread_workers:
		thread.terminate()

	# Evaluate the main thread.
	rewards = list()
	i_episode = 0

	plt.ion()
	plt.figure(1)
	plt.subplot(211)
	plt.title('Cumulative Reward')
	plt.subplot(212)
	plt.title('Average reward')

	while np.mean(rewards) < 480:
		curr_obs = env.reset()
		done = False
		cum_reward = 0
		print('===========>New Episode<==========')
		print('episode: ' + str(i_episode))
		while not done:
			env.render()
			action = target_network.final_action(curr_obs)
			next_obs, reward, done, info = env.step(action)
			cum_reward += reward
			curr_obs = next_obs
		i_episode += 1
		rewards.append(cum_reward)
		if len(rewards) > 100:
			rewards.pop(0)
		plt.subplot(211)
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.subplot(212)
		plt.scatter(x=i_episode, y=np.mean(rewards), c='r')
	plt.waitforbuttonpress()


# update_network = Q_Learner_N_Step.Network(
	# 	dense_layers=[128, 64, 32],
	# 	num_features=4,
	# 	num_actions=2,
	# 	exploration=np.random.random_sample() * 0.6 + 0.6,
	# 	max_steps=10,
	# )
	# env = gym.make('CartPole-v1')
	# for i in range(0, 100):
	# 	curr_obs = env.reset()
	# 	done = False
	# 	while not done:
	# 		action_space = env.action_space
	# 		action, exploration = update_network.propose_action(curr_obs, action_space)
	# 		next_obs, reward, done, info = env.step(action)
	# 		update_network.add_experience(curr_obs, action, reward, next_obs, done)
	# 		curr_obs = next_obs
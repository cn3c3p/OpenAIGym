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

class Q_Worker():

	def __init__(self, model, global_network, target_network, Id, target_num):
		self.model = model
		self.global_model = global_network
		self.target_network = target_network
		self.Id = Id
		self.target_num = target_num
		self.env = gym.make('CartPole-v1')

	def run(self, tick):
		done = False
		cum_reward = 0
		curr_obs = self.env.reset()
		while not done:
			action_space = self.env.action_space
			action, exploration = self.model.propose_action(curr_obs, self.global_model, action_space)
			next_obs, reward, done, info = self.env.step(action)
			cum_reward += reward
			if done:
				if cum_reward < 500:
					reward = -1
			if cum_reward >= 500:
				done = True
			inputs, targets = self.model.add_experience(curr_obs, action, reward, next_obs, done)
			if inputs and targets:
				self.global_model.train_with_batch(
					collective_inputs=inputs,
					collective_targets=targets
				)
			curr_obs = next_obs
			tick += 1
			if tick % self.target_num == 0:
				print('id: ', self.Id, ' update target network')
				self.target_network.copy_params(self.global_model)
		return tick

if __name__ == '__main__':
	num_workers = 3
	env = gym.make('CartPole-v1')
	# env = wrappers.Monitor(env, './CartPole-v1-exp-Q-Learner', force=True)
	target_network = Q_Learner_N_Step.Network(
		dense_layers=[128, 64, 32],
		num_features=4,
		num_actions=2,
		global_model=None,
		exploration=0.0,
		max_steps=1,
	)
	global_network = Q_Learner_N_Step.Network(
		dense_layers=[128, 64, 32],
		num_features=4,
		num_actions=2,
		global_model=None,
		exploration=0.0,
		max_steps=1
	)
	# Make worker Q networks
	Q_workers = list()
	for i in range(0, num_workers):
		update_network = Q_Learner_N_Step.Network(
			dense_layers=[128, 64, 32],
			num_features=4,
			num_actions=2,
			global_model=global_network,
			exploration=np.random.random_sample() * 0.6 + 0.4,
			max_steps=32,
		)
		worker = Q_Worker(
			model=update_network,
			global_network=global_network,
			target_network=target_network,
			Id=i,
			target_num=500)
		Q_workers.append(worker)

	goal_reached = False
	plt.ion()
	plt.figure(1)
	plt.subplot(111)
	plt.title('Target Cumulative Reward Training')
	i_episode = 0
	tick = 0
	while not goal_reached:
		cum_reward = 0
		# Run through the workers
		for worker in Q_workers:
			tick = worker.run(tick=tick)
		# Run the target network once
		i_episode += 1
		done = False
		curr_obs = env.reset()
		while not done:
			env.render()
			action = target_network.final_action(curr_obs)
			next_obs, reward, done, info = env.step(action)
			cum_reward += reward
			curr_obs = next_obs
		plt.subplot(111)
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.pause(0.01)

		if cum_reward >= 480:
			goal_reached = True
	plt.waitforbuttonpress()

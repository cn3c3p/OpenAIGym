import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import time

import threading

import os

# Append path to ffmpeg binary
os.environ['PATH'] += os.pathsep + '/usr/local/bin'
#os.environ['PATH'] += os.pathsep + 'C:\\Users\\Hooligan\\ffmpeg-20170202-08b0981-win64-static\\bin'

import Q_Learner_N_Step

class Q_Worker():

	def __init__(self, model, global_network, target_network, Id, target_num):
		self.model = model
		self.global_model = global_network
		self.target_network = target_network
		self.Id = Id
		self.target_num = target_num
		self.env = gym.make('LunarLander-v2')

	def run(self, tick):
		done = False
		cum_reward = 0
		curr_obs = self.env.reset()
		start_time = time.time()
		while not done:
			action_space = self.env.action_space
			action, exploration = self.model.propose_action(curr_obs, self.global_model, action_space)

			if time.time() - start_time >= 30:
				action = 0 # Do nothing

			next_obs, reward, done, info = self.env.step(action)

			reward /= 200

			inputs, targets = self.model.add_experience(curr_obs, action, reward, next_obs, done)

			if inputs and targets:
				# print('Id: ', self.Id, ' Update global')
				self.global_model.train_with_batch(
					collective_inputs=inputs,
					collective_targets=targets
				)
				if tick % self.target_num == 0:
					print('id: ', self.Id, ' update target network')
					self.target_network.copy_params(self.global_model)
				tick += 1

			cum_reward += reward
			curr_obs = next_obs
		return tick

if __name__ == '__main__':
	num_workers = 5
	env = gym.make('LunarLander-v2')
	env = wrappers.Monitor(env, './LunarLander-v2-exp-Q-Learner_n_steps', force=True)
	target_network = Q_Learner_N_Step.Network(
		dense_layers=[128, 64, 32],
		num_features=8,
		num_actions=4,
		exploration=0.0,
		max_steps=1,
		target_model=None,
	)

	global_network = Q_Learner_N_Step.Network(
		dense_layers=[128, 64, 32],
		num_features=8,
		num_actions=4,
		target_model=None,
		exploration=0.0,
		max_steps=1
	)

	# Make worker Q networks
	Q_workers = list()
	for i in range(0, num_workers):
		update_network = Q_Learner_N_Step.Network(
			dense_layers=[128, 64, 32],
			num_features=8,
			num_actions=4,
			target_model=global_network,
			exploration=np.random.random_sample() * 0.6 + 0.4,
			max_steps=1500,
		)
		worker = Q_Worker(
			model=update_network,
			global_network=global_network,
			target_network=target_network,
			Id=i,
			target_num=50)
		Q_workers.append(worker)

	goal_reached = False
	plt.ion()
	plt.figure(1)
	plt.subplot(111)
	plt.title('Target Cumulative Reward Training')
	i_episode = 0
	tick = 1
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
			reward /= 200
			cum_reward += reward
			curr_obs = next_obs
		plt.subplot(111)
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.pause(0.01)

		if reward >= 1:
			goal_reached = True
	plt.ion()
	plt.figure(2)
	plt.subplot(211)
	plt.title('Cumulative Reward')
	plt.subplot(212)
	plt.title('100 Average rewards')
	# ======== Run Target network
	rewards = [0]
	i_episode = 0
	while len(rewards) < 120 or np.mean(rewards) <= 500:
		curr_obs = env.reset()
		done = False
		cum_reward = 0
		while not done:
			env.render()
			action = target_network.final_action(curr_obs)
			next_obs, reward, done, info = env.step(action)
			cum_reward += reward
			curr_obs = next_obs
			cum_reward += reward
		i_episode += 1
		rewards.append(cum_reward)
		if len(rewards) > 120:
			rewards.pop(0)
		plt.subplot(211)
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.subplot(212)
		plt.scatter(x=i_episode, y=np.mean(rewards), c='r')
		plt.pause(0.01)
	plt.waitforbuttonpress()
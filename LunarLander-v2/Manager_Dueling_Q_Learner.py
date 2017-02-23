import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import time

import os

# Append path to ffmpeg binary
os.environ['PATH'] += os.pathsep + '/usr/local/bin'

import Dueling_Q_Learner

if __name__ == '__main__':

	env = gym.make('CartPole-v1')
	env = wrappers.Monitor(env, './CartPole-v1-exp-Duel-Q', force=True)
	target_network = Dueling_Q_Learner.network(
		adv_layers=[128, 128],
		val_layers=[128, 128],
		num_features=4,
		num_actions=2,
		learning_rate=0.0005
	)

	update_network = Dueling_Q_Learner.network(
		adv_layers=[128, 128],
		val_layers=[128, 128],
		num_features=4,
		num_actions=2,
		learning_rate=0.0005
	)

	tick = 1

	plt.ion()
	plt.figure(1)
	plt.subplot(211)
	plt.title('Cumulative Reward')
	plt.subplot(212)
	plt.title('100 episode mean Reward')

	rewards = list()
	final_form = False
	goal_reached = False
	i_episode = 0

	max_avg_reward = 0

	while not goal_reached:
		curr_obs = env.reset()
		done = False
		print('===========>New Episode<==========')
		print('episode: ' + str(i_episode))
		cum_reward = 0
		while not done:

			if i_episode % 50 == 0:
				print('EVALUATION MODE ~~~~~~~~~')
				eval_mode = True
			else:
				eval_mode = False

			tick += 1

			if tick % 300 == 0 and not final_form:
				print("update target")
				target_network.copy_params(update_network)

			action_space = env.action_space

			if final_form or eval_mode:
				env.render()
				action = target_network.final_action(curr_obs)
			else:
				action, exploration = update_network.propose_action(curr_obs, action_space)
			# action = env.action_space.sample()
			next_obs, reward, done, info = env.step(action)

			cum_reward += reward

			if done:
				print('cum_reward = ', cum_reward)
				if cum_reward <= 500:
					reward = -1
			# Add experience
			if not final_form and not eval_mode:
				update_network.add_experience(curr_obs, action, reward, next_obs, done, target_network)

			# value = update_network.evaluate_state(curr_obs)
			curr_obs = next_obs
		i_episode += 1
		rewards.append(cum_reward)

		if len(rewards) > 100:
			rewards.pop(0)
		print('rewards average: ', np.mean(rewards))

		if np.mean(rewards) >= 500:
			goal_reached = True

		if np.mean(rewards) >= 410.0:
			print('Final Form')
			final_form = True
			target_network.mode = 'max'

		plt.subplot(211)
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.subplot(212)
		plt.scatter(x=i_episode, y=np.mean(rewards), c='b', alpha=0.6)
		plt.pause(0.01)
	plt.waitforbuttonpress()
	env.close()
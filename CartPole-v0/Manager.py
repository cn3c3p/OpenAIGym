import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import time

import ActorCritic_DNN as ac

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	env = wrappers.Monitor(env, './CartPole-v0-exp')
	target_network = ac.ActorCriticDNN(
		actor_layers=[32, 32, 32],
		critic_layers=[32, 32, 32],
		num_action_output=2,
		num_features=4
	)

	update_network = ac.ActorCriticDNN(
		actor_layers=[32, 32, 32],
		critic_layers=[32, 32, 32],
		num_action_output=2,
		num_features=4
	)

	tick = 1

	plt.ion()
	plt.figure(1)
	plt.subplot(111)
	plt.title('Value')

	rewards = list()
	final_form = False
	for i_episode in range(0, 600):
		curr_obs = env.reset()
		done = False
		print('===========>New Episode<==========')
		print('episode: ' + str(i_episode))
		cum_reward = 0
		while not done:
			if tick % 100 == 0:
				print('Update everything!')
				target_network.copy_params(update_network)
			tick += 1
			env.render()
			action_space = env.action_space

			if final_form:
				action = target_network.final_action(curr_obs)
			else:
				action, exploration = target_network.propose_action(curr_obs, action_space)
			#action = env.action_space.sample()
			next_obs, reward, done, info = env.step(action)
			if done:
				reward = -1
			# Add experience
			if not final_form:
				update_network.add_experience_tuple(curr_obs, action, reward, next_obs, exploration)
			cum_reward += reward
			value = update_network.evaluate_state(curr_obs)
			curr_obs = next_obs
		rewards.append(cum_reward)
		if len(rewards) > 100:
			rewards.pop(0)
		print('rewards average: ', np.mean(rewards))
		if np.mean(rewards) > 195.0:
			# Fix the agent.
			final_form = True
			target_network.mode = 'max'
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.pause(0.01)
	plt.waitforbuttonpress()



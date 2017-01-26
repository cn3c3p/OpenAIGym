import gym
import numpy as np
import matplotlib.pyplot as plt
import time

import ActorCritic_DNN as ac

if __name__ == '__main__':
	env = gym.make('CartPole-v0')

	target_network = ac.ActorCriticDNN(
		actor_layers=[32, 32, 32],
		critic_layers=[32, 32, 32],
		num_action_output=2,
		num_features=4
	)

	update_network = ac.ActorCriticDNN(
		actor_layers=[128, 64, 32],
		critic_layers=[128, 64, 32],
		num_action_output=2,
		num_features=4
	)

	tick = 1

	plt.ion()
	plt.figure(1)
	plt.subplot(111)
	plt.title('Value')

	for i_episode in range(0, 5000):
		curr_obs = env.reset()
		done = False
		print('===========>New Episode<==========')
		print('episode: ' + str(i_episode))
		while not done:
			if tick % 500 == 0:
				target_network.copy_params(update_network)
			env.render()
			action_space = env.action_space
			action, exploration = target_network.propose_action(curr_obs, action_space)
			#action = env.action_space.sample()
			next_obs, reward, done, info = env.step(action)

			# Add experience
			update_network.add_experience_tuple(curr_obs, action, reward, next_obs, exploration)

			value = update_network.evaluate_state(curr_obs)
			if tick % 50 == 0:
				plt.subplot(111)
				plt.scatter(x=tick, y=value, alpha=0.5, c='b')
				plt.pause(0.01)

			curr_obs = next_obs

	plt.waitforbuttonpress()



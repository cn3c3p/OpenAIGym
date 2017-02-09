import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import time

import os

# Append path to ffmpeg binary
os.environ['PATH'] += os.pathsep + '/usr/local/bin'

import ActorCritic_DNN as ac

if __name__ == '__main__':
	env = gym.make('LunarLander-v2')
	env = wrappers.Monitor(env, './LunarLander-v2-exp-Actor-Critic', force=True)
	target_network = ac.ActorCriticDNN(
		actor_layers=[128, 64, 32],
		critic_layers=[128, 64, 32],
		num_action_output=4,
		num_features=8,
		#load_actor_model='actor/actor-226.h5',
		#load_critic_model='critic/critic-226.h5'
	)

	update_network = ac.ActorCriticDNN(
		actor_layers=[128, 64, 32],
		critic_layers=[128, 64, 32],
		num_action_output=4,
		num_features=8,
		# load_actor_model='actor/actor-226.h5',
		# load_critic_model='critic/critic-226.h5'
	)

	target_network.mode = 'stochastic'

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
		start_time = time.time()
		while not done:

			if i_episode % 50 == 0:
				print('EVALUATION MODE ~~~~~~~~~')
				eval_mode = True
				target_network.mode = 'max'
			else:
				target_network.mode = 'stochastic'
				eval_mode = False

			if tick % 150 == 0:
				if not final_form:
					print('Update everything!')
					target_network.copy_params(update_network)
			tick += 1
			action_space = env.action_space

			#env.render()

			if final_form or eval_mode:
				env.render()
				action = target_network.final_action(curr_obs)
			else:
				action, exploration = target_network.propose_action(curr_obs, action_space)
			#action = env.action_space.sample()
			next_obs, reward, done, info = env.step(action)

			reward /= 100

			cum_reward += reward

			if done:
				print('cum_reward = ', cum_reward)
			if cum_reward >= 200/100:
				final_form = True
			# Add experience
			if eval_mode and not final_form:
				update_network.add_experience_tuple(curr_obs, action, reward, next_obs, False)
			if not final_form and not eval_mode:
				update_network.add_experience_tuple(curr_obs, action, reward, next_obs, exploration)


			#value = update_network.evaluate_state(curr_obs)
			curr_obs = next_obs
			if time.time() - start_time > 90:
				print('Stuck')
				while not done:
					action = 0
					next_obs, reward, done, info = env.step(action)
					if done:
						reward = -100
					update_network.add_experience_tuple(curr_obs, action, reward, next_obs, False)
					curr_obs = next_obs
		i_episode += 1
		rewards.append(cum_reward)

		if np.mean(rewards) > max_avg_reward + 20:
			target_network.save(str(np.mean(rewards)))
			max_avg_reward = np.mean(rewards)

		if len(rewards) > 100:
			rewards.pop(0)
		print('rewards average: ', np.mean(rewards))

		if np.mean(rewards) >= 200:
			goal_reached = True

		if cum_reward == 200.0:
			print('Final Form')
			final_form = True
			target_network.mode = 'max'

		plt.subplot(211)
		plt.scatter(x=i_episode, y=cum_reward, c='b', alpha=0.6)
		plt.subplot(212)
		plt.scatter(x=i_episode, y=np.mean(rewards), c='b', alpha=0.6)
		plt.pause(0.01)
	plt.waitforbuttonpress()



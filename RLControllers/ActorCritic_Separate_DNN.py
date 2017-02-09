import time
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential, load_model

from keras.regularizers import l2, activity_l2


import math
import numpy as np
from random import shuffle
import pickle as pickle


class ActorCriticDNN:

	critic_experiences = list()
	actor_experiences = list()

	iterations = 0

	critic = None
	actor = None

	def __init__(self,
				 actor_layers, critic_layers,
				 num_action_output,
                 num_features,
				 critic_loss='mse',
				 actor_loss='mse',
				 critic_activation='linear',
				 actor_activation='linear',
				 actor_exploration = 0.9,
				 update_num=32,
				 batch_size=32,
				 discount_factor=0.9,
				 actor_buffer_len=50000,
				 critic_buffer_len=50000,
				 max_iter=10000000,
				 load_critic_model=None,
				 load_actor_model=None):

		self.update_num = update_num
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.actor_exploration = actor_exploration
		self.num_action_output = num_action_output
		self.init_exp = actor_exploration
		self.max_iter = max_iter
		self.iterations = 0
		self.actor_buffer_len = actor_buffer_len
		self.critic_buffer_len = critic_buffer_len

		if load_actor_model:
			print('Load actor')
			self.actor = load_model(load_actor_model)
		if load_critic_model:
			print('load critic')
			self.critic = load_model(load_critic_model)

		if not self.critic:
			self.critic = Sequential()
			# Critic Network

			self.critic.add(
				Dense(
					output_dim=critic_layers[0],
					input_dim=num_features
				)
			)

			for critic_layer in critic_layers[1:]:
				self.critic.add(
					Dense(
						output_dim=critic_layer,
						activation='relu'
					)
				)

			self.critic.add(
				Dense(
					output_dim=1,
					activation=critic_activation,
					name='critic_out'
				)
			)

			self.critic.compile(
				optimizer='rmsprop',
				loss=critic_loss
			)

		if not self.actor:
			self.actor = Sequential()
			# Actor network

			self.actor.add(
				Dense(
					output_dim=actor_layers[0],
					activation='relu',
					input_dim=num_features
				)
			)

			for actor_layer in actor_layers[1:]:
				self.actor.add(
					Dense(
						output_dim=actor_layer,
						activation='relu'
					)
				)

			self.actor.add(
				Dense (
					output_dim=num_action_output,
					activation=actor_activation,
					name='actor_out'
				)
			)

			self.actor.compile(
				optimizer='rmsprop',
				loss=actor_loss
			)

		print('actor layers: ')
		print(self.actor.summary())
		print('critic layers: ')
		print(self.critic.summary())

		print('actor.inputs: ')
		print(self.actor.input_shape)
		print('critic.inputs: ')
		print(self.actor.input_shape)

		print('actor.outputs: ')
		print(self.actor.output_shape)
		print('critic.outputs: ')
		print(self.critic.output_shape)

	def propose_action(self, s, action_space):

		# Sample epsilon
		p = np.random.random()
		action = self.evaluate_action(s)
		under_exploration = False
		if p < self.actor_exploration:
			under_exploration = True
			# ===== Let subclass to implement exploration ===== #
			action = self.explore_action(action, action_space)

		action = self.prepare_action_for_env(action)
		return action, under_exploration

	def add_experience_tuple(self, s, a, r, s_n, exploration):

		if exploration:
			self.actor_experiences.append((s, a, r, s_n))
			if len(self.actor_experiences) > self.actor_buffer_len:
				self.actor_experiences.pop(0)

		self.critic_experiences.append((s, a, r, s_n))
		if len(self.critic_experiences) > self.critic_buffer_len:
			self.critic_experiences.pop(0)

		self.iterations += 1
		# Anneal exploration factor
		self.actor_exploration = - (self.init_exp - 0.1) * self.iterations / self.max_iter + self.init_exp
		if self.actor_exploration < 0.1:
			self.actor_exploration = 0.1
		if self.iterations % 1000 == 0:
			print('>>> Iterations: ', self.iterations)
			print('>>> Exploration: ', self.actor_exploration)
		if self.iterations % self.update_num == 0:
			# Train the network
			self.update_critic()
			self.update_actor()

	def update_critic(self):
		# Shuffle experiences
		if len(self.critic_experiences) > self.batch_size:
			stuff = range(0, len(self.critic_experiences))
			ind = np.random.choice(
				a=stuff,
				size=self.batch_size,
				replace=False
			)
			experiences = [self.critic_experiences[i] for i in ind]
		else:
			experiences = self.critic_experiences
			if len(experiences) == 0:
				return

		states, actions, rewards, states_next = zip(*experiences)

		next_states_values = self.critic.predict_on_batch(
			x=np.asarray(states_next)
		)

		# Flatten..
		next_states_values = [item for sublist in next_states_values for item in sublist]

		targets = [r + self.discount_factor*s_n_val for (r, s_n_val) in zip(rewards, next_states_values)]

		loss = self.critic.train_on_batch(
			x=np.asarray(states),
			y=np.asarray(targets)
		)

		#print 'Critic loss: ' + str(np.mean(loss))

	def update_actor(self):
		# Shuffle experiences
		if len(self.actor_experiences) > self.batch_size:

			stuff = range(0, len(self.actor_experiences))

			ind = np.random.choice(
				a=stuff,
				size=self.batch_size,
				replace=False
			)
			experiences = [self.actor_experiences[i] for i in ind]
		else:
			experiences = self.actor_experiences

		if len(experiences) == 0:
			return

		s, a, r, s_n = zip(*experiences)

		current_predictions = self.critic.predict_on_batch(
			x=np.asarray(s)
		)

		next_predictions = self.critic.predict_on_batch(
			x=np.asarray(s_n)
		)

		# Flatten both lists

		current_state_values = [item for sublist in current_predictions for item in sublist]
		next_state_values = [item for sublist in next_predictions for item in sublist]

		current_state_values_p = [reward + self.discount_factor * q_s_n for reward, q_s_n in zip (r, next_state_values)]

		action_indices = [i for i, (q_s_n_p, q_s_p) in enumerate(zip(current_state_values_p, current_state_values)) if q_s_n_p > q_s_p]

		if len(action_indices) == 0:
			return

		targets = [a[action_index] for action_index in action_indices]
		# Modify the action to prepare it for training
		targets = [self.prepare_action_for_training(target) for target in targets]
		batch = [s[action_index] for action_index in action_indices]

		loss = self.actor.train_on_batch(
			x=np.asarray(batch),
			y=np.asarray(targets)
		)
		#print 'Actor Loss: ' + str(np.mean(loss))

	def evaluate_action(self, s):
		self.under_exploration = False
		prediction = self.actor.predict_on_batch(
			x=np.asarray([s])
		)
		action = prediction[0]
		return action

	def evaluate_state(self, s):
		prediction = self.critic.predict_on_batch(
			x=np.asarray([s])
		)
		value = prediction[0]
		return value

	def final_action(self, s):
		action = self.evaluate_action(s)
		action = self.prepare_action_for_env(action)
		return action

	def copy_params(self, other_network):
		self.critic.set_weights(other_network.critic.get_weights())
		self.actor.set_weights(other_network.actor.get_weights())

	def save(self, append):
		self.critic.save('critic/critic-' + append + '.h5')
		self.actor.save('actor/actor-' + append + '.h5')


	def explore_action(self, action, action_space):
		# TODO: Left for the subclass to implement
		pass

	def prepare_action_for_env(self, action):
		# TODO: Left for subclass to implement how to prepare action for environment
		pass

	def prepare_action_for_training(self, action):
		# TODO: Left for subclass to implement how to prepare action for training of network
		pass
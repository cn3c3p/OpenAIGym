import time
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model

import math
import numpy as np
from random import shuffle
import pickle as pickle


class ActorCriticDNN:

	critic_experiences = list()
	actor_experiences = list()

	iterations = 0

	def __init__(self,
				 common_layers, actor_layers, critic_layers,
				 num_action_output,
                 num_features,
				 loss='mse',
				 critic_activation='linear',
				 actor_activation='linear',
				 actor_exploration = 0.9,
				 update_num=32,
				 batch_size=32,
				 discount_factor=0.9,
				 max_iter=10000000):

		self.update_num = update_num
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.actor_exploration = actor_exploration
		self.num_action_output = num_action_output
		self.init_exp = actor_exploration
		self.max_iter = max_iter
		self.iterations = 0

		state_input = Input(
			shape=(num_features,),
			name='state_input'
		)

		common = state_input

		for common_layer in common_layers:

			common = Dense (
				output_dim=common_layer,
				activation='relu'
			)(common)

		# Critic subnetwork
		critic = common

		for critic_layer in critic_layers:
			critic = Dense(
				output_dim=critic_layer,
				activation='relu'
			)(critic)

		critic_out = Dense (
			output_dim=1,
			activation=critic_activation,
			name='critic_out'
		)(critic)

		# Actor subnetwork

		actor = common
		for actor_layer in actor_layers:
			actor = Dense(
				output_dim=actor_layer,
				activation='relu'
			)(actor)

		actor_out = Dense (
			output_dim=num_action_output,
			activation=actor_activation,
			name='actor_out'
		)(actor)

		self.model = Model(
			input=[state_input],
			output=[critic_out, actor_out]
		)

		start = time.time()

		self.model.compile(
			optimizer='rmsprop',
			loss=loss
		)

		print("Compilation Time : ", time.time() - start)

		print('model layers: ')
		print(self.model.summary())

		print('model.inputs: ')
		print(self.model.input_shape)

		print('model.outputs: ')
		print(self.model.output_shape)

	def propose_action(self, s, action_space):

		# Sample epsilon
		self.iterations += 1
		p = np.random.random()
		action = self.evaluate_action(s)
		under_exploration = False
		if p < self.actor_exploration:
			under_exploration = True
			# ===== Let subclass to implement exploration ===== #
			action = self.explore_action(action, action_space)

		# Anneal exploration factor
		self.actor_exploration = - (self.init_exp - 0.1) * self.iterations/self.max_iter  + self.init_exp
		if self.actor_exploration < 0.1:
			self.actor_exploration = 0.1
		if self.iterations % 1000 == 0:
			print('>>> Iterations: ', self.iterations)
			print('>>> Exploration: ',self.actor_exploration)

		action = self.prepare_action_for_env(action)
		return action, under_exploration

	def add_experience_tuple(self, s, a, r, s_n, exploration):

		if exploration:
			self.actor_experiences.append((s, a, r, s_n))
			if len(self.actor_experiences) > 50000:
				self.actor_experiences.pop(0)

		self.critic_experiences.append((s, a, r, s_n))
		if len(self.critic_experiences) > 50000:
			self.critic_experiences.pop(0)

		self.iterations += 1
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

		next_predictions = self.model.predict_on_batch(
			x={
				'state_input': np.asarray(states_next)
			}
		)
		next_states_values = next_predictions[0]

		targets = [r + self.discount_factor*s_n_val for (r, s_n_val) in zip(rewards, next_states_values)]

		loss = self.model.train_on_batch(
			x={
				'state_input':np.asarray(states)
			},
			y={
				'critic_out':np.asarray(targets),
				'actor_out':np.asarray(actions)
			}
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

		current_predictions = self.model.predict_on_batch(
			x={
				'state_input': np.asarray(s)
			}
		)

		next_predictions = self.model.predict_on_batch(
			x={
				'state_input': np.asarray(s_n)
			}
		)

		current_state_values = current_predictions[0]
		next_state_values = next_predictions[0]

		current_state_values_p = [reward + self.discount_factor * q_s_n for reward, q_s_n in zip (r, next_state_values)]

		action_indices = [i for i, (q_s_n_p, q_s_p) in enumerate(zip(current_state_values_p, current_state_values)) if q_s_n_p > q_s_p]

		if len(action_indices) == 0:
			return

		targets = [a[action_index] for action_index in action_indices]
		# Modify the action to prepare it for training
		targets = [self.prepare_action_for_training(target) for target in targets]
		batch = [s[action_index] for action_index in action_indices]
		critic_out = [current_state_values[action_index] for action_index in action_indices]

		loss = self.model.train_on_batch(
			x={
				'state_input':np.asarray(batch)
			},
			y= {
				'actor_out':np.asarray(targets),
				'critic_out':np.asarray(critic_out)
			}
		)
		#print 'Actor Loss: ' + str(np.mean(loss))

	def evaluate_action(self, s):
		self.under_exploration = False
		prediction = self.model.predict_on_batch(
			x={
				'state_input': np.asarray([s])
			}
		)
		action = prediction[1][0]
		return action

	def evaluate_state(self, s):
		prediction = self.model.predict_on_batch(
			x={
				'state_input': np.asarray([s])
			}
		)
		value = prediction[0][0]
		return value

	def copy_params(self, other_network):
		self.model.set_weights(other_network.get_weights())

	def explore_action(self, action, action_space):
		# TODO: Left for the subclass to implement
		pass

	def prepare_action_for_env(self, action):
		# TODO: Left for subclass to implement how to prepare action for environment
		pass

	def prepare_action_for_training(self, action):
		# TODO: Left for subclass to implement how to prepare action for training of network
		pass
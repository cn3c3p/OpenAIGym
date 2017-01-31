import time
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential

import math
import numpy as np
from random import shuffle
import pickle as pickle


class Network():

	iterations = 0
	experiences = list()

	def __init__(self, dense_layers,
				 num_features,
				 num_actions,
				 activation='linear',
				 loss='mse',
				 update_num=10,
				 batch_size=32,
				 discount_factor=0.9,
				 exploration=1.0,
				 experience_length=50000,
				 max_iter=100000):

		self.update_num = update_num
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.init_exp = exploration
		self.max_iter = max_iter
		self.iterations = 0
		self.experiences = list()
		self.experience_length = experience_length

		self.model = Sequential()

		self.model.add(
			Input(
				shape=(num_features,)
			)
		)

		for layer in dense_layers:
			self.model.add(
				Dense(
					output_dim=layer,
					activation='relu'
				)
			)
			self.model.add(
				Dropout(
					p=0.2
				)
			)

		self.model.add(
			Dense(
				output_dim=num_actions,
				activation=activation
			)
		)

		self.model.compile(
			optimizer='rmsprop',
			loss=loss
		)

		print('policy network layers: ')
		print(self.model.summary())

		print('model inputs: ')
		print(self.model.input_shape)

		print('model outputs: ')
		print(self.model.output_shape)

	def propose_action(self, s, action_space):
		p = np.random.random()
		under_exploration = False
		if p < self.actor_exploration:
			under_exploration = True
			# ===== Let subclass to implement exploration ===== #
			action = self.explore_action(action_space)
		else:
			Q_values = self.evaluate_Q_values(s)
			action = self.action_from_Q_values(q_values=Q_values)
		# Anneal exploration factor
		self.actor_exploration = - (self.init_exp - 0.1) * self.iterations / self.max_iter + self.init_exp
		if self.actor_exploration < 0.1:
			self.actor_exploration = 0.1
		if self.iterations % 1000 == 0:
			print('>>> Iterations: ', self.iterations)
			print('>>> Exploration: ', self.actor_exploration)
		return action, under_exploration

	def add_experience(self, s, a, r, s_n):
		self.experiences.append((s, a, r, s_n))

		if len(self.experiences) > self.experience_length:
			self.experiences.pop(0)

		self.iterations += 1
		if self.iterations % self.update_num == 0:
			# Train the network
			self.update()

	def update(self):
		if len(self.experiences) > self.batch_size:
			stuff = range(0, len(self.experiences))
			ind = np.random.choice(
				a=stuff,
				size=self.batch_size,
				replace=False
			)
			experiences = [self.experiences[i] for i in ind]
		else:
			experiences = self.experiences
			if len(experiences) == 0:
				return

		states, actions, rewards, states_next = zip(*experiences)

		Q_curr = self.model.predict_on_batch(
			x=np.asarray(states)
		)

		Q_next = self.model.predict_on_batch(
			x=np.asarray(states_next)
		)

		max_Q_next = [values[np.argmax(values)] for values in Q_next]

		targets = list()

		for curr_val, action, reward, next_val in zip(Q_curr, actions, rewards, max_Q_next):
			target = curr_val
			target[action] = reward + self.discount_factor * next_val
			targets.append(target)

		self.model.train_on_batch(
			x=np.asarray(states),
			y=np.asarray(targets)
		)

	def evaluate_Q_values(self, s):
		Q_values = self.model.predict_on_batch(
			x=np.asarray([s])
		) # Q values.
		return Q_values

	def explore_action(self, action_space):
		# TODO: To be implemented by subclass
		pass

	def action_from_Q_values(self, q_values):
		# TODO: To be implemented by subclass
		pass

	def prepare_action_for_training(self, action):
		# TODO: Left for subclass to implement how to prepare action for training of network
		pass
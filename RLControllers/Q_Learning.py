import time
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import rmsprop

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
				 learning_rate=0.001,
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
		self.exploration = exploration
		self.max_iter = max_iter
		self.iterations = 0
		self.experiences = list()
		self.experience_length = experience_length


		self.model = Sequential()

		self.model.add(
			Dense (
				output_dim=dense_layers[0],
				input_dim=num_features
			)
		)

		for layer in dense_layers[1:]:
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
			optimizer=rmsprop(lr=learning_rate),
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
		if p < self.exploration:
			under_exploration = True
			# ===== Let subclass to implement exploration ===== #
			action = self.explore_action(s, action_space)
		else:
			Q_values = self.evaluate_Q_values(s)
			action = self.action_from_Q_values(q_values=Q_values)
		return action, under_exploration

	def add_experience(self, s, a, r, s_n, done):
		self.experiences.append((s, a, r, s_n, done))
		self.iterations += 1
		# Anneal exploration factor
		self.exploration = - (self.init_exp - 0.1) * self.iterations / self.max_iter + self.init_exp
		if self.exploration < 0.1:
			self.exploration = 0.1
		if len(self.experiences) > self.experience_length:
			self.experiences.pop(0)

		if self.iterations % 1000 == 0:
			print('>>> Iterations: ', self.iterations)
			print('>>> Exploration: ', self.exploration)

		if self.iterations % self.update_num == 0:
			# Train the network
			for i in range(0, 10):
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

		states, actions, rewards, states_next, dones = zip(*experiences)

		Q_curr = self.model.predict_on_batch(
			x=np.asarray(states)
		)

		Q_next = self.model.predict_on_batch(
			x=np.asarray(states_next)
		)

		max_Q_next = [values[np.argmax(values)] for values in Q_next]

		targets = list()

		for curr_val, action, reward, next_val, done in zip(Q_curr, actions, rewards, max_Q_next, dones):
			target = curr_val
			if done:
				target[action] = reward
			else:
				target[action] = reward + self.discount_factor * next_val
			targets.append(target)

		self.model.fit(
			x=np.asarray(states),
			y=np.asarray(targets),
			batch_size=self.batch_size,
			nb_epoch=1,
			verbose=0
		)

	def evaluate_Q_values(self, s):
		Q_values = self.model.predict_on_batch(
			x=np.asarray([s])
		) # Q values.
		return Q_values

	def final_action(self, s):
		q_values = self.evaluate_Q_values(s)
		return self.action_from_Q_values(q_values)

	def save(self, name):
		self.model.save('Q_learning/model' + str(name) + '.h5')

	def copy_params(self, network):
		self.model.set_weights(network.model.get_weights())

	def explore_action(self, state, action_space):
		# TODO: To be implemented by subclass
		pass

	def action_from_Q_values(self, q_values):
		# TODO: To be implemented by subclass
		pass
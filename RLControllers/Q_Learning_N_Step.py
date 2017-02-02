import time
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential
import keras

import math
import numpy as np
from random import shuffle
import pickle as pickle


class Network():
	def __init__(self,
				 dense_layers,
				 num_features,
				 num_actions,
				 max_steps,
				 activation='linear',
				 loss='mse',
				 discount_factor=0.9,
				 exploration=1.0,
				 max_iter=100000):

		self.max_steps = max_steps
		self.discount_factor = discount_factor
		self.exploration = exploration
		self.init_exploration = exploration
		self.episode_experience = list()
		self.max_iter = max_iter
		self.iterations = 0
		self.model = Sequential()

		# ====== BUILD MODEL ====== #

		self.model.add(
			Dense(
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
			optimizer='rmsprop',
			loss=loss
		)

		print('policy network layers: ')
		print(self.model.summary())

		print('model inputs: ')
		print(self.model.input_shape)

		print('model outputs: ')
		print(self.model.output_shape)

	def propose_action(self, s, target_model, action_space):
		p = np.random.random()
		under_exploration = False
		if p < self.exploration:
			under_exploration = True
			# ===== Let subclass to implement exploration ===== #
			action = target_model.explore_action(action_space)
		else:
			Q_values = target_model.evaluate_Q_values(s)
			action = target_model.action_from_Q_values(q_values=Q_values)

		return action, under_exploration

	def add_experience(self, s, a, r, s_n, done):
		self.episode_experience.append((s, a, r, s_n, done))
		self.iterations += 1
		if self.iterations % 1000 == 0:
			print('>>> Iterations: ', self.iterations)
			print('>>> Exploration: ', self.exploration)
		# Anneal exploration factor
		self.exploration = - (self.init_exploration - 0.1) * self.iterations / self.max_iter + self.init_exploration
		if self.exploration < 0.1:
			self.exploration = 0.1
		if (len(self.episode_experience) >= self.max_steps) or done:
			self.update()
			self.reset()

	def update(self):
		experiences = self.episode_experience
		experiences.reverse()
		targets = list()
		inputs = list()
		for (s, a, r, s_n, done) in experiences:
			if done: # s_n is terminal state...
				R = r
			else:
				#  Pass through network, obtain max value for next state
				next_val_predictions = self.evaluate_Q_values(s_n)
				next_val_predictions = next_val_predictions[0]
				R = next_val_predictions[np.argmax(next_val_predictions)]
			inputs.append(s)
			curr_val_predictions = self.evaluate_Q_values(s)
			curr_val_predictions = curr_val_predictions[0]
			target = curr_val_predictions
			target[a] = r + self.discount_factor * R
			targets.append(target)
		self.model.fit(
			x=np.asarray(inputs),
			y=np.asarray(targets),
			nb_epoch=5,
			verbose=0
		)

	def reset(self):
		self.episode_experience = list()

	def final_action(self, s):
		q_values = self.evaluate_Q_values(s)
		return self.action_from_Q_values(q_values)

	def evaluate_Q_values(self, s):
		Q_values = self.model.predict_on_batch(
			x=np.asarray([s])
		)  # Q values.
		return Q_values

	def copy_params(self, other_network):
		self.model.set_weights(other_network.model.get_weights())

	def explore_action(self, action_space):
		# TODO: To be implemented by subclass
		pass

	def action_from_Q_values(self, q_values):
		# TODO: To be implemented by subclass
		pass

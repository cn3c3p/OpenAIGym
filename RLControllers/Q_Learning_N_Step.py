import time
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential

import math
import numpy as np
from random import shuffle
import pickle as pickle

class Network():

	def __init__(self, max_steps,
				 dense_layers,
				 num_features,
				 num_actions,
				 max_steps,
				 activation='linear',
				 loss='mse',
				 update_num=10,
				 batch_size=32,
				 discount_factor=0.9,
				 exploration=1.0,
				 max_iter=100000):


		self.max_steps = max_steps
		self.update_num = update_num
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.exploration = exploration
		self.init_exploration = exploration
		self.episode_experience = list()
		self.max_iter = max_iter

		# ====== BUILD MODEL ====== #

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

		def add_experience(self, s, a, r, s_n, done):
			self.episode_experience.append((s, a, r, s_n, done))
			if (len(self.episode_experience) > self.max_steps):
				self.update()
				self.reset()

		def update(self):
			experiences = self.episode_experience
			experiences.reverse()
			R = 0
			for (s, a, r, s_n, done) in experiences:
				if done:
					R = 0


		def reset(self):
			self.episode_experience = list()

		def evaluate_Q_values(self, s):
			Q_values = self.model.predict_on_batch(
				x=np.asarray([s])
			)  # Q values.
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
from keras.models import Model, load_model
from keras.layers import Input, Dense, merge, RepeatVector, Reshape, Dropout
from keras.backend import repeat_elements, sum
from keras.optimizers import rmsprop, Adam
from keras.regularizers import l2, activity_l2

import numpy as np

import sys
maj = sys.version_info
version = 2

if maj[0] >= 3:
	import _pickle as pickle
	import importlib.machinery
	import types
	version = 3
else:
	import cPickle as pickle
	import imp

import os
import re

cwd = os.getcwd()
split_str = '\\\\+|\/'
while True:
	tokens = re.split(split_str , cwd)
	if tokens[-1] == 'OpenAIGym':
		break
	else:
		l = tokens[:-1]
		s = '/'
		cwd = s.join(l)

data_path = cwd

if version == 3:
	loader = importlib.machinery.SourceFileLoader('Layer_Advantage', cwd + '/RLControllers/Layer_Advantage.py')
	module = types.ModuleType(loader.name)
	loader.exec_module(module)

else:
	module = imp.load_source('Layer_Advantage', cwd + '/RLControllers/Layer_Advantage.py')


class Dueling_Q_Network():

	def __init__(self, adv_layers, val_layers, num_features, num_actions, update_num, batch_size, max_iter, experience_length, discount_factor=0.9, exploration=1.0, f_name=None, learning_rate=0.001):

		self.exploration = exploration

		self.update_num = update_num
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.init_exp = exploration
		self.max_iter = max_iter
		self.iterations = 0
		self.experiences = list()
		self.experience_length = experience_length

		self.loaded = False
		if f_name:
			self.model = load_model(f_name)
			self.loaded = True


		if not self.loaded:
			# Common Layers
			input_layer = Input(shape=(num_features,))

			# common_layer = input_layer
			#
			# for layer in dense_layers:
			# 	common_layer = Dense(
			# 		output_dim=layer,
			# 		activation='relu'
			# 	)(common_layer)
			#
			# # Advantage layers
			# adv_layer = common_layer

			adv_layer = input_layer
			val_layer = input_layer

			for layer in adv_layers:
				adv_layer = Dense(
					output_dim=layer,
					activation='relu',
					W_regularizer=l2(0.01)
				)(adv_layer)
				adv_layer = Dropout(
					p=0.2
				)(adv_layer)

			# Advantage Output
			adv_layer = module.Advantage_Layer(
				output_dim=num_actions,
				activation='linear'
			)(adv_layer)

			# Value Layers
			for layer in val_layers:
				val_layer = Dense(
					output_dim=layer,
					activation='relu',
				)(val_layer)
				val_layer = Dropout(
					p=0.2
				)(val_layer)
			# Value Output
			val_layer = Dense(
				output_dim=1,
				activation='linear'
			)(val_layer)

			val_layer = RepeatVector(num_actions)(val_layer)
			val_layer = Reshape(target_shape=(num_actions,),input_shape=(num_actions,1,))(val_layer)
			merge_layer = merge(
				inputs=[adv_layer, val_layer],
				mode='sum'
			)

			self.model = Model(
				input=[input_layer],
				output=[merge_layer]
			)

			self.model.compile(
				optimizer=Adam(lr=learning_rate),
				loss='mse'
			)

			print('network layers: ')
			print(self.model.summary())

			print('network inputs: ')
			print(self.model.input_shape)

			print('network outputs: ')
			print(self.model.output_shape)

	def propose_action(self, s, action_space):
		p = np.random.random()
		under_exploration = False
		if p < self.exploration:
			under_exploration = True
			# ===== Let subclass to implement exploration ===== #
			action = self.explore_action(s, action_space)
		else:
			Q_values = self.evaluate_q_values(s)
			action = self.action_from_Q_values(Q_values)
		return action, under_exploration

	def add_experience(self, s, a, r, s_n, done, target_net=None):
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
			self.update(target_net)

	def update(self, Q_net=None):
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

		# Double Deep Q
		Q_curr = self.model.predict_on_batch(
			x=np.asarray(states)
		) # Q(s,a)

		Q_next_me = self.model.predict_on_batch(
			x=np.asarray(states_next)
		) # Q(s', a')

		if not Q_net:
			Q_next = self.model.predict_on_batch(
				x=np.asarray(states_next)
			) # Q'(s', a')
		else:
			Q_next = Q_net.model.predict_on_batch(
				x=np.asarray(states_next)
			) # Q(s', a')

		next_max_as = [np.argmax(values) for values in Q_next_me] # max_a Q(s', a)

		max_Q_next = [values[max_a] for values, max_a in zip(Q_next, next_max_as)] # Q'(s', max_a Q(s',a))

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

	def evaluate_q_values(self, s):
		Q_values = self.model.predict_on_batch(
			x=np.asarray([s])
		)  # Q values.
		return Q_values

	def final_action(self, s):
		q_values = self.evaluate_q_values(s)
		return self.action_from_Q_values(q_values)

	def save(self, name):
		self.model.save('Duel_Q_Learning/model' + str(name) + '.h5')

	def copy_params(self, network):
		self.model.set_weights(network.model.get_weights())


	def explore_action(self, state, action_space):
		# TODO: To be implemented by subclass
		pass

	def action_from_Q_values(self, q_values):
		# TODO: To be implemented by subclass
		pass

if __name__ == '__main__':
	network = Dueling_Q_Network(
		adv_layers=[512],
		val_layers=[512],
		num_features=8,
		num_actions=4,
		update_num=10,
		batch_size=512,
		max_iter=100,
		experience_length=100,
		learning_rate=0.01
	)
import time
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model

import math
import numpy as np
from random import shuffle
import pickle as pickle

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

if version == 3:
	loader = importlib.machinery.SourceFileLoader('ActorCritic_Separate_DNN', '../RLControllers/ActorCritic_Separate_DNN.py')
	module = types.ModuleType(loader.name)
	loader.exec_module(module)

else:
	module = imp.load_source('ActorCritic_Separate_DNN', '../RLControllers/ActorCritic_Separate_DNN.py')


class ActorCriticDNN(module.ActorCriticDNN):

	def __init__(self,
				 actor_layers, critic_layers,
				 num_action_output, num_features,
				 mode='stochastic',
				 actor_exploration = 0.9,
				 learning_rate=0.003,
				 update_num=10,
				 batch_size=512,
				 discount_factor=0.9,
				 load_actor_model=None,
				 load_critic_model=None):
		module.ActorCriticDNN.__init__(self,
										actor_layers,
									   	critic_layers,
										num_action_output,
									   	num_features,
									   	actor_loss='categorical_crossentropy',
									   	actor_activation='softmax',
									   	actor_exploration=actor_exploration,
									    learning_rate=learning_rate,
									   	update_num=update_num,
									   	batch_size=batch_size,
									   	discount_factor=discount_factor,
										actor_buffer_len = 1000,
									   	critic_buffer_len = 1000,
									   	max_iter=140000,
									   	load_actor_model=load_actor_model,
									   	load_critic_model=load_critic_model)

		self.mode = mode

	def explore_action(self, action, action_space):
		# Randomly choose an action.
		action = np.ones(action_space.n)/action_space.n
		return action

	def prepare_action_for_env(self, action):
		# Just guessing for now
		if self.mode == 'stochastic':
			action = np.random.choice(range(0, self.num_action_output), p=action)
		else:
			action = np.argmax(action)
		#action = np.argmax(action)
		return action

	def prepare_action_for_training(self, action):
		# One hot encoding for action
		vec = np.zeros(self.num_action_output)
		vec[action] = 1
		return vec
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
	loader = importlib.machinery.SourceFileLoader('Dueling_Q_Learning', '../RLControllers/Dueling_Q_Learning.py')
	module = types.ModuleType(loader.name)
	loader.exec_module(module)

else:
	module = imp.load_source('Dueling_Q_Learning', '../RLControllers/Dueling_Q_Learning.py')


class network(module.Dueling_Q_Network):

	def __init__(self,
				 adv_layers,
				 val_layers,
				 num_features,
				 num_actions,
				 learning_rate=0.01):
		module.Dueling_Q_Network.__init__(
			self,
			adv_layers,
			val_layers,
			num_features,
			num_actions,
			update_num=10,
			batch_size=512,
			max_iter=60000,
			experience_length=5000,
			discount_factor=0.99,
			exploration=1.0,
			f_name=None,
			learning_rate=learning_rate
		)

	def explore_action(self, state, action_space):
		# Boltzmann Exploration..

		return np.random.choice(range(0, action_space.n))

	def action_from_Q_values(self, q_values):
		return np.argmax(q_values)
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
	loader = importlib.machinery.SourceFileLoader('Q_Learning', '../RLControllers/Q_Learning.py')
	module = types.ModuleType(loader.name)
	loader.exec_module(module)

else:
	module = imp.load_source('Q_Learning', '../RLControllers/Q_Learning.py')

class Network(module.Network):

	mode = 'stochastic'

	def __init__(self, dense_layers, num_features, num_actions):
		module.Network.__init__(self,
								dense_layers,
								num_features,
								num_actions,
								activation='linear',
								loss='mse',
								learning_rate=0.001,
								update_num=5,
								batch_size=250,
								discount_factor=0.99,
								exploration=1.0,
								end_exploration=0.01,
								experience_length=3000,
								max_iter=30000)

	def explore_action(self, state, action_space):
		return np.random.choice(range(0, action_space.n))

	def action_from_Q_values(self, q_values):
		return np.argmax(q_values)

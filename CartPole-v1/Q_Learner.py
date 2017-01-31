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

	def __init__(self, dense_layers, num_features, num_actions):
		module.Network.__init__(self,
								dense_layers,
								num_features,
								num_actions,
								activation='linear',
								loss='mse',
								update_num=10,
								batch_size=32,
								discount_factor=0.9,
								exploration=1.0,
								experience_length=10000,
								max_iter=100000)

	def explore_action(self, action_space):
		# TODO: To be implemented by subclass
		np.choice(np.range(action_space.len))
		pass

	def action_from_Q_values(self, q_values):
		# TODO: To be implemented by subclass
		pass

	def prepare_action_for_training(self, action):
		# TODO: Left for subclass to implement how to prepare action for training of network
		pass
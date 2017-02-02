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
	loader = importlib.machinery.SourceFileLoader('Q_Learning_N_Step', '../RLControllers/Q_Learning_N_Step.py')
	module = types.ModuleType(loader.name)
	loader.exec_module(module)

else:
	module = imp.load_source('Q_Learning_N_Step', '../RLControllers/Q_Learning_N_Step.py')

class Network(module.Network):

	def __init__(self, dense_layers, num_features, num_actions, max_steps, exploration):
		module.Network.__init__(self,
								dense_layers,
								num_features,
								num_actions,
								exploration=exploration,
								max_steps=max_steps,
								activation='linear',
								loss='mse',
								discount_factor=0.9,
								max_iter=80000)

	def explore_action(self, action_space):
		return np.random.choice(range(0, action_space.n))

	def action_from_Q_values(self, q_values):
		return np.argmax(q_values)

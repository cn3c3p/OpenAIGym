from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense
import numpy as np
from keras.backend import mean


class Advantage_Layer(Dense):

	def call(self, x, mask=None):
		advantage = super(Advantage_Layer, self).call(x, mask)
		mean_adv = mean(advantage)
		return advantage - mean_adv
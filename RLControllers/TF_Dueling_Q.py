import tensorflow as tf
import math

class Qnetwork():

	def __init__(self, num_features, num_actions):
		# The network recieves a frame from the game, flattened into an array.
		self.input = tf.placeholder(shape=[None, num_features], dtype=tf.float32)

		# Common layers
		input_weights = tf.Variable(
			tf.truncated_normal(shape=(num_features, 128),
								stddev=1.0/math.sqrt(num_features))
		)

		hidden_weights = tf.Variable(
			tf.truncated_normal(shape=(128, 128),
								stddev=1.0/math.sqrt(num_features))
		)

		biases = tf.Variable(
			tf.zeros([128]),
			name='biases'
		)

		hidden_1 = tf.nn.relu(tf.matmul(self.input, input_weights) + biases)
		hidden_2 = tf.nn.relu(tf.matmul(hidden_1, hidden_weights) + biases)

		self.AW = tf.Variable(tf.random_normal([128, num_actions]))
		self.VW = tf.Variable(tf.random_normal([128, 1]))

		self.Advantage = tf.matmul(hidden_2, self.AW)
		self.Value = tf.matmul(hidden_2, self.VW)

		# Then combine them together to get our final Q-values.
		self.Qout = self.Value + tf.sub(self.Advantage,
										tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
		self.predict = tf.argmax(self.Qout, 1)

		# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

		self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)

		self.td_error = tf.square(self.targetQ - self.Q)
		self.loss = tf.reduce_mean(self.td_error)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.updateModel = self.trainer.minimize(self.loss)

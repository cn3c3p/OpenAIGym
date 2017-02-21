import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import time

import os

import sys
maj = sys.version_info
version = 2

import tensorflow as tf

if maj[0] >= 3:
	import _pickle as pickle
	import importlib.machinery
	import types
	version = 3
else:
	import cPickle as pickle
	import imp

import re

import random

# Append path to ffmpeg binary
os.environ['PATH'] += os.pathsep + '/usr/local/bin'

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
	loader = importlib.machinery.SourceFileLoader('TF_Dueling_Q', cwd + '/RLControllers/TF_Dueling_Q.py')
	module = types.ModuleType(loader.name)
	loader.exec_module(module)

else:
	module = imp.load_source('TF_Dueling_Q', cwd + '/RLControllers/TF_Dueling_Q.py')

class experience_buffer():
	def __init__(self, buffer_size=50000):
		self.buffer = []
		self.buffer_size = buffer_size

	def add(self, experience):
		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
		self.buffer.extend(experience)

	def sample(self, size):
		return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars/2]):
		op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)


batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 15000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 1000 #How many steps of random actions before training begins.
max_epLength = 1000000 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./Dueling Q" #The path to save our model to.
tau = 0.001 #Rate to update target network toward primary network

eval_mode = False
final_form = False


if __name__ == '__main__':

	tf.reset_default_graph()
	mainQN = module.Qnetwork(num_features=4, num_actions=2)
	targetQN = module.Qnetwork(num_features=4, num_actions=2)

	init = tf.initialize_all_variables()

	saver = tf.train.Saver()

	trainables = tf.trainable_variables()

	targetOps = updateTargetGraph(trainables, tau)

	myBuffer = experience_buffer()

	# Set the rate of random action decrease.
	e = startE
	stepDrop = (startE - endE) / anneling_steps

	# create lists to contain total rewards and steps per episode
	jList = []
	rList = []
	total_steps = 0

	# Make a path for our model to be saved in.
	if not os.path.exists(path):
		os.makedirs(path)

	env = gym.make('CartPole-v1')
	env = wrappers.Monitor(env, './CartPole-v1-exp-Q-Learner', force=True)

	tick = 1

	plt.ion()
	plt.figure(1)
	plt.subplot(211)
	plt.title('Cumulative Reward')
	plt.subplot(212)
	plt.title('100 episode mean Reward')

	rewards = list()
	final_form = False
	goal_reached = False
	i_episode = 0

	max_avg_reward = 0

	with tf.Session() as sess:
		if load_model == True:
			print 'Loading Model...'
			ckpt = tf.train.get_checkpoint_state(path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		sess.run(init)
		updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
		for i in range(num_episodes):
			episodeBuffer = experience_buffer()
			# Reset environment and get first new observation
			s = env.reset()
			d = False
			rAll = 0
			j = 0
			# The Q-Network
			while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
				j += 1

				if num_episodes % 50 == 0:
					eval_mode = True
				else:
					eval_mode = False

				if eval_mode:
					a = sess.run(mainQN.predict, feed_dict={mainQN.input: [s]})[0]
				elif np.random.rand(1) < e or total_steps < pre_train_steps:
					a = np.random.randint(0, env.action_space.n)
				else:
					a = sess.run(mainQN.predict, feed_dict={mainQN.input: [s]})[0]
				s1, r, d, _ = env.step(a)
				env.render()
				total_steps += 1

				if d:
					r = -1

				episodeBuffer.add(
					np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

				if total_steps > pre_train_steps:
					if e > endE:
						e -= stepDrop

					if total_steps % (update_freq) == 0:
						trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
						# Below we perform the Double-DQN update to the target Q-values
						Q1 = sess.run(mainQN.predict, feed_dict={mainQN.input: np.vstack(trainBatch[:, 3])}) # a'
						Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.input: np.vstack(trainBatch[:, 3])}) # Q(s',a)
						end_multiplier = -(trainBatch[:, 4] - 1)
						doubleQ = Q2[range(batch_size), Q1]
						targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
						# Update the network with our target values.
						_ = sess.run(mainQN.updateModel,
									 feed_dict={
										 mainQN.input: np.vstack(trainBatch[:, 0]),
										 mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]
									 }
									)
						updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
				rAll += r
				s = s1
				if len(rewards) > 100:
					rewards.pop(0)
				if d:
					break

			# Get all experiences from this episode and discount their rewards.
			myBuffer.add(episodeBuffer.buffer)
			jList.append(j)
			rList.append(rAll)
			if len(rList) > 100:
				rList.pop(0)
			# Periodically save the model.
			if i % 1000 == 0:
				saver.save(sess, path + '/model-' + str(i) + '.cptk')
				print "Saved Model"
			if len(rList) % 10 == 0:
				print total_steps, np.mean(rList[-10:]), e

			plt.subplot(211)
			plt.scatter(i, rAll)
			plt.subplot(212)
			plt.scatter(i, np.mean(rList))
			plt.pause(0.01)
		plt.waitforbuttonpress()
		saver.save(sess, path + '/model-' + str(i) + '.cptk')
	print "Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%"


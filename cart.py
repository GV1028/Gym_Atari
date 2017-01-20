import gym
import tensorflow as tf
from PIL import Image
import numpy as np
import random

class airraid:
	def __init__(self):
		self.nactions = 2
		self.gamma = 0.9
		self.nobs = 1000
		self.currobs = 0
		self.state = 0
		self.imgdir = '/home/vignesh/Desktop/gymtry/images/'
		self.memory = np.zeros((self.nobs, 1, 1, 1, 1, 1))
		self.batch_size = 32	
		self.learning_rate = 1e-6
		self.nb_epochs = 100
		self.sess = tf.Session()
		self.create_placeholders()
		self.convlayers()
		self.fullylayers()
		self.create_environment()
		self.dqn()
	
	def create_placeholders(self):
		self.imgs = tf.placeholder(tf.float32, shape = [None, 250, 160, 3])
		#self.target = tf.placeholder(tf.float32, shape = [self.batch_size])
		
	def convlayers(self):

		with tf.name_scope('conv1') as scope:
			kernel = tf.Variable(tf.truncated_normal([8, 8, 3, 32], dtype = tf.float32, stddev = 1e-1), name = 'weights')
			conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding = 'SAME')
			bias = tf.Variable(tf.constant(0.0, shape = [32], dtype = tf.float32), trainable = True, name = 'biases')
			out = tf.nn.bias_add(conv, bias)
			self.conv1 = tf.nn.relu(out, name = scope)

		with tf.name_scope('pool1') as scope:
			self.pool1 = tf.nn.max_pool(self.conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = scope)

		with tf.name_scope('conv2') as scope:
			kernel = tf.Variable(tf.truncated_normal([4, 4, 32, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding = 'SAME')
			bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
			out = tf.nn.bias_add(conv, bias)
			self.conv2 = tf.nn.relu(out, name = scope)

		with tf.name_scope('pool2') as scope:
			self.pool2 = tf.nn.max_pool(self.conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = scope)
		
		with tf.name_scope('conv3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding = 'SAME')
			bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
			out = tf.nn.bias_add(conv, bias)
			self.conv3 = tf.nn.relu(out, name = scope)

		with tf.name_scope('pool3') as scope:
			self.pool3 = tf.nn.max_pool(self.conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = scope)
	
		with tf.name_scope('conv4') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding = 'SAME')
			bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
			out = tf.nn.bias_add(conv, bias)
			self.conv4 = tf.nn.relu(out, name = scope)

		with tf.name_scope('pool4') as scope:
			self.pool4 = tf.nn.max_pool(self.conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = scope)
		
	def fullylayers(self):

		with tf.name_scope('fc1') as scope:
			shape = int(np.prod(self.pool4.get_shape()[1:]))
			poolflat = tf.reshape(self.pool4, [-1, shape])
			weights_fc = tf.Variable(tf.truncated_normal([shape, 6], dtype=tf.float32, stddev=1e-1), trainable = True, name = 'weights')
			bias_fc = tf.Variable(tf.constant(1.0, shape = [6], dtype = tf.float32), trainable = True, name = 'biases')
			self.fc1 = tf.nn.bias_add(tf.matmul(poolflat, weights_fc), bias_fc)

	def create_environment(self):
		self.env = gym.make('AirRaid-v0')

	def set_experience_relay(self, inital_state, action, reward, final_state, terminal):
		self.memory[self.currobs%self.nobs,0,:,:,:,:] = inital_state
		self.memory[self.currobs%self.nobs,:,0,:,:,:] = action
		self.memory[self.currobs%self.nobs,:,:,0,:,:] = reward
		self.memory[self.currobs%self.nobs,:,:,:,0,:] = final_state
		self.memory[self.currobs%self.nobs,:,:,:,:,0] = terminal
		self.currobs += 1

	def get_experience_relay(self):
		index = np.random.randint(0, self.nobs, self.batch_size)
		return self.memory[index, :, :, :, :, :]


	def save_state(self, img, state_no):
		image = Image.fromarray(img.astype(np.uint8))
		state_no = np.reshape(state_no,(1))
		image.save(self.imgdir + 'img' + str(int(state_no)) + '.jpg')

	def get_state(self, state_no):
		state_no = np.reshape(state_no,(1))
		img = Image.open(self.imgdir + 'img' + str(int(state_no[0])) + '.jpg')
		img = np.array(img)
		return img

	def create_loss(self, y):
		self.loss = tf.reduce_mean(tf.square(y - self.target))

	def create_optim(self):
		self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)	

	def train(self):
		x = self.get_experience_relay()
		x_train = np.zeros((self.batch_size, 250, 160, 3))
		y = np.zeros(self.batch_size, 6)
		target = np.zeros(self.batch_size, 6)
		for i in range(x.shape[0]):
			x_train[i,:,:,:] = self.get_state(x[i,:,:,:,0,:])
			if(x[i,:,:,:,:,0] == 1):
				y[i] = x[i,:,:,0,:,:]
			else:
				rewards = self.sess.run([self.fc1], feed_dict = {self.imgs: np.reshape(self.get_state(x[i,:,:,:,0,:]),(1, 250, 160, 3))})
				y[i] = x[i,:,:,0,:,:] + self.gamma*np.amax(rewards)
			rewards = self.sess.run([self.fc1], feed_dict = {self.imgs:np.reshape(self.get_state(x[i,0,:,:,:,:]),(1, 250, 160, 3))})
			#print rewards
			ind = np.reshape(x[i,:,0,:,:,:],(1))[0]
			#print rewards[0][int(ind)][0] 
			target[i, int(ind)[0]] = rewards[0][int(ind)][0] 	

		self.create_loss(y)
		self.create_optim()
		for i in range(self.nb_epochs):
			_,curr_loss,rewards = self.sess.run([self.optim, self.loss, self.fc1], feed_dict = {self.imgs: x_train})
			print "Iteration: {0} Current Loss: {1} Rewards:{2}".format(i, curr_loss, rewards)
	def dqn(self):
		self.sess.run(tf.global_variables_initializer())
		for i_episode in range(200):
			re = 0
			observation = self.env.reset()
			done = False
			i_count = 0
			while done is not True:
				self.env.render()
				if i_count<10:
					if random.uniform(0.0,1.0)<0.35:
						self.save_state(observation, self.state)
						action = self.env.action_space.sample()
						observation, reward, done, info = self.env.step(action)
						#self.env.render()
						re += reward
						self.set_experience_relay(self.state, action, reward, self.state + 1, done*1)
						self.state = self.state + 1
						self.save_state(observation, self.state)
						i_count += 1
					else:
						self.save_state(observation, self.state)
						observation = np.reshape(observation, (1, 250,160,3))
						rewards = self.sess.run([self.fc1], feed_dict = {self.imgs:observation})
						action = np.argmax(rewards)
						observation, reward, done, info = self.env.step(action)
						#self.env.render()
						re += reward
						self.set_experience_relay(self.state, action, reward, self.state + 1, done*1)
						self.state = self.state + 1
						self.save_state(observation, self.state)
						i_count += 1	
				else:
					self.save_state(observation, self.state)
					observation = np.reshape(observation, (1, 250,160,3))
					rewards = self.sess.run([self.fc1], feed_dict = {self.imgs:observation})
					action = np.argmax(rewards)
					observation, reward, done, info = self.env.step(action)
					#self.env.render()
					re += reward
					self.set_experience_relay(self.state, action, reward, self.state + 1, done*1)
					self.state = self.state + 1
					self.save_state(observation, self.state)
					i_count += 1	 
				if i_episode>0:
					self.train()
			print "Total reward at end of episode {0} is {1}".format(i_episode, re)			 

if __name__ == '__main__':
	airraid()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
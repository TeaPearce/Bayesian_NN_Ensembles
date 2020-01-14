# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# import edward as ed
# from edward.models import Normal, Empirical
from tensorflow_probability import edward2 as ed
import tensorflow_probability as tfp
from scipy.special import erf

import importlib
import utils
importlib.reload(utils)
from utils import *



class hmc_model:
	def __init__(self, activation_fn, data_noise, 
		b_0_var=1., w_0_var=1., u_var=1., g_var=1.,
		hidden_size = 100,
		step_size=0.001, n_steps=40, n_samples=1000, burn_in=200, n_predict=50, deep_NN = False):

		''' create object that will be a Bayesian NN w inference done by HMC '''

		self.name_ = 'hmc_NN_h' + str(hidden_size)
		self.activation_fn = activation_fn
		self.data_noise = data_noise	
		self.hidden_size = hidden_size
		self.deep_NN = deep_NN

		# inference params
		self.step_size = step_size		# size of steps
		self.n_steps = n_steps			# no steps in between samples
		self.n_samples = n_samples		# no samples to collect
		self.burn_in = burn_in			# drop this number of burn in samples
		self.n_predict = n_predict		# take this number of when doing predictions

		if self.n_samples < self.burn_in:
			raise Exception('no. samples is less than burn in samples!')

		if self.deep_NN == True:
			print('going deep...')

		# variance for step fn, relu, erf
		self.b_0_var = b_0_var # first layer bias variance		
		self.w_0_var = w_0_var # first layer weight variance

		# variance for rbf - we use williams 1996 notation
		# i.e. node = exp(-(x-u)^2 / 2*var_g)
		self.g_var = g_var # param of rbf fn (fixed)
		self.u_var = u_var # var of centers, as -> inf, goes to stationary cov dist

		return

	def train(self, X_train, y_train, X_val, is_print=True):
		''' set up BNN and run HMC inference '''

		def neural_network(X):
			# set up the BNN structure using tf

			if self.activation_fn == 'relu':
				h = tf.maximum(tf.matmul(X, W_0) + b_0,0) # relu
			elif self.activation_fn == 'Lrelu':
				a=0.2
				h = tf.maximum(tf.matmul(X, W_0) + b_0,a*(tf.matmul(X, W_0) + b_0)) # leakly relu
			elif self.activation_fn == 'erf':
				h = tf.erf(tf.matmul(X, W_0) + b_0)
			elif self.activation_fn == 'tanh':
				h = tf.tanh(tf.matmul(X, W_0) + b_0)
				# h = tf.tanh(1.23*tf.matmul(X, W_0) + b_0) # add 1.23 for close to GP erf
			elif self.activation_fn == 'sigmoid':
				h = tf.sigmoid(tf.matmul(X, W_0) + b_0)
			elif self.activation_fn == 'softplus':
				self.c=2. # if this is bigger -> relu behaviour, but less 'soft'
				h = tf.divide(tf.log(tf.exp(tf.multiply(tf.matmul(X, W_0) + b_0,c)) + 1),c)
			elif self.activation_fn == 'rbf':
				self.beta_2 = 1/(2*self.g_var)
				h = tf.exp(-self.beta_2*tf.square(X - W_0))

			h = tf.matmul(h, W_1) #+ b_1
			return tf.reshape(h, [-1])

		def neural_network_deep(X):
			# set up the BNN structure using tf

			if self.activation_fn == 'relu':
				h1 = tf.maximum(tf.matmul(X, W_0) + b_0,0) # relu
				h = tf.maximum(tf.matmul(h1, W_1) + b_1,0) # relu
			elif self.activation_fn == 'Lrelu':
				a=0.2
				h1 = tf.maximum(tf.matmul(X, W_0) + b_0,a*(tf.matmul(X, W_0) + b_0)) # leakly relu
				h = tf.maximum(tf.matmul(h1, W_1) + b_1,a*(tf.matmul(h1, W_1) + b_1)) # leakly relu
			elif self.activation_fn == 'erf':
				h1 = tf.erf(tf.matmul(X, W_0) + b_0)
				h = tf.erf(tf.matmul(h1, W_1) + b_1)
			else:
				raise Exception('tp: activation not implemented')

			h = tf.matmul(h, W_2) #+ b_2
			return tf.reshape(h, [-1])

		if self.activation_fn == 'relu' or self.activation_fn == 'softplus' or self.activation_fn == 'Lrelu': 
			init_stddev_0_w = np.sqrt(self.w_0_var) # /d_in
			init_stddev_0_b = np.sqrt(self.b_0_var) # /d_in
			init_stddev_1_w = 1.0/np.sqrt(self.hidden_size) #*np.sqrt(10) # 2nd layer init. dist
		elif self.activation_fn == 'tanh' or self.activation_fn == 'erf': 
			init_stddev_0_w = np.sqrt(self.w_0_var) # 1st layer init. dist for weights
			init_stddev_0_b = np.sqrt(self.b_0_var) # for bias
			init_stddev_1_w = 1.0/np.sqrt(self.hidden_size) # 2nd layer init. dist
		elif self.activation_fn == 'rbf':
			init_stddev_0_w = np.sqrt(self.u_var)		# centres = sig_u
			init_stddev_0_b = np.sqrt(self.g_var) 		# fixed /beta
			init_stddev_1_w = 1.0/np.sqrt(self.hidden_size) # 2nd layer init. dist


		n = X_train.shape[0]
		X_dim = X_train.shape[1]
		y_dim = 1 #y_train.shape[1]

		with tf.name_scope("model"):
			W_0 = Normal(loc=tf.zeros([X_dim, self.hidden_size]), scale=init_stddev_0_w*tf.ones([X_dim, self.hidden_size]),
				name="W_0")
			if self.deep_NN == False:
				W_1 = Normal(loc=tf.zeros([self.hidden_size, y_dim]), scale=init_stddev_1_w*tf.ones([self.hidden_size, y_dim]), 
					name="W_1")
				b_0 = Normal(loc=tf.zeros(self.hidden_size), scale=init_stddev_0_b*tf.ones(self.hidden_size), 
					name="b_0")
				b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1), 
					name="b_1")
			else:
				W_1 = Normal(loc=tf.zeros([self.hidden_size, self.hidden_size]), scale=init_stddev_1_w*tf.ones([self.hidden_size, y_dim]), 
					name="W_1")
				b_0 = Normal(loc=tf.zeros(self.hidden_size), scale=init_stddev_0_b*tf.ones(self.hidden_size), 
					name="b_0")
				W_2 = Normal(loc=tf.zeros([self.hidden_size, y_dim]), scale=init_stddev_1_w*tf.ones([self.hidden_size, y_dim]), 
					name="W_2")
				b_1 = Normal(loc=tf.zeros(self.hidden_size), scale=init_stddev_1_w*tf.ones(self.hidden_size), 
					name="b_1")
				b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), 
					name="b_2")

			X = tf.placeholder(tf.float32, [n, X_dim], name="X")
			if self.deep_NN == False:
				y = Normal(loc=neural_network(X), scale=np.sqrt(self.data_noise) * tf.ones(n), name="y")
			else:
				y = Normal(loc=neural_network_deep(X), scale=np.sqrt(self.data_noise) * tf.ones(n), name="y")
		# inference
		if self.deep_NN == False:
			qW_0 = Empirical(tf.Variable(tf.zeros([self.n_samples, X_dim, self.hidden_size])))
			qW_1 = Empirical(tf.Variable(tf.zeros([self.n_samples, self.hidden_size, y_dim])))
			qb_0 = Empirical(tf.Variable(tf.zeros([self.n_samples, self.hidden_size])))
			qb_1 = Empirical(tf.Variable(tf.zeros([self.n_samples, y_dim])))
		else:
			qW_0 = Empirical(tf.Variable(tf.zeros([self.n_samples, X_dim, self.hidden_size])))
			qW_1 = Empirical(tf.Variable(tf.zeros([self.n_samples, self.hidden_size, self.hidden_size])))
			qW_2 = Empirical(tf.Variable(tf.zeros([self.n_samples, self.hidden_size, y_dim])))
			qb_0 = Empirical(tf.Variable(tf.zeros([self.n_samples, self.hidden_size])))
			qb_1 = Empirical(tf.Variable(tf.zeros([self.n_samples, self.hidden_size])))
			qb_2 = Empirical(tf.Variable(tf.zeros([self.n_samples, y_dim])))


		# get some priors
		### !!! TODO, turn this into a proper function
		# X_pred = X_val.astype(np.float32).reshape((X_val.shape[0], 1))
		# self.y_priors = tf.stack([nn_predict(X_pred, W_0.sample(), W_1.sample(),b_0.sample(), b_1.sample())
		# 	for _ in range(10)])

		# Neal 2012
		# Too large a stepsize will result in a very low acceptance rate for states 
		# proposed by simulating trajectories. Too small a stepsize will either waste 
		# computation time, by the same factor as the stepsize is too small, or (worse) 
		# will lead to slow exploration by a random walk,

		# https://stats.stackexchange.com/questions/304942/how-to-set-step-size-in-hamiltonian-monte-carlo
		# If ϵ is too large, then there will be large discretisation error and low acceptance, if ϵ
		# is too small then more expensive leapfrog steps will be required to move large distances.
		# Ideally we want the largest possible value of ϵ
		# that gives reasonable acceptance probability. Unfortunately this may vary for different values of the target variable.
		# A simple heuristic to set this may be to do a preliminary run with fixed L,
		# gradually increasing ϵ until the acceptance probability is at an appropriate level.

		# Setting the trajectory length by trial and error therefore seems necessary. 
		# For a problem thought to be fairly difficult, a trajectory with L = 100 might be a 
		# suitable starting point. If preliminary runs (with a suitable ε; see above) show that HMC 
		# reaches a nearly independent point after only one iteration, a smaller value of L might be 
		# tried next. (Unless these “preliminary” runs are actually sufficient, in which case there is 
		# of course no need to do more runs.) If instead there is high autocorrelation in the run 
		# with L = 100, runs with L = 1000 might be tried next
		# It may also be advisable to randomly sample ϵ
		# and L form suitable ranges to avoid the possibility of having paths that are close to periodic as this would slow mixing.

		if self.deep_NN == False:
			# inference = tfp.mcmc.HamiltonianMonteCarlo({W_0: qW_0, b_0: qb_0,
			# 			 W_1: qW_1, b_1: qb_1}, 
			# 			 data={X: X_train, y: y_train.ravel()})

			hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo( 
					target_log_prob_fn=target_log_prob_fn,
					step_size=self.step_size,
					num_leapfrog_steps=self.n_steps)

			# think this actually runs it
			states, kernels_results = tfp.mcmc.sample_chain(
				num_results=self.n_samples,
				current_state=[qW_0, qb_0, qW_1, qb_1],
				kernel=hmc_kernel,
				num_burnin_steps=self.burn_in)
		else:
			inference = ed.HMC({W_0: qW_0, b_0: qb_0,
						 W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2}, 
						 data={X: X_train, y: y_train.ravel()})
		# inference.run(step_size=self.step_size, n_steps=self.n_steps) # logdir='log'

		max_steps = 10000  # number of training iterations
		model_dir = None  # directory for model checkpoints

		sess = tf.Session()
		summary = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
		start_time = time.time()

		sess.run(tf.global_variables_initializer())
		for step in range(max_steps):
		  start_time = time.time()
		  _, elbo_value = sess.run([train_op, elbo])
		  if step % 500 == 0:
		    duration = time.time() - start_time
		    print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
		        step, elbo_value, duration))
		    summary_str = sess.run(summary)
		    summary_writer.add_summary(summary_str, step)
		    summary_writer.flush()


		# drop first chunk of burn in samples
		if self.deep_NN == False:
			self.qW_0_keep = qW_0.params[self.burn_in:].eval()
			self.qW_1_keep = qW_1.params[self.burn_in:].eval()
			self.qb_0_keep = qb_0.params[self.burn_in:].eval()
			self.qb_1_keep = qb_1.params[self.burn_in:].eval()
		else:
			self.qW_0_keep = qW_0.params[self.burn_in:].eval()
			self.qW_1_keep = qW_1.params[self.burn_in:].eval()
			self.qb_0_keep = qb_0.params[self.burn_in:].eval()
			self.qW_2_keep = qW_2.params[self.burn_in:].eval()
			self.qb_1_keep = qb_1.params[self.burn_in:].eval()
			self.qb_2_keep = qb_2.params[self.burn_in:].eval()

		return


	def predict(self, X_pred):
		''' do predict on new data '''

		def nn_predict_np(X, W_0, W_1, b_0, b_1):
			if self.activation_fn == 'relu':
				h = np.maximum(np.matmul(X, W_0) + b_0,0)
			elif self.activation_fn == 'Lrelu':
				a=0.2
				h = np.maximum(np.matmul(X, W_0) + b_0,a*(np.matmul(X, W_0) + b_0))
			elif self.activation_fn == 'erf':
				h = erf(np.matmul(X, W_0) + b_0)
			elif self.activation_fn == 'softplus':
				h = np.log(1+np.exp(self.c*(np.matmul(X, W_0) + b_0) ))/self.c
			elif self.activation_fn == 'tanh':
				h = np.tanh(np.matmul(X, W_0) + b_0)
			elif self.activation_fn == 'rbf':
				h = np.exp(-self.beta_2*np.square(X - W_0))

			h = np.matmul(h, W_1) #+ b_1
			return np.reshape(h, [-1])

		def nn_predict_np_deep(X, W_0, W_1, W_2, b_0, b_1, b_2):
			if self.activation_fn == 'relu':
				h1 = np.maximum(np.matmul(X, W_0) + b_0,0)
				h = np.maximum(np.matmul(h1, W_1) + b_1,0)
			elif self.activation_fn == 'Lrelu':
				a=0.2
				h1 = np.maximum(np.matmul(X, W_0) + b_0,a*(np.matmul(X, W_0) + b_0))
				h = np.maximum(np.matmul(h1, W_1) + b_1,a*(np.matmul(h, W_1) + b_1))
			elif self.activation_fn == 'erf':
				h1 = erf(np.matmul(X, W_0) + b_0)
				h = erf(np.matmul(h1, W_1) + b_1)
			else:
				raise Exception('tp: other activations not implemented')

			h = np.matmul(h, W_2) #+ b_2
			return np.reshape(h, [-1])

		# predictive sampling with burn in
		y_preds=[]
		print('\nsampling predictions...')
		for _ in range(self.n_predict):
			# if _%5 == 0:
			# 	print('sampling:',_, 'of', self.n_predict)
			if self.n_predict == self.qW_0_keep.shape[0]:
				id = _
			else:
				id = np.random.randint(0,self.qW_0_keep.shape[0]) # sample from posterior
			
			# if sample from same index it will be joint, this is why we don't do sample

			# use np instead of tf to speed up!
			if self.deep_NN == False:
				temp = nn_predict_np(X_pred,self.qW_0_keep[id],self.qW_1_keep[id],self.qb_0_keep[id],self.qb_1_keep[id])
			else:
				temp = nn_predict_np_deep(X_pred,self.qW_0_keep[id],self.qW_1_keep[id],self.qW_2_keep[id],self.qb_0_keep[id],self.qb_1_keep[id],self.qb_2_keep[id])
			y_preds.append(temp)
			
		y_preds = np.array(y_preds)

		y_pred_mu = np.mean(y_preds,axis=0)
		y_pred_std = np.std(y_preds,axis=0)
		y_pred_std = np.sqrt(np.square(y_pred_std) + self.data_noise) # add on data noise

		y_pred_mu = np.atleast_2d(y_pred_mu).T
		y_pred_std = np.atleast_2d(y_pred_std).T

		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std

		return y_preds, y_pred_mu, y_pred_std






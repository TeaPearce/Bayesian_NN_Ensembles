# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Beta, Normal
import edward as ed
from scipy.special import erf

import importlib
import utils
importlib.reload(utils)
from utils import *



class vi_model:
	def __init__(self, activation_fn, data_noise, 
		b_0_var=1., w_0_var=1., u_var=1., g_var=1.,
		hidden_size = 100,
		n_predict=50, n_iter=1000, n_samples_vi=100):

		''' create object that will be a Bayesian NN w inference done by VI '''

		self.name_ = 'vi_NN_h' + str(hidden_size)
		self.activation_fn = activation_fn
		self.data_noise = data_noise	
		self.hidden_size = hidden_size

		# inference params
		self.n_predict = n_predict		# take this number of when doing predictions
		self.n_iter = n_iter
		self.n_samples_vi = n_samples_vi

		# variance for step fn, relu, erf
		self.b_0_var = b_0_var # first layer bias variance		
		self.w_0_var = w_0_var # first layer weight variance

		# variance for rbf - we use williams 1996 notation
		# i.e. node = exp(-(x-u)^2 / 2*var_g)
		self.g_var = g_var # param of rbf fn (fixed)
		self.u_var = u_var # var of centers, as -> inf, goes to stationary cov dist

		return


	def train(self, X_train, y_train, X_val, is_print=True):
		''' set up BNN and run VI inference '''
		# v similar to the HMC training method

		tf.reset_default_graph()

		def neural_network(X, W_0, W_1, b_0, b_1):
			# set up the BNN structure using tf

			if self.activation_fn == 'relu':
				h = tf.maximum(tf.matmul(X, W_0) + b_0,0) # relu
			elif self.activation_fn == 'Lrelu':
				a=0.2
				h = tf.maximum(tf.matmul(X, W_0) + b_0,a* (tf.matmul(X, W_0) + b_0)) # leakly relu
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

		# with tf.name_scope("model"):
		W_0 = Normal(loc=tf.zeros([X_dim, self.hidden_size]), scale=init_stddev_0_w*tf.ones([X_dim, self.hidden_size]),
			name="W_0")
		W_1 = Normal(loc=tf.zeros([self.hidden_size, y_dim]), scale=init_stddev_1_w*tf.ones([self.hidden_size, y_dim]), 
			name="W_1")
		b_0 = Normal(loc=tf.zeros(self.hidden_size), scale=init_stddev_0_b*tf.ones(self.hidden_size), 
			name="b_0")
		b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1), 
			name="b_1")
		X = tf.placeholder(tf.float32, [n, X_dim], name="X")
		y = Normal(loc=neural_network(X, W_0, W_1, b_0, b_1), scale=np.sqrt(self.data_noise) * tf.ones(n), name="y")

		# inference
		qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [X_dim, self.hidden_size], initializer=tf.zeros_initializer),
		              scale= tf.nn.softplus(np.log(np.exp(init_stddev_0_w)-1) * tf.get_variable("qW_0/scale", [X_dim, self.hidden_size], initializer=tf.ones_initializer)))
		qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [self.hidden_size, y_dim], initializer=tf.zeros_initializer),
		              scale= tf.nn.softplus(np.log(np.exp(init_stddev_1_w)-1)* tf.get_variable("qW_1/scale", [self.hidden_size, y_dim], initializer=tf.ones_initializer)))
		qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [self.hidden_size], initializer=tf.zeros_initializer),
		              scale= tf.nn.softplus( np.log(np.exp(init_stddev_0_b)-1) * tf.get_variable("qb_0/scale", [self.hidden_size], initializer=tf.ones_initializer)))
		qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [y_dim], initializer=tf.zeros_initializer),
		              scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [y_dim], initializer=tf.ones_initializer)))

		### !!! TODO, get some priors
		# n_samples: int.
		# Number of samples from variational model for calculating
		# stochastic gradients.

		# inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_train.ravel()})
		inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={X: X_train, y: y_train.ravel()})
		inference.run(n_iter=self.n_iter, n_samples=self.n_samples_vi)

		# save learnt distributions
		self.qW_0 = qW_0
		self.qb_0 = qb_0
		self.qW_1 = qW_1
		self.qb_1 = qb_1

		# self.outputs_vi = tf.stack([neural_network(X_pred, self.qW_0.sample(), self.qW_1.sample(), self.qb_0.sample(), self.qb_1.sample())
		# 	for _ in range(self.n_predict)])
		def outputs_vi_fn(X_pred):
			return tf.stack([neural_network(tf.cast(X_pred, dtype=tf.float32), self.qW_0.sample(), self.qW_1.sample(), self.qb_0.sample(), self.qb_1.sample()) for _ in range(self.n_predict)])
		self.outputs_vi = outputs_vi_fn
		return


	def predict(self, X_pred):
		''' do predict on new data '''
		# again, v similar to hmc method

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

		# this was really slow...
		# y_preds=[]
		# print('\nsampling predictions...')
		# for _ in range(self.n_predict):
		# 	# use np instead of tf to speed up!
		# 	temp = nn_predict_np(X_pred,self.qW_0.sample().eval(),self.qW_1.sample().eval(),self.qb_0.sample().eval(),self.qb_1.sample().eval())
		# 	y_preds.append(temp)

		# made this instead
		y_preds = self.outputs_vi(X_pred).eval()
		y_preds = np.array(y_preds)

		y_pred_mu = np.mean(y_preds,axis=0)
		y_pred_std = np.std(y_preds,axis=0)
		y_pred_std = np.sqrt(np.square(y_pred_std) + self.data_noise) # add on data noise

		y_pred_mu = np.atleast_2d(y_pred_mu).T
		y_pred_std = np.atleast_2d(y_pred_std).T

		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std

		return y_preds, y_pred_mu, y_pred_std






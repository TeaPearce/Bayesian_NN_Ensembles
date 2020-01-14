
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import datetime
from scipy.special import erf

import importlib
import utils
importlib.reload(utils)
from utils import *


class NN():
	def __init__(self, 
		activation_fn, x_dim, y_dim, hidden_size, 
		init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, optimiser_in, n, learning_rate, decay_rate=1.0,
		drop_out=False, deep_NN=False):
		'''set up one single-layer NN'''
		''' unless deep_NN = True, in which case add two layers'''

		self.activation_fn = activation_fn
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.hidden_size = hidden_size
		self.optimiser_in = optimiser_in
		self.n = n
		self.learning_rate = learning_rate
		self.decay_rate = decay_rate
		self.drop_out = drop_out
		self.deep_NN = deep_NN

		if activation_fn == 'tanh':
			fn_use = tf.nn.tanh
		elif activation_fn == 'relu':
			fn_use = tf.nn.relu
		elif activation_fn == 'Lrelu':
			def tp_Lrelu(x): 
				a=0.2
				return tf.maximum(a*x,x)
			# not sure why but this didn't work
			# fn_use = tf.nn.leaky_relu # alpha=0.2 by default
			fn_use = tp_Lrelu
		elif activation_fn == 'erf':
			fn_use = tf.erf
		elif activation_fn == 'softplus':
			def tp_softplus(x):
				# manually adjust so it is more similar to relu
				c=3. # if this is bigger -> relu behaviour, but less 'soft'
				return tf.divide(tf.log(tf.exp(tf.multiply(x,c)) + 1),c)
				# https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
				# to avoid overflow we could do something like if x>30/c, return x
				# return tf.cond(x>30/c, lambda: tf.divide(tf.log(tf.exp(tf.multiply(x,c)) + 1),c), lambda: x)
				# def f1(): return tf.divide(tf.log(tf.exp(tf.multiply(x,c)) + 1),c)
				# def f2(): return x
				# return tf.cond(tf.less(x,30/c), f1,  f2)
			fn_use = tp_softplus

		# used to have these as float32
		self.inputs = tf.placeholder(tf.float64, [None, x_dim], name='inputs')
		self.y_target = tf.placeholder(tf.float64, [None, y_dim], name='target')

		anchor_factor = 1 # could inflate the init. dist. by this
		
		if activation_fn != 'rbf':
			# we use 'Dense' instead of 'dense' - so can access weights more easily
			self.layer_1_w = tf.layers.Dense(hidden_size,
				activation=fn_use, #trainable=False,
				kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=anchor_factor*init_stddev_1_w),
				bias_initializer=tf.random_normal_initializer(mean=0.,stddev=anchor_factor*init_stddev_1_b))
			self.layer_1 = self.layer_1_w.apply(self.inputs)

			if self.drop_out:
				self.layer_1 = tf.layers.dropout(self.layer_1,rate=0.4,training=True)

			self.output_w = tf.layers.Dense(y_dim, 
				activation=None, use_bias=False,
				kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=anchor_factor*init_stddev_2_w))


			if self.deep_NN:
				print('going deep...')
				# add an extra hidden layer

				self.layer_2_w = tf.layers.Dense(hidden_size,
					activation=fn_use, #trainable=False,
					kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=anchor_factor*init_stddev_2_w),
					bias_initializer=tf.random_normal_initializer(mean=0.,stddev=anchor_factor*init_stddev_2_w))
				self.layer_2 = self.layer_2_w.apply(self.layer_1)

				if self.drop_out:
					self.layer_2 = tf.layers.dropout(self.layer_2,rate=0.4,training=True)
				self.output = self.output_w.apply(self.layer_2)
			else:
				self.output = self.output_w.apply(self.layer_1)

		else:
			self.c = tf.Variable(tf.random_normal([x_dim,hidden_size],mean=0.,stddev=anchor_factor*init_stddev_1_w, dtype=tf.float64),trainable=True, dtype=tf.float64)
			self.beta = tf.Variable(initial_value=[init_stddev_1_b],trainable=False, dtype=tf.float64)
			self.beta_2 = tf.pow(2*tf.square(self.beta),-1)
			self.w2_rbf = tf.Variable(tf.random_normal([hidden_size,y_dim],mean=0.,stddev=anchor_factor*init_stddev_2_w, dtype=tf.float64), dtype=tf.float64)

			self.layer_1_rbf = tf.exp(-self.beta_2*tf.square(self.inputs - self.c))
			if self.drop_out:
				self.layer_1_rbf = tf.layers.dropout(self.layer_1_rbf,rate=0.4,training=True)

			self.output = tf.matmul(self.layer_1_rbf,self.w2_rbf)
			# self.output = tf.matmul(tf.exp(-tf.multiply(self.beta_2,tf.square(self.inputs - self.c))),self.w2_rbf)

		self.batch = tf.Variable(0, trainable=False)
		# decayed_learning_rate = learning_rate *
  		#                       decay_rate ^ (global_step / decay_steps)
		self.l_rate_decay = tf.train.exponential_decay(
			  self.learning_rate,                # Base learning rate.
			  global_step=self.batch,  # Current index into the dataset.
			  decay_steps=20,          # Decay step.
			  decay_rate=self.decay_rate,                # Decay rate.
			  staircase=True)
		if optimiser_in == 'adam':
			# self.opt_method = tf.train.AdamOptimizer(learning_rate)
			self.opt_method = tf.train.AdamOptimizer(self.l_rate_decay)
		elif optimiser_in == 'SGD':
			self.opt_method = tf.train.GradientDescentOptimizer(learning_rate)
		elif optimiser_in == 'AdaDel':
			self.opt_method = tf.train.AdadeltaOptimizer(learning_rate)
		elif optimiser_in == 'RMS':
			self.opt_method = tf.train.RMSPropOptimizer(learning_rate)
		elif optimiser_in == 'AdaGrad':
			self.opt_method = tf.train.AdagradOptimizer(learning_rate)


		# self.loss_ = tf.reduce_mean(tf.square(self.y_target - self.output))
		# self.loss_ = 1/self.n * tf.reduce_sum(tf.square(self.y_target - self.output))
		self.loss_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] * tf.reduce_sum(tf.square(self.y_target - self.output))
		# self.mse_loss = 1/self.n * tf.reduce_sum(tf.square(self.y_target - self.output)) # useful for val
		self.mse_loss = 1/tf.shape(self.inputs, out_type=tf.int64)[0] * tf.reduce_sum(tf.square(self.y_target - self.output)) # useful for val
		# self.loss_ = 1/tf.cast(tf.size(self.inputs),tf.float32) * tf.reduce_sum(tf.square(self.y_target - self.output))
		self.optimizer = self.opt_method.minimize(self.loss_, global_step=self.batch)
		return


	def get_weights(self, sess):
		'''method to return current params - yes it rly does seem this hard..'''

		if self.activation_fn != 'rbf':
			ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.output_w.kernel]
		else:
			ops = [self.c, self.beta, self.w2_rbf]
		w1, b1, w2 = sess.run(ops)
		# b2 = sess.run(self.output_w.bias)
		return w1, b1, w2


	def anchor(self, sess, lambda_anchor, regularise=False, unconstrain=False):
		'''method to set loss to account for anchoring'''

		# lambda_anchor=[0,0,0] # hardcode for testing effect of anchoring
		# regularise = True ### hardcode for testing effect of anchoring
		# lambda_anchor = lambda_anchor*0.01
		# print('\nlambda_anchor',lambda_anchor)

		if unconstrain:
			# turn off effect of prior
			lambda_anchor=[0,0,0]
			print('unconstraining!!!')

		if regularise:
			# to do normal regularisation
			print('regularising!!!')
			w1, b1, w2 = self.get_weights(sess)

			self.w1_init, self.b1_init, self.w2_init = np.zeros_like(w1),np.zeros_like(b1),np.zeros_like(w2) # overwrite for normal regulariser
		else:
			# get weights
			w1, b1, w2 = self.get_weights(sess)

			# set around initial params
			self.w1_init, self.b1_init, self.w2_init = w1, b1, w2

		# print('w1_init',self.w1_init)
		# print('b1_init',self.b1_init)
		# print('w2_init',self.w2_init)

		if self.activation_fn != 'rbf':
			# set squared loss around it
			loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
			loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
			loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.output_w.kernel))
		else:
			loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.c))
			loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.beta))
			loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.w2_rbf))

		# combine with original loss
		self.loss_ = self.loss_ + 1/tf.shape(self.inputs, out_type=tf.int64)[0] * loss_anchor 
		# I spent a long time analysing if we need to divide this by n
		# although we should in the eqn, actually tf doesn't repeat loss_anchor
		# n times, so no need!
		# 25 aug, actually I got this wrong - do need to. cost me a lot of time...

		# reset optimiser
		self.optimizer = self.opt_method.minimize(self.loss_, global_step=self.batch)
		return


	def get_weights_deep(self, sess):
		'''method to return current params - yes it rly does seem this hard..'''
		ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.layer_2_w.kernel, self.layer_2_w.bias, self.output_w.kernel]
		w1, b1, w2, b2, w3 = sess.run(ops)
		# b2 = sess.run(self.output_w.bias)
		return w1, b1, w2, b2, w3


	def anchor_deep(self, sess, lambda_anchor, regularise=False, unconstrain=False):
		'''method to set loss to account for anchoring for a deep NN'''

		if unconstrain:
			# turn off effect of prior
			lambda_anchor=[0,0,0,0,0]
			print('unconstraining!!!')

		if regularise:
			# to do normal regularisation
			print('regularising!!!')
			w1, b1, w2, b2, w3 = self.get_weights_deep(sess)

			self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w3_init = np.zeros_like(w1),np.zeros_like(b1),np.zeros_like(w2),np.zeros_like(b2),np.zeros_like(w3)
		else:
			# get weights
			w1, b1, w2, b2, w3 = self.get_weights_deep(sess)

			# set around initial params
			self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w3_init  = w1, b1, w2, b2, w3

		# print('w1_init',self.w1_init)

		if self.activation_fn == 'rbf':
			raise Exception('tp: deep NN not set up for rbf activations')

		# set squared loss around it
		loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
		loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
		loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
		loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))
		loss_anchor += lambda_anchor[2]*tf.reduce_sum(tf.square(self.w3_init - self.output_w.kernel))

		# combine with original loss
		self.loss_ = self.loss_ + 1/tf.shape(self.inputs, out_type=tf.int64)[0] * loss_anchor 
		# I spent a long time analysing if we need to divide this by n
		# although we should in the eqn, actually tf doesn't repeat loss_anchor
		# n times, so no need!
		# 25 aug, actually I got this wrong - do need to. cost me a lot of time...

		# reset optimiser
		self.optimizer = self.opt_method.minimize(self.loss_, global_step=self.batch)
		return

	def predict(self, x, sess):
		feed = {self.inputs: x}
		y_pred = sess.run(self.output, feed_dict=feed)
		return y_pred



class NN_last:
	def __init__(self, 
				activation_fn,
				data_noise, 
				b_0_var=1.0, w_0_var=1.0, u_var=1.0, g_var=1.0,
				optimiser_in = 'adam', 
				learning_rate = 0.001, 
				hidden_size = 100, 
				n_epochs = 100, 
				cycle_print = 10, 
				n_ensembles = 3, 
				regularise = False,
				unconstrain = False,
				drop_out = False,
				deep_NN = False,
				batch_size = 32,
				total_trained=0,
				decay_rate=1.0
				):
		''' create object that will do analytical inference in last weight layer of NN '''
		''' our starting point for this was the ensemble module, so it won't be the 
		cleanest code, sorry! '''

		self.activation_fn = activation_fn
		self.data_noise = data_noise
		self.optimiser_in = optimiser_in
		self.learning_rate = learning_rate
		self.decay_rate = decay_rate
		self.hidden_size = hidden_size
		self.n_epochs = n_epochs
		self.cycle_print = cycle_print
		self.n_ensembles = 1
		self.regularise = regularise 	# regularise around zero, not anchor
		self.unconstrain = unconstrain 	# set regularisation lambdas to zero
		self.drop_out = drop_out		# use dropout for training and test time
		self.deep_NN = deep_NN 			# use more than one hidden layer
		self.total_trained = total_trained
		
		self.batch_size = batch_size
		self.drop_out = drop_out

		if self.drop_out:
			self.name_ = 'NN_drop_h' + str(hidden_size) + '_ens' + str(n_ensembles+total_trained)
		elif self.regularise:
			self.name_ = 'NN_regular_h' + str(hidden_size) + '_ens' + str(n_ensembles+total_trained)
		elif self.unconstrain:
			self.name_ = 'NN_uncons_h' + str(hidden_size) + '_ens' + str(n_ensembles+total_trained)
		elif self.deep_NN:
			self.name_ = 'NN_deepanch_h' + str(hidden_size) + '_ens' + str(n_ensembles+total_trained)
		else:
			self.name_ = 'NN_anch_h' + str(hidden_size) + '_ens' + str(n_ensembles+total_trained)

		# variance for step fn, relu, erf
		self.b_0_var = b_0_var # first layer bias variance		
		self.w_0_var = w_0_var # first layer weight variance

		# variance for rbf - we use williams 1996 notation
		# i.e. node = exp(-(x-u)^2 / 2*var_g)
		self.g_var = g_var # param of rbf fn (fixed)
		self.u_var = u_var # var of centers, as -> inf, goes to stationary cov dist

		# place holders
		self.mse_unnorm = 0.
		self.rmse = 0.
		self.nll = 0.
		return

	def nn_last_layer_np(self, X, W_0, W_1, b_0):
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

		# just ignore last layer
		# h = np.matmul(h, W_1) #+ b_1
		return h

	def nn_last_layer_np_deep(self, X, W_0, W_1, W_2, b_0, b_1):
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

		return h


	def train(self, X_train, y_train, X_val=None, y_val=None, is_print=True):
		''' train an ensemble of NNs '''
		# note we use different notation in this file, 
		# so b_1 is first bias - elsewhere we call this b_0

		if self.activation_fn == 'relu' or self.activation_fn == 'softplus' or self.activation_fn == 'Lrelu': 
			init_stddev_1_w = np.sqrt(self.w_0_var) # /np.sqrt(self.hidden_size)
			init_stddev_1_b = np.sqrt(self.b_0_var) # /np.sqrt(self.hidden_size)
			init_stddev_2_w = 1.0/np.sqrt(self.hidden_size)#*np.sqrt(10) # 2nd layer init. dist
			lambda_anchor = self.data_noise/(np.array([init_stddev_1_w,init_stddev_1_b,init_stddev_2_w*1])**2)#/X_train.shape[0]
			# lambda_anchor = [0.,0.,0.]
		elif self.activation_fn == 'tanh' or self.activation_fn == 'erf': 
			init_stddev_1_w = np.sqrt(self.w_0_var) # 1st layer init. dist for weights
			init_stddev_1_b = np.sqrt(self.b_0_var) # for bias
			init_stddev_2_w = 1.0/np.sqrt(self.hidden_size) # 2nd layer init. dist
			# lambda_anchor = [0.,0.,0.] # lambda for weight layer 1, bias layer 1, weight layer 2
			lambda_anchor = self.data_noise/(np.array([init_stddev_1_w,init_stddev_1_b,init_stddev_2_w])**2)
		elif self.activation_fn == 'rbf': 
			init_stddev_1_w = np.sqrt(self.u_var)		# centres = sig_u
			init_stddev_1_b = np.sqrt(self.g_var) 		# fixed /beta
			init_stddev_2_w = 1.0/np.sqrt(self.hidden_size) # 2nd layer init. dist
			lambda_anchor = self.data_noise/(np.array([init_stddev_1_w,init_stddev_1_b,init_stddev_2_w])**2)

		n = X_train.shape[0]
		X_dim = X_train.shape[1]
		y_dim = 1 #y_train.shape[1]
		# batch_size = n

		# --- ensembles w proper anchoring! ---
		NNs=[]
		y_pred=[]
		y_prior=[]
		tf.reset_default_graph()
		sess = tf.Session()	
		for ens in range(0,self.n_ensembles):

			if is_print:
				print('\n\n-- working on ensemble number '+ str(self.total_trained + ens) + ' ---')
			else:
				print('-- working on ensemble number '+ str(self.total_trained + ens) + ' ---', end='\r')

			# create a NN
			NNs.append(NN(self.activation_fn, X_dim, y_dim, self.hidden_size, 
					init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, 
					self.optimiser_in, n, self.learning_rate, decay_rate=self.decay_rate, drop_out=self.drop_out, deep_NN=self.deep_NN))
			# sess.run(tf.global_variables_initializer()) # must do this after NN created
			# sess.run(tf.initialize_variables([NNs[ens].layer_1_w.kernel, NNs[ens].layer_1_w.bias, NNs[ens].output_w.kernel]))

			# initialise only unitialized variables
			global_vars = tf.global_variables()
			is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
			not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
			if len(not_initialized_vars):
				sess.run(tf.variables_initializer(not_initialized_vars))

			# set anchoring
			if self.deep_NN == False:
				NNs[ens].anchor(sess, lambda_anchor, regularise=self.regularise, unconstrain=self.unconstrain)
			else:
				NNs[ens].anchor_deep(sess, lambda_anchor, regularise=self.regularise, unconstrain=self.unconstrain)

			# prior
			# if X_val != None:
			y_prior.append(NNs[ens].predict(X_val, sess))

			# do training
			feed = {}
			feed[NNs[ens].inputs] = X_train
			feed[NNs[ens].y_target] = y_train
			# feed[NNs[ens].l_rate_in] = 0.1
			# print('\n\nhhhhh\n\n\n')
			# print(self.learning_rate)

			# if (X_val!=None)[0,0]:
			feed_val = {}
			feed_val[NNs[ens].inputs] = X_val
			feed_val[NNs[ens].y_target] = y_val
			# feed_val[NNs[ens].l_rate_in] = self.learning_rate

			ep_ = 0; train_complete=False
			while not train_complete:
			# for ep_ in range(0,self.n_epochs):
				if False and ep_==0:
					# view training as it progresses
					y_pred_temp = NNs[ens].predict(X_val,sess)
					plot_1d_grid(X_val, y_pred_temp, 0.01, X_train, y_train,title='ep '+str(ep_))

				# blank = sess.run(NNs[ens].optimizer, feed_dict=feed)

				# train in batches
				perm = np.random.permutation(X_train.shape[0])
				X_train_shuff = X_train[perm]
				y_train_shuff = y_train[perm]
				n_batches = int(np.ceil(X_train.shape[0]/self.batch_size))
				for b in range(0,n_batches):

					# if not final batch
					if b != n_batches:
						X_train_b = X_train_shuff[b*self.batch_size:(b+1)*self.batch_size]
						y_train_b = y_train_shuff[b*self.batch_size:(b+1)*self.batch_size]
					else: # use whatever is left
						X_train_b = X_train_shuff[b*self.batch_size:]
						y_train_b = y_train_shuff[b*self.batch_size:]
					feed_b = {}
					feed_b[NNs[ens].inputs] = X_train_b
					feed_b[NNs[ens].y_target] = y_train_b
					blank = sess.run(NNs[ens].optimizer, feed_dict=feed_b)

				if ep_ % self.cycle_print == 0 or ep_ == self.n_epochs-1:
					if False:
						# view training as it progresses
						y_pred_temp = NNs[ens].predict(X_val,sess)
						plot_1d_grid(X_val, y_pred_temp, 0.01, X_train, y_train,title='ep '+str(ep_))

					loss = sess.run(NNs[ens].loss_, feed_dict=feed)
					# if (X_val!=None)[0,0]:
					loss_val = sess.run(NNs[ens].mse_loss, feed_dict=feed_val)
					l_rate_curr = sess.run(NNs[ens].l_rate_decay, feed_dict=feed_val)

					if is_print:
						print('ep:',ep_,'/', self.n_epochs-1, 'train:',np.round(loss,5), 'val mse:',np.round(loss_val,5), 'lr',np.round(l_rate_curr,5))#, end='\r')
					# useful to do a stability check here
					# if one NN doesnt train perfectly it can mess up whole ensemble
					if ep_ == self.n_epochs-1: # if last run
						# train further if increased since last check
						if (loss - loss_old)/loss > 0.02:
							ep_ = np.max(ep_-int(self.n_epochs/10),0)
							print(' !!! one was unstable !!!, continuing training')
							continue
					loss_old = loss.copy()
				ep_+=1
				if ep_ == self.n_epochs: train_complete=True

			# make prediction - used to do here so don't worry about reinit other NNs
			# but we found a way to only reinit new variables now
			# y_pred.append(NNs[ens].predict(x_s))

		self.NNs = NNs
		self.sess = sess

		# priors
		# if X_val != None:
		y_priors = np.array(y_prior)
		y_priors = y_priors[:,:,0]
		y_prior_mu = np.mean(y_prior,axis=0)
		y_prior_std = np.std(y_prior,axis=0, ddof=1)

		# -- last layer inference here -- above is same as NN_ens (except nn_last_layer fns)

		# first need input two last layer
		if not self.deep_NN:
			w1,b1,w2 = NNs[0].get_weights(sess)
			X_last = self.nn_last_layer_np(X_train, w1, w2, b1)
		else:
			w1,b1,w2,b2,w3 = NNs[0].get_weights_deep(sess)
			X_last = self.nn_last_layer_np_deep(X_train, w1,w2,w3,b1,b2)

		w_last_prior_var = init_stddev_2_w**2 # could inflate this artificially to get closer to true post
		print('w_last_prior_var',w_last_prior_var)

		w_last_post_cov = np.linalg.inv(np.matmul(X_last.T,X_last)/self.data_noise + np.identity(self.hidden_size)/w_last_prior_var) 
		w_last_post_mu = np.matmul(np.matmul(w_last_post_cov, X_last.T),y_train)/self.data_noise

		# print('\nX_last\n',X_last)
		# print('\nw_last_post_cov\n',w_last_post_cov)
		# print('\nw_last_post_mu\n',w_last_post_mu)

		# print('\nw1\n',w1)
		# print('\nb1\n',b1)
		# print('\nw2\n',w2)

		self.w_last_post_cov = w_last_post_cov
		self.w_last_post_mu = w_last_post_mu

		return y_priors, y_prior_mu, y_prior_std


	def predict(self, X_val):

		if not self.deep_NN:
			w1,b1,w2 = self.NNs[0].get_weights(self.sess)
			X_last = self.nn_last_layer_np(X_val, w1, w2, b1)
		else:
			w1,b1,w2,b2,w3 = self.NNs[0].get_weights_deep(self.sess)
			X_last = self.nn_last_layer_np_deep(X_val, w1,w2,w3,b1,b2)

		# numerically sampling
		if False:
			y_pred=[]
			for ens in range(0,20):
				w_last_sample = np.random.multivariate_normal(self.w_last_post_mu.flatten(), self.w_last_post_cov)
				y_pred.append(np.matmul(X_last, np.atleast_2d(w_last_sample).T))
			y_preds = np.array(y_pred)
			y_preds = y_preds[:,:,0]
			y_pred_mu = np.mean(y_preds,axis=0)
			y_pred_std = np.std(y_preds,axis=0, ddof=1)

		# analytically
		y_pred_mu = np.matmul(X_last, np.atleast_2d(self.w_last_post_mu)).flatten()
		y_pred_std=[]
		for i in range(X_last.shape[0]):
			y_pred_var = np.matmul(X_last[i], np.matmul(self.w_last_post_cov,X_last[i].T))
			y_pred_std.append(np.sqrt(y_pred_var))
		y_pred_std = np.array(y_pred_std)
		y_preds = []

		# shapes don't work doing it in matrices...
		# y_pred_var = np.matmul(X_last, np.matmul(self.w_last_post_cov,X_last.T))
		# y_pred_std = np.sqrt(y_pred_var)

		# print('X_last', X_last.shape)
		# print('y_pred_mu', y_pred_mu.shape)
		# print('y_pred_std', y_pred_std.shape)
		# print('self.w_last_post_cov', self.w_last_post_cov.shape)
		# print('w2_sample', w2_sample.shape)
		# print('y_preds', y_preds.shape)
		# print('a', a.shape)

		# add on data noise
		y_pred_std = np.sqrt(np.square(y_pred_std) + self.data_noise)

		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std
		self.y_preds = y_preds

		return y_preds, y_pred_mu, y_pred_std















# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import importlib

import utils
importlib.reload(utils)
from utils import *


# --- step fn kernel ---

def np_k_0(x,x2):
	k = b_0 + w_0*(x*x2)
	return k

def np_step(x,x2):
	# do the kernel for two scalars
	# only appropriate for 1-D
	# return 1 - np.arccos(x * x2 / (np.linalg.norm(x) * np.linalg.norm(x2)))/np.pi
	k_s = np_k_0(x,x2) / np.sqrt( (np_k_0(x,x) * np_k_0(x2,x2)) )
	theta = np.arccos(k_s)
	w_1 = 1.0
	return w_1*(1 - theta/np.pi)

def np_step_kernel(X,X2=None):
	# cho and saul step kernel?
	cov = np.zeros([X.shape[0],X2.shape[0]])
	if X2 is None:
		X2 = X
	for i in range(X.shape[0]):
		for j in range(X2.shape[0]):
			cov[i,j] = np_step(X[i],X2[j])
	return cov



# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class gp_model:
	def __init__(self, kernel_type, data_noise, b_0_var=1., w_0_var=1., u_var=1., g_var=1.):

		self.kernel_type = kernel_type
		self.data_noise = data_noise
		self.name_ = 'GP'

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


	# def __relu_kernel_matrix(self, X, X2=None):
	# 	# this approach is only good for same matrices
	# 	if X2 is None:
	# 		X2 = X

	# 	d_in = X.shape[1]
	# 	K = self.b_0_var + self.w_0_var*np.matmul(X,X2.T) / d_in
	# 	cov = np.zeros([X.shape[0],X2.shape[0]])

	# 	b_1 = 0.
	# 	w_1 = 1.

	# 	for i in range(X.shape[0]):
	# 		for j in range(X2.shape[0]):
	# 			k_x_x = K[i,i]
	# 			k_x2_x2 = K[j,j]
	# 			k_x_x2 = K[i,j]

	# 			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
	# 			theta = np.arccos(k_s)
	# 			x_bar = np.sqrt(k_x_x)
	# 			x2_bar = np.sqrt(k_x2_x2)

	# 			cov[i,j] = b_1 + w_1/(2*np.pi) * x_bar * x2_bar * (np.sin(theta) + (np.pi-theta)*np.cos(theta))

	# 	return cov


	def __relu_kernel(self, X, X2=None):
		# relu kernel from cho and saul, 
		# also Lee 2010 (2018?) helped communicate where to put bias and w_0 variance
		# eq. 6 & 11, Lee, Bahri et al. 2018, could add + b_1

		def relu_inner(x, x2):
			# actually these should be 1/d_in going by Lee. But we leave it normal
			# to be equivalent to our NN ens implementation
			k_x_x = self.b_0_var + self.w_0_var*(np.matmul(x,x.T))/1#/d_in
			k_x2_x2 = self.b_0_var + self.w_0_var*(np.matmul(x2,x2.T))/1
			k_x_x2 = self.b_0_var + self.w_0_var*(np.matmul(x,x2.T))/1

			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
			if k_s>1.0: k_s=1.0 # occasionally get some overflow errors

			theta = np.arccos(k_s)
			
			x_bar = np.sqrt(k_x_x)
			x2_bar = np.sqrt(k_x2_x2)

			w_1 = 1.0 # variance of last layers
			# w_1 = 10.0 
			b_1 = 0.0

			return b_1 + w_1/(2*np.pi) * x_bar * x2_bar * (np.sin(theta) + (np.pi-theta)*np.cos(theta))
			
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(X2.shape[0]):
					cov[i,j] = relu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = relu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov


	def __relu_kernel_tf(self, X, X2=None):
		''' doing a tensorflow version of relu_kernel to try to speed up '''
		### development on hold - not as easy as i thought ###
		
		if X2 is None:
			same_inputs=False
			X2 = X
		else:
			same_inputs=False ### change later ###

		cov = np.zeros([X.shape[0],X2.shape[0]])

		inputs_X = []; inputs_X2 = []
		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					# collect all possible combinations of the inputs
					inputs_X.append(X[i]); inputs_X2.append(X2[j])
			inputs_X = np.array(inputs_X); inputs_X2 = np.array(inputs_X2)

			# print('inputs_X.shape',inputs_X.shape)

			# now set up tf stuff
			tf.reset_default_graph()
			sess = tf.Session()

			# set up computation graph
			tf_inputs_X = tf.placeholder(tf.float32, [None, X.shape[1]], name='X')
			tf_inputs_X2 = tf.placeholder(tf.float32, [None, X.shape[1]], name='X2')

			tf_k_x_x = (self.b_0_var + self.w_0_var*(tf.matmul(tf_inputs_X,tf_inputs_X,transpose_b=True)))
			tf_k_x2_x2 = (self.b_0_var + self.w_0_var*(tf.matmul(tf_inputs_X2,tf_inputs_X2,transpose_b=True)))
			tf_k_x_x2 = (self.b_0_var + self.w_0_var*(tf.matmul(tf_inputs_X,tf_inputs_X2,transpose_b=True)))

			tf_theta = tf.acos(tf_k_x_x2 / tf.sqrt(tf_k_x_x * tf_k_x2_x2))

			w_1 = 1.0 # variance of last layers
			b_1 = 0.0

			tf_output = b_1 + w_1/(2*np.pi) * tf.sqrt(tf_k_x_x) * tf.sqrt(tf_k_x2_x2) * (tf.sin(tf_theta) + (np.pi-tf_theta)*tf.cos(tf_theta))
			
			# pick a component of graph to test the output of
			test=tf_inputs_X
			# test=tf.matmul(tf_inputs_X,tf_inputs_X,transpose_b=True)

			sess.run(tf.global_variables_initializer())
			feed = {}
			feed[tf_inputs_X] = inputs_X
			feed[tf_inputs_X2] = inputs_X2
			cov_out = sess.run(tf_output, feed_dict=feed)
			test_out = sess.run(test, feed_dict=feed)
			sess.close()

			print('cov_out',cov_out)
			print('cov_out.shape',cov_out.shape)
			print('\n\ntest_out.shape',test_out.shape)
			print('\ntest_out',test_out)

			# now repack the cov matrix in same order as inputs
			k=0
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = cov_out[k]; k+=1

		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = relu_inner(X[i], X2[j])
		return

	def __Lrelu_kernel(self, X, X2=None):
		# leaky relu kernel from Tsuchida, 2018, eq. 6

		def Lrelu_inner(x, x2):
			# actually these should be 1/d_in going by Lee. But we leave it normal
			# to be equivalent to our NN ens implementation
			k_x_x = self.b_0_var + self.w_0_var*(np.matmul(x,x.T))/1#/d_in
			k_x2_x2 = self.b_0_var + self.w_0_var*(np.matmul(x2,x2.T))/1
			k_x_x2 = self.b_0_var + self.w_0_var*(np.matmul(x,x2.T))/1

			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
			theta = np.arccos(k_s)
			
			x_bar = np.sqrt(k_x_x)
			x2_bar = np.sqrt(k_x2_x2)

			w_1 = 1.0 # variance of last layers
			b_1 = 0.0

			a = 0.2 # leaky param

			return b_1 + w_1 * x_bar * x2_bar * ( np.square(1-a)/(2*np.pi) * (np.sin(theta) + (np.pi-theta)*np.cos(theta)) + a*np.cos(theta))
			
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = Lrelu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = Lrelu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov


	def __erf_kernel(self, X, X2=None):
		# erf kernel from Williams 1996, eq. 11

		def erf_inner(x,x2):	
			# actually these should be 1/d_in
			k_x_x = 2*(self.b_0_var + self.w_0_var*(np.matmul(x,x.T)))
			k_x2_x2 = 2*(self.b_0_var + self.w_0_var*(np.matmul(x2,x2.T)))
			k_x_x2 = 2*(self.b_0_var + self.w_0_var*(np.matmul(x,x2.T)))

			a = k_x_x2 / np.sqrt((1+k_x_x)*(1+k_x2_x2))
			
			w_1 = 1.0 # variance of last layers
			b_1 = 0.0

			return b_1 + w_1*2*np.arcsin(a)/np.pi

		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = erf_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = erf_inner(X[i],X2[j])
			# now just reflect - saves recomputing half the matrix
			cov += np.tril(cov,k=-1).T
		return cov


	def __rbf_kernel(self, X, X2=None):
		# rbf kernel from Williams, 1996, eq. 13
		# don't think we use biases here, not sure about input weights

		def rbf_inner(x, x2):
			# do the kernel for two scalars
			# only appropriate for 1-D, for now...

			var_e = 1/(2/self.g_var + 1/self.u_var)
			var_s = 2*self.g_var + (self.g_var**2)/self.u_var
			var_m = 2*self.u_var + self.g_var

			# williams eq 13
			term1 = np.sqrt(var_e/self.u_var)
			term2 = np.exp(-np.matmul(x,x.T)/(2*var_m))
			term3 = np.exp((np.matmul((x-x2),(x2-x).T))/(2*var_s))
			term4 = np.exp(-np.matmul(x2,x2.T)/(2*var_m))

			w_1 = 1.0 # variance of last layers
			b_1 = 0.0

			return b_1 + w_1*term1*term2*term3*term4
			# return only term3 gives stationary
		
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = rbf_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = rbf_inner(X[i],X2[j])
			cov += np.tril(cov,k=-1).T
		return cov


	def run_inference(self, x_train, y_train, x_predict, print=False):
		''' this is why we're here - do inference '''

		if self.kernel_type == 'relu' or self.kernel_type == 'softplus':
			kernel_fn = self.__relu_kernel
			# kernel_fn = self.__relu_kernel_tf
		elif self.kernel_type == 'Lrelu':
			kernel_fn = self.__Lrelu_kernel
		elif self.kernel_type == 'rbf':
			kernel_fn = self.__rbf_kernel
		elif self.kernel_type == 'erf':
			kernel_fn = self.__erf_kernel
		elif self.kernel_type == 'step':
			kernel_fn = step_kernel

		# d is training data, x is test data
		if print: print_w_time('beginning inference')
		cov_dd = kernel_fn(x_train) + np.identity(x_train.shape[0])*self.data_noise
		if print: print_w_time('compiled cov_dd')
		cov_xd = kernel_fn(x_predict, x_train)
		if print: print_w_time('compiled cov_xd')
		cov_xx = kernel_fn(x_predict,x_predict)
		if print: print_w_time('compiled cov_xx')

		# if print: print_w_time('inverting matrix dims: '+ str(cov_dd.shape))
		# cov_dd_inv = np.linalg.inv(cov_dd) # could speed this up w cholesky or lu decomp
		# if print: print_w_time('matrix inverted')

		# cov_pred = cov_xx - np.matmul(np.matmul(cov_xd,cov_dd_inv),cov_xd.T)
		# y_pred_mu = np.matmul(np.matmul(cov_xd,cov_dd_inv),y_train)
		# # y_pred_var = np.atleast_2d(np.diag(cov_pred)).T
		# y_pred_var = np.atleast_2d(np.diag(cov_pred) + self.data_noise).T
		# y_pred_std = np.sqrt(y_pred_var)

		# p 19 of rasmussen
		L = np.linalg.cholesky(cov_dd)
		alpha = np.linalg.solve(L.T,np.linalg.solve(L,y_train))
		y_pred_mu = np.matmul(cov_xd,alpha)
		v = np.linalg.solve(L,cov_xd.T)
		cov_pred = cov_xx - np.matmul(v.T,v)

		y_pred_var = np.atleast_2d(np.diag(cov_pred) + self.data_noise).T
		y_pred_std = np.sqrt(y_pred_var)


		if print: print_w_time('calculating log likelihood')
		# marg_log_like = - np.matmul(y_train.T,np.matmul(cov_dd_inv,y_train))/2 - np.log(np.linalg.det(cov_dd))/2 - x_train.shape[0]*np.log(2*np.pi)/2
		# have problems with this going to zero

		# marg_log_like = - np.matmul(y_train.T,np.matmul(cov_dd_inv,y_train))/2 - np.sum(np.log(np.diag(L))) - x_train.shape[0]*np.log(2*np.pi)/2
		marg_log_like = - np.matmul(y_train.T,alpha)/2 - np.sum(np.log(np.diag(L))) - x_train.shape[0]*np.log(2*np.pi)/2
		
		# print_w_time(L)
		# a = np.sum(np.log(np.diag(L)))
		# print_w_time(a)
		# a = np.matmul(y_train.T,np.matmul(cov_dd_inv,y_train))/2
		# print_w_time(a)
		# a = np.linalg.det(cov_dd) # this goes to zero sometimes...
		# print_w_time(a)
		# a = np.log(np.linalg.det(cov_dd))/2 
		# print_w_time(a)
		if print: print_w_time('matrix ops complete')

		self.cov_xx = cov_xx
		self.cov_dd = cov_dd
		# self.cov_dd_inv = cov_dd_inv
		self.cov_xd = cov_xd
		self.cov_xx = cov_xx
		self.cov_pred = cov_pred
		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std
		self.y_pred_var = y_pred_var
		self.x_train = x_train
		self.y_train = y_train
		self.x_predict = x_predict

		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std
		self.marg_log_like = marg_log_like

		return y_pred_mu, y_pred_std


	def cov_visualise(self):
		''' display heat map of cov matrix over 1-d input '''

		# plot cov matrix
		fig = plt.figure()
		plt.imshow(self.cov_xx, cmap='hot', interpolation='nearest')
		if self.kernel_type != 'rbf':
			title = self.kernel_type + ', cov matrix, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
		else:
			title = self.kernel_type + ', cov matrix, g_var: ' + str(self.g_var) + ', u_var: ' + str(self.u_var)
		plt.title(title)
		plt.colorbar()
		fig.show()

		return


	def priors_visualise(self, n_draws=10):
		# 1-D only, plot priors
		# we currently have data noise included in this, could remove it to get smooth

		print_w_time('getting priors')
		# get some priors
		y_samples_prior = np.random.multivariate_normal(
			np.zeros(self.x_predict.shape[0]), self.cov_xx, n_draws).T # mean, covariance, size

		# plot priors
		fig = plt.figure()
		plt.plot(self.x_predict, y_samples_prior, 'k',lw=0.5, label=u'Priors')
		plt.xlabel('$x$')
		plt.ylabel('$f(x)$')
		title = self.kernel_type + ', priors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
		plt.title(title)
		# plt.xlim(-6, 6)
		fig.show()

		return


	def posts_draw_visualise(self, n_draws=10, is_graph=True):
		# 1-D only, plot posteriors
		# we currently have data noise included in this, could remove it to get smooth

		# sample from posterior
		y_samples_post = np.random.multivariate_normal(
			self.y_pred_mu.ravel(), self.cov_pred, n_draws).T # mean, covariance, size

		# plot priors
		if is_graph:
			fig = plt.figure()
			plt.plot(self.x_predict, y_samples_post, color='k',alpha=0.5,lw=0.5, label=u'Priors')
			plt.plot(self.x_train, self.y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
			plt.xlabel('$x$')
			plt.ylabel('$f(x)$')
			title = self.kernel_type + ', posteriors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
			plt.title(title)
			# plt.xlim(-6, 6)
			fig.show()

		self.y_preds = y_samples_post.T

		y_pred_mu_draws = np.mean(self.y_preds,axis=0)
		y_pred_std_draws = np.std(self.y_preds,axis=0, ddof=1)

		# add on data noise
		# do need to add on for GP!
		y_pred_std_draws = np.sqrt(np.square(y_pred_std_draws) + self.data_noise)

		self.y_pred_mu_draws = y_pred_mu_draws
		self.y_pred_std_draws = y_pred_std_draws

		return






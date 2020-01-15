from __future__ import print_function
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
import pandas as pd
import datetime
import pickle

# =================================================================
# Uncertainty in Neural Networks: Approximately Bayesian Ensembling
#
# this code creates figure 6 plot, comparing loss choices for an 
# ensemble for 2D classification task
# =================================================================

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True

n_ensemble = 10
# reg = 'anc' # type of regularisation to use - anc reg free
batch_size = int(128*2)
num_classes = 2
epochs = 100
# activation_in = 'relu'
l_rate = 0.001
decay_rate = 0.0
n_hidden = 50

# options
is_plot_train = 0 # plot loss through training
is_plot_params = 0 # plot parameter distributions pre and post training
verbose_in = 1 # how much text to output to console, 0 is least

is_BNN = 1 # run HMC as ground truth
n_pred_samples = 20 # use 
n_inf_samples = 1000 # collect these samples during inference
drops = int(n_inf_samples/2) # burn ins to manually drop

n_runs = 1
is_save_graph = 0



def lr_schedule(epoch):

	# fashion mnist
	lrate = 0.005

	return lrate


start_time = datetime.datetime.now()
print('\nstart_time:', start_time.strftime('%H:%M:%S'))


from sklearn.datasets import make_blobs
x_train, y_train = make_blobs(n_samples=30, centers=2, n_features=2,random_state=0)

x_test = x_train.copy()
y_test = y_train.copy()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_mu = np.mean(x_train)
x_std = np.std(x_train)
x_train = (x_train - x_mu)/x_std
x_test = (x_test - x_mu)/x_std

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

n_in=x_train.shape[1]
num_classes=y_train.shape[1]

x_range = 5
y_range = 3.4
n_points = 20
x = np.linspace(-x_range,x_range,n_points+1)
y = np.linspace(-y_range,y_range,n_points+1)
x_grid_row,x_grid_col = np.meshgrid(x,y)
x_grid = np.vstack([x_grid_row.ravel(), x_grid_col.ravel()]).T

color_in = []
for i in range(y_train.shape[0]):
	if y_train[i,0] == 0:
		# color_in.append('r')
		color_in.append('salmon')
	else:
		# color_in.append('b')
		# color_in.append('dodgerblue')
		color_in.append('deepskyblue')

def np_softmax(z):
	# e = np.exp(z - np.max(z)) # need this to avoid overflow
	return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
	


def fn_my_loss(y_true,y_pred):
	return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

n_out = num_classes
n_train = x_train.shape[0]

def fn_make_NN(reg='none', type='conv_anch', is_conv_train=True, conv_Ws=None, activation_in='relu',seed_in=4):
	np.random.seed(seed_in) 
	model = Sequential()

	# we need to manually choose weight initialisations, so we can set a custom regulariser
	# n_in = 512 # for conv extracted faatures
	# n_in = 3072 # for full cifar
	# n_in = 1152 # with stride [2,2], 32 filters
	# n_in = 2704 # no stride
	# n_in = 800 # stride [2,2], 16 then 32

	# get initialisations, and regularisation values
	W1_var = 15/n_in # flattened, 20 for relu, if use 200 for tanh get noiser fn and catches noise ood better
	W1_lambda = 1/(2*W1_var)
	W1_init = np.random.normal(loc=0,scale=np.sqrt(W1_var),size=[n_in,n_hidden])
	W1_init_b = np.random.normal(loc=0,scale=np.sqrt(W1_var),size=[n_in,n_hidden])

	b1_var = W1_var
	b1_lambda =  1/(2*b1_var) # W1_lambda
	b1_init = np.random.normal(loc=0,scale=np.sqrt(b1_var),size=[n_hidden])
	b1_init_b = np.random.normal(loc=0,scale=np.sqrt(b1_var),size=[n_hidden])

	W2_var = 1/n_hidden
	W2_lambda = 1/(2*W2_var)
	W2_init = np.random.normal(loc=0,scale=np.sqrt(W2_var),size=[n_hidden, n_hidden])
	W2_init_b = np.random.normal(loc=0,scale=np.sqrt(W2_var),size=[n_hidden, n_hidden])

	b2_var = W2_var
	b2_lambda = W2_lambda 
	b2_init = np.random.normal(loc=0,scale=np.sqrt(b2_var),size=[n_hidden])
	b2_init_b = np.random.normal(loc=0,scale=np.sqrt(b2_var),size=[n_hidden])

	# more hidden layers
	W2b_var = W2_var
	W2b_lambda = W2_lambda
	W2b_init = np.random.normal(loc=0,scale=np.sqrt(W2b_var),size=[n_hidden, n_hidden])
	W2b_init_b = np.random.normal(loc=0,scale=np.sqrt(W2b_var),size=[n_hidden, n_hidden])

	b2b_var = b2_var
	b2b_lambda = b2_lambda 
	b2b_init = np.random.normal(loc=0,scale=np.sqrt(b2b_var),size=[n_hidden])
	b2b_init_b = np.random.normal(loc=0,scale=np.sqrt(b2b_var),size=[n_hidden])

	W3_var = 10/n_hidden
	W3_lambda = 1/(2*W2_var) # !!! wrong...
	W3_init = np.random.normal(loc=0,scale=np.sqrt(W3_var),size=[n_hidden, n_out])
	W3_init_b = np.random.normal(loc=0,scale=np.sqrt(W3_var),size=[n_hidden, n_out])

	print('W1_lambda',W1_lambda/n_in)
	print('W2_lambda',W2_lambda/n_train)
	print('W2b_lambda',W2b_lambda/n_train)
	print('W3_lambda',W3_lambda/n_train)

	print('activation_in: ',activation_in)

	def anchored_reg_W1(weight_matrix):
		if reg == 'reg':
			print('regularising')
			return K.sum(K.square(weight_matrix)) * W1_lambda/n_train
		elif reg == 'free':
			print('free')
			return 0.
		elif reg == 'anc':
			print('anchoring')
			return K.sum(K.square(weight_matrix - W1_init)) * W1_lambda/n_train

	def anchored_reg_b1(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b1_lambda/n_train
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b1_init)) * b1_lambda/n_train

	def anchored_reg_W2(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W2_lambda/n_train
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W2_init)) * W2_lambda/n_train

	def anchored_reg_b2(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b2_lambda/n_train
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b2_init)) * b2_lambda/n_train

	def anchored_reg_W2b(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W2b_lambda/n_train
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W2b_init)) * W2b_lambda/n_train

	def anchored_reg_b2b(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * b2b_lambda/n_train
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - b2b_init)) * b2b_lambda/n_train

	def anchored_reg_W3(weight_matrix):
		if reg == 'reg':
			return K.sum(K.square(weight_matrix)) * W3_lambda/n_train
		elif reg == 'free':
			return 0.
		elif reg == 'anc':
			return K.sum(K.square(weight_matrix - W3_init)) * W3_lambda/n_train

	# model.add(Flatten(input_shape=(n_in,)))
	# model.add(Flatten(input_shape=x_train.shape[1:]))

	model.add(Dense(n_hidden, activation=activation_in, input_shape=(n_in,),
		kernel_initializer=keras.initializers.Constant(value=W1_init_b),
		bias_initializer=keras.initializers.Constant(value=b1_init_b),
		kernel_regularizer=anchored_reg_W1,
		bias_regularizer=anchored_reg_b1))

	model.add(Dense(n_hidden, activation=activation_in,
		kernel_initializer=keras.initializers.Constant(value=W2_init_b),
		bias_initializer=keras.initializers.Constant(value=b2_init_b),
		kernel_regularizer=anchored_reg_W2,
		bias_regularizer=anchored_reg_b2))

	# model.add(Dense(n_hidden, activation=activation_in,
	# 	kernel_initializer=keras.initializers.Constant(value=W2b_init_b),
	# 	bias_initializer=keras.initializers.Constant(value=b2b_init_b),
	# 	kernel_regularizer=anchored_reg_W2b,
	# 	bias_regularizer=anchored_reg_b2b))

	model.add(Dense(num_classes, activation='linear',use_bias=False,
		kernel_initializer=keras.initializers.Constant(value=W3_init_b),
		kernel_regularizer=anchored_reg_W3)) # , trainable=False

	# cross entropy is same as log likelihood
	# model.compile(loss=keras.losses.categorical_crossentropy,
	model.compile(loss=fn_my_loss,
			  optimizer=keras.optimizers.Adam(lr=0.0,decay=0.0),
			  # optimizer=keras.optimizers.rmsprop(lr=0.0, decay=0),
			  # optimizer=keras.optimizers.Adadelta(lr=2.0),
			  # metrics=[keras.metrics.categorical_accuracy])
			  # metrics=[acc_logits])
			  # metrics=[keras.metrics.categorical_accuracy])
			  metrics=['accuracy'])
			# even though it calcs accuracy from logits still works as largest logit is largest prob

	return model


def fn_predict_ensemble(NNs, x_test):
	''' fn to predict given a list of NNs (an ensemble)''' 
	y_logit_preds = []
	for m in range(len(NNs)):
		y_logit_preds.append(NNs[m].predict(x_test, verbose=0))
	y_logit_preds = np.array(y_logit_preds)
	y_mean = np.mean(y_logit_preds,axis=0)
	y_var = np.var(y_logit_preds,axis=0)
	# y_prob_preds = np_softmax(y_logit_preds)

	# I need the softmax output to be exactly as in keras
	# I used to use a numpy version but I wasn't certain it was the same
	# also gave me overflow errors
	sess = tf.Session()
	y_prob_preds = sess.run(tf.nn.softmax(y_logit_preds))
	sess.close()

	# now sample from logits to get predicted prob
	# I decided against using the below method, prefer to use y_prob_preds now
	# seems unnecessary as get m samples from it anyway. besides the logits
	# are unidentifiable so the same logit from a different NN isn't necessarily
	# the same distribution, I think would need to consider JOINT distribution
	if False:
		y_logit_samples = np.array([np.random.normal(loc=y_mean,scale=np.sqrt(y_var)) for _ in range(50)])
		# might look better to select a quantile randomly, then follow this the whole way
		# this would reduce noise visually and mean less sampling required
		# could probably compute the integral analytically if used erf activation fn
		y_prob_samples = np_softmax(y_logit_samples)
		y_prob_final = np.mean(y_prob_samples,axis=0)
	else:
		y_prob_final = np.mean(y_prob_preds,axis=0)

	return y_logit_preds, y_mean, y_var, y_prob_preds, y_prob_final


def fn_display_weights(model,title_in=' '):
	params = model.get_weights() # has structure of a list, with w1, b1, w2, b2

	W1_final = params[0].flatten()
	b1_final = params[1].flatten()
	W2_final = params[2].flatten()
	fig = plt.figure(figsize=(6, 4))
	ax = fig.add_subplot(311)
	ax.set_title(title_in)
	ax.hist(W1_final,bins=20)
	ax.set_ylabel('W1')
	ax = fig.add_subplot(312)
	ax.hist(b1_final,bins=20)
	ax.set_ylabel('b1')
	ax = fig.add_subplot(313)
	ax.hist(W2_final,bins=20)
	ax.set_ylabel('W2')
	fig.show()
	return

for reg in ['free','reg','anc']:
	for activation_in in ['relu']: 

		result_NN_acc=[]; result_ens_acc=[]; result_corrupt=[]
		for run in range(n_runs):
			print('\n\n ==== run', run+1, 'of', n_runs,' ====')
			print(reg,activation_in)

			np.random.seed(run) 

			# just ensemble training
			NNs=[]
			for m in range(n_ensemble):
				NNs.append(fn_make_NN(reg=reg, is_conv_train=True, conv_Ws=None, activation_in=activation_in, seed_in=run+m+17))

			if verbose_in != 0:
				print(NNs[-1].summary())

			if is_plot_params:
				fn_display_weights(NNs[-1],title_in='before training')

			NNs_hist_train=[]; NNs_hist_val=[]
			for m in range(n_ensemble):
				print('\n\n-- training: ' + str(m+1) + ' of ' + str(n_ensemble) + ' NNs --')
				hist = NNs[m].fit(x_train, y_train,
						  batch_size=batch_size,
						  epochs=epochs,
						  verbose=verbose_in,
						  shuffle=True,
						  callbacks=[LearningRateScheduler(lr_schedule)],
						  validation_data=(x_test[0:2000], y_test[0:2000]))
				NNs_hist_train.append(hist.history['loss'])
				NNs_hist_val.append(hist.history['val_loss'])

			if is_plot_train:
				NNs_hist_train=np.array(NNs_hist_train)
				NNs_hist_val=np.array(NNs_hist_val)
				fig = plt.figure(figsize=(4, 3))
				ax = fig.add_subplot(111)
				for m in range(n_ensemble):
					ax.plot(NNs_hist_train[m], color='b',label='train')
					ax.plot(NNs_hist_val[m], color='r',label='val')
				# ax.set_legend()
				fig.legend(loc='upper right')
				fig.show()
				plt.show(block=False)

			# print(history.losses)

			print('\n\n-- NN training finished --\n\n')

			if is_plot_params:
				fn_display_weights(NNs[-1],title_in='after training')

			y_logit_preds, y_mean, y_var, y_prob_preds, y_prob_final = fn_predict_ensemble(NNs,x_test)

			# evaluate accuracy
			for m in range(n_ensemble): # individual members
				score = NNs[m].evaluate(x_test, y_test, verbose=0)
				# print('NN ' +str(m) + ' test loss:', score[0])
				print('NN ' + str(m) + ' test accuracy:', score[1])
				result_NN_acc.append(score)


			ens_acc = np.mean(np.equal(np.argmax(y_prob_final,axis=-1), np.argmax(y_test,axis=-1)))
			print('total ens test accuracy:', ens_acc)
			result_ens_acc.append(ens_acc)

			# calc entropy
			y_logits_mnist = y_logit_preds.copy()
			y_outs_mnist = y_prob_final.copy()
			entropies_mnist = -np.sum(y_outs_mnist * np.log(y_outs_mnist+1e-10),axis=1)
			max_probs_mnist = np.max(y_outs_mnist,axis=1)

			print('\n\n-- NN evaluation finished --\n\n')

			# print('reg:',reg)
			# print('n_ensemble:',n_ensemble)



		# ==========================================================================================
		# ==========================================================================================
		# 									-- plotting --


		y_logit_preds, y_mean, y_var, y_prob_preds, y_prob_final = fn_predict_ensemble(NNs,x_grid)

		y_prob_final_grid = y_prob_final[:,0].reshape(x_grid_row.shape)


		
		# for render in ['cartoon','full']:
		for render in ['cartoon']: # use full to get smoother plot
			fig = plt.figure(figsize=(5, 4))
			ax = fig.add_subplot(111)
			# ax.set_title(title_in)
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,levels=1,colors=['red','blue'])
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,cmap='coolwarm')
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,cmap='RdBu')

			# fix for the white lines between contour levels
			if render == 'cartoon':
				cnt = ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid, cmap='RdBu')
			elif render == 'full':
				cnt = ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid, levels=500, cmap='RdBu')
			# remove white lines
			for c in cnt.collections:
				# c.set_edgecolor("k")
				c.set_edgecolor("face")

			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,levels=20,cmap='RdBu')
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,levels=100,cmap='PuRd')
			ax.scatter(x_train[:,0],x_train[:,1],c=color_in, s=50,linewidths=1,edgecolors='k')
			ax.set_ylim([-y_range,y_range])
			ax.set_xlim([-x_range,x_range])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			fig.show()
			plt.show(block=False)

			if is_save_graph:
				title = str(datetime.datetime.now().strftime('%H-%M')) + render + '_NNs_' + str(n_ensemble) + '_type_' + reg + '_actfn_' + activation_in
				fig.savefig('00_outputs/2d_plot_' + title +'.eps', format='eps', dpi=1000, bbox_inches='tight')


end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))



# ==========================================================================================
# ==========================================================================================
# 								-- exact Bayesian HMC NN --
if is_BNN:
	print('\n\n -- running BNN --')

	W1_var = 15/n_in
	b1_var = W1_var
	W1_var_2 = 1/n_hidden
	b1_var_2 = W1_var_2
	W2_var = 10/n_hidden

	import theano
	floatX = theano.config.floatX
	import pymc3 as pm
	import theano.tensor as T
	from scipy.special import erf

	activation_fn = 'relu'

	ann_input = theano.shared(x_train)
	ann_output = theano.shared(y_train)

	# Initialize random weights between each layer
	# W1_var = constant_var_W1
	# b1_var = W1_var
	# W2_var = constant_var_W2/n_hidden
	init_w1_bnn = np.random.normal(loc=0, scale=np.sqrt(W1_var), size=[n_in, n_hidden]).astype(floatX)
	init_b1_bnn = np.random.normal(loc=0, scale=np.sqrt(b1_var), size=[n_hidden]).astype(floatX)

	init_w1_bnn_2 = np.random.normal(loc=0, scale=np.sqrt(W1_var), size=[n_hidden, n_hidden]).astype(floatX)
	init_b1_bnn_2 = np.random.normal(loc=0, scale=np.sqrt(b1_var), size=[n_hidden]).astype(floatX)

	init_w2_bnn = np.random.normal(loc=0, scale=np.sqrt(W2_var), size=[n_hidden,n_out]).astype(floatX)

	def build_BNN(ann_input, ann_output):
		with pm.Model() as model:
			weights_in_w1 = pm.Normal('w_in_1', 0, sd=np.sqrt(W1_var),
									 shape=(n_in, n_hidden),
									 testval=init_w1_bnn)

			weights_in_b1 = pm.Normal('b_in_1', 0, sd=np.sqrt(b1_var),
									 shape=(n_hidden),
									 testval=init_b1_bnn)

			weights_in_w1_2 = pm.Normal('w_in_1_2', 0, sd=np.sqrt(W1_var_2),
									 shape=(n_hidden, n_hidden),
									 testval=init_w1_bnn_2)

			weights_in_b1_2 = pm.Normal('b_in_1_2', 0, sd=np.sqrt(b1_var_2),
									 shape=(n_hidden),
									 testval=init_b1_bnn_2)

			weights_2_out = pm.Normal('w_2_out', 0, sd=np.sqrt(W2_var),
									  shape=(n_hidden,n_out),
									  testval=init_w2_bnn)

			if activation_fn =='relu':
				act_0 = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1,0)
				act_1 = pm.math.maximum(pm.math.dot(act_0, weights_in_w1_2) + weights_in_b1_2,0)

			elif activation_fn =='erf':
				act_1 = pm.math.erf(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			elif activation_fn =='tanh':
				act_0 = pm.math.tanh(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
				act_1 = pm.math.tanh(pm.math.dot(act_0, weights_in_w1_2) + weights_in_b1_2)

			elif activation_fn =='sigmoid':
				act_1 = pm.math.sigmoid(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			
			act_out = pm.math.sigmoid(pm.math.dot(act_1, weights_2_out)*1.)
			# could multiply by constant here to make sigmoid sharper -> prior variance more reasonable

			out = pm.Bernoulli('out',
							   act_out,
							   observed=ann_output,
							   total_size=n_train)

		return model, out

	def nn_predict_np(X, W_0, W_1, W_2, b_0, b_1, b_2=None):
		if activation_fn == 'relu':
			h0 = np.maximum(np.matmul(X, W_0) + b_0,0)
			h = np.maximum(np.matmul(h0, W_1) + b_1,0)
		elif activation_fn == 'Lrelu':
			a=0.2
			h = np.maximum(np.matmul(X, W_0) + b_0,a*(np.matmul(X, W_0) + b_0))
		elif activation_fn == 'erf':
			h = erf(np.matmul(X, W_0) + b_0)
		elif activation_fn == 'softplus':
			h = np.log(1+np.exp(c*(np.matmul(X, W_0) + b_0) ))/c
		elif activation_fn == 'tanh':
			h0 = np.tanh(np.matmul(X, W_0) + b_0)
			h = np.tanh(np.matmul(h0, W_1) + b_1)
		elif activation_fn == 'cosine':
			h = np.cos(np.matmul(X, W_0) + b_0)
		elif activation_fn == 'linear':
			h = np.matmul(X, W_0) + b_0
		elif activation_fn == 'sigmoid':
			h = sigmoid(np.matmul(X, W_0) + b_0)

		y_pred_logits = np.matmul(h, W_1)
		y_pred_probs = sigmoid(y_pred_logits)
		return y_pred_logits, y_pred_probs


	# build BNN
	BNN, out = build_BNN(ann_input, ann_output)

	is_plot_priors=False
	if is_plot_priors:
		# priors
		priors_BNN = pm.sample_prior_predictive(n_ensemble, model=BNN) 
		w1_priors = priors_BNN['w_in_1']
		b1_priors = priors_BNN['b_in_1']
		w1_priors_2 = priors_BNN['w_in_1_2']
		b1_priors_2 = priors_BNN['b_in_1_2']
		w2_priors = priors_BNN['w_2_out']

		y_preds_logits=[]; y_preds_probs=[];
		for m in range(n_ensemble):
			temp_y_pred_logits, temp_y_pred_probs = nn_predict_np(np.atleast_2d(x_grid).T,w1_priors[m],w2_priors[m],b1_priors[m])
			y_preds_logits.append(temp_y_pred_logits)
			y_preds_probs.append(temp_y_pred_probs)
		y_preds_logits = np.squeeze(np.array(y_preds_logits))

		# y_logit_preds = np.squeeze(np.array(y_preds_logits))
		fig = plt.figure(figsize=(6, 4))
		ax = fig.add_subplot(111)
		ax.plot(x_grid,y_preds_logits.T)
		ax.set_ylabel('logit priors, BNN')
		ax.set_xlim(-x_view,x_view)
		fig.show()
		plt.show(block=False)

		# param distributions
		priors_BNN_dist = pm.sample_prior_predictive(1000, model=BNN) 
		w1_priors_dist = priors_BNN_dist['w_in_1']
		b1_priors_dist = priors_BNN_dist['b_in_1']
		w2_priors_dist = priors_BNN_dist['w_2_out']
		fig = plt.figure(figsize=(6, 4))
		ax = fig.add_subplot(311)
		# ax.hist(w1_priors_dist.flatten()*np.sqrt(1/40),bins=20)
		ax.hist(w1_priors_dist.flatten(),bins=20)
		ax = fig.add_subplot(312)
		ax.hist(b1_priors_dist.flatten(),bins=20)
		ax = fig.add_subplot(313)
		ax.hist(w2_priors_dist.flatten(),bins=20)
		fig.show()
		plt.show(block=False)


	# run inference
	# step = pm.Metropolis(model=BNN)
	step = pm.HamiltonianMC(path_length=1., adapt_step_size=True, step_scale=0.04,
		gamma=0.3, k=0.9, t0=10, target_accept=0.95, model=BNN)
	trace = pm.sample(n_inf_samples, step=step, model=BNN, chains=1, n_jobs=1, tune=300)

	# https://docs.pymc.io/api/inference.html

	# 	path_length : float, default=2
	#     total length to travel
	# step_rand : function float -> float, default=unif
	#     A function which takes the step size and returns an new one used to randomize the step size at each iteration.
	# step_scale : float, default=0.25
	#     Initial size of steps to take, automatically scaled down by 1/n**(1/4).
	# scaling : array_like, ndim = {1,2}
	#     The inverse mass, or precision matrix. One dimensional arrays are interpreted as diagonal matrices. If is_cov is set to True, this will be interpreded as the mass or covariance matrix.
	# is_cov : bool, default=False
	#     Treat the scaling as mass or covariance matrix.
	# potential : Potential, optional
	#     An object that represents the Hamiltonian with methods velocity, energy, and random methods. It can be specified instead of the scaling matrix.
	# target_accept : float, default .8
	#     Adapt the step size such that the average acceptance probability across the trajectories are close to target_accept. Higher values for target_accept lead to smaller step sizes. Setting this to higher values like 0.9 or 0.99 can help with sampling from difficult posteriors. Valid values are between 0 and 1 (exclusive).
	# gamma : float, default .05
	# k : float, default .75
	#     Parameter for dual averaging for step size adaptation. Values between 0.5 and 1 (exclusive) are admissible. Higher values correspond to slower adaptation.
	# t0 : int, default 10
	#     Parameter for dual averaging. Higher values slow initial adaptation.
	# adapt_step_size : bool, default=True
	#     Whether step size adaptation should be enabled. If this is disabled, k, t0, gamma and target_accept are ignored.


	# do prediction
	ann_input.set_value(x_grid.astype('float32'))
	ann_output.set_value(x_grid.astype('float32'))
	if True:
		ppc = pm.sample_ppc(trace, model=BNN, samples=n_pred_samples*50) # this does new set of preds per point
		y_preds = ppc['out']
		y_pred_mu = y_preds.mean(axis=0)
		y_pred_std = y_preds.std(axis=0)

		y_pred_mu_grid = y_pred_mu[:,0].reshape(x_grid_row.shape)

		# for render in ['cartoon','full']:
		for render in ['cartoon']:
			# fig = plt.figure(figsize=(4, 4))
			fig = plt.figure(figsize=(5, 4))
			ax = fig.add_subplot(111)
			# ax.set_title(title_in)
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,levels=1,colors=['red','blue'])
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,cmap='coolwarm')
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,cmap='RdBu')

			# fix for the white lines between contour levels
			if render == 'cartoon':
				cnt = ax.contourf(x_grid_row, x_grid_col, y_pred_mu_grid, cmap='RdBu')
			elif render == 'full':
				cnt = ax.contourf(x_grid_row, x_grid_col, y_pred_mu_grid, levels=500, cmap='RdBu')
			# remove white lines
			for c in cnt.collections:
				c.set_edgecolor("face")

			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,levels=20,cmap='RdBu')
			# ax.contourf(x_grid_row, x_grid_col, y_prob_final_grid,levels=100,cmap='PuRd')
			ax.scatter(x_train[:,0],x_train[:,1],c=color_in, s=50,linewidths=1,edgecolors='k')
			ax.set_ylim([-y_range,y_range])
			ax.set_xlim([-x_range,x_range])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			fig.show()
			plt.show(block=False)

			if is_save_graph:
				title = str(datetime.datetime.now().strftime('%H-%M')) + render + '_HMC' + '_actfn_' + activation_in
				fig.savefig('00_outputs/2d_plot_' + title +'.eps', format='eps', dpi=1000, bbox_inches='tight')


	else:

		# manual sampling
		w1_s = trace.get_values('w_in_1')
		b1_s = trace.get_values('b_in_1')
		w1_s_2 = trace.get_values('w_in_1_2')
		b1_s_2 = trace.get_values('b_in_1_2')
		w2_s = trace.get_values('w_2_out')
		w1_s = w1_s[drops:] # drop extra burn ins
		b1_s = b1_s[drops:]
		w2_s = w2_s[drops:]
		total_samples = w1_s.shape[0]

		if is_plot_priors:
			# param distributions
			fig = plt.figure(figsize=(6, 4))
			ax = fig.add_subplot(311)
			# ax.hist(w1_priors_dist.flatten()*np.sqrt(1/40),bins=20)
			ax.hist(w1_s.flatten(),bins=20)
			ax = fig.add_subplot(312)
			ax.hist(b1_s.flatten(),bins=20)
			ax = fig.add_subplot(313)
			ax.hist(w2_s.flatten(),bins=20)
			fig.show()
			plt.show(block=False)

		# make predictions
		y_preds_logits=[]; y_preds_probs=[];
		# print('\nsampling predictions...')
		for _ in range(n_pred_samples):
			id = np.random.randint(0,total_samples) # sample from posterior
			temp_y_pred_logits, temp_y_pred_probs = nn_predict_np(np.atleast_2d(x_grid).T,w1_s[id],w2_s[id],b1_s[id])
			y_preds_logits.append(temp_y_pred_logits)
			y_preds_probs.append(temp_y_pred_probs)

		y_preds_logits = np.array(y_preds_logits)
		y_preds_probs = np.array(y_preds_probs)

		y_preds_logits_mu = np.mean(y_preds_logits,axis=0)
		y_preds_logits_std = np.std(y_preds_logits,axis=0)

		y_preds_probs_mu = np.mean(y_preds_probs,axis=0)

plt.show()


# -*- coding: utf-8 -*-
import numpy as np

"""
contains params to run experiments on
"""

def get_hyperparams(data_set, activation_fn,hidden_size,is_deep_NN=False):
	# create a dict of hyperparams
	hyp = {}
	hyp = 	{'batch_size': 64, 'optimiser_in': 'adam', 'learning_rate': 0.001}

	if data_set == '~boston':
		X_dim=13
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.06
			hyp['b_0_var'] = 10
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.05
			hyp['b_0_var'] = 25

		# specifically for NN optimisation
		if hidden_size == 50:
			hyp['n_epochs'] = 3000
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.995
			# if is_deep_NN:
			# 	hyp['n_epochs'] = 4000
			# 	hyp['learning_rate'] = 0.01
			# 	hyp['decay_rate'] = 0.997
		elif hidden_size == 1000:
			hyp['n_epochs'] = 3000
			hyp['learning_rate'] = 0.02
			hyp['decay_rate'] = 0.995
		hyp['single_data_n_std'] = np.sqrt(0.2) # size of single NN noise std dev


	elif data_set == '~yacht':
		X_dim=6
		if activation_fn == 'relu':
			hyp['data_noise'] = 1e-7
			hyp['b_0_var'] = 15
		elif activation_fn == 'erf':
			hyp['data_noise'] = 1e-5
			hyp['b_0_var'] = 25

		if hidden_size == 50:
			hyp['n_epochs'] = 3000
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.997
		elif hidden_size ==1000:
			hyp['n_epochs'] = 3000
			hyp['learning_rate'] = 0.01
			hyp['decay_rate'] = 0.99
		hyp['single_data_n_std'] = np.sqrt(0.01)


	elif data_set == '~energy':
		X_dim=8
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 1e-7
			hyp['b_0_var'] = 12
		elif activation_fn == 'erf':
			hyp['data_noise'] = 1e-7
			hyp['b_0_var'] = 10

		if hidden_size == 50:
			hyp['n_epochs'] = 2000
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.997
		elif hidden_size == 1000:
			hyp['n_epochs'] = 2000
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.995
		hyp['single_data_n_std'] = np.sqrt(0.001)


	elif data_set == '~concrete':
		X_dim=8
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.05
			hyp['b_0_var'] = 40
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.06
			hyp['b_0_var'] = 100

		if hidden_size == 50:
			hyp['n_epochs'] = 2000
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.997
		elif hidden_size == 1000:
			hyp['n_epochs'] = 2000
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.995
		hyp['single_data_n_std'] = np.sqrt(0.08)


	elif data_set == '~wine':
		X_dim=11
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.5
			hyp['b_0_var'] = 20
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.5
			hyp['b_0_var'] = 50

		if hidden_size == 50:
			hyp['n_epochs'] = 500
			hyp['learning_rate'] = 0.05
			hyp['decay_rate'] = 0.997
		elif hidden_size == 1000:
			hyp['n_epochs'] = 500
			hyp['learning_rate'] = 0.02
			hyp['decay_rate'] = 0.995
		hyp['single_data_n_std'] = np.sqrt(0.6)


	elif data_set == '~kin8':
		X_dim=8
		hyp['batch_size'] = 256
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.02
			hyp['b_0_var'] = 40
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.04
			hyp['b_0_var'] = 40

		if hidden_size == 50:
			hyp['n_epochs'] = 2000
			hyp['learning_rate'] = 0.1
			hyp['decay_rate'] = 0.998
		elif hidden_size == 1000:
			hyp['n_epochs'] = 200
			hyp['learning_rate'] = 0.02
			hyp['decay_rate'] = 0.995
		hyp['single_data_n_std'] = np.sqrt(0.1)


	elif data_set == '~power':
		X_dim=4
		hyp['batch_size'] = 256
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.05 # 0.01
			hyp['b_0_var'] = 4
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.05
			hyp['b_0_var'] = 5

		if hidden_size == 50:
			hyp['n_epochs'] = 1000
			hyp['learning_rate'] = 0.2
			hyp['decay_rate'] = 0.995
		elif hidden_size == 1000:
			hyp['n_epochs'] = 300
			hyp['learning_rate'] = 0.01
			hyp['decay_rate'] = 0.999
		hyp['single_data_n_std'] = np.sqrt(0.06)


	elif data_set == '~naval':
		X_dim=16
		hyp['batch_size'] = 256
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 1e-7
			hyp['b_0_var'] = 200
		elif activation_fn == 'erf':
			hyp['data_noise'] = 1e-7
			hyp['b_0_var'] = 1000

		if hidden_size == 50:
			hyp['n_epochs'] = 1000
			hyp['learning_rate'] = 0.1
			hyp['decay_rate'] = 0.997
		elif hidden_size == 1000:
			hyp['n_epochs'] = 300 # not done!
			hyp['learning_rate'] = 0.01
			hyp['decay_rate'] = 0.999
		hyp['single_data_n_std'] = np.sqrt(0.0007)


	elif data_set == '~protein':
		X_dim=9
		hyp['batch_size'] = 8192
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.5
			hyp['b_0_var'] = 50
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.5
			hyp['b_0_var'] = 200

		if hidden_size == 100:
			hyp['n_epochs'] = 3000
			hyp['learning_rate'] = 0.1
			hyp['decay_rate'] = 0.995
		elif hidden_size == 1000:
			hyp['n_epochs'] = 3000
			hyp['learning_rate'] = 0.1
			hyp['decay_rate'] = 0.995
		elif hidden_size == 50:
			raise Exception('too few nodes for this data')
		hyp['single_data_n_std'] = np.sqrt(0.5)


	elif data_set == '~song':
		X_dim=90
		hyp['batch_size'] = 8192*4
		# general params shared by all
		if activation_fn == 'relu':
			hyp['data_noise'] = 0.7
			hyp['b_0_var'] = 2
		elif activation_fn == 'erf':
			hyp['data_noise'] = 0.7
			hyp['b_0_var'] = 6

		if hidden_size == 100:
			hyp['n_epochs'] = 500
			hyp['learning_rate'] = 0.01
			hyp['decay_rate'] = 0.996
		elif hidden_size == 1000:
			hyp['n_epochs'] = 1000
			hyp['learning_rate'] = 0.001
			hyp['decay_rate'] = 0.998
		elif hidden_size == 50:
			raise Exception('too few nodes for this data')
		hyp['single_data_n_std'] = np.sqrt(0.7)

	# for a two layer NN would be good to retune hyperparams
	# but we just want rough results, so doing some rough rules 
	# of thumb instead
	if is_deep_NN:
		hyp['n_epochs'] = hyp['n_epochs']*1.5
		hyp['learning_rate'] = hyp['learning_rate']/4
		# hyp['decay_rate'] = 0.997


	# enforcing this constraint
	hyp['w_0_var'] = hyp['b_0_var']/X_dim
	hyp['cycle_print'] = hyp['n_epochs']/10
	return hyp

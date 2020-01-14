# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import importlib
import DataGen
import utils
import module_gp
import module_NN_ens
import module_HMC
import module_VI
import hyperparams

# useful to have reloads when developing
importlib.reload(DataGen)
importlib.reload(utils)
importlib.reload(module_gp)
importlib.reload(module_NN_ens)
importlib.reload(module_HMC)
importlib.reload(module_VI)
importlib.reload(hyperparams)

from DataGen import DataGenerator
from utils import *
import module_gp
import module_NN_ens
import module_HMC
import module_VI
import hyperparams

import numpy as np
import tensorflow as tf
import datetime
import pickle

start_time = datetime.datetime.now()
print_w_time('started')

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = True

np.random.seed(101)

# this script produces banchmark results for table 1

# -- inputs --
data_set =  '~boston' # 'drunk_bow_tie' '~boston' favourite_fig, toy_2, bias_fig, test_2D, test_2D_gap, x_cubed_gap
n_samples = 12
# activation_fn = 'relu' 		# activation type - relu, erf, rbf, softplus, Lrelu (Leaky ReLu)
data_noise = 0.001 			# data noise variance
b_0_var = 10.				# var of b_0
w_0_var = b_0_var			# var of w_0
# w_0_var = 4			
u_var = 1.0					# var for rbf params as -> inf, goes to stationary cov dist
g_var = 1					# var for rbf params
#n_runs = 20					# no. runs to average over

# -- NN model inputs --
optimiser_in = 'adam' 		# optimiser: adam, SGD
learning_rate = 0.005		# learning rate
# hidden_size = 1000			# no. hidden neurons
n_epochs = 4000				# no. epochs to train for
cycle_print = n_epochs/10 	# print info every cycle no. of epochs
batch_size = 64
# n_ensembles = 5				# no. NNs in ensemble

# -- HMC model inputs --
step_size = 0.001			# size of 'leapfrog steps' for HMC inference
n_steps = 150				# no. 'leapfrog steps' in between samples for HMC inference
burn_in = 500				# no. samples to drop
n_predict_hmc = 50				# no. samples used to make predictions
# n_samples_hmc = burn_in+n_predict	# total no. of samples to collect - should be > burn_in
n_samples_hmc = 1000

# -- VI model inputs --
n_predict_vi = 50
n_iter_vi = 2000
n_samples_vi = 100		# Number of samples from variational model for calculating stochastic gradients

# -- misc model inputs --
# single_data_n_std = 0.05 	# when training a single NN, const. data noise std dev!

# plotting options
is_try_plot = 0
is_save_graphs = 0

# key experimental inputs
activation_fn = 'relu'
hidden_size = 50
n_ensembles = 5
n_runs = 1
is_deep_NN = False

if n_runs > 1 or data_set == '~song':
	train_prop=0.9
else:
	train_prop=0.8

# which to run
is_gp_run = 0
is_ens_run = 1

# for the experimental runs, don't really use the below
is_mc_run = 0
is_hmc_run = 0
is_vi_run = 0
is_unconstrained_run = 0
is_reg_run = 0
is_single_run = 0
is_sk_run = 0


hyp = hyperparams.get_hyperparams(data_set,activation_fn,hidden_size,is_deep_NN)

gp_results=[]; ens_results=[]; single_results=[]; mc_results=[]; 
hmc_results=[]; sk_results=[]; unc_ens_results=[]; 
run_kls=[]
for run_ in range(n_runs):
	print('\n\n  ====== run:',run_, '======\n')

	# -- create data --
	Gen = DataGenerator(type_in=data_set)
	X_train, y_train, X_val, y_val = Gen.CreateData(n_samples=n_samples, seed_in=run_+1000, 
		train_prop=train_prop)
	n_samples = X_train.shape[0]
	X_dim = X_train.shape[1]
	y_dim = 1

	# for larger datasets when running GP only use 10k rows to train
	if is_gp_run == 1 and data_set=='~protein':
		X_train = X_train[0:10000]
		y_train = y_train[0:10000]
	if is_gp_run == 1 and data_set=='~song':
		X_train = X_train[0:20000]
		y_train = y_train[0:20000]
		X_val = X_val[0:5000]
		y_val = y_val[0:5000]

	# mesh the input space for evaluations
	if X_dim == 1:
		X_grid = np.atleast_2d(np.linspace(-3, 3, 100)).T
		X_val = X_grid
		y_val = np.expand_dims(X_grid[:,0],1)
	elif X_dim == 2:
		x_s = np.atleast_2d(np.linspace(-3, 3, 20)).T
		X_grid = np.array(np.meshgrid(x_s,x_s))
		X_grid = np.stack((X_grid[1].ravel(), X_grid[0].ravel()),axis=-1)
		X_val = X_grid
		y_val = np.expand_dims(X_grid[:,0],1)
	else:
		X_grid = X_val


	if is_gp_run:
		# -- gp model --
		gp = module_gp.gp_model(kernel_type=activation_fn, data_noise=hyp['data_noise'], 
			b_0_var=hyp['b_0_var'], w_0_var=hyp['w_0_var'], u_var=5., g_var=1.)
		y_pred_mu, y_pred_std = gp.run_inference(x_train=X_train, y_train=y_train, x_predict=X_val, print=False)

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, hyp['b_0_var'], hyp['w_0_var'], hyp['data_noise'], gp, is_print=True)
		gp_results.append(np.array((gp.mse_unnorm, gp.rmse, gp.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, gp, save=is_save_graphs)


	if is_vi_run:
		# -- BNN with variational inference model --
		vi = module_VI.vi_model(activation_fn=activation_fn, data_noise=data_noise, 
			b_0_var=b_0_var, w_0_var=w_0_var, u_var=5., g_var=1., hidden_size = hidden_size,
			n_predict=n_predict_vi, n_iter=n_iter_vi, n_samples_vi=n_samples_vi)

		vi.train(X_train=X_train, y_train=y_train, X_val=X_val,is_print=False)

		y_preds, y_pred_mu, y_pred_std = vi.predict(X_val)

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, vi, is_print=True)
		hmc_results.append(np.array((vi.mse_unnorm, vi.rmse, vi.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, vi, save=is_save_graphs)#, y_preds)


	if is_hmc_run:
		# -- hmc model --
		hmc = module_HMC.hmc_model(activation_fn=activation_fn, data_noise=data_noise, 
			b_0_var=b_0_var, w_0_var=w_0_var, u_var=5., g_var=1., hidden_size = hidden_size,
			step_size=step_size, n_steps=n_steps, burn_in=burn_in, n_samples=n_samples_hmc, n_predict=n_predict_hmc)

		hmc.train(X_train=X_train, y_train=y_train, X_val=X_val,is_print=False)

		y_preds, y_pred_mu, y_pred_std = hmc.predict(X_val)

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, hmc, is_print=True)
		hmc_results.append(np.array((hmc.mse_unnorm, hmc.rmse, hmc.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, hmc, save=is_save_graphs)#, y_preds)


	if is_ens_run:
		# -- NN ensemble model --
		total_ens_run = 0; y_preds_list=[]
		while total_ens_run < n_ensembles:
			print('\n\ntotal_ens_run:',total_ens_run)
			n_ensembles_in = np.min((n_ensembles-total_ens_run,5))
			ens = module_NN_ens.NN_ens(activation_fn=activation_fn, 
				data_noise=hyp['data_noise'],
				b_0_var=hyp['b_0_var'], w_0_var=hyp['w_0_var'], u_var=u_var, g_var=g_var,
				optimiser_in = hyp['optimiser_in'], 
				learning_rate = hyp['learning_rate'], 
				hidden_size = hidden_size, 
				n_epochs = hyp['n_epochs'], 
				cycle_print = hyp['cycle_print'], 
				n_ensembles = n_ensembles_in,
				total_trained=total_ens_run,
				batch_size = hyp['batch_size'],
				decay_rate = hyp['decay_rate'],
				deep_NN = is_deep_NN
				)

			y_priors, y_prior_mu, y_prior_std = ens.train(X_train, y_train, X_val, y_val, is_print=True)
			# plot priors
			if False:
				if is_try_plot: try_plot(X_dim, X_grid, y_prior_mu, y_prior_std, X_train, y_train, ens, y_priors)
			y_preds_temp, _mu, _std = ens.predict(X_val)
			total_ens_run += n_ensembles_in
			y_preds_list.append(y_preds_temp)
			# y_predsnp.concatenate((y_preds[0],y_preds[1]))

		y_preds = y_preds_list[0].copy()
		for i in range(1,len(y_preds_list)):
			y_preds = np.concatenate((y_preds,np.atleast_2d(y_preds_list[i])))
		# y_preds = np.array(y_preds).T
		y_pred_mu = np.atleast_2d(np.mean(y_preds,axis=0)).T
		y_pred_std = np.atleast_2d(np.std(y_preds,axis=0, ddof=1)).T
		y_pred_std = np.sqrt(np.square(y_pred_std) + hyp['data_noise'])

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, hyp['b_0_var'], hyp['w_0_var'], hyp['data_noise'], ens, is_print=True)
		ens_results.append(np.array((ens.mse_unnorm, ens.rmse, ens.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, ens, y_preds, save=is_save_graphs)
		

	if is_gp_run and is_ens_run:
		kl_avg = compare_dist(gp.y_pred_mu, gp.y_pred_std, y_pred_mu, y_pred_std)
		print('\n\nkl(gp,NN_ens)',np.round(kl_avg,4))
		run_kls.append(kl_avg)

	if is_gp_run and is_hmc_run:
		kl_avg = compare_dist(gp.y_pred_mu, gp.y_pred_std, hmc.y_pred_mu, hmc.y_pred_std)
		print('\n\nkl(gp,hmc)',np.round(kl_avg,4))
		# run_kls.append(kl_avg)

	if is_hmc_run and is_ens_run:
		kl_avg = compare_dist(hmc.y_pred_mu, hmc.y_pred_std, y_pred_mu, y_pred_std)
		print('\n\nkl(hmc,NN_ens)',np.round(kl_avg,4))
		# run_kls.append(kl_avg)

	if is_mc_run:
		mc_NN = module_NN_ens.NN_ens(activation_fn=activation_fn, 
			data_noise=data_noise,
			b_0_var=b_0_var, w_0_var=w_0_var, u_var=u_var, g_var=g_var,
			optimiser_in = optimiser_in, 
			learning_rate = learning_rate, 
			hidden_size = hidden_size, 
			n_epochs = n_epochs, 
			cycle_print = cycle_print, 
			n_ensembles = 1,
			regularise=True,
			batch_size = batch_size,
			drop_out=True
			)
		mc_NN.train(X_train, y_train, X_val, y_val, is_print=True)

		y_preds=[]
		for i in range(200):
			y_preds_temp, y_pred_mu, y_pred_std = mc_NN.predict(X_val)
			y_preds.append(y_preds_temp)
		y_preds = np.array(y_preds).squeeze()
		y_pred_mu = np.mean(y_preds,axis=0)
		y_pred_std = np.std(y_preds,axis=0, ddof=1)

		# add on data noise
		y_pred_std = np.sqrt(np.square(y_pred_std) + data_noise)
		y_pred_mu = np.atleast_2d(y_pred_mu).T
		y_pred_std = np.atleast_2d(y_pred_std).T

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, mc_NN, is_print=True)
		mc_results.append(np.array((mc_NN.mse_unnorm, mc_NN.rmse, mc_NN.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, mc_NN, save=is_save_graphs)

	if is_unconstrained_run:
		# -- NN ensemble model, unconstrained --
		total_ens_run = 0; y_preds_list=[]
		while total_ens_run < n_ensembles:
			print('\n\ntotal_ens_run:',total_ens_run)
			n_ensembles_in = np.min((n_ensembles-total_ens_run,5))
			unc_ens = module_NN_ens.NN_ens(activation_fn=activation_fn, 
				data_noise=data_noise,
				b_0_var=b_0_var, w_0_var=w_0_var, u_var=u_var, g_var=g_var,
				optimiser_in = optimiser_in, 
				learning_rate = learning_rate, 
				hidden_size = hidden_size, 
				n_epochs = n_epochs, 
				cycle_print = cycle_print, 
				n_ensembles = n_ensembles_in,
				total_trained=total_ens_run,
				batch_size = batch_size,
				unconstrain = True
				)

			y_priors, y_prior_mu, y_prior_std = unc_ens.train(X_train, y_train, X_val, y_val, is_print=True)
			y_preds_temp, _mu, _std = unc_ens.predict(X_val)
			total_ens_run += n_ensembles_in
			y_preds_list.append(y_preds_temp)
			# y_predsnp.concatenate((y_preds[0],y_preds[1]))

		y_preds = y_preds_list[0].copy()
		for i in range(1,len(y_preds_list)):
			y_preds = np.concatenate((y_preds,np.atleast_2d(y_preds_list[i])))
		# y_preds = np.array(y_preds).T
		y_pred_mu = np.atleast_2d(np.mean(y_preds,axis=0)).T
		y_pred_std = np.atleast_2d(np.std(y_preds,axis=0, ddof=1)).T
		y_pred_std = np.sqrt(np.square(y_pred_std) + data_noise)

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, unc_ens, is_print=True)
		unc_ens_results.append(np.array((unc_ens.mse_unnorm, unc_ens.rmse, unc_ens.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, unc_ens, y_preds, save=is_save_graphs)
		
	if is_reg_run:
		# -- NN ensemble model, regularised --
		total_ens_run = 0; y_preds_list=[]
		while total_ens_run < n_ensembles:
			print('\n\ntotal_ens_run:',total_ens_run)
			n_ensembles_in = np.min((n_ensembles-total_ens_run,5))
			reg_ens = module_NN_ens.NN_ens(activation_fn=activation_fn, 
				data_noise=data_noise,
				b_0_var=b_0_var, w_0_var=w_0_var, u_var=u_var, g_var=g_var,
				optimiser_in = optimiser_in, 
				learning_rate = learning_rate, 
				hidden_size = hidden_size, 
				n_epochs = n_epochs, 
				cycle_print = cycle_print, 
				n_ensembles = n_ensembles_in,
				total_trained=total_ens_run,
				batch_size = batch_size,
				regularise=True
				)

			y_priors, y_prior_mu, y_prior_std = reg_ens.train(X_train, y_train, X_val, y_val, is_print=True)
			y_preds_temp, _mu, _std = reg_ens.predict(X_val)
			total_ens_run += n_ensembles_in
			y_preds_list.append(y_preds_temp)
			# y_predsnp.concatenate((y_preds[0],y_preds[1]))

		y_preds = y_preds_list[0].copy()
		for i in range(1,len(y_preds_list)):
			y_preds = np.concatenate((y_preds,np.atleast_2d(y_preds_list[i])))
		# y_preds = np.array(y_preds).T
		y_pred_mu = np.atleast_2d(np.mean(y_preds,axis=0)).T
		y_pred_std = np.atleast_2d(np.std(y_preds,axis=0, ddof=1)).T
		y_pred_std = np.sqrt(np.square(y_pred_std) + data_noise)

		# metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, unc_ens, is_print=True)
		# unc_ens_results.append(np.array((unc_ens.mse_unnorm, unc_ens.rmse, unc_ens.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, reg_ens, y_preds, save=is_save_graphs)
		

	if is_single_run:
		single_NN = module_NN_ens.NN_ens(activation_fn=activation_fn, 
			data_noise=hyp['data_noise'],
			b_0_var=hyp['b_0_var'], w_0_var=hyp['w_0_var'], u_var=u_var, g_var=g_var,
			optimiser_in = hyp['optimiser_in'], 
			learning_rate = hyp['learning_rate'], 
			hidden_size = hidden_size, 
			n_epochs = hyp['n_epochs'], 
			cycle_print = hyp['cycle_print'], 
			n_ensembles = 1,
			regularise=True,
			batch_size = hyp['batch_size'],
			decay_rate = hyp['decay_rate']
			)
		single_NN.train(X_train, y_train, X_val, y_val, is_print=True)
		# y_preds_temp, y_pred_mu, y_pred_std = single_NN.predict(X_train)
		# single_data_noise = np.mean(np.square(y_train - np.atleast_2d(y_pred_mu).T))
		y_preds_temp, y_pred_mu, y_pred_std = single_NN.predict(X_val)
		# manually add constant noise here
		# y_pred_std = np.zeros_like(y_pred_std) + np.sqrt(data_noise)
		# print('\nsingle_data_noise:',np.round(single_data_noise,4),'\n')
		# y_pred_std = np.zeros_like(y_pred_std) + np.sqrt(single_data_noise)
		y_pred_std = np.zeros_like(y_pred_std) + hyp['single_data_n_std']
		y_pred_mu = np.atleast_2d(y_pred_mu).T
		y_pred_std = np.atleast_2d(y_pred_std).T

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, hyp['b_0_var'], hyp['w_0_var'], hyp['data_noise'], single_NN, is_print=True)
		single_results.append(np.array((single_NN.mse_unnorm, single_NN.rmse, single_NN.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, single_NN, save=is_save_graphs)

	if is_sk_run:
		from sklearn import linear_model
		# sk_model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True)

		from sklearn.neighbors import KNeighborsRegressor
		sk_model = KNeighborsRegressor(n_neighbors=1, weights='uniform', algorithm='brute', leaf_size=30, p=2, metric='minkowski')

		sk_model.fit(X_train,y_train)
		y_pred_train = sk_model.predict(X_train)
		sk_data_noise = np.mean(np.square(y_train - y_pred_train))
		y_pred_mu = sk_model.predict(X_val)
		# y_pred_std = np.zeros_like(y_pred_mu) +  np.sqrt(sk_data_noise)*1.1 # add for overfit
		y_pred_std = np.zeros_like(y_pred_mu) +  0.2 # add for overfit
			
		# for completely naive
		# y_pred_mu = np.zeros_like(y_pred_mu)
		# y_pred_std = np.zeros_like(y_pred_mu) + 1.

		mse_unnorm, rmse, nll = metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, is_print=True)
		sk_results.append(np.array((mse_unnorm, rmse, nll)))


if is_gp_run:
	gp_results = np.array(gp_results)
	print('\n\n___ GP RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(gp_results[:,i])
		std_dev = np.std(gp_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))


if is_ens_run:
	ens_results = np.array(ens_results)
	print('\n\n___ NN ens RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(ens_results[:,i])
		std_dev = np.std(ens_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))


if is_gp_run and is_ens_run:
	print('\n\nKL avg:',np.round(np.mean(run_kls),4), ', std err:',np.round(np.std(run_kls,ddof=1)/np.sqrt(n_runs),5))


if is_single_run:
	single_results = np.array(single_results)
	print('\n\n___ single NN RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(single_results[:,i])
		std_dev = np.std(single_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))

if is_mc_run:
	mc_results = np.array(mc_results)
	print('\n\n___ mc drop out NN RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(mc_results[:,i])
		std_dev = np.std(mc_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))

if is_unconstrained_run:
	unc_ens_results = np.array(unc_ens_results)
	print('\n\n___ unconstrained NN RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(unc_ens_results[:,i])
		std_dev = np.std(unc_ens_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))

if is_sk_run:
	sk_results = np.array(sk_results)
	print('\n\n___ sk learn RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(sk_results[:,i])
		std_dev = np.std(sk_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))


if False:
	w1, b1, w2 = ens.NNs[0].get_weights(ens.sess)
	act_point = -b1.T/w1
	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(111)
	ax.hist(act_point.ravel(),bins=100, range=(-8,8))
	fig.show()

# view dataset
if False:
	data_set_all = ['boston','concrete','energy','kin8','power','protein','wine','yacht']#,'song', 'naval']
	for data_set_loop in data_set_all:
		print(data_set_loop)

		# -- create data --
		Gen = DataGenerator(type_in='~'+data_set_loop)
		X_train, y_train, X_val, y_val = Gen.CreateData(n_samples=n_samples, seed_in=run_+1000, 
			train_prop=0.5)
		n_samples = X_train.shape[0]
		X_dim = X_train.shape[1]
		y_dim = 1

		# fig = plt.figure(figsize=(5, 4))
		# for i in range(X_train.shape[1]):
		# 	ax = fig.add_subplot(int(X_train.shape[1]/2 +1),2,i+1)
		# 	ax.scatter(X_train[:,i],y_train,s=2)
		# fig.show()

		n_rows=400
		X_train_df = X_train[0:n_rows]
		y_train_df = y_train[0:n_rows]

		df = np.zeros((X_train_df.shape[0], X_train_df.shape[1]+1))
		df[:,:-1]=X_train_df
		df[:,-1]=y_train_df[:,0]
		new_names = []
		for i in range(X_train.shape[1]):
			new_names.append(('x'+str(i)))
		new_names.append('y')
		
		df = pd.DataFrame(df, columns=new_names)

		fig = plt.figure(figsize=(5, 5))
		ax = fig.add_subplot(111)
		scatter_matrix(df, alpha = 1, s=6,ax=ax, diagonal = 'kde')
		# fig.suptitle(data_set_loop)
		for ax_i in fig.get_axes():
			ax_i.set_xticks([])
			ax_i.set_yticks([])
		fig.show()

		fig.savefig('00_outputs_graph/datasets_vis/visualise_' + data_set_loop +'.eps', format='eps', dpi=500, bbox_inches='tight')




# -- tidy up --
print_w_time('finished')
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))


# ------------------------------------------------------------------------------------------------
# print of hyperparams
if False:
	data_set_all = ['boston','concrete','energy','kin8','naval','power','protein','wine','yacht','song']
	activation_fn_loop = 'relu'

	for k in hyp.keys():
		print(k, end='\t')

	for data_set_loop in data_set_all:
		if data_set_loop == 'protein': 
			hidden_size_loop = 100
		elif data_set_loop == 'song': 
			hidden_size_loop = 100
		else:
			hidden_size_loop = 50
		hyp = hyperparams.get_hyperparams('~'+data_set_loop,activation_fn_loop,hidden_size_loop)
		print('\n'+data_set_loop, end='\t')
		for k in hyp.keys():
			print(hyp[k], end='\t')
			# print(k, ':',hyp[k])






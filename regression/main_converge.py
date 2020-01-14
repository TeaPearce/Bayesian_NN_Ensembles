# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import importlib
import DataGen
import utils
import module_gp
import module_NN_ens

importlib.reload(DataGen)
importlib.reload(utils)
importlib.reload(module_gp)
importlib.reload(module_NN_ens)

from DataGen import DataGenerator
from utils import *
import module_gp
import module_NN_ens

import numpy as np
import tensorflow as tf
import datetime
import pickle

start_time = datetime.datetime.now()
print_w_time('started')

# avoid the dreaded type 3 fonts...
# http://phyletica.org/matplotlib-fonts/
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True
# plt.rcParams['axes.unicode_minus'] = False

# http://nerdjusttyped.blogspot.com/2010/07/type-1-fonts-and-matplotlib-figures.html
# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True

np.random.seed(101)

# this script produces results for figure 7
# show progressive Gaussian behaviour on Boston dataset for increasing ensemble number and width

# -- inputs --
data_set =  '~boston' # 'drunk_bow_tie' '~boston' favourite_fig, test_2D, test_2D_gap, x_cubed_gap
n_samples = 10				# no data points to create - only for some synthetic datasets
activation_fn = 'relu' 		# activation type - relu, erf, rbf
data_noise = 0.1 			# data noise variance
b_0_var = 2				# var of b_0
w_0_var = b_0_var			# var of w_0
u_var = 10.0					# var for rbf params as -> inf, goes to stationary cov dist
g_var = 10					# var for rbf params
n_runs = 10					# no. runs to average over

# -- NN model inputs --
optimiser_in = 'adam' 		# optimiser: adam, SGD, AdaDel, AdaGrad
learning_rate = 0.001		# learning rate
hidden_size = 4			# no. hidden neurons
n_epochs = 400				# no. epochs to train for
cycle_print = n_epochs/10 	# print info every cycle no. of epochs
batch_size = 128
n_ensembles = 5				# no. NNs in ensemble

# plotting options
is_try_plot = 0

# convergence test options
h_size_vals=[4,16,64,256,1024]
# h_size_vals=[1024]
# h_size_vals=[hidden_size]
n_ens_vals=[3,5,10,20,40]
# n_ens_vals=[40]
# n_ens_vals=[n_ensembles]

# -- create data --
Gen = DataGenerator(type_in=data_set)
X_train, y_train, X_val, y_val = Gen.CreateData(n_samples=n_samples, seed_in=3, 
	train_prop=0.5)

n_samples = X_train.shape[0]
X_dim = X_train.shape[1]
y_dim = 1

# mesh the input space for evaluations
if X_dim == 1:
	X_grid = np.atleast_2d(np.linspace(-1.5, 1.5, 200)).T
	X_val = X_grid
	y_val = np.expand_dims(X_grid[:,0],1)
elif X_dim == 2:
	x_s = np.atleast_2d(np.linspace(-3, 3, 20)).T
	X_grid = np.array(np.meshgrid(x_s,x_s))
	X_grid = np.stack((X_grid[1].ravel(), X_grid[0].ravel()),axis=-1)
	X_val = X_grid
else:
	X_grid = X_val

# -- gp model --
gp = module_gp.gp_model(kernel_type=activation_fn, data_noise=data_noise, 
	b_0_var=b_0_var, w_0_var=w_0_var, u_var=5., g_var=1.)
y_pred_mu, y_pred_std = gp.run_inference(x_train=X_train, y_train=y_train, x_predict=X_val, print=True)

metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, 
	b_0_var, w_0_var, data_noise, gp, is_print=True)
if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, gp)

results=[]
for n_ensembles in n_ens_vals:
	for hidden_size in reversed(h_size_vals): # better to reverse so can tune eps etc
		run_metrics=[]; run_kls=[]

		if True: # necessary to modify hyperparams based on NN size...
			if hidden_size < 20:# need less epochs for larger NNs
				n_epochs=1000
			else:
				n_epochs=1000
			cycle_print = n_epochs/5
			if hidden_size < 200: # need lower lr for larger NNs
				learning_rate = 0.001
			else:
				learning_rate = 0.0002

		for run_ in range(n_runs):
			print('\n\n  ====== run:',run_, 'n_ens:',n_ensembles, 'h_size:',hidden_size,'======\n')

			# -- NN ensemble model --
			# v slow when do large no. NNs in same object,
			# we break them into sets to speed up
			total_ens_run = 0; y_preds_list=[]
			while total_ens_run < n_ensembles:
				# print('\n\ntotal_ens_run:',total_ens_run)
				n_ensembles_in = np.min((n_ensembles-total_ens_run,10))
				ens = module_NN_ens.NN_ens(activation_fn=activation_fn, 
					data_noise=data_noise, 
					b_0_var=b_0_var, w_0_var=w_0_var, u_var=u_var, g_var=g_var,
					optimiser_in = optimiser_in, 
					learning_rate = learning_rate, 
					hidden_size = hidden_size, 
					n_epochs = n_epochs, 
					cycle_print = cycle_print, 
					n_ensembles = n_ensembles_in,
					total_trained=total_ens_run,
					batch_size = batch_size
					)

				ens.train(X_train, y_train, X_val, y_val, is_print=True)
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
			y_pred_std = np.sqrt(np.square(y_pred_std) + data_noise)

			metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, ens, is_print=True)
			if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, ens, y_preds=y_preds)
			run_metrics.append([ens.mse_unnorm, ens.rmse, ens.nll])

			kl_avg = compare_dist(gp.y_pred_mu, gp.y_pred_std, y_pred_mu, y_pred_std)
			print('\n\nkl(gp,NN_ens)',np.round(kl_avg,4))
			run_kls.append(kl_avg)

		# results: [n_ensembles, hidden_size, kl_mean, kl_std_err]
		results.append(np.array([n_ensembles, hidden_size, np.mean(run_kls), np.std(run_kls,ddof=1)/np.sqrt(n_runs)]))

		# print results so far in form that can copy/paste across to new array
		results_temp=np.array(results)
		print('\n\n== intermediate results == \n')
		print("[",end="")
		for i in range(results_temp.shape[0]):
			print("[",end="")
			for j in range(results_temp.shape[1]):
				print(results_temp[i,j],end="")
				if j+1 != results_temp.shape[1]:
					print(",",end="")
			print("]",end="")
			if i+1 != results_temp.shape[0]:
				print(",",end="")
		print("]",end="")
		# a=np.array()

results=np.array(results)
# results=list(results)
print('\n\nresults\n',results)

if n_runs > 1:
	# pickle.dump(results, open( "02_outputs_data/01_KL_converge/KL_"+ data_set + "_" + activation_fn + "_b0_" + str(b_0_var) +'_' + get_time() +".p", "wb" ) )
	name="BEST_KL_" + get_time() + '_' + data_set + "_" + activation_fn + "_b0_" + str(b_0_var) + ".p"
	pickle.dump([results,n_ens_vals, h_size_vals], open( "02_outputs_data/01_KL_converge/"+name, "wb" ) )
	print('saved data as', name)
	# results = pickle.load( open( '02_outputs_data/01_KL_converge/KL_~boston_relu_b0_0.1_08_20_16_40_31.p', "rb" ) )


if False:
	# plot of KL-divergence over hidden size and ensemble no.
	# my_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	# my_colors = [(0.1, 0.2, 1), (0.1, 0.2, 0.8), (0.1, 0.2, 0.6), (0.1, 0.2, 0.4)]

	my_colors = []
	my_dashes = []
	for i in range(len(n_ens_vals)):
		# my_colors.append( ( i/(len(n_ens_vals)-1), 0., (len(n_ens_vals)-1-i)/(len(n_ens_vals)-1))) # (R,B,G)
		my_colors.append( ( 0.2, 0.2, 1- i/(len(n_ens_vals)+2))) # (R,B,G)
		my_dashes.append((i+1,i+1)) # (length of dash, space to next dash)
		# my_dashes.append((1,0.001)) # (length of dash, space to next dash)
	# my_dashes=my_dashes[::-1]
	if True:
		fig = plt.figure(figsize=(5, 4))
		ax = fig.add_subplot(111)
		for i in range(0,len(n_ens_vals)):
			temp=results[i*len(h_size_vals):(i+1)*len(h_size_vals)]
			ax.plot(temp[:,1], temp[:,2], linestyle='--', dashes=my_dashes[i], color=my_colors[i],linewidth=1.5,label=str(int(temp[0,0])) + 'xNNs')
			ax.errorbar(temp[:,1], temp[:,2], yerr=temp[:,3], linestyle='none',markersize=3, marker='o', ecolor=my_colors[i], color=my_colors[i], alpha=1., elinewidth=1., capsize=2., capthick=1.)
		# ax.title(title)
		plt.xlabel('Log hidden size')
		plt.ylabel('Log KL divergence(GP, NN ens)')
		# ax.set_ylim(0.1,10)
		ax.set_ylim(0)
		# ax.set_ylim(top=10)
		# ax.set_xlim(0)
		ax.set_xscale("log", nonposx='clip')
		# ax.set_yscale("log", nonposy='clip')
		ax.grid(color='k', linestyle='-', linewidth=.2)
		ax.legend(loc='upper right')
		fig.show()
		if False:
			fig.savefig('00_outputs_graph/' + 'KL_conv_' + data_set + '_' + get_time() +'.eps', format='eps', dpi=1000, bbox_inches='tight')


if False: # now plotting n_ens vs KL

	# compute empirically the best could hope for if each ens was from true GP posterior
	gp_draw_results=[]
	gp = module_gp.gp_model(kernel_type=activation_fn, data_noise=data_noise, 
		b_0_var=b_0_var, w_0_var=w_0_var, u_var=5., g_var=1.)
	y_pred_mu, y_pred_std = gp.run_inference(x_train=X_train, y_train=y_train, x_predict=X_val, print=True)
	for n_ensembles in n_ens_vals:
		print('ensemble',n_ensembles)
		run_kls=[]; n_runs_gp=400
		for run_ in range(n_runs_gp):
			print(run_,end='\r')
			gp.posts_draw_visualise(n_draws=n_ensembles, is_graph=False)
			kl_avg = compare_dist(gp.y_pred_mu, gp.y_pred_std, np.atleast_2d(gp.y_pred_mu_draws).T, np.atleast_2d(gp.y_pred_std_draws).T)
			run_kls.append(kl_avg)

			# debugging
			# try_plot(X_dim, X_grid, np.atleast_2d(gp.y_pred_mu_draws).T, np.atleast_2d(gp.y_pred_std_draws).T, X_train, y_train, gp, y_preds=gp.y_preds)
			# print('kl_avg',kl_avg)
		gp_draw_results.append(np.array([n_ensembles, np.mean(run_kls), np.std(run_kls,ddof=1)/np.sqrt(n_runs_gp)]))
	gp_draw_results=np.array(gp_draw_results)
	print(gp_draw_results)

	is_log_y=True
	my_colors = []
	my_dashes = []
	for i in range(len(h_size_vals)):
		my_colors.append( ( 0.2, 0.2, 1- i/(len(n_ens_vals)+2))) # (R,B,G)
		my_dashes.append((i+1,i+1)) # (length of dash, space to next dash)

	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(111)
	res_sorted = results[results[:,1].argsort()]
	for i in range(0,len(h_size_vals)):
		temp = res_sorted[i*len(n_ens_vals):(i+1)*len(n_ens_vals)]
		temp = temp[temp[:,0].argsort()]
		ax.plot(temp[:,0], temp[:,2], linestyle='--', dashes=my_dashes[i], color=my_colors[i],linewidth=1.5,label=str(int(temp[0,1])) + ' nodes')
		ax.errorbar(temp[:,0], temp[:,2], yerr=temp[:,3], linestyle='none',markersize=3, marker='o', ecolor=my_colors[i], color=my_colors[i], alpha=1., elinewidth=1., capsize=2., capthick=1.)
	# ax.title(title)
	# ax.plot(gp_draw_results[:,0], gp_draw_results[:,1], linestyle='-', color='r',linewidth=1.0,label='Performance limit')
	ax.plot(gp_draw_results[:,0], gp_draw_results[:,1], linestyle='-', color='r',markersize=3, marker='o',linewidth=1.0,label='Ideal')
	plt.xlabel('Log no. NNs in ensemble')
	if is_log_y:
		plt.ylabel('Log KL divergence(GP, NN ens)')
	else:
		plt.ylabel('KL divergence(GP, NN ens)')

	# ax.set_ylim(0.1,10)
	# ax.set_ylim(0.,3.5)

	# ax.set_ylim(top=10)
	# ax.set_xlim(0)
	ax.set_xscale("log", nonposx='clip')
	if is_log_y:
		ax.set_yscale("log", nonposy='clip')
	else:
		ax.set_ylim(0.)
	# ax.grid(color='k', linestyle='-', linewidth=.2)
	ax.legend(loc='upper right')
	ax.legend(loc='lower left')
	fig.show()
	if False:
		fig.savefig('00_outputs_graph/' + 'BEST_KL_conv_vsnens_' + data_set + '_' + get_time() +'.eps', format='eps', dpi=1000, bbox_inches='tight')


# -- tidy up --
print_w_time('finished')
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))






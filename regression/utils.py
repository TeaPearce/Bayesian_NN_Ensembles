# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime

"""
contains some handy functions
- 
- 
- some plotting fns
"""


# utils
def print_w_time(string_in):
	print('\n--',string_in, '-- ', datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3],'\n')
	return

def get_time():
	return datetime.datetime.now().strftime('%m_%d_%H_%M_%S')


def gauss_neg_log_like(y_true, y_pred_gauss_mid, y_pred_gauss_dev, scale_c):
	"""
	return negative gaussian log likelihood
	"""

	n = y_true.shape[0]
	y_true=y_true.reshape(-1)*scale_c
	y_pred_gauss_mid=y_pred_gauss_mid*scale_c
	y_pred_gauss_dev=y_pred_gauss_dev*scale_c
	neg_log_like = -np.sum(stats.norm.logpdf(y_true.squeeze(), loc=y_pred_gauss_mid.squeeze(), scale=y_pred_gauss_dev.squeeze()))
	neg_log_like = neg_log_like/n

	return neg_log_like

def metrics_calc(y_val, y_pred_mu, y_pred_std, scale_c, b_0_var, w_0_var, data_noise, model=None, is_print=True):
	''' computes and returns predictive metrics of interest '''

	mse_unnorm = np.mean(np.square(y_val - y_pred_mu)) # for data noise
	rmse = np.sqrt(np.mean(np.square(scale_c*(y_val - y_pred_mu))))
	neg_log_like = gauss_neg_log_like(y_val, y_pred_mu, y_pred_std, scale_c)

	if is_print:
		print('\n\n-- '+model.name_+' --')
		print('b_0=' + str(b_0_var) + ', w_0=' + str(w_0_var) + ', noise var=' + str(data_noise))
		print('RMSE\t', np.round(rmse,4), '\nNLL\t', np.round(neg_log_like,4))
		print('data noise est\t', np.round(mse_unnorm,4))

	if not model is None:
		# place holders
		model.mse_unnorm = mse_unnorm
		model.rmse = rmse
		model.nll = neg_log_like

	return mse_unnorm, rmse, neg_log_like


def compare_dist(y_pred_mu_1, y_pred_std_1, y_pred_mu_2, y_pred_std_2):
	''' return metrics of comparing two gaussians'''
	# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
	# http://www.allisons.org/ll/MML/KL/Normal/
	# https://stats.stackexchange.com/questions/323069/can-kl-divergence-ever-be-greater-than-1
	kl = np.log(y_pred_std_2/y_pred_std_1) + (np.square(y_pred_std_1) - np.square(y_pred_std_2) + np.square(y_pred_mu_1-y_pred_mu_2))/(2*np.square(y_pred_std_2))
	return np.mean(kl)


def try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, model, y_preds=None,save=False, type='blank'):
	''' try to plot the predicted stuff if X_dim appropriate '''
	# title = model.name_ + ', b_0=' + str(model.b_0_var) + ', w_0=' + str(model.w_0_var) + ', noise var=' + str(model.data_noise)
	# type='panel/favfig_erf_low_noise_'
	# type='panel/favfig_relu_low_noise_'
	# type='converge_favfig_relu_low_noise_'
	# type='need_ensembles'
	title = model.name_# + type
	if X_dim == 1:
		plot_1d_grid(X_grid, y_pred_mu, y_pred_std, X_train, y_train, title, save=save, y_preds=y_preds, type=type)
	elif X_dim == 2:
		plot_2d_grid(X_grid, y_pred_mu, y_pred_std, X_train, y_train, title)
	return


def plot_1d_grid(x_s, y_mean, y_std, X_train, y_train, title=None, save=False, name=None, y_preds=None, type=None):

	if name is None: name = title[0:3]

	# plot predictions
	# fig = plt.figure(figsize=(3, 2))
	# fig = plt.figure(figsize=(5, 3)) # intro panel
	fig = plt.figure(figsize=(5, 4)) # usual
	ax = fig.add_subplot(111)
	# plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')

	if True:
		ax.plot(x_s, y_mean, 'b-', linewidth=2.,label=u'Prediction')
		ax.plot(x_s, y_mean + 2 * y_std, 'b', linewidth=0.5)
		ax.plot(x_s, y_mean - 2 * y_std, 'b', linewidth=0.5)
		ax.plot(x_s, y_mean + 1 * y_std, 'b', linewidth=0.5)
		ax.plot(x_s, y_mean - 1 * y_std, 'b', linewidth=0.5)
		ax.fill(np.concatenate([x_s, x_s[::-1]]),
				 np.concatenate([y_mean - 2 * y_std,
								(y_mean + 2 * y_std)[::-1]]),
				 alpha=1, fc='lightskyblue', ec='None')
		ax.fill(np.concatenate([x_s, x_s[::-1]]),
				 np.concatenate([y_mean - 1 * y_std,
								(y_mean + 1 * y_std)[::-1]]),
				 alpha=1, fc='deepskyblue', ec='None')

	# if not y_preds is None and (type[0:4]=='conv' or type[0:4]=='intr'):
	if False and not y_preds is None: # make true to plot individual NN predictions
		# ax.plot(x_s, y_preds.T, 'k', linewidth=2/np.sqrt(y_preds.T.shape[1])) # for intro panel
		ax.plot(x_s, y_preds.T, 'k', linewidth=1/np.sqrt(y_preds.T.shape[1]))

	if True:
		ax.plot(X_train[:,0], y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
	# ax.plot(X, y, 'r.', markersize=10, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)

	# plt.title(title)
	# plt.xlabel('$x$')
	# plt.ylabel('$f(x)$')
	if save:
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		if False:
			ax.set_yticks([])
			ax.set_xticks([])
	# ax.set_ylim(-2.3, 1.2)
	# ax.set_xlim(-2.5, 2.5)
	# ax.set_ylim(-3, 15)
	
	if True:
		if type=='panel/favfig_erf_low_noise_':
			ax.set_ylim(-2.3, 1.2) # panel, favourite_fig with erf, low data noise
			ax.set_xlim(-2.5, 2.5)
		elif type=='panel/favfig_relu_low_noise_':
			ax.set_ylim(-3, 0.7)
			ax.set_xlim(-1.7, 1.5)
		elif type=='converge_favfig_relu_low_noise_':
			ax.set_ylim(-3, 1)
			ax.set_xlim(-1.7, 1.7)
		elif type=='panel/favfig_Lrelu_low_noise_':
			ax.set_ylim(-3, 1)
			ax.set_xlim(-1.7, 1.7)
		elif type=='panel/favfig_rbf_low_noise_':
			ax.set_ylim(-2.5, 1)
			ax.set_xlim(-6,6)
		elif type=='need_ensembles':
			ax.set_ylim(-3, 2.5)
			ax.set_xlim(-3, 3)
		elif type=='intro_panel':
			# ax.set_ylim(-3, 2.5)
			ax.set_xlim(-3, 3)
	# else:
		# ax.set_ylim(-7, 1)
		# ax.set_xlim(-3,3)

	# ax.set_xlim(np.min(x_s), np.max(x_s))
	# ax.grid(color='k', linestyle='--', linewidth=.3, dashes=(2,7)) # these don't come out well on pdf
	# plt.legend(loc='upper left')
	fig.show()
	plt.show(block = False)

	if save:
		fig.savefig('00_outputs_graph/'+type+'/' + title +'.eps', format='eps', dpi=1000, bbox_inches='tight')
		# fig.savefig('00_outputs_graph/converge_favfig_relu_low_noise_/' + title +'.eps', format='eps', dpi=1000, bbox_inches='tight')

	return


def plot_2d_grid(X_grid, y_pred_mu, y_pred_std, X_train, y_train, title):
	from mpl_toolkits.mplot3d import Axes3D

	fig = plt.figure()
	# plt.title(title)
	views = [(5,90),(40,45),(70,20),(20,120)]
	# first param is kind of tilt of camera (looking down on it or up)
	# second is rotation
	# views = [(-165,117),(-165,88),(60,30),(-181,147)]
	# fig, axarr = plt.subplots(len(views), projection='3d')
	for i,view_set_ in enumerate(views):
		ax = fig.add_subplot(2,2,i+1, projection='3d')
		# ax.scatter(X_grid[:,0], X_grid[:,1], y_pred_mu, color='b')

		ax.plot_trisurf(X_grid[:,0], X_grid[:,1], y_pred_mu[:,0]+2*y_pred_std[:,0], color='r', alpha=0.3)
		ax.scatter(X_grid[:,0], X_grid[:,1], y_pred_mu[:,0]+2*y_pred_std[:,0], color='r', alpha=0.5, s=5)
		ax.plot_trisurf(X_grid[:,0], X_grid[:,1], y_pred_mu[:,0]-2*y_pred_std[:,0], color='b', alpha=0.3)
		ax.scatter(X_grid[:,0], X_grid[:,1], y_pred_mu[:,0]-2*y_pred_std[:,0], color='b', alpha=0.5, s=5)
		ax.scatter(X_train[:,0], X_train[:,1], y_train[:,0], color='k', s=20)
		ax.set_xlabel('X[0]')
		ax.set_ylabel('X[1]')
		ax.set_zlabel('y')
		ax.set_xlim(-3, 3)
		ax.set_ylim(-3, 3)
		ax.set_zlim(-2, 2)

		# ax.view_init(0, 90) # elev, azim
		# ax.view_init(90, 0) # elev, azim
		# ax.view_init(95, -4) # elev, azim
		# ax.view_init(-179,164) # elev, azim

		ax.view_init(view_set_[0],view_set_[1]) # elev, azim
	fig.show()
	plt.show(block = False)

	# print('ax.azim {}'.format(ax.azim))
	# print('ax.elev {}'.format(ax.elev))

	return

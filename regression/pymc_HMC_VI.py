import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from scipy.special import erf
from sklearn.datasets import fetch_openml


# this code produces HMC and VI plots for figure 4
# use activation_fn = 'erf' or 'relu'
# and inference_method = 'hmc' or 'vi'

# note that the original paper plots were created using Edward
# so this file produces plots with some differences.
# In particular, the VI case appears slightly worse for relu
# If use smaller data_noise with VI, get better looking results

# inputs
data_noise = 0.001 # variance
n_hidden = 100
n_inf_samples = 2000 # number samples to take during inference
drops = 100
n_pred_samples = 200 # number samples to take during prediction
vi_steps = 200000 # number optimisation steps to run for VI 
w1_var = 10
b1_var = 10 # w1_var
activation_fn = 'erf' # relu, erf, tanh, mixed, cosine, linear
inference_method = 'hmc' # hmc, vi

print('\n\n --running ' + inference_method + ' on activation fn ' + activation_fn + '-- \n\n')

# create data, favourite_fig
X_train = np.atleast_2d([1., 4.5, 5.1, 6., 8., 9.]).T
X_train = X_train/5. - 1
Y_train = X_train * np.sin(X_train*5.)

# X_grid = np.atleast_2d(np.linspace(np.min(X_train), np.max(X_train)+9., 1000)).T
X_grid = np.atleast_2d(np.linspace(-3, 3, 600)).T
Y_grid = np.ones_like(X_grid)

# set up
ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)
total_size = len(Y_train)
n_in = X_train.shape[1]

# Initialize random weights between each layer


init_w1 = np.random.normal(loc=0, scale=np.sqrt(w1_var), size=[n_in, n_hidden]).astype(floatX)
init_b1 = np.random.normal(loc=0, scale=np.sqrt(b1_var), size=[n_hidden]).astype(floatX)
# init_out = np.random.normal(loc=0, scale=np.sqrt(np.sqrt(comp_a)/n_hidden), size=[n_hidden,1]).astype(floatX)
init_out = np.random.normal(loc=0, scale=np.sqrt(1/n_hidden), size=[n_hidden,1]).astype(floatX)

def build_model(ann_input, ann_output):
	with pm.Model() as model:

		# first head of NN
		weights_in_w1 = pm.Normal('w_in_1', 0, sd=np.sqrt(w1_var),
								 shape=(n_in, n_hidden),
								 testval=init_w1)

		weights_in_b1 = pm.Normal('b_in_1', 0, sd=np.sqrt(b1_var),
								 shape=(n_hidden),
								 testval=init_b1)

		weights_2_out = pm.Normal('w_2_out', 0, sd=np.sqrt(1/n_hidden),
								  shape=(n_hidden,1),
								  testval=init_out)

		# Build neural-network using tanh activation function
		if activation_fn == 'relu':
			act_1 = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1,0)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='tanh':
			act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='erf':
			act_1 = pm.math.erf(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='cosine':
			act_1 = pm.math.cos(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='linear':
			act_1 = pm.math.dot(ann_input, weights_in_w1) + weights_in_b1
			act_out = pm.math.dot(act_1, weights_2_out)

		
		out = pm.Normal('out', act_out,sd=np.sqrt(data_noise),
						   observed=ann_output,
						   total_size=total_size)

	return model, out


# build BNN
BNN, out = build_model(ann_input, ann_output)
 
# run inference
if inference_method == 'hmc':
	step = pm.HamiltonianMC(path_length=0.5, adapt_step_size=True, step_scale=0.04,
		gamma=0.05, k=0.9, t0=1, target_accept=0.95, model=BNN)
	trace = pm.sample(n_inf_samples, step=step, model=BNN, chains=1, n_jobs=1, tune=300)
	# reduce path_length if failing - 5.0 is ok with cos_lin data
elif inference_method == 'vi':
	# https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html
	inference = pm.ADVI(model=BNN)
	approx = pm.fit(n=vi_steps, method=inference, model=BNN)
	trace = approx.sample(draws=n_inf_samples)

	if True:
		fig = plt.figure(figsize=(8, 4))
		ax = fig.add_subplot(111)
		ax.plot(-inference.hist, label='new ADVI', alpha=.3)
		ax.plot(approx.hist, label='old ADVI', alpha=.3)
		ax.set_ylabel('ELBO');
		ax.set_xlabel('iteration');
		fig.show()


def nn_predict_np(X, W_0, W_1, b_0, b_1=0, W_0_2=None, W_1_2=None, b_0_2=None, comp_a_learn=None, comp_b_learn=None):
	if activation_fn == 'relu':
		h = np.maximum(np.matmul(X, W_0) + b_0,0)
	elif activation_fn == 'Lrelu':
		a=0.2
		h = np.maximum(np.matmul(X, W_0) + b_0,a*(np.matmul(X, W_0) + b_0))
	elif activation_fn == 'erf':
		h = erf(np.matmul(X, W_0) + b_0)
	elif activation_fn == 'softplus':
		h = np.log(1+np.exp(c*(np.matmul(X, W_0) + b_0) ))/c
	elif activation_fn == 'tanh':
		h = np.tanh(np.matmul(X, W_0) + b_0)
	elif activation_fn == 'cosine':
		h = np.cos(np.matmul(X, W_0) + b_0)
	elif activation_fn == 'linear':
		h = np.matmul(X, W_0) + b_0
	elif activation_fn == 'rbf':
		h = np.exp(-beta_2*np.square(X - W_0))

	h = np.matmul(h, W_1) #+ b_1
	return np.reshape(h, [-1])

# make predictions
ann_input.set_value(X_grid.astype('float32'))
ann_output.set_value(X_grid.astype('float32'))

ppc = pm.sample_ppc(trace, model=BNN, samples=n_pred_samples) # this does new set of preds per point
y_preds = ppc['out']
y_pred_mu = y_preds.mean(axis=0)
y_pred_std = y_preds.std(axis=0)

# plot predictions
x_s = X_grid; y_mean = y_pred_mu; y_std = y_pred_std
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
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

ax.plot(X_train[:,0], Y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)

if activation_fn=='erf':
	ax.set_ylim(-2.3, 1.2) # panel, favourite_fig with erf, low data noise
	ax.set_xlim(-2.5, 2.5)
elif activation_fn=='relu':
	ax.set_ylim(-3, 0.7)
	ax.set_xlim(-1.7, 1.5)
fig.show()
plt.show(block = False)

plt.show()





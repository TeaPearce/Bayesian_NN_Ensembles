# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import importlib

def sigmoid_array(x):
	"""
	sigmoid function for an array
	needed for 5-D dopolous data
	"""                                     
	return 1 / (1 + np.exp(-x))

class DataGenerator:
	def __init__(self, type_in, n_feat=1):

		# not really using no. feat anymore

		self.n_feat = n_feat
		self.type_in = type_in

		return


	def CreateData(self, n_samples, seed_in=5, 
		train_prop=0.9, bound_limit=6., n_std_devs=1.96,**kwargs):

		# seed_in=5

		np.random.seed(seed_in)
		scale_c=1.0 # default
		shift_c=1.0

		# for ideal boundary
		X_ideal = np.linspace(start=-bound_limit,stop=bound_limit, num=500)
		y_ideal_U = np.ones_like(X_ideal)+1. # default
		y_ideal_L = np.ones_like(X_ideal)-1.
		y_ideal_mean = np.ones_like(X_ideal)+0.5

		if self.type_in=="regression":
			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,x_size))
			y = X[:,0] + 2*X[:,1]
			y = y.reshape([-1,1])
			ind = int(X.shape[0]*2/3)
			X_train = X[:ind]
			X_val = X[ind:]		

		elif self.type_in=="bow_tie":
			"""
			creates a bow tie shape with changing variance
			"""
			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,self.n_feat))
			y = X[:,0] + np.random.normal(loc=0.,scale=np.abs(X[:,0])/1.)
			y = y.reshape([-1,1])/5.
			X_train = X
			y_train = y	
			X_val = X_train
			y_val = y_train
			y_ideal_U = X_ideal/5. + n_std_devs * np.abs(X_ideal)/5.
			y_ideal_L = X_ideal/5. - n_std_devs * np.abs(X_ideal)/5.

		elif self.type_in=="periodic_1":
			"""
			creates a bow tie shape with changing variance
			"""
			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,self.n_feat))
			y=2.1*np.cos(3.2*X[:,0]) #+ 0.7*np.cos(20.1*X[:,0])
			y = y.reshape([-1,1])/1.
			X_train = X
			y_train = y	
			X_val = X_train
			y_val = y_train
			# y_ideal_U = X_ideal/5. + n_std_devs * np.abs(X_ideal)/5.
			# y_ideal_L = X_ideal/5. - n_std_devs * np.abs(X_ideal)/5.

		elif self.type_in=="pareto_smile":
			"""
			creates a bow tie shape with changing variance
			"""
			# X = np.random.normal(loc=0.,scale=1.,size=(n_samples,self.n_feat))
			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,self.n_feat))
			# y = np.power(X[:,0]*4.,2) - 1. - np.random.pareto(a=X[:,0]/4.+2.0) + np.random.exponential(scale=X[:,0]/2.+4.0)
			y = np.power(X[:,0]*4.,2) - np.random.pareto(a=X[:,0]+3.0)*2. + np.random.exponential(scale=X[:,0]+2.)*2.
			y = (y.reshape([-1,1]) - 30.)/20.
			X_train = X
			y_train = y			
			X_val = X_train
			y_val = y_train	

		elif self.type_in=="thin_cross":
			"""
			creates a thin cross
			trying to make lube fail
			has a tiny bit of noise in most places
			then large noise in over 5% of X range
			"""
			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,self.n_feat))
			y = np.zeros_like(X)
			y = np.random.normal(loc=0.,scale=y+0.01)
			y[(X<0.1) & (X>-0.1)] = np.random.normal(loc=0.,
				scale=np.zeros_like(y[(X<0.1) & (X>-0.1)]) + 1.)
			y = y.reshape([-1,1])
			X_train = X
			y_train = y			
			X_val = X_train
			y_val = y_train		

		elif self.type_in=="5D_dopolous":
			"""
			5-D input variables with variable noise
			as used in Papadopolous 2000 (orig. Friedman 1991 p37)
			"""
			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,5))
			noise_sd = 2.*sigmoid_array(X[:,0]+X[:,1]-2*X[:,2]-5*X[:,3]+2*X[:,4]) # sigmoidal fn
			noise_val = np.random.normal(loc=0.,scale=noise_sd)
			y = 10*sp.sin(X[:,0]*X[:,1]*np.pi) + 20*((X[:,2]-0.5)**2) + 10*X[:,3] + 5*X[:,4] + noise_val
			y = y.reshape([-1,1])/100.
			X_train = X
			y_train = y			
			X_val = X_train
			y_val = y_train

		elif self.type_in=="pred_intervals_simple":
			"""
			simple line of points all at X=1
			"""
			y = np.linspace(start=(1./n_samples),stop=1.,num=n_samples).reshape(-1,1)
			X = np.ones_like(y,dtype=np.float).reshape(-1,1)
			X_train = X
			y_train = y			
			X_val = X_train
			y_val = y_train

		elif self.type_in=="drunk_bow_tie":
			"""
			similar to bow tie but less linear
			"""	

			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.
			X_train = X
			y_train = y	

			X = np.random.uniform(low=-2.,high=2.,size=(int(10*n_samples),1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.		
			X_val = X
			y_val = y

			y_ideal_U = 1.5*np.sin(np.pi*X_ideal) + n_std_devs*np.power(X_ideal,2)
			y_ideal_U = y_ideal_U/5.
			y_ideal_L = 1.5*np.sin(np.pi*X_ideal) - n_std_devs*np.power(X_ideal,2)
			y_ideal_L = y_ideal_L/5.
			y_ideal_mean = 1.5*np.sin(np.pi*X_ideal)
			y_ideal_mean = y_ideal_mean/5.	

			# overwrite!
			# X_val = X_train
			# y_val = y_train

		elif self.type_in=="drunk_bow_tie_exp":
			"""
			similar to bow tie but less linear, now with non-gaussian noise
			"""	

			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.exponential(scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.
			X_train = X
			y_train = y	

			X = np.random.uniform(low=-2.,high=2.,size=(int(10*n_samples),1))
			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.exponential(scale=1.*np.power(X[:,0],2))
			y = y.reshape([-1,1])/5.		
			X_val = X
			y_val = y

			# for exponential quantile = ln(1/quantile) /lambda
			# note that np inputs beta = 1/lambda
			y_ideal_U = 1.5*np.sin(np.pi*X_ideal) + np.log(1/(1-0.95))*np.power(X_ideal,2)
			y_ideal_U = y_ideal_U/5.
			y_ideal_L = 1.5*np.sin(np.pi*X_ideal)
			y_ideal_L = y_ideal_L/5.
			y_ideal_mean = 1.5*np.sin(np.pi*X_ideal)
			y_ideal_mean = y_ideal_mean/5.	

			# overwrite!
			# X_val = X_train
			# y_val = y_train

		elif self.type_in=="x_cubed":
			"""
			toy data problem from Probabilistic Backprop (Lobato) & 
			deep ensembles (Blundell)

			"""
			scale_c = 50.
			X = np.random.uniform(low=-4.,high=4.,size=(n_samples,1))
			y = X[:,0]**3 + np.random.normal(loc=0.,scale=3., size=X[:,0].shape[0])
			y = y.reshape([-1,1])/scale_c
			X_train = X
			y_train = y

			y_ideal_U = X_ideal**3 + n_std_devs*3.
			y_ideal_U = y_ideal_U/scale_c
			y_ideal_L = X_ideal**3 - n_std_devs*3.
			y_ideal_L = y_ideal_L/scale_c
			y_ideal_mean = X_ideal**3
			y_ideal_mean = y_ideal_mean/scale_c			

			# make sure always same val data
			np.random.seed(10)
			X = np.random.uniform(low=-4.,high=4.,size=(500,1))
			y = X[:,0]**3 + np.random.normal(loc=0.,scale=3., size=X[:,0].shape[0])
			y = y.reshape([-1,1])/scale_c
			X_val = X
			y_val = y

		elif self.type_in=="x_cubed_gap":
			"""
			toy data problem from Probabilistic Backprop (Lobato) & 
			deep ensembles (Blundell)

			"""
			scale_c = 50.
			half_samp = int(round(n_samples/2))
			X_1 = np.random.uniform(low=-4.,high=-1.,size=(half_samp,1))
			X_2 = np.random.uniform(low=1,high=4.,size=(n_samples - half_samp,1))
			X = np.concatenate((X_1, X_2)) /4
			y = X[:,0]**3 + np.random.normal(loc=0.,scale=0.1, size=X[:,0].shape[0])#*5
			y = y.reshape([-1,1])/3#/scale_c
			X_train = X
			y_train = y	

			# overwrite		
			X_val = X_train
			y_val = y_train

			y_ideal_U = X_ideal**3 + n_std_devs*3.#*5
			y_ideal_U = y_ideal_U/scale_c
			y_ideal_L = X_ideal**3 - n_std_devs*3.#*5
			y_ideal_L = y_ideal_L/scale_c
			y_ideal_mean = X_ideal**3
			y_ideal_mean = y_ideal_mean/scale_c		

		elif self.type_in=="xor":
			"""
			creates the 4 data points for xor
			"""
			X_train = np.array([[1,0],[0,0],[0,1],[1,1]]) # xor
			y_train = np.array([[1],[0],[1],[0]])
			X_val = X_train
			y_val = y_train

		elif self.type_in=="favourite_fig":
			"""
			creates the 6 data points used in anchored ens fig
			"""
			X_train = np.atleast_2d([1., 4.5, 5.1, 6., 8., 9.]).T
			X_train = X_train/5. - 1
			y_train = X_train * np.sin(X_train*5.)
			X_val = X_train
			y_val = y_train

		elif self.type_in=="need_ens":
			"""
			creates the 6 data points used in anchored ens fig
			"""
			# X_train = np.atleast_2d([1., 4.5, 5.1, 6., 8., 9.5,9.1]).T
			# X_train = np.atleast_2d([1., 4.5, 5.1, 5.3, 5.9,6., 6.5, 8., 9.5,9.1]).T
			X_train = np.atleast_2d(np.linspace(-1.5,1.5,20)).T
			# X_train = X_train/5. - 1
			y_train =  np.sin(X_train*2) + np.random.normal(loc=0.,scale=0.2, size=X_train.shape)
			y_train[-1] = y_train[-1]-.4 # make one outlier
			# y_train[4] = y_train[-1]+.1 # make one outlier
			X_val = X_train
			y_val = y_train

		elif self.type_in=="bias_fig":
			"""
			c
			"""
			# X_train = np.atleast_2d([1., 7, 9.]).T
			X_train = np.atleast_2d([0.,  10.]).T
			X_train = X_train/5. - 1
			y_train = X_train * np.sin(X_train*5.) * -5 +10
			# y_train = X_train *0.5 +1#* np.sin(X_train*5.) * -5 +10
			X_val = X_train
			y_val = y_train

		elif self.type_in=="toy_2":
			"""
			toy 1-d data with gap in the middle
			"""
			X_train = np.atleast_2d([1,1.4,2.3,0.1, 8, 9.,9.3,8.3]).T
			X_train = X_train/2.5 - 2
			y_train = X_train * np.sin(X_train*1.) * 0.5 + np.random.normal(loc=0.,scale=0.1, size=X_train.shape)
			y_train[-1] = y_train[-1]-.2
			X_val = X_train
			y_val = y_train

		elif self.type_in=="test_2D":
			"""
			as before, but adds a nonsense extra variable
			"""
			X_train = np.random.uniform(low=-1.,high=1.,size=(n_samples,2))
			X_train[:,1] = X_train[:,1] / 1.
			y_train = np.sin(X_train[:,0]) + np.sin(X_train[:,1]) + np.random.normal(loc=0.,scale=0.1**0.5, size=X_train[:,0].shape[0])
			y_train = np.atleast_2d(y_train).T
			X_val = X_train
			y_val = y_train

		elif self.type_in=="test_2D_gap":
			"""
			as before, but adds a nonsense extra variable
			"""
			X_1 = np.random.uniform(low=-2,high=-1.,size=(n_samples,2))
			X_2 = np.random.uniform(low=1,high=1.5,size=(n_samples,2))
			X_train = np.concatenate((X_1, X_2))
			X_train[:,1] = X_train[:,1] / 1.
			y_train = np.sin(X_train[:,0]) + np.sin(0.2*X_train[:,1]) #+ np.random.normal(loc=0.,scale=0.1**0.5, size=X_train[:,0].shape[0])
			y_train = np.atleast_2d(y_train).T
			X_val = X_train
			y_val = y_train

		# use single char at start to identify
		# real data sets
		elif self.type_in[:1] == '~':

			if self.type_in=="~boston":
				path = '01_data//01_boston_house//housing_data.csv'
				data = np.loadtxt(path,skiprows=0)

			elif self.type_in=="~concrete":
				path = '01_data//02_concrete//Concrete_Data.csv'
				data = np.loadtxt(path, delimiter=',',skiprows=1)

			elif self.type_in=="~energy":
				path = '01_data//03_energy//ENB2012_data.csv'
				data = np.loadtxt(path, delimiter=',', skiprows=1)

				data = data[:,:-1] # have 2 y variables, ignore last

			elif self.type_in=="~kin8":
				path = '01_data//04_kin8nm//Dataset.csv'
				data = np.loadtxt(path, skiprows=0)

			elif self.type_in=="~naval":
				path = '01_data//05_naval//data.csv'
				data = np.loadtxt(path, skiprows=0)
				data = data[:,:-1] # have 2 y variables, ignore last

			elif self.type_in=="~power":
				path = '01_data//06_power_plant//Folds5x2_pp.csv'
				data = np.loadtxt(path, delimiter=',',skiprows=1)

			elif self.type_in=="~protein":
				path = '01_data//07_protein//CASP.csv'
				data_1 = np.loadtxt(path, delimiter=',',skiprows=1)
				# we are predicting the first column
				# put first column at end
				data = np.c_[data_1[:,1:], data_1[:,0]]
			
			elif self.type_in=="~wine":
				path = '01_data//08_wine//winequality-red.csv'
				data = np.loadtxt(path, delimiter=';',skiprows=1)

			elif self.type_in=="~yacht":
				path = '01_data//09_yacht//yacht_hydrodynamics.csv'
				data = np.loadtxt(path,skiprows=0)

			elif self.type_in=="~song":
				# path = '01_data//10_year_msd//YearPredictionMSD.csv'
				# absolute path for this one - v large so causes problem w git
				path = '//Users//tpearce/Documents//03_random//YearPredictionMSD.csv'
				path = 'YearPredictionMSD.csv' # for linux vm
				data_1 = np.loadtxt(path, delimiter=',',skiprows=0)
				# predicting the first column
				data = np.c_[data_1[:,1:], data_1[:,0]]
				# data = data[0:10000] # first 10k for now

			elif self.type_in=="~warranty":
				data = get_warranty_data()
				data = data[0:20000]

			scale_c = np.std(data[:,-1])
			shift_c = np.mean(data[:,-1])

			# visualise
			# for i in range(0, data.shape[1]):
			# 	fig, ax = plt.subplots(1)
			# 	ax.hist(data[0:1000,i],bins=10)
			# 	fig.show()

			# normalise data
			for i in range(0,data.shape[1]):
				# avoid zero variance features (exist one or two)
				sdev_norm = np.std(data[:,i])
				sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm
				data[:,i] = (data[:,i] - np.mean(data[:,i]) )/sdev_norm

			# split into train test
			perm = np.random.permutation(data.shape[0])
			train_size = int(round(train_prop*data.shape[0]))
			train = data[perm[:train_size],:]
			test = data[perm[train_size:],:]

			y_train = train[:,-1].reshape(-1,1)
			X_train = train[:,:-1]
			y_val = test[:,-1].reshape(-1,1)
			X_val = test[:,:-1]

			# override
			# y_val = y_train
			# X_val = X_train

		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.X_ideal = X_ideal
		self.y_ideal_U = y_ideal_U
		self.y_ideal_L = y_ideal_L
		self.y_ideal_mean = y_ideal_mean
		self.scale_c = scale_c
		self.shift_c = shift_c

		return X_train, y_train, X_val, y_val


	def ViewData(self, n_rows=5, hist=False, plot=False, print_=True):
		"""
		print first few rows of data
		option to view histogram of x and y
		option to view scatter plot of x vs y
		"""
		if print_:
			print("\nX_train\n",self.X_train[:n_rows], 
				"\ny_train\n", self.y_train[:n_rows], 
				"\nX_val\n", self.X_val[:n_rows], 
				"\ny_val\n", self.y_val[:n_rows])

		if hist:
			fig, ax = plt.subplots(1, 2)
			ax[0].hist(self.X_train)
			ax[1].hist(self.y_train)
			ax[0].set_title("X_train")
			ax[1].set_title("y_train")
			fig.show()

		if plot:
			n_feat = self.X_train.shape[1]
			fig, ax = plt.subplots(n_feat, 1) # create an extra
			if n_feat == 1:	ax = [ax] # make into list
			for i in range(0,n_feat):
				ax[i].scatter(self.X_train[:,i],self.y_train,
					alpha=0.5,s=2.0)
				# ax[i].scatter(self.X_train[self.censor_R_ind==True],
				# 	self.y_train[self.censor_R_ind==True], alpha=0.5,s=8.0, marker='^', c='r')
				# ax[i].scatter(self.X_train[self.censor_R_ind==False],
				# 	self.y_train[self.censor_R_ind==False], alpha=0.5,s=8.0, marker='o', c='b')
				
				# ax[i].scatter(self.X_train[:,i],self.y_train,
				# 	alpha=0.5,s=2.0, c=self.censor_R_ind)
				ax[i].set_xlabel('x_'+str(i))
				ax[i].set_ylabel('y')
			# fig.delaxes(ax.flatten()[-1])
			fig.show()
			# plt.tight_layout()

		return

		
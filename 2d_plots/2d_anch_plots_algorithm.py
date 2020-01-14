
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(104)

# ===============================
# this code creates figure 2 plot
# from 
# Uncertainty in Neural Networks: 
# Approximately Bayesian Ensembling
# ===============================

# inputs
n_samples = 50
is_save = 0
# fig_size_in = (3,3)

# prior
prior_mean = [0.,0.]
lambda_1 = 0.7 # from eq. 31
prior_cov = lambda_1*np.array([[1.,0.],[0.,1.]])
prior_cov_inv = np.linalg.inv(prior_cov)

# likelihood
like_mean = [3.,-1] # use as of 5 sep 2018
# like_mean = [0,0.5]
like_cov = 0.5*np.identity(2) # nice isotropic one
# like_cov = np.array([[0.2,0.1],[0.1,0.2]])*2
like_cov = np.array([[0.15,0.1],[0.1,0.25]])*3 # use as of 5 sep 2018
# like_cov = np.array([[0.25,0.249],[0.249,0.25]])*3 # shows convergence to prior
# like_cov = np.array([[0.25,0.12],[0.12,0.25]])*10 # shows convergence to prior
# like_cov = np.array([[0.25,0.25*0.99999],[0.25*0.99999,0.25]])*3000
# like_cov = np.array([[0.2,0.1],[0.1,0.25]]) # used to use
# like_cov = np.array([[0.3,0.1],[0.1,0.3]])
like_cov_inv = np.linalg.inv(like_cov)

# play with matrices
if False:
    a = np.array([[0.2,0.1],[0.1,0.25]])*100
    a = np.array([[1,1*0.99999999],[1*0.99999999,1]])*10000
    # a = np.array([[1,0.1,0.3],[0.1,0.9,0.6],[0.1,0.6,1]])*100
    a = np.array([[1,1*0.99999999,0.3],[1*0.99999999,1,0.6],[0.3,0.6,1]])*10000
    
    larg_cor = 1*0.999
    # a = np.array([[1,larg_cor,larg_cor],[larg_cor,1,larg_cor],[larg_cor,larg_cor,1]])*1
    # a = np.array([[1,larg_cor,0],[larg_cor,1,0.01],[0,0.01,5]])*1000
    a = np.array([[1,larg_cor,0],[larg_cor,1,0.01],[0,0.01,10]])*1000
    print('matrix\n',a)
    print('\ninverse\n',np.linalg.inv(a))
    prior_cov_l = np.identity(a.shape[0])
    print('\nanchor\n',prior_cov_l + np.matmul(prior_cov_l**2, np.linalg.inv(a)))

# set up plot
rc('legend',**{'fontsize':19})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 32})
plt.rcParams['text.usetex'] = True


# xlimits = [-1.6, 3.8]
xlimits = [-2.1, 3.8]
ylimits = [-2.7, 2.7]
x = np.linspace(*xlimits, num=200)
y = np.linspace(*ylimits, num=200)
X, Y = np.meshgrid(x, y)
fig = plt.figure(figsize=(5, 5)); fig.clf()
ax = fig.add_subplot(111)

# prior
rv = multivariate_normal(prior_mean, prior_cov)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
prior = rv.pdf(pos)
CS2 = plt.contour(X, Y, prior, 2, colors='r',alpha=1.0,lw=1)
CS2.collections[0].set_label('Prior')

# likelihood
rv = multivariate_normal(like_mean, like_cov)
like = rv.pdf(pos)
CS1 = plt.contour(X, Y, like, 2, colors='k',alpha=1.0, lw=1)
CS1.collections[0].set_label('Likelihood')

# numerical posterior
if False:
	post = prior * like
	post = post / np.sum(post)

# calculate anchor dist
ident = np.identity(2)
# anch_cov_analy = np.matmul(np.matmul((ident + lambda_1*like_cov_inv),np.linalg.inv(( (1/lambda_1)*ident + like_cov_inv))),(ident + lambda_1*like_cov_inv))
anch_cov_analy = prior_cov + np.matmul(np.matmul(prior_cov,prior_cov) , like_cov_inv)
# anch_cov_analy = lambda_1*ident + lambda_1**2 * like_cov_inv

print('exact anchor:\n',anch_cov_analy)
if 0:
    anch_cov_analy = prior_cov # overwrite
    print('using approx anchor:\n',anch_cov_analy)

anch_cov_analy_inv = np.linalg.inv(anch_cov_analy)
anch_mean = prior_mean.copy()
# disp_name = 'Anchor dist. exact'
disp_name = 'Initialisation dist.'
save_name = 'MAP_recon_exact_2.eps'

# isotropic likelihood dist
if False:
    lambda_2 = np.trace(like_cov) / like_cov.shape[0]
    like_cov_iso = lambda_2*np.identity(2)
    like_cov_iso_inv = np.linalg.inv(like_cov_iso)
    # anch_cov_analy = np.matmul(np.matmul((ident + lambda_1*like_cov_iso_inv),np.linalg.inv(( (1/lambda_1)*ident + like_cov_iso_inv))),(ident + lambda_1*like_cov_iso_inv))
    # anch_cov_analy = ((lambda_1/lambda_2) + 1)**2 * (1/((1/lambda_1) + (1/lambda_2))*np.identity(2)
    anch_cov_analy = (lambda_1**2/lambda_2 + lambda_1) *np.identity(2)
    print('isotropic anchor:\n',anch_cov_analy)
    disp_name = 'Anchor dist. isotropic'
    save_name = 'MAP_recon_approx_2.eps'

# approximate with diagonal cov matrix - directly on ANCHOR
if False:
    anch_avg = np.trace(anch_cov_analy) / anch_cov_analy.shape[0]
    anch_cov_analy = anch_avg*np.identity(2)
    anch_cov_analy_inv = np.linalg.inv(anch_cov_analy)
    print('isotropic anchor:\n',anch_cov_analy)
    disp_name = 'Anchor dist. isotropic'
    save_name = 'MAP_recon_approx_2.eps'

# anchor
rv = multivariate_normal(anch_mean, anch_cov_analy)
anchor = rv.pdf(pos)

# analytical posterior
post_cov_analy = np.linalg.inv(prior_cov_inv+like_cov_inv)
post_mean_analy = np.matmul(np.matmul(post_cov_analy,prior_cov_inv),prior_mean)\
    + np.matmul(np.matmul(post_cov_analy,like_cov_inv),like_mean)
rv_post_analy = multivariate_normal(post_mean_analy, post_cov_analy)
CS4 = plt.contour(X, Y, rv_post_analy.pdf(pos), 2, colors='g',alpha=1.0,lw=1.)
ax.contour(X, Y, rv_post_analy.pdf(pos), 2, colors='g',alpha=1.0,lw=1.)
CS4.collections[0].set_label('Posterior') # analytical

# centre of prior
ax.scatter(prior_mean[0],prior_mean[1],c='k', marker='+', s=100)

# ax.set_xlabel('Parameter 1')
# ax.set_ylabel('Parameter 2')
# ax.set_xlabel(r'$\theta_1$', fontsize=12)
# ax.set_ylabel(r'$\theta_2$', fontsize=12)
ax.legend(loc='upper right')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
# fig.set_size_inches((5,4))
fig.show()
plt.show(block=False)
if is_save:
    fig.savefig('00_outputs_graph/2d_MAP_anchor/Bayesian_step0.eps', format='eps', dpi=1000, bbox_inches='tight')



# now do MAP samples
# sample from prior
prior_samples = np.random.multivariate_normal(anch_mean, anch_cov_analy, size=n_samples)
X_MAP=[]; Y_MAP=[]
for i in range(prior_samples.shape[0]): # for each anchor sample
    if i%1000==0:
        print('running anchor sample: ',i)
    anchor_mean = prior_samples[i]
    anchor_dist = multivariate_normal(anchor_mean, prior_cov)
    anchor_prior = anchor_dist.pdf(pos)

    # work out anchor posterior
    anchor_post = anchor_prior * like
    anchor_post = anchor_post / np.sum(anchor_post)

    # work out x,y coords of MAP
    coord_MAP = np.argmax(anchor_post)
    X_MAP.append(X.ravel()[coord_MAP])
    Y_MAP.append(Y.ravel()[coord_MAP])

X_MAP = np.array(X_MAP)
Y_MAP = np.array(Y_MAP)


# prior anchor samples plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# scatter prior anchor samples
CS = plt.contour(X, Y, anchor, 2, colors='b',alpha=1.0,lw=1)
CS.collections[0].set_label(disp_name)
ax.scatter(prior_samples[:,0], prior_samples[:,1],alpha=1.0, c='b',s=10.0, marker='+',label='Samples from anchor dist.')
# CS = ax.contour(X, Y, prior, 2, colors='r',alpha=1.)
# CS.collections[0].set_label('Prior')

# dummy figure so can generate counts etc
fig_temp = plt.figure()
ax_temp = fig_temp.add_subplot(111)
counts,ybins,xbins,image = ax_temp.hist2d(X_MAP,Y_MAP,bins=30,normed=True,range=[xlimits,ylimits])
fig_temp.show()
plt.close()

# scatter posterior anchor samples
CS2 = ax.contour(X, Y, rv_post_analy.pdf(pos), 2, colors='g',alpha=1.)
ax.scatter(X_MAP, Y_MAP,alpha=0.5, s=10.0, marker='x',c='g',label='Anchored MAP estimates')
CS2.collections[0].set_label('Target Posterior')
ax.legend(loc='upper right')
ax.set_xlim(xlimits)
ax.set_ylim(ylimits)
# ax.set_xlabel('Parameter 1')
# ax.set_ylabel('Parameter 2')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
fig.show()
plt.close()
# if is_save:
#     fig.savefig('00_outputs_graph/2d_MAP_anchor/'+save_name, format='eps', dpi=1000, bbox_inches='tight')
# plt.close() # get bored of opening this one

# posterior anchor samples - contour
if False:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	CS1 = ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
	    lw=1,colors='b')
	CS2 = ax.contour(X, Y, post, colors='g',alpha=1.)
	CS1.collections[0].set_label('MAP recon posterior')
	CS2.collections[0].set_label('True posterior')
	ax.legend()
	fig.show()
	# bored of this one




# step 1 plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# prior centre
ax.scatter(prior_mean[0],prior_mean[1],c='k', marker='+', s=100, label='Original prior centre')

# anchor dist
CS = plt.contour(X, Y, anchor, 2, colors='b',alpha=1.0,lw=0.5, linestyles='--')
CS.collections[0].set_label(disp_name)

# one sample from anchor
ax.scatter(prior_samples[0,0], prior_samples[0,1],alpha=1.0, c='b',s=80, marker='x',label='Samples from init. dist.')

ax.legend(loc='upper right')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
# ax.set_xlabel(r'$\theta_1$', fontsize=12)
# ax.set_ylabel(r'$\theta_2$', fontsize=12)
# fig.set_size_inches((5,4))
fig.show()
plt.show(block=False)
if is_save:
    fig.savefig('00_outputs_graph/2d_MAP_anchor/Bayesian_step1.eps', format='eps', dpi=1000, bbox_inches='tight')



# step 2 plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# prior centre
ax.scatter(prior_mean[0],prior_mean[1],c='k', marker='+', s=100)#,m='+')

# anchor dist
CS = plt.contour(X, Y, anchor, 2, colors='b',alpha=1.0,lw=0.5, linestyles='--')
# CS.collections[0].set_label(disp_name)

# prior dist over the top
rv = multivariate_normal([prior_samples[0,0], prior_samples[0,1]], prior_cov)
prior_samp = rv.pdf(pos)
CS2 = plt.contour(X, Y, prior_samp, 2, colors='r',alpha=1.0,lw=1)
CS2.collections[0].set_label('Recentered prior')

# likelihood
rv = multivariate_normal(like_mean, like_cov)
like = rv.pdf(pos)
CS1 = plt.contour(X, Y, like, 2, colors='k',alpha=1.0, lw=1)
CS1.collections[0].set_label('Original likelihood')

# one sample from anchor
ax.scatter(prior_samples[0,0], prior_samples[0,1],alpha=1.0, c='b',s=70, marker='x')

# MAP point estimate for that sample
ax.scatter(X_MAP[0], Y_MAP[0],alpha=1, s=90, marker='*',c='g',label='Anchored MAP estimates')

ax.legend(loc='upper right')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
# ax.set_xlabel(r'$\theta_1$', fontsize=12)
# ax.set_ylabel(r'$\theta_2$', fontsize=12)
# fig.set_size_inches((5,4))
fig.show()
plt.show(block=False)
if is_save:
    fig.savefig('00_outputs_graph/2d_MAP_anchor/Bayesian_step2.eps', format='eps', dpi=1000, bbox_inches='tight')




# step 3 plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# prior centre
ax.scatter(prior_mean[0],prior_mean[1],c='k', marker='+', s=80)#,m='+')

# anchor dist
CS = plt.contour(X, Y, anchor, 2, colors='b',alpha=1.0,lw=0.5, linestyles='--')
# CS.collections[0].set_label(disp_name)

# all samples from anchor
ax.scatter(prior_samples[:,0], prior_samples[:,1],alpha=1.0, c='b',s=50, marker='x')

# plot all MAP estimates
CS2 = ax.contour(X, Y, rv_post_analy.pdf(pos), 2, colors='g',alpha=1.)
ax.scatter(X_MAP, Y_MAP,alpha=1, s=50, marker='*',c='g')#,label='Anchored MAP estimates')
CS2.collections[0].set_label('Original Posterior')

# find and plot analytical anchored ensemble posterior
if True:
    A = np.matmul(np.linalg.inv(prior_cov_inv+like_cov_inv),prior_cov_inv)
    anch_cov_analy_post = np.matmul(np.matmul(anch_cov_analy,A),A.T)
    rv_anch_post_analy = multivariate_normal(post_mean_analy, anch_cov_analy_post)
    CS3 = ax.contour(X, Y, rv_anch_post_analy.pdf(pos), 2, colors='k',alpha=1.,lw=2)
    CS3.collections[0].set_label('MAP Posterior')
    print('\n Actual posterior\n',post_cov_analy)
    print('\n Anch. Ens. posterior\n',anch_cov_analy_post)
    np.matmul(np.matmul(post_cov_analy, post_cov_analy),prior_cov_inv)

    # i'm not convince kl divergence is out metric of interest
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    # sig1 = post_cov_analy
    # sig2 = anch_cov_analy_post
    # mu1 = mu2 = post_mean_analy
    # kl1 = np.log(np.linalg.det(sig1)/np.linalg.det(sig2)) 
    # kl2 = np.shape(sig1)[0]
    # kl3 = np.trace(np.matmul(np.linalg.inv(sig2),sig1))
    # kl4 = 0 # since our means are alligned
    # kl_div = 1/2 * (kl1 - kl2 + kl3 + kl4)
    # print('\n KL Divergence:', kl_div)

ax.legend(loc='upper right')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(xlimits)
ax.set_ylim(ylimits)
# ax.set_xlabel(r'$\theta_1$', fontsize=12)
# ax.set_ylabel(r'$\theta_2$', fontsize=12)
# fig.set_size_inches((5,4))
fig.show()
plt.show(block=False)
if is_save:
    fig.savefig('00_outputs_graph/2d_MAP_anchor/Bayesian_step3.eps', format='eps', dpi=1000, bbox_inches='tight')


# do analytical test with the posterior i produce for differing correlation levels
prior_mean = [0.,0.]
lambda_1 = 0.7 # from eq. 31
prior_cov = lambda_1*np.array([[1.,0.],[0.,1.]])
prior_cov_inv = np.linalg.inv(prior_cov)

# likelihood
like_mean = [3.,-1]
like_cov = np.array([[0.25,-0.2],[-0.2,0.25]])*3
like_cov_inv = np.linalg.inv(like_cov)

plt.show()


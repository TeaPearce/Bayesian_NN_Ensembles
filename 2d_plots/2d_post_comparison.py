
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(104)

# ===============================
# this code creates figure 3 plot
# from 
# Uncertainty in Neural Networks: 
# Approximately Bayesian Ensembling
# ===============================

# inputs
n_samples = 50
is_save = 0

type_post = 'general' # general, correlated, extrapolation

# prior
prior_mean = [0.,0.]
lambda_1 = 1. # from eq. 31
prior_cov = lambda_1*np.array([[1.,0.],[0.,1]])
prior_cov_inv = np.linalg.inv(prior_cov)

for type_inf in ['analytical','anchor','vi']:
    # likelihood
    # 
    if type_post == 'general':
        like_mean = [1.,-1]
        corr = 0.8
        a = 1.*1.5
        d = 1.*1.5
    elif type_post == 'extrapolation':
        like_mean = [1.*0.5,-1*0.5] # use for vi plots
        corr = 0.
        a = 1.*2000
        d = 1.*2000
    elif type_post == 'correlated':
        like_mean = [1.*0.5,-1*0.5] # use for vi plots
        corr = 0.9999
        a = 1.*2
        d = 1.*2
    else:
        raise Exception

    cov_off_diag = corr*np.sqrt(a*d)
    like_cov = np.array([[a,cov_off_diag],[cov_off_diag,d]]) # use for vi
    like_cov_inv = np.linalg.inv(like_cov)

    # set up plot
    rc('legend',**{'fontsize':19})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 32})
    plt.rcParams['text.usetex'] = True


    xlimits = [-2.1, 3.8]
    ylimits = [-2.7, 2.7]
    x = np.linspace(*xlimits, num=2000)
    y = np.linspace(*ylimits, num=2000)
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
    # CS2.collections[0].set_label('Prior')

    # likelihood
    rv = multivariate_normal(like_mean, like_cov)
    like = rv.pdf(pos)
    # CS1 = plt.contour(X, Y, like, 2, colors='k',alpha=1.0, lw=1)
    # CS1 = plt.contour(X, Y, like, 2, colors='k',alpha=1.0, linewidths=2., levels = [ 0.09])
    # CS1.collections[0].set_label('Likelihood')


    # calculate anchor dist
    ident = np.identity(2)
    # anch_cov_analy = np.matmul(np.matmul((ident + lambda_1*like_cov_inv),np.linalg.inv(( (1/lambda_1)*ident + like_cov_inv))),(ident + lambda_1*like_cov_inv))
    # anch_cov_analy = prior_cov + np.matmul(np.matmul(prior_cov,prior_cov) , like_cov_inv)
    anch_cov_analy = prior_cov + np.matmul(np.matmul(prior_cov,like_cov_inv),prior_cov)
    anch_cov_analy = prior_cov  # if want to overwrite and use workaround
    # anch_cov_analy = lambda_1*ident + lambda_1**2 * like_cov_inv


    anch_cov_analy_inv = np.linalg.inv(anch_cov_analy)
    anch_mean = prior_mean.copy()
    # disp_name = 'Anchor dist. exact'
    disp_name = 'Initialisation dist.'
    save_name = 'MAP_recon_exact_2.eps'


    # anchor
    rv = multivariate_normal(anch_mean, anch_cov_analy)
    anchor = rv.pdf(pos)

    # analytical posterior
    post_cov_analy = np.linalg.inv(prior_cov_inv+like_cov_inv)
    post_mean_analy = np.matmul(np.matmul(post_cov_analy,prior_cov_inv),prior_mean)\
        + np.matmul(np.matmul(post_cov_analy,like_cov_inv),like_mean)
    rv_post_analy = multivariate_normal(post_mean_analy, post_cov_analy)

    # posterior here
    if type_inf=='analytical':
        if type_post == 'correlated':
            CS4 = plt.contour(X, Y, rv_post_analy.pdf(pos)/rv_post_analy.pdf(pos).max(), 2, colors='g',alpha=1.0,linewidths=3.,levels = [0.2])
        elif type_post == 'extrapolation':
            CS4 = plt.contour(X, Y, rv_post_analy.pdf(pos)/rv_post_analy.pdf(pos).max(), 2, colors='g',alpha=1.0,linewidths=3.,levels = [0.35])
        else:
            CS4 = plt.contour(X, Y, rv_post_analy.pdf(pos)/rv_post_analy.pdf(pos).max(), 2, colors='g',alpha=1.0,linewidths=3.,levels = [0.35])
        CS4.collections[0].set_label('True post.') # analytical

    # anchored post
    elif type_inf=='anchor':
        rv_post_anch_analy = multivariate_normal(post_mean_analy, np.matmul(np.matmul(post_cov_analy,prior_cov_inv),post_cov_analy))
        if type_post == 'correlated':
            CS4 = plt.contour(X, Y, rv_post_anch_analy.pdf(pos)/rv_post_anch_analy.pdf(pos).max(), 2, colors='purple',alpha=1.0,linewidths=5.,levels = [0.2])
        elif type_post == 'extrapolation':
            CS4 = plt.contour(X, Y, rv_post_anch_analy.pdf(pos)/rv_post_anch_analy.pdf(pos).max(), 2, colors='purple',alpha=1.0,linewidths=5.,levels = [0.35])
        else:
            CS4 = plt.contour(X, Y, rv_post_anch_analy.pdf(pos)/rv_post_anch_analy.pdf(pos).max(), 2, colors='purple',alpha=1.0,linewidths=3.,levels = [0.35])  
        CS4.collections[0].set_label('RMS approx post.') # analytical

    elif type_inf=='vi':
        # VI post
        vi_cov_analy_post = np.eye(post_cov_analy.shape[0])
        post_cov_analy_inv = np.linalg.inv(post_cov_analy)
        vi_cov_analy_post[0,0] = 1/post_cov_analy_inv[1,1]
        vi_cov_analy_post[1,1] = 1/post_cov_analy_inv[0,0]

        rv_vi_post_analy = multivariate_normal(post_mean_analy, vi_cov_analy_post)
        if type_post == 'correlated':
            CS5 = ax.contour(X, Y, rv_vi_post_analy.pdf(pos)/rv_vi_post_analy.pdf(pos).max(), 2, colors='blue',alpha=1.,lw=2,linewidths=3.,levels = [0.2])
        elif type_post == 'extrapolation':
            CS5 = ax.contour(X, Y, rv_vi_post_analy.pdf(pos)/rv_vi_post_analy.pdf(pos).max(), 2, colors='blue',alpha=1.,lw=2,linewidths=3.,levels = [0.35])
        else:
            CS5 = ax.contour(X, Y, rv_vi_post_analy.pdf(pos)/rv_vi_post_analy.pdf(pos).max(), 2, colors='blue',alpha=1.,lw=2,linewidths=3.,levels = [0.35])
        CS5.collections[0].set_label('MFVI post.')

    else:
        raise Exception

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
    plt.show(block = False)

    

    if is_save:
        fig.savefig('general_vi'+ type_inf +'.pdf', format='pdf', dpi=1000, bbox_inches='tight')

plt.show()



from sklearn.neighbors import KernelDensity
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

from engine import get_data

#####################################
##### Kernel Density Estimation #####
#####################################

def estimate_kde(dist_type: str, log_returns):
	'''
	Kernel Density estimation based on historical data.
	:log_returns: series of returns for the pair of interest.
	'''
	#Convert to the right format for the Kernel Density estimation
	X = log_returns.to_numpy().reshape(-1,1)
	kde = KernelDensity(kernel=dist_type, bandwidth=0.003).fit(X)
	return kde

#####################################
######## KDE plotting ###############
#####################################

def plot_kdes(log_returns):
	'''
	Plot KDEs estimated with three different estimation methods. 
	:log_returns: series of returns for the pair of interest.
	'''
	#Convert to the right format for the Kernel Density estimation
	X = log_returns.to_numpy().reshape(-1,1)
	X_plot = np.linspace((X.min()-1), X.max()+1, 1000)[:, np.newaxis]

	fig, ax = plt.subplots()
	colors = ['navy', 'cornflowerblue', 'darkorange']
	kernels = ['gaussian', 'tophat', 'epanechnikov']
	lw = 2

	for color, kernel in zip(colors, kernels):
	    kde = KernelDensity(kernel=kernel, bandwidth=0.003).fit(X)
	    log_dens = kde.score_samples(X_plot)
	    ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
	            linestyle='-', label="kernel = '{0}'".format(kernel))

	ax.legend(loc='upper left')
	ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
	ax.set_xlim(-0.15, 0.15)
	plt.show()
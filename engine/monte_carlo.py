import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from engine import get_data

def monte_carlo_sim(df, num_simulations, predicted_periods):

	returns = df['log_returns']
	prices = df['close']
	last_price = prices.iloc[-1]

	#Create Each Simulation as a Column in df
	simulation_df = pd.DataFrame()
	for x in range(num_simulations):
		count = 0
		period_vol = returns.std()
		price_series = []

		#Append Start Value
		price = last_price * (1 + np.random.normal(0, period_vol))
		price_series.append(
			price)

		#Series for predicted Days
		for i in range(predicted_periods):
			price = price_series[count] * (1 + np.random.normal(0, period_vol))
			price_series.append(price)
			count += 1

		simulation_df[x] = price_series

	return simulation_df


def brownian_motion(df, num_simulations, predicted_periods):

	returns = df['log_returns']
	prices = df['close']
	last_price = prices.iloc[-1]

	#Note we are assuming drift here
	simulation_df = pd.DataFrame()

	#Create Each Simulation as a Column in df
	for x in range(num_simulations):

		#Inputs
		count = 0
		avg_period_ret = returns.mean()
		variance = returns.var()

		period_vol = returns.std()
		period_drift = avg_period_ret - (variance/2)
		drift = period_drift - 0.5 * period_vol ** 2

		#Append Start Value
		prices = []

		shock = drift + period_vol * np.random.normal()
		last_price * math.exp(shock)
		prices.append(last_price)

		for i in range(predicted_periods):

			shock = drift + period_vol * np.random.normal()
			price = prices[count] * math.exp(shock)
			prices.append(price)
			count += 1

		simulation_df[x] = prices

	return simulation_df

def compute_percent_crisis(df):
	df = get_data.fetch_data()

def monte_carlo_empirical(df, num_simulations, predicted_periods):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	'''
	log_returns = df['log_returns']*100
	#Convert to the right format for the Kernel Density estimation
	X = log_returns.to_numpy().reshape(-1,1)

	prices = df['close']
	last_price = prices.iloc[-1]

	#Note we are assuming drift here
	simulation_df = pd.DataFrame()

	#Create Each Simulation as a Column in df
	for x in range(num_simulations):

		#Inputs
		count = 0
		avg_period_ret = log_returns.mean()
		variance = log_returns.var()

		period_vol = log_returns.std()
		period_drift = avg_period_ret - (variance/2)
		drift = period_drift - 0.5 * period_vol ** 2

		#Append Start Value
		prices = []

		shock = drift + period_vol * np.random.normal()
		last_price * math.exp(shock)
		prices.append(last_price)

		for i in range(predicted_periods):

			shock = drift + period_vol * np.random.normal()
			price = prices[count] * math.exp(shock)
			prices.append(price)
			count += 1

		simulation_df[x] = prices

	return simulation_df




X_plot = np.linspace((array.min()-1), array.max()+1, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')
colors = ['navy', 'cornflowerblue', 'darkorange']
kernels = ['gaussian', 'tophat', 'epanechnikov']
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
            linestyle='-', label="kernel = '{0}'".format(kernel))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

plt.show()

#Can only sample from 'gaussian', 'tophat' --> Go with tophat as closer to epanechnikov
kde = KernelDensity(kernel='tophat', bandwidth=0.5).fit(X)
#Sample from the data
new_data = kde.sample(1000)



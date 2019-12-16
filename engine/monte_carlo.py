import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from engine import get_data
from engine import kernel_estimation

# def monte_carlo_sim(df, num_simulations, predicted_periods):

# 	returns = df['log_returns']
# 	prices = df['close']
# 	last_price = prices.iloc[-1]

# 	#Create Each Simulation as a Column in df
# 	simulation_df = pd.DataFrame()
# 	for x in range(num_simulations):
# 		count = 0
# 		period_vol = returns.std()
# 		price_series = []

# 		#Append Start Value
# 		price = last_price * (1 + np.random.normal(0, period_vol))
# 		price_series.append(
# 			price)

# 		#Series for predicted Days
# 		for i in range(predicted_periods):
# 			price = price_series[count] * (1 + np.random.normal(0, period_vol))
# 			price_series.append(price)
# 			count += 1

# 		simulation_df[x] = price_series

# 	return simulation_df


# def brownian_motion(df, num_simulations, predicted_periods):

# 	returns = df['log_returns']
# 	prices = df['close']
# 	last_price = prices.iloc[-1]

# 	#Note we are assuming drift here
# 	simulation_df = pd.DataFrame()

# 	#Create Each Simulation as a Column in df
# 	for x in range(num_simulations):

# 		#Inputs
# 		count = 0
# 		avg_period_ret = returns.mean()
# 		variance = returns.var()

# 		period_vol = returns.std()
# 		period_drift = avg_period_ret - (variance/2)
# 		drift = period_drift - 0.5 * period_vol ** 2

# 		#Append Start Value
# 		prices = []

# 		shock = drift + period_vol * np.random.normal()
# 		last_price * math.exp(shock)
# 		prices.append(last_price)

# 		for i in range(predicted_periods):

# 			shock = drift + period_vol * np.random.normal()
# 			price = prices[count] * math.exp(shock)
# 			prices.append(price)
# 			count += 1

# 		simulation_df[x] = prices

# 	return simulation_df

# def compute_percent_crisis(df):
# 	df = get_data.fetch_data()

def monte_carlo_simulator(df, num_simulations, predicted_periods, dist_type: str):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	'''
	kde = kernel_estimation.estimate_kde(dist_type, df)
	log_returns = df['log_returns'].dropna()
	#Convert to the right format for the Kernel Density estimation

	initial_price = df['close'].iloc[-1]

	#Note we are assuming drift here
	simulation_df = pd.DataFrame()

	#Create Each Simulation as a Column in df
	for x in range(num_simulations):

		#Inputs
		count = 0
		avg_period_ret = log_returns.mean()
		period_vol = log_returns.std()
		period_var = log_returns.var()

		#Analytical params
		period_drift = avg_period_ret - (period_var/2)
		drift = period_drift - 0.5 * period_vol ** 2

		#Append Start Value
		prices = []
		prices.append(initial_price)

		for i in range(predicted_periods):

			shock = drift + period_vol * kde.sample(1)[0][0]
			price = prices[count] * math.exp(shock)
			prices.append(price)
			count += 1

		simulation_df[x] = prices

	return simulation_df
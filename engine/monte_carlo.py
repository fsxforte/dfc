import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from engine import get_data
from engine import kernel_estimation

def monte_carlo_simulator(from_sym, to_sym, start_time_historical, end_time_historical, dist_type, num_simulations, predicted_periods):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	Monte carlo equation: St = St-1* exp((μ-(σ2/2))*t + σWt).
	:df:
	:dist_type: KDE estimator to use, choices are 'gaussian' or 'tophat'.
	:num_simulations: number of runs of simulation to perform. 
	:predicted periods: number of periods to predict into the future. 
	'''
	df = get_data.create_df(from_sym, to_sym)
	df = df[start_time_historical:end_time_historical]

	log_returns = get_data.create_logrets_series(from_sym, to_sym)
	log_returns = log_returns[start_time_historical:end_time_historical]

	#Perform kernel density estimation
	kde = kernel_estimation.estimate_kde(dist_type, log_returns)

	#Price level data
	initial_price = df['close'].iloc[-1]

	mu = log_returns.mean()

	sigma = log_returns.std()

	simulation = {}
	for sim in range(num_simulations):
		simulation['Simulation' + str(sim)] = [initial_price]
		for days in range(predicted_periods):
			next_day = simulation['Simulation' + str(sim)][-1]*np.exp((mu-(sigma**2)/2) + sigma*kde.sample(1)[0][0])
			simulation['Simulation' + str(sim)].append(next_day)

	return simulation

simulations = pd.DataFrame(simulations)
import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from engine import get_data
from engine import kernel_estimation

def multivariate_monte_carlo(historical_prices, num_simulations, T, dt):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	Monte carlo equation: St = St-1* exp((μ-(σ2/2))*t + σWt).
	:historical_prices: dataframe where columns are assets and rows are time
	:num_simulations: number of runs of simulation to perform. 
	:T: length of time prediction horizon (in units of dt, i.e. days)
	:dt: time increment, i.e. frequency of data (using daily data here)
	'''
	#Set seed to ensure same simulation run
	np.random.seed(137)

	num_periods_ahead = int(T / dt)

	#From prices, calculate log returns
	log_returns = np.log(historical_prices) - np.log(historical_prices.shift(1))
	log_returns = log_returns.dropna()

	#Parameter assignment

	#Initial asset price
	S0 = historical_prices.iloc[0]
	#S0 = historical_prices.iloc[-1]


	#Mean log return
	mu = np.mean(log_returns)
	print('mu: ' + str(mu))

	#Standard deviation of log return
	sigma = np.std(log_returns)
	print('sigma: ' + str(sigma))
	#Diagonal sigmas
	#sd = np.diag(sigma)

	#Compute covariance matrix from historical prices
	corr_matrix = log_returns.corr()
	cov_matrix = log_returns.cov()
	print(corr_matrix)
	#cov_matrix = np.dot(sd, np.dot(corr_matrix, sd))

	#Cholesky decomposition
	Chol = np.linalg.cholesky(cov_matrix) 

	#Time index for predicted periods
	t = np.arange(1, int(num_periods_ahead) + 1)

	#Generate uncorrelated random sequences
	b = {str(simulation): np.random.normal(0, 1, (len(S0), num_periods_ahead)) for simulation in range(1, num_simulations + 1)}

	#Correlate them with Cholesky
	b_corr = {str(simulation): Chol.dot(b[str(simulation)]) for simulation in range(1, num_simulations + 1)}

	#Cumulate the shocks
	#W is keyed by simulations, within which rows correspond to assets and columns to periods ahead
	W = {}
	for simulation in range(1, num_simulations + 1):
		W[str(simulation)] = [b_corr[str(simulation)][asset].cumsum() for asset in range(len(S0))]

	#Drift
	#Drift is keyed by simulation, within which rows correspond to assets and colummns to periods ahead
	#Drift should grow linearly
	drift = {}
	for simulation in range(1, num_simulations + 1):
		drift[str(simulation)] = [(mu - 0.5 * sigma**2)[asset]*t for asset in range(len(S0))]

	#Diffusion
	diffusion = {}
	for simulation in range(1, num_simulations + 1):
		diffusion[str(simulation)] = [sigma[asset] * W[str(simulation)][asset] for asset in range(len(S0))]

	#Making the predictions
	simulations = {}
	for simulation in range(1, num_simulations + 1):
		simulations[str(simulation)] = [np.append(S0[asset], S0[asset] * np.exp(drift[str(simulation)][asset] + diffusion[str(simulation)][asset])) for asset in range(len(S0))]
		#simulations[str(simulation)] = [np.append(S0[asset], S0[asset] * np.exp(diffusion[str(simulation)][asset])) for asset in range(len(S0))]

	return simulations

def asset_extractor_from_sims(simulations, asset_index_in_basket):
	'''
	Function to pull out simulations for a particular asset.
	:simulations: input dictionary (output from multivariate monte carlo function)
	:asset_index_in_basket: if token basket is ['ETH', 'MKR', 'BAT'], then the index 0 refers to ETH
	'''
	asset_sims = {}
	
	for simulation in range(1, len(simulations)+1):
		asset_sims[str(simulation)] = simulations[str(simulation)][asset_index_in_basket]
	
	return asset_sims

def crash_simulator(simulations, DAI_DEBT, INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS, COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET, LIQUIDITY_DRYUP):
	'''
	Simulate the behaviour of a system collateralized to exactly 150% which faces downturn such that all debt sold off
	:param_of_interest: whether to return margins, dai_liabilities or eth_collateral in the output
	:simulations: monte carlo simulations of correlated price movements
	:DAI_DEBT: amount of system DAI DEBT
	:INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS: maximum liquidity supportable by market at start of crash, decays exponentially
	:COLLATERALIZATION_RATIO: system collateralization ratio
    '''
	sims = {}
	for simulation in range(1, len(simulations) + 1):
		eth_sim_prices = simulations[str(simulation)][0]
		mkr_sim_prices = simulations[str(simulation)][1]
		total_margins = []
		debt = []
		eth_collateral = []
		for index, price in enumerate(eth_sim_prices):
			if index == 0:

				#Set the initial base case from the first price where a sell off of all DAI_DEBT is triggered
				eth_price = eth_sim_prices[index] # ETH/USD

				#Calculate the ETH holdings corresponding to the assumption of exactly 150% collateralization at the start
				starting_eth_collateral = DAI_DEBT * COLLATERALIZATION_RATIO / eth_sim_prices[index] #ETH
				
				#Assets
				eth_collateral.append(starting_eth_collateral) #ETH

				#Liabilities
				debt.append(DAI_DEBT) #USD

				#MARGIN
				total_eth = starting_eth_collateral * eth_sim_prices[index]  #USD
				#print('Total ETH: ' + str(total_eth))
				total_mkr = QUANTITY_RESERVE_ASSET * mkr_sim_prices[index] #USD
				#print('Total MKR: ' + str(total_mkr))
				margin = total_eth + total_mkr - DAI_DEBT #USD
				#print('Debt: ' + str(DAI_DEBT))
				#print('Total margin: ' + str(margin))
				total_margins.append(margin)
				
			if (index > 0) & (index < len(eth_sim_prices) - 1):
				
				#Debt
				debt_start_period = debt[index - 1] # USD

				#Calculate how many USD of ETH can be sold each period
				avg_eth_price = (eth_sim_prices[index-1] + eth_sim_prices[index]) / 2
				max_daily_eth_liquidation_usd = INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS*np.math.exp(-1 * LIQUIDITY_DRYUP * index) * avg_eth_price #USD

				if debt_start_period > max_daily_eth_liquidation_usd:
					#Assets
					eth_collateral_end_period = eth_collateral[index - 1] - INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS*np.math.exp(-1 * LIQUIDITY_DRYUP * index) #ETH
					eth_collateral.append(eth_collateral_end_period)
					#Liabilities
					debt_end_period = debt_start_period - max_daily_eth_liquidation_usd # USD
					debt.append(debt_end_period)
				else:
					#Assets
					eth_collateral_end_period = eth_collateral[index - 1] - debt_start_period/avg_eth_price
					eth_collateral.append(eth_collateral_end_period)
					debt_end_period = 0
					#Liabilities
					debt.append(debt_end_period)
				
				#MARGIN
				total_eth = eth_collateral_end_period * eth_sim_prices[index] #USD
				#print('Total ETH: ' + str(total_eth))
				total_mkr = QUANTITY_RESERVE_ASSET * mkr_sim_prices[index] #USD
				#print('Total MKR: ' + str(total_mkr))
				total_margin = total_eth + total_mkr - debt_end_period #USD
				#print('Debt: ' + str(debt_end_period))
				#print('Total margin: ' + str(total_margin))
				total_margins.append(total_margin)
		
		sims[str(simulation)] = (total_margins, debt)

	return sims
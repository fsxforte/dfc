import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from engine import get_data
from engine import kernel_estimation

def multivariate_monte_carlo(historical_prices, num_simulations, num_periods_ahead):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	Monte carlo equation: St = St-1* exp((μ-(σ2/2))*t + σWt).
	:historical_prices: dataframe where columns are assets and rows are time
	:num_simulations: number of runs of simulation to perform. 
	:predicted_periods: number of periods to predict into the future. 
	'''

	#From prices, calculate log returns
	log_returns = np.log(historical_prices) - np.log(historical_prices.shift(1))
	log_returns = log_returns.dropna()

	#Parameter assignment

	#Initial asset price
	S0 = historical_prices.iloc[-1]

	#Mean log return
	mu = np.mean(log_returns)

	#Standard deviation of log return
	sigma = np.std(log_returns)
	#Diagonal sigmas
	sd = np.diag(sigma)

	#Compute covariance matrix from historical prices
	corr_matrix = log_returns.corr()
	cov_matrix = np.dot(sd, np.dot(corr_matrix, sd))

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

def crash_simulator(simulations, DAI_DEBT, MAX_ETH_SELLABLE_IN_24HOURS, COLLATERALIZATION_RATIO):
	'''
	Simulate the behaviour of a system collateralized to exactly 150% which faces downturn such that all debt sold off
	:param_of_interest: whether to return margins, dai_liabilities or eth_collateral in the output
	:simulations: monte carlo simulations of correlated price movements
	:DAI_DEBT: amount of system DAI DEBT
	:MAX_ETH_SELLABLE_IN_24_HOURS: maximum liquidity supportable by market
	:COLLATERALIZATION_RATIO: system collateralization ratio
    '''
	sims = {}
	for simulation in range(1, len(simulations) + 1):
		sim_version = simulations[str(simulation)]
		new_sim_version = []
		for asset_array in sim_version:
			margins = []
			dai_liability = []
			eth_collateral = []
			for index, price in enumerate(asset_array):

				if index == 0:

					#Set the initial base case from the first price where a sell off of all DAI_DEBT is triggered
					dai_balance_outstanding = DAI_DEBT
					avg_eth_price = (asset_array[index] + asset_array[index + 1]) / 2

					#Calculate amount of ETH needed to be at 150% collateralization
					starting_eth_collateral = DAI_DEBT * COLLATERALIZATION_RATIO / asset_array[index]
					max_eth_liquidation_usd = MAX_ETH_SELLABLE_IN_24HOURS * avg_eth_price

					if dai_balance_outstanding > max_eth_liquidation_usd:
						#Assets
						remaining_collateral = starting_eth_collateral - MAX_ETH_SELLABLE_IN_24HOURS
						eth_collateral.append(remaining_collateral)
						#Liabilities                    
						residual_dai = dai_balance_outstanding - max_eth_liquidation_usd
						dai_liability.append(residual_dai)                    
					else:
						#Assets
						remaining_collateral = starting_eth_collateral - dai_balance_outstanding/avg_eth_price
						eth_collateral.append(remaining_collateral)
						#Liabilities
						residual_dai = 0
						dai_liability.append(residual_dai)

					#MARGIN
					margin = remaining_collateral * asset_array[index + 1] - residual_dai
					margins.append(margin)
					
				if (index < len(asset_array) - 1) & (index > 0):
					#DAI liabilities
					dai_balance_outstanding = dai_liability[index - 1]
					avg_eth_price = (asset_array[index] + asset_array[index + 1]) / 2
					max_eth_liquidation_usd = MAX_ETH_SELLABLE_IN_24HOURS * avg_eth_price

					if dai_balance_outstanding > max_eth_liquidation_usd:
						#Assets
						remaining_collateral = eth_collateral[index - 1] - MAX_ETH_SELLABLE_IN_24HOURS
						eth_collateral.append(remaining_collateral)
						#Liabilities
						residual_dai = dai_balance_outstanding - max_eth_liquidation_usd
						dai_liability.append(residual_dai)
					else:
						#Assets
						remaining_collateral = eth_collateral[index - 1] - dai_balance_outstanding/avg_eth_price
						eth_collateral.append(remaining_collateral)
						#Liabilities
						residual_dai = 0
						dai_liability.append(residual_dai)
					
					#MARGIN
					margin = remaining_collateral * asset_array[index + 1] - residual_dai
					margins.append(margin)

			new_sim_version.append(margins)

		sims[str(simulation)] = new_sim_version

	return sims
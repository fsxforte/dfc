import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import uniform

from engine import get_data
from engine import kernel_estimation

def multivariate_monte_carlo_normal(historical_prices, num_simulations, T, dt):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	Monte carlo equation: St = St-1* exp((μ-(σ2/2))*t + σWt).
	:historical_prices: dataframe where columns are assets and rows are time
	:length_of_crash: number of days the worst ETH shock (-22.45\%) occurs
	:shock_size: size of worst eth shock
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
	S0 = historical_prices.iloc[-1]

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

	#Cholesky decomposition
	Chol = np.linalg.cholesky(corr_matrix)
	#Chol = np.linalg.cholesky(cov_matrix)

	#Time index for predicted periods
	t = np.arange(1, int(num_periods_ahead) + 1)

	#Generate uncorrelated random sequences
	b = {}
	for simulation in range(1, num_simulations + 1):
		sim = []
		for asset in log_returns.columns:
			random_sequence = []
			for period in range(num_periods_ahead):
				random_sequence.append(np.random.normal(0, 1))
			sim.append(random_sequence)
		b[str(simulation)] = sim

	#Correlate them with Cholesky
	b_corr = {str(simulation): Chol.dot(b[str(simulation)]) for simulation in range(1, num_simulations + 1)}
	
	#Cumulate the shocks
	#W is keyed by simulations, within which rows correspond to assets and columns to periods ahead
	W = {}
	for simulation in range(1, num_simulations + 1):
		W[str(simulation)] = [np.cumsum(b_corr[str(simulation)][asset]) for asset in range(len(S0))]

	simulations = {}
	for simulation in range(1, num_simulations + 1):

		sim = []

		for asset in range(len(S0)):
			
			#Calculate drift
			drift = (mu[asset] - 0.5 * (sigma[asset]**2)) * t

			#Calculate Diffusion
			diffusion = W[str(simulation)][asset]  * sigma[asset]

			#Make predictions
			predicted_path = np.append(S0[asset], S0[asset]*np.exp(diffusion+drift))

			sim.append(predicted_path)	

		simulations[str(simulation)] = sim

	return simulations


def multivariate_monte_carlo_historical(historical_prices, num_simulations, T, dt):
	'''
	Perform Monte Carlo Simulation using empirical distribution of log returns (via Kernel Density Estimate).
	Monte carlo equation: St = St-1* exp((μ-(σ2/2))*t + σWt).
	:historical_prices: dataframe where columns are assets and rows are time
	:length_of_crash: number of days the worst ETH shock (-22.45\%) occurs
	:shock_size: size of worst eth shock
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

	#Mean log return
	mu = np.mean(log_returns)
	print('mu: ' + str(mu))

	#Standard deviation of log return
	sigma = np.std(log_returns)
	print('sigma: ' + str(sigma))
	#Diagonal sigmas
	#sd = np.diag(sigma)

	#Time index for predicted periods
	t = np.arange(1, int(num_periods_ahead) + 1)

	#Initial asset price
	S0 = historical_prices.iloc[-1]

	b = {}
	for simulation in range(1, num_simulations + 1):
		sim = []
		for asset in log_returns.columns:
			random_sequence = []
			for period in range(num_periods_ahead):
				random_sequence.append(log_returns[asset].sample(1).values[0])
			sim.append(random_sequence)
		b[str(simulation)] = sim
	
	#Cumulate the shocks
	#W is keyed by simulations, within which rows correspond to assets and columns to periods ahead
	W = {}
	for simulation in range(1, num_simulations + 1):
		W[str(simulation)] = [np.cumsum(b[str(simulation)][asset]) for asset in range(len(S0))]

	simulations = {}
	for simulation in range(1, num_simulations + 1):

		sim = []

		for asset in range(len(S0)):
			
			#Calculate drift
			drift = (mu[asset] - 0.5 * (sigma[asset]**2)) * t

			#Calculate Diffusion
			diffusion = W[str(simulation)][asset]

			#Make predictions
			predicted_path = np.append(S0[asset], S0[asset]*np.exp(diffusion+drift))

			sim.append(predicted_path)	

		simulations[str(simulation)] = sim

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

def crash_simulator(simulations, initial_debt, initial_eth_vol, collateralization_ratio, quantity_reserve_asset, liquidity_dryup, token_basket):
	'''
	Simulate the behaviour of a system collateralized to exactly 150% which faces downturn such that all debt sold off
	:simulations: monte carlo simulations of correlated price movements
	:initial_debt: amount of initial system debt
	:initial_liquidities: maximum liquidity supportable by markets at start of crash
	:collateralization_ratio: system collateralization ratio
    '''
	sims = {}
	for simulation in range(1, len(simulations) + 1):
		eth_index = token_basket.index('ETH')
		eth_sim_prices = simulations[str(simulation)][eth_index]
		mkr_index = token_basket.index('MKR')
		mkr_sim_prices = simulations[str(simulation)][mkr_index]
		total_margins = []
		debts = []
		eth_collateral = []
		for index, price in enumerate(eth_sim_prices):
			if index == 0:

				#Set the initial base case from the first price where a sell off of all debt is triggered
				eth_price = eth_sim_prices[index] # ETH/USD

				#Calculate the ETH holdings corresponding to the assumption of exactly 150% collateralization at the start
				starting_eth_collateral = initial_debt * collateralization_ratio / eth_sim_prices[index] #ETH
				
				#Assets
				eth_collateral.append(starting_eth_collateral) #ETH

				#Liabilities
				debts.append(initial_debt) #USD

				#MARGIN
				total_eth = starting_eth_collateral * eth_sim_prices[index]  #USD
				#print('Total ETH: ' + str(total_eth))
				total_mkr = quantity_reserve_asset * mkr_sim_prices[index] #USD
				#print('Total MKR: ' + str(total_mkr))
				margin = total_eth + total_mkr - initial_debt #USD
				#print('Debt: ' + str(debt))
				#print('Total margin: ' + str(margin))
				total_margins.append(margin)
				
			if (index > 0) & (index < len(eth_sim_prices) - 1):
				
				#Debt
				debt_start_period = debts[index - 1] # USD

				#Calculate how many USD of ETH can be sold each period
				avg_eth_price = (eth_sim_prices[index-1] + eth_sim_prices[index]) / 2
				max_daily_eth_liquidation_usd = initial_eth_vol*np.math.exp(-1 * liquidity_dryup * index) * avg_eth_price #USD

				if debt_start_period > max_daily_eth_liquidation_usd:
					#Assets
					eth_collateral_end_period = eth_collateral[index - 1] - initial_eth_vol*np.math.exp(-1 * liquidity_dryup * index) #ETH
					eth_collateral.append(eth_collateral_end_period)
					#Liabilities
					debt_end_period = debt_start_period - max_daily_eth_liquidation_usd # USD
					debts.append(debt_end_period)
				else:
					#Assets
					eth_collateral_end_period = eth_collateral[index - 1] - debt_start_period/avg_eth_price
					eth_collateral.append(eth_collateral_end_period)
					debt_end_period = 0
					#Liabilities
					debts.append(debt_end_period)
				
				#MARGIN
				total_eth = eth_collateral_end_period * eth_sim_prices[index] #USD
				#print('Total ETH: ' + str(total_eth))
				total_mkr = quantity_reserve_asset * mkr_sim_prices[index] #USD
				#print('Total MKR: ' + str(total_mkr))
				total_margin = total_eth + total_mkr - debt_end_period #USD
				#print('Debt: ' + str(debt_end_period))
				#print('Total margin: ' + str(total_margin))
				total_margins.append(total_margin)
		
		sims[str(simulation)] = (total_margins, debts)

	return sims

def undercollateralized_debt(price_simulations, sim_results, point_evaluate_eth_price):
	'''
	For the worst eth price outcome, extract the amount of debt that is undercollateralized when the margin goes negative.
	'''
	worst_eth_outcome = get_data.extract_index_of_worst_eth_sim(price_simulations, point_evaluate_eth_price = point_evaluate_eth_price)
	total_margins = sim_results[worst_eth_outcome.values[0]][0]
	debts = sim_results[worst_eth_outcome.values[0]][1]

	negative_margins = []
	#Loop through margins to find the first negative one
	for index, margin in enumerate(total_margins):
		if margin < 0:
			negative_margins.append(index)

	if len(negative_margins) > 0:
		first_negative_margin_index = negative_margins[0]		
		debt_when_negative_margin = debts[first_negative_margin_index]
	elif len(negative_margins) == 0:
		debt_when_negative_margin = 0

	return debt_when_negative_margin

def crash_debts(debt_levels, liquidity_levels, price_simulations, initial_eth_vol, collateralization_ratio, quantity_reserve_asset, token_basket, point_evaluate_eth_price):
	'''
	For the considered range of debts and liquidities, create a DataFrame of the debt at the point of collapse. 
	'''
	df = pd.DataFrame(index = debt_levels, columns = liquidity_levels)

	for i in debt_levels:
		for j in liquidity_levels:
			sim_results = crash_simulator(simulations = price_simulations, initial_debt = i, initial_eth_vol = initial_eth_vol, collateralization_ratio = collateralization_ratio, quantity_reserve_asset = quantity_reserve_asset, liquidity_dryup = j, token_basket = token_basket)
			debt_when_negative_margin = undercollateralized_debt(price_simulations = price_simulations, sim_results = sim_results, point_evaluate_eth_price = point_evaluate_eth_price)
			df.loc[int(i)][float(j)] = debt_when_negative_margin
	
	return df

def protocol_composer(max_number_of_protocols, crash_debts_df, max_oc_requirement):
	'''
	Multiplier in the case of multiple DeFi protocols. 
	'''
	debt_shares_master = []
	for no_of_protocols in range(1, max_number_of_protocols + 1):
		per_protocol_debt = crash_debts_df / no_of_protocols
		debt_shares = []
		for i in range(1, no_of_protocols + 1):
			debt_shares.append(per_protocol_debt)
		debt_shares_master.append(debt_shares)

	collateralization_ratios_master = []
	for no_of_protocols in range(1, max_number_of_protocols + 1):
		collateralization_ratios = []
		for i in range(1, no_of_protocols + 1):
			collateralization_ratio = uniform(1.0, float(max_oc_requirement))
			collateralization_ratios.append(collateralization_ratio)
		collateralization_ratios_master.append(collateralization_ratios)
	
	max_levered_debt = []
	for i, protocol_debts in enumerate(debt_shares_master):
		for j, protocol_ratios in enumerate(collateralization_ratios_master):
			levered_debts = []
			if i == j:
				for q in range(len(protocol_debts)):
					a = protocol_debts[q]/protocol_ratios[q]
					r = 1/protocol_ratios[q]
					levered_debt = a/(1-r)
					levered_debts.append(levered_debt)
				max_levered_debt.append(levered_debts)
				
	total_protocol_debt = []
	for index, debts in enumerate(max_levered_debt):
		total_protocol_debt.append(sum(debts))
	
	return total_protocol_debt
		
def worst_case_per_protocol_number(sims, debt_size, liquidity_param):
	'''
	Taking the simulated levered debt losses, find the worst case loss for each economy size.
	:sims: output of protocol composer
	'''
	worst_cases = []
	number_of_protocols = len(sims)
	for index in range(number_of_protocols):
		#List of defaults per protocol size
		default_size = []
		for sim in sims:
			economy_in_sim = sims[index]
			params_in_economy = economy_in_sim.loc[debt_size][liquidity_param]
			default_size.append(params_in_economy)
		index_max = max(range(len(default_size)), key=default_size.__getitem__)
		value_max = default_size[index_max]
		worst_cases.append(value_max)

	df = pd.DataFrame(worst_cases)
	df.index +=1
	return df
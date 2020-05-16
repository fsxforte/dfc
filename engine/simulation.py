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

def multivariate_monte_carlo(close_prices, returns_distribution: str, num_simulations: int, T: int, dt: int, correlation: float, res_vol: float, collateral_asset: str):
	'''
	Perform Monte Carlo Simulation.
	Monte carlo equation: St = St-1* exp((μ-(σ2/2))*t + σWt).
	:close_prices: collateral asset close prices.
	:num_simulations: number of runs of simulation to perform. 
	:T: length of time prediction horizon (in units of dt, i.e. days)
	:dt: time increment, i.e. frequency of data (using daily data here)
	:correlation: strength of correlation between ETH and RES asset
	:res_vol: the relative volatility of the Reserve asset to ETH. a value of 0.5 indicates the Reserve asset has half the std dev.
	:collateral_asset: asset used as collateral, i.e. ETH.
	'''
	#Set seed to ensure same simulation run
	np.random.seed(150)

	#Create prices DataFrame
	prices_df = pd.DataFrame()
	prices_df[str(collateral_asset)] = close_prices
	prices_df['RES'] = close_prices

	num_periods_ahead = int(T / dt)

	log_returns = get_data.compute_log_returns(prices_df)

	#Initial asset price
	S0 = prices_df.iloc[-1]

	#Mean log return
	mu = np.mean(log_returns)
	print('mu: ' + str(mu))

	#Standard deviation of log return
	sigma = np.std(log_returns)

	#Set relative volatility of Reserve asset to ETH
	sigma['RES'] = sigma['ETH'] * res_vol
	print('sigma: ' + str(sigma))

	#Create correlation matrix
	corr_matrix = np.array([[1, correlation], [correlation, 1]])

	#Cholesky decomposition
	Chol = np.linalg.cholesky(corr_matrix)

	#Time index for predicted periods
	t = np.arange(1, int(num_periods_ahead) + 1)

	#Generate uncorrelated random sequences
	if returns_distribution == 'normal':
		b = {}
		for simulation in range(1, num_simulations + 1):
			sim = []
			for asset in S0.index:
				random_sequence = []
				for period in range(num_periods_ahead):
					random_sequence.append(np.random.normal(0, 1))
				sim.append(random_sequence)
			b[str(simulation)] = sim

	if returns_distribution == 'historical':
		b = {}
		for simulation in range(1, num_simulations + 1):
			sim = []
			for asset in S0.index:
				random_sequence = []
				for period in range(num_periods_ahead):
					random_sequence.append(log_returns[asset].sample(1).values[0])
				sim.append(random_sequence)
			b[str(simulation)] = sim

	#Correlate them with Cholesky
	b_corr = {str(simulation): Chol.dot(b[str(simulation)]) for simulation in range(1, num_simulations + 1)}
	
	b_corr_dict = {}
	for simulation in range(1, num_simulations + 1):
		sim = {}
		for asset in S0.index:
			sim[asset] = b_corr[str(simulation)][list(S0.index).index(asset)]
		b_corr_dict[str(simulation)] = sim

	#Cumulate the shocks
	#W is keyed by simulations, within which rows correspond to assets and columns to periods ahead
	W = {}
	for simulation in range(1, num_simulations + 1):
		cumulated_shocks = {}
		for asset in S0.index:
			cumulated_shocks[asset] = np.cumsum(b_corr_dict[str(simulation)][asset])
		W[str(simulation)] = cumulated_shocks

	simulations = {}
	for simulation in range(1, num_simulations + 1):

		sim = {}

		for asset in S0.index:

			drift = mu[asset] + 0.5 * sigma[asset]**2
			print(drift)
			
			#Calculate drift
			drift_component = (drift - 0.5 * (sigma[asset]**2)) * t
			
			#Calculate Diffusion
			if returns_distribution == 'normal':
				diffusion = W[str(simulation)][asset]  * sigma[asset]
			if returns_distribution == 'historical':
				#Original
				diffusion = W[str(simulation)][asset]
				#New
				#diffusion = W[str(simulation)][asset] - sigma[asset] * (mu[asset] - 0.5 * sigma[asset]**2)

			#Make predictions
			predicted_path = np.append(S0[asset], S0[asset]*np.exp(diffusion+drift_component))

			sim[asset] = predicted_path

		simulations[str(simulation)] = sim

	return simulations

def asset_extractor_from_sims(simulations, asset):
	'''
	Function to pull out simulations for a particular asset.
	:simulations: input dictionary (output from multivariate monte carlo function)
	:asset: e.g. 'ETH'
	'''
	asset_sims = {}	
	for simulation in range(1, len(simulations)+1):
		asset_sims[str(simulation)] = simulations[str(simulation)][asset]
	
	return asset_sims

def crash_simulator(price_simulations, 
					initial_debt, 
					initial_eth_vol, 
					collateralization_ratio, 
					quantity_reserve_asset, 
					liquidity_dryup):
	'''
	Simulate the behaviour of a system collateralized to exactly 150% which faces downturn such that all debt sold off
	:price_simulations: monte carlo simulations of correlated price movements
	:initial_debt: amount of initial system debt (USD)
	:initial_liquidities: maximum liquidity supportable by markets at start of crash
	:collateralization_ratio: system collateralization ratio
    '''
	sims = {}
	for simulation in range(1, len(price_simulations) + 1):
		eth_sim_prices = price_simulations[str(simulation)]['ETH']
		res_sim_prices = price_simulations[str(simulation)]['RES']
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
				total_res = quantity_reserve_asset * res_sim_prices[index] #USD
				#print('Total MKR: ' + str(total_res))
				margin = total_eth + total_res - initial_debt #USD
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
					#Liabilities
					debt_end_period = 0
					debts.append(debt_end_period)
				
				#MARGIN
				total_eth = eth_collateral_end_period * eth_sim_prices[index] #USD
				#print('Total ETH: ' + str(total_eth))
				total_res = quantity_reserve_asset * res_sim_prices[index] #USD
				#print('Total MKR: ' + str(total_res))
				total_margin = total_eth + total_res - debt_end_period #USD
				#print('Debt: ' + str(debt_end_period))
				#print('Total margin: ' + str(total_margin))
				total_margins.append(total_margin)
		
		sims[str(simulation)] = {'total_margin': total_margins, 'debt': debts, 'eth_collateral': eth_collateral}

	return sims

def extract_sim_fastest_default(crash_sims):
	'''
	Extract the simulation number of the simulation that results in the fastest default. 
	'''
	sim_default_dict = {}

	for simulation in crash_sims:
		margins = crash_sims[simulation]['total_margin']
		for k, v in enumerate(margins):
			if v < 0:
				sim_default_dict[simulation] = k
				break

	if sim_default_dict:
		fastest_default_sim = min(sim_default_dict, key=sim_default_dict.get)
	
		#Tuple of sim number and how many days the default took
		return (fastest_default_sim, sim_default_dict[fastest_default_sim])

def crash_searcher(debt_levels, 
					liquidity_levels, 
					price_simulations,
					initial_eth_vol,
					collateralization_ratio,
					quantity_reserve_asset):
	'''
	For a set of price simulations, search over the debt and liquidity levels
	to find the worst outcome. 
	'''
	worst_sim_master = []

	for debt in debt_levels:

		for liquidity in liquidity_levels:

			crash_sims = crash_simulator( \
				price_simulations = price_simulations, 
				initial_debt = debt, 
				initial_eth_vol = initial_eth_vol, 
				collateralization_ratio = collateralization_ratio, 
				quantity_reserve_asset = quantity_reserve_asset, 
				liquidity_dryup = liquidity)

			worst_sim = extract_sim_fastest_default(crash_sims)

			print('For debt ' + str(debt) + ' and liquidity ' + str(liquidity)+ \
					' the worst sim is ' + str(worst_sim))
			
			crash_tuple = (debt, liquidity, worst_sim)
			
			worst_sim_master.append(crash_tuple)

	return worst_sim_master
			


# def undercollateralized_debt(price_simulations, 
# 								sim_results, 
# 								point_evaluate_eth_price):
# 	'''
# 	For the worst eth price outcome, 
# 	extract the amount of debt that is undercollateralized when the margin goes negative.
# 	'''
# 	worst_eth_outcome = get_data.extract_index_of_worst_eth_sim(price_simulations, 
# 																point_evaluate_eth_price = point_evaluate_eth_price)
	
# 	total_margins = sim_results[worst_eth_outcome.values[0]]['total_margin']

# 	debts = sim_results[worst_eth_outcome.values[0]]['debt']

# 	negative_margins = []
# 	#Loop through margins to find the first negative one
# 	for index, margin in enumerate(total_margins):
# 		if margin < 0:
# 			negative_margins.append(index)

# 	if len(negative_margins) > 0:
# 		first_negative_margin_index = negative_margins[0]		
# 		debt_when_negative_margin = debts[first_negative_margin_index]
# 	elif len(negative_margins) == 0:
# 		debt_when_negative_margin = 0

# 	return debt_when_negative_margin

# def crash_debts(debt_levels, 
# 				liquidity_levels, 
# 				price_simulations, 
# 				initial_eth_vol, 
# 				collateralization_ratio, 
# 				quantity_reserve_asset, 
# 				point_evaluate_eth_price):
# 	'''
# 	For the considered range of debts and liquidities, 
# 	create a DataFrame of the debt at the point of collapse. 
# 	'''
# 	df = pd.DataFrame(index = debt_levels, columns = liquidity_levels)

# 	for i in debt_levels:

# 		for j in liquidity_levels:

# 			sim_results = crash_simulator(price_simulations = price_simulations, 
# 											initial_debt = i, 
# 											initial_eth_vol = initial_eth_vol, 
# 											collateralization_ratio = collateralization_ratio, 
# 											quantity_reserve_asset = quantity_reserve_asset, 
# 											liquidity_dryup = j)

# 			debt_when_negative_margin = undercollateralized_debt(price_simulations = price_simulations, 
# 																	sim_results = sim_results, 
# 																	point_evaluate_eth_price = point_evaluate_eth_price)

# 			df.loc[int(i)][float(j)] = debt_when_negative_margin
	
# 	return df

# def protocol_composer(max_number_of_protocols, crash_debts_df, max_oc_requirement):
# 	'''
# 	Multiplier in the case of multiple DeFi protocols. 
# 	'''
# 	debt_shares_master = []
# 	for no_of_protocols in range(1, max_number_of_protocols + 1):
# 		per_protocol_debt = crash_debts_df / no_of_protocols
# 		debt_shares = []
# 		for i in range(1, no_of_protocols + 1):
# 			debt_shares.append(per_protocol_debt)
# 		debt_shares_master.append(debt_shares)

# 	collateralization_ratios_master = []
# 	for no_of_protocols in range(1, max_number_of_protocols + 1):
# 		collateralization_ratios = []
# 		for i in range(1, no_of_protocols + 1):
# 			collateralization_ratio = uniform(1.0, float(max_oc_requirement))
# 			collateralization_ratios.append(collateralization_ratio)
# 		collateralization_ratios_master.append(collateralization_ratios)
	
# 	max_levered_debt = []
# 	for i, protocol_debts in enumerate(debt_shares_master):
# 		for j, protocol_ratios in enumerate(collateralization_ratios_master):
# 			levered_debts = []
# 			if i == j:
# 				for q in range(len(protocol_debts)):
# 					a = protocol_debts[q]/protocol_ratios[q]
# 					r = 1/protocol_ratios[q]
# 					levered_debt = a/(1-r)
# 					levered_debts.append(levered_debt)
# 				max_levered_debt.append(levered_debts)
				
# 	total_protocol_debt = []
# 	for debts in max_levered_debt:
# 		total_protocol_debt.append(sum(debts))
	
# 	return total_protocol_debt
		
# def worst_case_per_protocol_number(sims, debt_size, liquidity_param):
# 	'''
# 	Taking the simulated levered debt losses, find the worst case loss for each economy size.
# 	:sims: output of protocol composer
# 	'''
# 	worst_cases = []
# 	number_of_protocols = len(sims)
# 	for index in range(number_of_protocols):
# 		#List of defaults per protocol size
# 		default_size = []
# 		for sim in sims:
# 			economy_in_sim = sims[index]
# 			params_in_economy = economy_in_sim.loc[debt_size][liquidity_param]
# 			default_size.append(params_in_economy)
# 		index_max = max(range(len(default_size)), key=default_size.__getitem__)
# 		value_max = default_size[index_max]
# 		worst_cases.append(value_max)

# 	df = pd.DataFrame(worst_cases)
# 	df.index +=1
# 	return df
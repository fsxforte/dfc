import numpy as np
import datetime as dt
import pandas as pd
from scipy.stats import norm


from engine import kernel_estimation
from engine import get_data

######################################
###### Get historical data ###########
######################################

TOKEN_BASKET = ['ETH', 'MKR', 'BAT']

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2020,1,6)

#Get the data
df_eth = get_data.create_df(TOKEN_BASKET[0], 'USD')[start_date:end_date]
df_mkr = get_data.create_df(TOKEN_BASKET[1], 'USD')[start_date:end_date]
df_bat = get_data.create_df(TOKEN_BASKET[2], 'USD')[start_date:end_date]

eth_prices = df_eth['close']
eth_prices.rename('ETH', inplace=True)

mkr_prices = df_mkr['close']
mkr_prices.rename('MKR', inplace=True)

bat_prices = df_bat['close']
bat_prices.rename('BAT', inplace=True)

all_prices = pd.concat([eth_prices, mkr_prices, bat_prices], axis = 1)

corr_matrix = all_prices.corr()

def make_correlated_brownian(nb_assets, nb_simulation, correlation_matrix):
    """
    Function that returns a matrix with all the correlated brownian 
    for all the simulations by proceeding a Cholesky decomposition.
    :nb_assets: number of assets
    :nb_simulation: number of simulation runs
    :correlation_matrix: correlation matric of assets

    Return: an array number of sims x number of assets          
    """
    #Generate random number array of size nb_simulation x nb assets
    X = np.random.randn(nb_simulation, nb_assets)
    #Perform Cholesky decomposition of matrix
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    for i in range(nb_simulation):
        #Calculate matrix dot product
        X[i,:] = np.dot(cholesky_matrix, X[i,:])
    return X

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
	M = [t for i in range(len(S0))] 

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
	#diffusion = {str(simulation): sigma * W[str(simulation)] for simulation in range(1, num_simulations + 1)}
	diffusion = {}
	for simulation in range(1, num_simulations + 1):
		diffusion[str(simulation)] = [sigma[asset] * W[str(simulation)][asset] for asset in range(len(S0))]

	#Making the predictions
	simulations = {}
	for simulation in range(1, num_simulations + 1):
		simulations[str(simulation)] = [np.append(S0[asset], S0[asset] * np.exp(drift[str(simulation)][asset] + diffusion[str(simulation)][asset])) for asset in range(len(S0))]

	return simulations

simulations_df = pd.DataFrame(simulations['1'])
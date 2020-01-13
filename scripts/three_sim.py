import numpy as np
import datetime as dt
import pandas as pd

from engine import get_data
from engine import monte_carlo

#Token basket
TOKEN_BASKET = ['ETH', 'MKR', 'BAT']

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2018,6,1)

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

prices = pd.concat([eth_prices, mkr_prices, bat_prices], axis = 1)

simulations = monte_carlo.multivariate_monte_carlo(prices, 1000, 100)

#Plot for a particular asset
sims = monte_carlo.asset_extractor_from_sims(simulations, 0)
df = pd.DataFrame(sims)
df.plot()

#Plot for a simulation
df_sim = pd.DataFrame(simulations['10']).transpose()
df_sim = df_sim/df_sim.loc[0]
df_sim.plot()

#Of the simulations, find the worst 1%
sims_eth = monte_carlo.asset_extractor_from_sims(simulations, 0)
df_eth = pd.DataFrame(sims)
worst_eth_outcomes = df_eth.iloc[-1].nsmallest(10).index
worst_eth = df_eth.loc[:, worst_eth_outcomes]
worst_eth.plot()

#Find corresponding bad MKR outcomes
sims_mkr = monte_carlo.asset_extractor_from_sims(simulations, 1)
df_mkr = pd.DataFrame(sims_mkr)
corresponding_mkr_sims = df_mkr.loc[:, worst_eth_outcomes]
corresponding_mkr_sims.plot()

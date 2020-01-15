import numpy as np
import datetime as dt
import pandas as pd

from engine import get_data
from engine import monte_carlo

#CONSTANTS
TOKEN_BASKET = ['ETH', 'MKR', 'BAT']
num_simulations = 1000
num_periods = 100

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2018,4,1)

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

simulations = monte_carlo.multivariate_monte_carlo(prices, num_simulations, num_periods)

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

#Assuming starting at 150% system CR, find the amount of ETH that has to be liquidated to stay at 150% CR
#Find the price change deltas for each asset
price_deltas = {}
for simulation in range(1, num_simulations + 1):
	price_deltas[str(simulation)] = [np.diff(simulations[str(simulation)][asset]) for asset in range(len(TOKEN_BASKET))]

#Calculate implied ETH vol to be liquidated for these price changes
#Assuming CDPs are on cusp of liquidation at 150% by arbitrage
sims = {}
DAI_DEBT = 300000000
PERC_FEE = 1 # e.g. could be 1.13
MAX_ETH_SELLABLE_IN_24HOURS = 100

for simulation in range(1, num_simulations + 1):
    sim_version = simulations[str(simulation)]
    new_sim_version = []
    for asset_array in sim_version:
        new_asset_array_margin = []
        dai_liability = []
        for index, price in enumerate(asset_array):

            if index == 0:
                #Set the initial base case from the first price where a sell off of all DAI_DEBT is triggered
                avg_eth_price = (asset_array[index] + asset_array[index + 1]) / 2
                eth_equivalent = DAI_DEBT * PERC_FEE / avg_eth_price
                unliquidated_eth = eth_equivalent - MAX_ETH_SELLABLE_IN_24HOURS
                unliquidated_eth_usd = unliquidated_eth * asset_array[index + 1]
                residual_dai = DAI_DEBT - unliquidated_eth_usd
                dai_liability.append(residual_dai)

                #MARGIN
                safety_margin = unliquidated_eth_usd - residual_dai
                new_asset_array_margin.append(safety_margin)
                
            if (index < num_periods - 1) & (index > 0):
                dai_balance_outstanding = dai_liability[index - 1]
                avg_eth_price = (asset_array[index] + asset_array[index + 1]) / 2
                max_eth_liquidation_usd = MAX_ETH_SELLABLE_IN_24HOURS * avg_eth_price
                if dai_balance_outstanding > max_eth_liquidation_usd:                    
                    residual_dai = dai_balance_outstanding - max_eth_liquidation_usd
                    dai_liability.append(residual_dai)
                else:
                    residual_dai = 0
                    dai_liability.append(residual_dai)
                
                #MARGIN
                safety_margin = unliquidated_eth_usd - residual_dai
                new_asset_array_margin.append(safety_margin)

        new_sim_version.append(dai_liability)

    sims[str(simulation)] = new_sim_version

sims['1'][0]

#Extract just the arrays corresponding to the ETH asset
# sims_eth_liquidated = monte_carlo.asset_extractor_from_sims(eth_vols, 0)
# df_liquidated = pd.DataFrame(sims_eth_liquidated)
# worst_sell_off = df_liquidated.loc[:, worst_eth_outcomes]
# worst_sell_off.plot()


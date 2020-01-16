import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib import cm


sns.set(style="darkgrid")

from engine import get_data
from engine import monte_carlo


##############################################################
#########                PARAMETERS                  #########
##############################################################

#CONSTANTS
TOKEN_BASKET = ['ETH', 'MKR', 'BAT']
NUM_SIMULATIONS = 1000
DAYS_AHEAD = 100
TIME_INCREMENT = 1 # Frequency of data 

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,13)
end_date = dt.datetime(2018,4,7)

DAI_DEBT = 300000000000
MAX_ETH_SELLABLE_IN_24HOURS = 100000
COLLATERALIZATION_RATIO = 1.5
QUANTITY_RESERVE_ASSET = 1000000

###############################################################
#############           GET INPUT DATA                #########
###############################################################

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

###############################################################
###########      EXPLORATORY PLOTS                    #########
###############################################################

simulations = monte_carlo.multivariate_monte_carlo(prices, NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT)

# #Plot for a particular asset
# sims = monte_carlo.asset_extractor_from_sims(simulations, 0)
# df = pd.DataFrame(sims)
# df.plot()

#Plot for a simulation
df_sim = pd.DataFrame(simulations['1000']).transpose()
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

#################################################################
################       SIMULATION              ##################
#################################################################

#Run multivariate monte carlo using selected input parameters
price_simulations = monte_carlo.multivariate_monte_carlo(prices, NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT)

#Run the system simulator
system_simulations = monte_carlo.crash_simulator(simulations = price_simulations, DAI_DEBT = DAI_DEBT, MAX_ETH_SELLABLE_IN_24HOURS = MAX_ETH_SELLABLE_IN_24HOURS, COLLATERALIZATION_RATIO = COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET = QUANTITY_RESERVE_ASSET)

df = pd.DataFrame(system_simulations)
worst_cases = df.loc[:, worst_eth_outcomes]
worst_cases.plot()


#################################################################
#########      EXPLORE THE MARGIN SPACE #########################
#################################################################
import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


sns.set(style="darkgrid")

from engine import get_data
from engine import monte_carlo


##############################################################
#########                PARAMETERS                  #########
##############################################################

#CONSTANTS
TOKEN_BASKET = ['ETH', 'MKR'] # Can have n tokens in here
NUM_SIMULATIONS = 1000
DAYS_AHEAD = 50
TIME_INCREMENT = 1 # Frequency of data 

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2018,4,20)

INITIAL_MAX_ETH_SELLABLE_IN_24HOURS = 3447737 # Over period 13 Jan 2018 to 7 April 2018, avg vol
COLLATERALIZATION_RATIO = 1.5 # Exactly right
QUANTITY_RESERVE_ASSET = 1000000 # About the right amount of MKR Reserve asset at the moment
#go with BoE leverage ratio to govern the proportion of MKR token reserve to DAI debt

###############################################################
#############           GET INPUT DATA                #########
###############################################################

#Get the data
df_eth = get_data.create_df(TOKEN_BASKET[0], 'USD')[start_date:end_date]
df_mkr = get_data.create_df(TOKEN_BASKET[1], 'USD')[start_date:end_date]
eth_prices = df_eth['close']
eth_prices.rename('ETH', inplace=True)
mkr_prices = df_mkr['close']
mkr_prices.rename('MKR', inplace=True)

#Extract period of ETH price crash
start_date_crash = eth_prices.idxmax()
end_date_crash = eth_prices.idxmin()

#Filter to this period
eth_prices = eth_prices[start_date_crash:end_date_crash]
mkr_prices = mkr_prices[start_date_crash:end_date_crash]

prices = pd.concat([eth_prices, mkr_prices], axis = 1)

###############################################################
###########      EXPLORATORY PLOTS                    #########
###############################################################

#########
## Price simulation
#########
price_simulations = monte_carlo.multivariate_monte_carlo(prices, NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT)

##########
# 1. Plots of price paths
##########

##ETH
sims = monte_carlo.asset_extractor_from_sims(price_simulations, 0)
df = pd.DataFrame(sims)
fig, ax = plt.subplots()
df.plot(ax=ax)
ax.set_ylabel('ETH/USD', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('ETH/USD: 1000 Monte Carlo Simulations', fontsize = 14)
ax.set_xlabel('Time steps (days)', fontsize = 14)
ax.get_legend().remove()
fig.savefig('../5d8dd7887374be0001c94b71/images/eth_monte_carlo.png', bbox_inches = 'tight', dpi = 600)

##MKR
sims = monte_carlo.asset_extractor_from_sims(price_simulations, 1)
df = pd.DataFrame(sims)
fig, ax = plt.subplots()
df.plot(ax=ax)
ax.set_ylabel('MKR/USD', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('MKR/USD: 1000 Monte Carlo Simulations', fontsize = 14)
ax.set_xlabel('Time steps (days)', fontsize = 14)
ax.get_legend().remove()
fig.savefig('../5d8dd7887374be0001c94b71/images/mkr_monte_carlo.png', bbox_inches = 'tight', dpi = 600)

#########
### 2. Plot of worst outcome
#########

###Of the simulations, plot the worst ETH outcome one
sims_eth = monte_carlo.asset_extractor_from_sims(price_simulations, 0)
df_eth = pd.DataFrame(sims_eth)
worst_eth_outcomes = df_eth.iloc[-1].nsmallest(1).index
worst_eth = df_eth.loc[:, worst_eth_outcomes]
worst_eth = worst_eth.rename(columns = {"275": "ETH"})

#Find corresponding bad MKR outcomes
sims_mkr = monte_carlo.asset_extractor_from_sims(price_simulations, 1)
df_mkr = pd.DataFrame(sims_mkr)
corresponding_mkr_sims = df_mkr.loc[:, worst_eth_outcomes]
corresponding_mkr_sims = corresponding_mkr_sims.rename(columns = {"275": "MKR"})

#Join and plot to see correlated movements
df_joined = pd.concat([worst_eth, corresponding_mkr_sims], axis = 1)
df_normalized = df_joined/df_joined.loc[0]
fig, ax = plt.subplots()
df_normalized.plot(ax=ax)
ax.set_ylabel('Price evolution, normalized to 1', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('The co-evolution of the ETH and MKR price', fontsize = 14)
ax.set_xlabel('Time steps (days)', fontsize = 14)
fig.savefig('../5d8dd7887374be0001c94b71/images/co-evolution.png', bbox_inches = 'tight', dpi = 600)

#################################################################
#########          SYSTEM MARGIN SIMULATIONS      ###############
#################################################################

system_simulations = monte_carlo.crash_simulator(simulations = price_simulations, DAI_DEBT = 3000000000, INITIAL_MAX_ETH_SELLABLE_IN_24HOURS = INITIAL_MAX_ETH_SELLABLE_IN_24HOURS, COLLATERALIZATION_RATIO = COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET = QUANTITY_RESERVE_ASSET, LIQUIDITY_DRYUP = 0)

#Plot evolution of margins over time steps
#Vary liquidation rate parameter
#Vary amount of DAI
#Vary reserve size

df = pd.DataFrame(system_simulations)
df.plot()

#Plot distribution of margins at different time steps - violin plots
#Vary liquidation rate parameter
#Vary amount of DAI
#Vary reserve size
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
DAYS_AHEAD = 100
TIME_INCREMENT = 1 # Frequency of data 

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2018,4,20)

COLLATERALIZATION_RATIO = 1.5 # Exactly right
QUANTITY_RESERVE_ASSET = 988787 # The right amount of MKR Reserve asset at the moment
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

#Extract ETH volume on 13 January 2018
INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS = df_eth.loc[dt.datetime(2018,1,13)]['volumefrom']

#Extract period of ETH price crash
start_date_crash = eth_prices.idxmax() # 13 January 2018
end_date_crash = eth_prices.idxmin() # 6 April 2018

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
worst_eth = worst_eth.rename(columns = {"13": "ETH"})

#Find corresponding bad MKR outcomes
sims_mkr = monte_carlo.asset_extractor_from_sims(price_simulations, 1)
df_mkr = pd.DataFrame(sims_mkr)
corresponding_mkr_sims = df_mkr.loc[:, worst_eth_outcomes]
corresponding_mkr_sims = corresponding_mkr_sims.rename(columns = {"13": "MKR"})

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

#####
#### Plot evolution of margins over time steps
#####

debts = [30000000, 300000000, 3000000000, 30000000000]
liquidities = [0.5, 0.6, 0.7, 0.8]

#Create 1 x 4 plot
fig, ax = plt.subplots(1, 4, figsize=(16,4))
for i, debt in enumerate(debts):
    debt_master_df_margin = pd.DataFrame(index = range(DAYS_AHEAD))
    debt_master_df_debt = pd.DataFrame(index = range(DAYS_AHEAD))
    for liquidity in liquidities:
        system_simulations = monte_carlo.crash_simulator(simulations = price_simulations, DAI_DEBT = debt, INITIAL_MAX_ETH_SELLABLE_IN_24HOURS = INITIAL_MAX_ETH_SELLABLE_IN_24HOURS, COLLATERALIZATION_RATIO = COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET = QUANTITY_RESERVE_ASSET, LIQUIDITY_DRYUP = liquidity)
        
        #Reconstruct dictionaries for margin and dai liability separately
        margin_simulations = {}
        dai_simulations = {}
        for x in range(1, NUM_SIMULATIONS+1):
            margin_simulations[str(x)] = system_simulations[str(x)][0]
            dai_simulations[str(x)] = system_simulations[str(x)][1]
        
        #Total margins
        df_margins = pd.DataFrame(margin_simulations)
        worst_path_margin = df_margins.loc[:, worst_eth_outcomes]
        worst_path_margin = worst_path_margin.rename(columns={'13': str(liquidity)})
        debt_master_df_margin['Margin, '+ 'liq. ' + str(liquidity)] = worst_path_margin

        #Remaining debt
        df_debt = pd.DataFrame(dai_simulations)
        worst_path_debt = df_debt.loc[:, worst_eth_outcomes]
        worst_path_debt = worst_path_debt.rename(columns={'13': str(liquidity)})
        debt_master_df_debt['Debt, ' + 'liq. ' + str(liquidity)] = worst_path_debt

    #Of the plots, look at the worst 0.1% (worst ETH outcome)
    debt_master_df_margin.plot(ax = ax[i])
    debt_master_df_debt.plot(ax = ax[i], style = '--')

    #Graph polish
    ax[i].set_title('Debt: ' + str(f'{debt:,}'))
    ax[0].set_ylabel('Total margin (USD)', fontsize = 14)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    ax[0].set_xlabel('Time steps (days)', fontsize = 14)
    if i<len(debts)-1:
        ax[i].get_legend().remove()
    else:
        handles, labels = ax[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol = len(debt_master_df_debt.columns) + len(debt_master_df_margin.columns))
        ax[i].get_legend().remove()
    fig.savefig('../5d8dd7887374be0001c94b71/images/total_margin_debt.png', bbox_inches = 'tight', dpi = 600)
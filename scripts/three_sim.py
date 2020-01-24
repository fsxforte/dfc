import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle


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
fig.savefig('../5d8dd7887374be0001c94b71/images/eth_monte_carlo.png', bbox_inches = 'tight', dpi = 300)

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
fig.savefig('../5d8dd7887374be0001c94b71/images/mkr_monte_carlo.png', bbox_inches = 'tight', dpi = 300)

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
fig.savefig('../5d8dd7887374be0001c94b71/images/co-evolution.png', bbox_inches = 'tight', dpi = 300)

#################################################################
#########          SYSTEM MARGIN SIMULATIONS      ###############
#################################################################

#####
#### Plot evolution of margins over time steps
#####

debts = [30000000, 20000000000, 25000000000, 30000000000]
liquidities = [0, 0.01, 0.02]

#Create 1 x 4 plot
markers = ['s', 'p', 'v',]
colors = ['g', 'k', 'r']
fig, ax = plt.subplots(1, 4, figsize=(18,8))
for i, debt in enumerate(debts):
    debt_master_df_margin = pd.DataFrame(index = range(DAYS_AHEAD))
    debt_master_df_debt = pd.DataFrame(index = range(DAYS_AHEAD))
    for liquidity in liquidities:
        system_simulations = monte_carlo.crash_simulator(simulations = price_simulations, DAI_DEBT = debt, INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS = INITIAL_MAX_ETH_SELLABLE_IN_24_HOURS, COLLATERALIZATION_RATIO = COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET = QUANTITY_RESERVE_ASSET, LIQUIDITY_DRYUP = liquidity)
        
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
        debt_master_df_debt['Initial debt, ' + 'liq. ' + str(liquidity)] = worst_path_debt

    #Margin
    debt_master_df_margin.plot(ax = ax[i], markevery = 10)
    for k, line in enumerate(ax[i].get_lines()):
        line.set_marker(markers[k])
        line.set_color(colors[k])

    #Debt
    debt_master_df_debt.plot(ax = ax[i], style = '--', markevery = 10)
    for k, line in enumerate(ax[i].get_lines()[len(debt_master_df_margin.columns):]):
        line.set_marker(markers[k])
        line.set_color(colors[k])

    #Graph polish
    ax[i].set_title('Initial debt: ' + str(f'{debt:,}'), fontsize = 10.5)
    ax[i].tick_params(axis='both', which='major', labelsize=14)

    #Shading
    ax_lims = ax[i].get_ylim()
    ax[i].axhspan(ax_lims[0], 0, facecolor='red', alpha=0.5)

    if i<len(debts)-1:
        ax[i].get_legend().remove()
    else:
        handles, labels = ax[i].get_legend_handles_labels()
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        fig.legend([extra, handles[0], handles[3], extra, handles[1], handles[4], extra, handles[2], handles[5]], ['Liquidity: '+ str(liquidities[0]), 'Margin', 'Remaining debt', 'Liquidity: '+ str(liquidities[1]), 'Margin', 'Remaining debt', 'Liquidity: '+ str(liquidities[2]), 'Margin', 'Remaining debt'], loc = 'lower center', ncol=3, borderaxespad=0., fontsize = 14)#, bbox_to_anchor=(0.5,-0.0005))
        ax[i].get_legend().remove()

fig.subplots_adjust(bottom=0.2) 
ax[0].set_ylabel('USD', fontsize = 14)
ax[0].set_xlabel('Time steps (days)', fontsize = 14)
fig.suptitle('A Decentralized Financial Crisis: illiquidity causing negative margins', fontsize = 18)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
fig.savefig('../5d8dd7887374be0001c94b71/images/total_margin_debt.png', dpi = 300)
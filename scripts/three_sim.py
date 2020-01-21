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
start_date = dt.datetime(2018,1,13)
end_date = dt.datetime(2018,4,7)

DAI_DEBT = 300000000000
MAX_ETH_SELLABLE_IN_24HOURS = 3447737 # Over period 13 Jan 2018 to 7 April 2018, avg vol
COLLATERALIZATION_RATIO = 1.5
QUANTITY_RESERVE_ASSET = 1000000 # About the right amount of MKR Reserve asset at the moment
#go with BoE leverage ratio to govern the proportion of MKR token reserve to DAI debt

###############################################################
#############           GET INPUT DATA                #########
###############################################################

#Get the data
df_eth = get_data.create_df(TOKEN_BASKET[0], 'USD')[start_date:end_date]
df_mkr = get_data.create_df(TOKEN_BASKET[1], 'USD')[start_date:end_date]
#df_bat = get_data.create_df(TOKEN_BASKET[2], 'USD')[start_date:end_date]
eth_prices = df_eth['close']
eth_prices.rename('ETH', inplace=True)
mkr_prices = df_mkr['close']
mkr_prices.rename('MKR', inplace=True)
#bat_prices = df_bat['close']
#bat_prices.rename('BAT', inplace=True)
prices = pd.concat([eth_prices, mkr_prices], axis = 1)

###############################################################
###########      EXPLORATORY PLOTS                    #########
###############################################################

simulations = monte_carlo.multivariate_monte_carlo(prices, NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT)

# #Plot for a particular asset
# sims = monte_carlo.asset_extractor_from_sims(simulations, 0)
# df = pd.DataFrame(sims)
# df.plot()

# #Plot for a simulation
# df_sim = pd.DataFrame(simulations['1000']).transpose()
# df_sim = df_sim/df_sim.loc[0]
# df_sim.plot()

#Of the simulations, find the worst 1
sims_eth = monte_carlo.asset_extractor_from_sims(simulations, 0)
df_eth = pd.DataFrame(sims_eth)
worst_eth_outcomes = df_eth.iloc[-1].nsmallest(1).index
worst_eth = df_eth.loc[:, worst_eth_outcomes]
worst_eth.plot()

# #Find corresponding bad MKR outcomes
# sims_mkr = monte_carlo.asset_extractor_from_sims(simulations, 1)
# df_mkr = pd.DataFrame(sims_mkr)
# corresponding_mkr_sims = df_mkr.loc[:, worst_eth_outcomes]
# corresponding_mkr_sims.plot()

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

#Remember that each monte carlo will yield different results unless same seed used

#Set up the X axis
#About 30,000,000 at present, so a range around this
dai_debts = []
for i in range(100000000000, 600000000000, 100000000000):
    dai_debts.append(i)

#Set up the Y axis
max_eth_sellable = []
#3,000,000 seems about right, looking at current data. So range around this.
for i in range(0, 50000000, 10000000):
    max_eth_sellable.append(i)

#Make skeleton dataframe to fill
df_3d = pd.DataFrame(index = dai_debts, columns = max_eth_sellable)

#Compute the margins at a certain number of days for these parameters (need a scalar)
#Run multivariate monte carlo using selected input parameters
price_simulations = monte_carlo.multivariate_monte_carlo(prices, NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT)

#Of the simulations, find the worst 1
sims_eth = monte_carlo.asset_extractor_from_sims(price_simulations, 0)
df_eth = pd.DataFrame(sims_eth)
worst_eth_outcomes = df_eth.iloc[-1].nsmallest(1).index

def plotter(day):
    for i in dai_debts:
        for j in max_eth_sellable:
            #Run the system simulator
            system_simulations = monte_carlo.crash_simulator(simulations = price_simulations, DAI_DEBT = i, MAX_ETH_SELLABLE_IN_24HOURS = j, COLLATERALIZATION_RATIO = COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET = QUANTITY_RESERVE_ASSET)

            df = pd.DataFrame(system_simulations)
            worst_case = df.loc[:, worst_eth_outcomes]
            margin_on_day = worst_case.loc[day][0]

            df_3d.at[i, j] = margin_on_day

    df_unstacked=df_3d.unstack().reset_index()
    df_unstacked.columns=["eth_sellable","dai_debt","margin"]
    df_unstacked = df_unstacked.astype('int')

    # Make the plot
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    surf=ax.plot_trisurf(df_unstacked['eth_sellable'], df_unstacked['dai_debt'], df_unstacked['margin'], cmap=plt.cm.terrain_r, linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev = 30, azim = 220)
    ax.set_xlabel('\n' + 'MAX ETH SELLABLE')
    ax.set_ylabel('\n' + 'DAI DEBT')
    ax.set_zlabel('\n' + 'TOTAL MARGIN')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()

for day in range(100):
    plotter(day)

plotter(40)
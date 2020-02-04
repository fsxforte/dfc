import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
import numpy as np
import seaborn as sns

from engine import get_data, plots, simulation
from constants import NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT, DEBTS, LIQUIDITIES, COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET, OC_LEVELS

START_DATE_DATA = dt.datetime(2018,1,1) # First data for MKR token
#END_DATE_DATA = dt.datetime(2018,6,1)
END_DATE_DATA = dt.datetime.now()

sns.set(style="darkgrid")

###THE LIQUIDITY ISSUE IS THE BIG ONE. HOW MUCH ETH/DAI and MKR/DAI can be traded?
###Coingecko just pulling in volume!

#1. Get times corresponding to high and low of ETH price
#start_crisis, end_crisis = get_data.get_crisis_endpoints(start_date = START_DATE_DATA, end_date = END_DATE_DATA)

#2. Make dataframe of close prices for the crash period
#crash_df = get_data.create_close_df()[start_crisis:end_crisis]
crash_df = get_data.create_close_df()[START_DATE_DATA:END_DATE_DATA]

#3. Extract the ETH volume at the start of the crash
#initial_eth_vol = get_data.liquidity_on_date(token = 'ETH', start_date_data = START_DATE_DATA, end_date_data = END_DATE_DATA, date_of_interest = start_crisis)
initial_eth_vol = get_data.coingecko_volumes('ETH')

#4. Find the worst 1% of shocks
log_rets = get_data.create_logrets_df()[START_DATE_DATA:END_DATE_DATA]

#4. Using dataframe for crash period, perform Monte Carlo simulation
price_simulations = simulation.multivariate_monte_carlo(historical_prices = crash_df, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, length_of_crash = 1)

#5. Plot the simulated prices
plots.plot_monte_carlo_simulations(price_simulations)

#6. Plot the worst (joint) path
plots.plot_worst_simulation(price_simulations)

#7. Plot simulation outputs for debts and liquidities
plots.plot_crash_sims(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations, initial_eth_vol = initial_eth_vol)

#8. Plot heatmap
debts = []
for x in range(0, 100000000000, 10000000000):
    debts.append(x)

liquidities = []
for x in np.arange(0, 0.11, 0.01):
    liquidities.append(x)

plots.plot_heatmap(debt_levels = debts, liquidity_levels = liquidities, price_simulations = price_simulations, initial_eth_vol = initial_eth_vol)

#9. Get sim results (free-standing)
sim_results = simulation.crash_simulator(simulations = price_simulations, initial_debt=23000000000, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = 0.5)

#10. Find the debt outstanding when market crashes
debts_outstanding = simulation.crash_debts(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET)

#11. Examine what the worst case is from #11 (for each economy size)
plots.plot_protocol_universe_default(max_number_of_protocols = 30, crash_debts_df = debts_outstanding, number_of_simulations = 100, oc_levels = OC_LEVELS)
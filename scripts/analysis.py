import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
import numpy as np
import seaborn as sns

from engine import get_data, plots, simulation
from constants import NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT, DEBTS, LIQUIDITIES, COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET, OC_LEVELS, TOKEN_BASKET, INITIAL_ETH_VOL

START_DATE = dt.datetime(2018,1,1) # First data for MKR token
END_DATE = dt.datetime(2020,2,6)

#1. Make dataframe of close prices since 1 January 2018 for all tokens in basket
close_prices = get_data.create_close_df()[START_DATE:END_DATE]

#2. Plot ETH and MKR prices
plots.plot_close_prices(start_date = START_DATE, end_date = END_DATE)

#2. Extract the current DAI/ETH volume, use this to approximate liquidity
#initial_eth_vol = get_data.coingecko_volumes('ETH') / 2 #Assume half is selling of ETH

#3. Find the worst ETH price shock
worst_eth_shock = get_data.create_logrets_df()[START_DATE:END_DATE]['ETH'].sort_values()[0]

#4. Using dataframe for crash period, perform Monte Carlo simulation
price_simulations = simulation.multivariate_monte_carlo(historical_prices = close_prices, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT)

#5. Plot the simulated prices
plots.plot_monte_carlo_simulations(price_simulations)

#6. Plot the worst (joint) path
plots.plot_worst_simulation(price_simulations)

#7. Get simulation results
sim_results = simulation.crash_simulator(simulations = price_simulations, initial_debt=23000000000, initial_eth_vol = INITIAL_ETH_VOL, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = 0.01, token_basket = TOKEN_BASKET)

#8. Plot simulation outputs for debts and liquidities
plots.plot_crash_sims(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations, initial_eth_vol = INITIAL_ETH_VOL)

#9. Plot heatmap for debt vs liquidity dry-up
debts = []
for x in range(500000000, 1000000000, 50000000):
    debts.append(x)

liquidities = []
for x in np.arange(0, 0.11, 0.01):
    liquidities.append(x)

plots.plot_heatmap_liquidities(debt_levels = debts, liquidity_params = liquidities, price_simulations = price_simulations, initial_eth_vol = initial_eth_vol)

#10. Plot heatmap for debt vs initial eth liquidity
eth_vols = []
for x in range(0, 20000, 2000):
    eth_vols.append(x)

plots.plot_heatmap_initial_volumes(debt_levels = debts, liquidity_param = 0.01, price_simulations = price_simulations, initial_eth_vols = eth_vols)

#11. Find the debt outstanding when market crashes
debts_outstanding = simulation.crash_debts(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations, initial_eth_vol = INITIAL_ETH_VOL, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, token_basket = TOKEN_BASKET)

#12. Examine what the worst case is from #11 (for each economy size)
plots.plot_protocol_universe_default(max_number_of_protocols = 30, crash_debts_df = debts_outstanding, number_of_simulations = 100, oc_levels = OC_LEVELS, debt_size = 500000000, liquidity_param = 0.01)
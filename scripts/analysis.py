import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
import numpy as np
import seaborn as sns

from engine import get_data, plots, simulation
from constants import NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT, DEBTS, LIQUIDITIES, COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET, OC_LEVELS, TOKEN_BASKET, POINT_EVALUATE_ETH_PRICE

START_DATE = dt.datetime(2018,1,1) # First data for MKR token
END_DATE = dt.datetime(2020,2,7)

#1. Make dataframe of close prices since 1 January 2018 for all tokens in basket
close_prices = get_data.create_close_df()[START_DATE:END_DATE]

#2. Plot ETH and MKR prices
plots.plot_close_prices(start_date = START_DATE, end_date = END_DATE)

#3. Extract the current DAI/ETH volume, use this to approximate liquidity
#initial_eth_vol = get_data.coingecko_volumes('ETH') #Assume half is selling of ETH
initial_eth_vol = 34006.73863635782 #Value 7th Feb

#4. Using dataframe for crash period, perform Monte Carlo simulation using normal distribution and historical distribution
price_simulations_normal = simulation.multivariate_monte_carlo_normal(historical_prices = close_prices, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT)
#price_simulations_historical = simulation.multivariate_monte_carlo_historical(historical_prices = close_prices, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT)

#5. Plot the simulated prices
plots.plot_monte_carlo_simulations(price_simulations_normal)
#plots.plot_monte_carlo_simulations(price_simulations_historical)

#6. Plot the worst (joint) path
plots.plot_worst_simulation(price_simulations_normal, point_evaluate_eth_price = 30)
#plots.plot_worst_simulation(price_simulations_historical, point_evaluate_eth_price = 30)

#7. Plot simulation outputs for debts and liquidities
plots.plot_crash_sims(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations_normal, initial_eth_vol = initial_eth_vol, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)
#plots.plot_crash_sims(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations_historical, initial_eth_vol = initial_eth_vol, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

#8. Plot heatmap for debt vs liquidity dry-up
debts = []
for x in range(400000000, 800000000, 50000000):
    debts.append(x)

liquidities = []
for x in np.arange(0, 0.03, 0.005):
    liquidities.append(x)

plots.plot_heatmap_liquidities(debt_levels = debts, liquidity_params = liquidities, price_simulations = price_simulations_normal, initial_eth_vol = initial_eth_vol, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)
#plots.plot_heatmap_liquidities(debt_levels = debts, liquidity_params = liquidities, price_simulations = price_simulations_historical, initial_eth_vol = initial_eth_vol, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

#9. Plot heatmap for debt vs initial eth liquidity
eth_vols = []
for x in range(0, 50000, 5000):
    eth_vols.append(x)

plots.plot_heatmap_initial_volumes(debt_levels = debts, liquidity_param = 0.01, price_simulations = price_simulations_normal, initial_eth_vols = eth_vols, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)
#plots.plot_heatmap_initial_volumes(debt_levels = debts, liquidity_param = 0.01, price_simulations = price_simulations_historical, initial_eth_vols = eth_vols, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

#10. Find the debt outstanding when market crashes
debts_outstanding = simulation.crash_debts(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations_normal, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, token_basket = TOKEN_BASKET)

#11. Examine what the worst case is from #11 (for each economy size)
plots.plot_protocol_universe_default(max_number_of_protocols = 30, crash_debts_df = debts_outstanding, number_of_simulations = 100, oc_levels = OC_LEVELS, debt_size = 400000000, liquidity_param = 0.01)
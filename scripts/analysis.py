import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
import numpy as np
import seaborn as sns

from engine import get_data, plots, simulation
from constants import NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT, DEBTS, LIQUIDITIES, COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET, OC_LEVELS, POINT_EVALUATE_ETH_PRICE, INITIAL_ETH_VOL, COLLATERAL_ASSET

START_DATE = dt.datetime(2018,1,1)
END_DATE = dt.datetime(2020,2,7)

#1. Make dataframe of close prices since 1 January 2018 for all tokens in basket
close_prices = get_data.create_close_df()[START_DATE:END_DATE]

#2. Plot ETH price
plots.plot_close_prices(start_date = START_DATE, end_date = END_DATE)

#3. Using dataframe for crash period, perform Monte Carlo simulation using normal distribution and historical distribution
price_simulations_normal = simulation.multivariate_monte_carlo_normal(historical_prices = close_prices, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = 0.1, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)

#4. Plot the simulated prices
plots.plot_monte_carlo_simulations(price_simulations_normal)

#5. Plot the worst (joint) path
plots.plot_worst_simulation(price_simulations_normal, point_evaluate_eth_price = 100)

#6. Plot simulation outputs for debts and liquidities
plots.plot_crash_sims(debt_levels = DEBTS, liquidity_levels = LIQUIDITIES, price_simulations = price_simulations_normal, initial_eth_vol = INITIAL_ETH_VOL, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE, correlation = 0.1)

#7. Plot heatmap for debt vs liquidity dry-up
debts = []
for x in range(400000000, 800000000, 50000000):
    debts.append(x)

liquidities = []
for x in np.arange(0, 0.03, 0.005):
    liquidities.append(x)

plots.plot_heatmap_liquidities(debt_levels = debts, liquidity_params = liquidities, price_simulations = price_simulations_normal, initial_eth_vol = INITIAL_ETH_VOL, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

#8. Plot heatmap for debt vs initial eth liquidity
eth_vols = []
for x in range(0, 50000, 5000):
    eth_vols.append(x)

plots.plot_heatmap_initial_volumes(debt_levels = debts, liquidity_param = 0.01, price_simulations = price_simulations_normal, initial_eth_vols = eth_vols, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

#9. Find the debt outstanding when market crashes
debts_outstanding = simulation.crash_debts(debt_levels = debts, liquidity_levels = liquidities, price_simulations = price_simulations_normal, initial_eth_vol = INITIAL_ETH_VOL, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

#10. Examine what the worst case is from #10 (for each economy size)
plots.plot_protocol_universe_default(max_number_of_protocols = 30, crash_debts_df = debts_outstanding, oc_levels = OC_LEVELS, debt_size = 400000000, liquidity_param = 0.01)
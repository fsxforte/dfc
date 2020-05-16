import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
import numpy as np
import seaborn as sns
import os

os.chdir('../')

from engine import get_data, plots, simulation, import_matlab_sims
from constants import NUM_SIMULATIONS, DAYS_AHEAD, TIME_INCREMENT, DEBTS, LIQUIDITIES, COLLATERALIZATION_RATIO, QUANTITY_RESERVE_ASSET, OC_LEVELS, POINT_EVALUATE_ETH_PRICE, INITIAL_ETH_VOL, COLLATERAL_ASSET

START_DATE = dt.datetime(2018,1,1)
END_DATE = dt.datetime(2020,2,7)

# CORRELATIONS = [-0.9, 0.1, 0.9]
# RETURNS_DISTRIBUTIONS = ['normal', 'historical']

#1. Get close price data for ETH
close_prices = get_data.get_prices(from_sym=COLLATERAL_ASSET, to_sym='USD', start_date=START_DATE, end_date=END_DATE)
# close_prices.to_csv('ethprices.csv')

#2. Plot ETH price
plots.plot_close_prices(close_prices)

#3. Compute log returns
log_returns = get_data.compute_log_returns(close_prices)

#Jarque-Bera test for normality
stats.jarque_bera(log_returns)

#3. Plot log returns
plots.plot_log_returns(log_returns)

#4. Plot histogram log returns
plots.plot_histogram_log_returns(log_returns)

#5. ALL Price plots
# for distribution in RETURNS_DISTRIBUTIONS:
#     for correlation in CORRELATIONS:
#         price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = distribution, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = correlation, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
#         plots.plot_monte_carlo_simulations(price_simulations, returns_distribution = distribution, correlation = str(correlation))
#         plots.plot_worst_simulation(price_simulations, returns_distribution = distribution, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE, correlation = str(correlation))

##New --> Use Matlab correlated sims

price_simulations_A = import_matlab_sims.get_matlab_sims('A')
price_simulations_B = import_matlab_sims.get_matlab_sims('B')
price_simulations_C = import_matlab_sims.get_matlab_sims('C')

#Find the fastest crashes for the range of debts and liquidities
fastest_crashes = simulation.crash_searcher(debt_levels=DEBTS, 
                                            liquidity_levels=LIQUIDITIES,
                                            price_simulations=price_simulations_A,
                                            initial_eth_vol=INITIAL_ETH_VOL,
                                            collateralization_ratio=COLLATERALIZATION_RATIO,
                                            quantity_reserve_asset=QUANTITY_RESERVE_ASSET)

fastest_crash_sim_index = fastest_crashes[-1][2][0]

#6. Plots for paper
#Normally distributed shocks, 0.9 correlation
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = 'normal', num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = 0.9, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
plots.plot_monte_carlo_simulations(price_simulations_A, 
                                    returns_distribution = 'normal', 
                                    correlation = str(0.9))

plots.plot_worst_simulation(price_simulations_A, 
                                returns_distribution = 'normal', 
                                point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE, 
                                correlation = str(0.9))

#Normally distributed, 0.1 correlation
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = 'normal', num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = 0.1, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
plots.plot_worst_simulation(price_simulations_B, 
                            returns_distribution='normal', 
                            fastest_crash_sim_index=fastest_crash_sim_index, 
                            correlation = str(0.1))

#Normally distributed, -0.9 correlation
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = 'normal', num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = -0.9, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
plots.plot_worst_simulation(price_simulations_C, 
                            returns_distribution = 'normal', 
                            fastest_crash_sim_index=fastest_crash_sim_index, 
                            correlation = str(-0.9))

#6. Crash plots
correlation = 0.9
returns_distribution = 'normal'
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = returns_distribution, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = correlation, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
plots.plot_crash_sims(debt_levels = DEBTS, 
                        liquidity_levels = LIQUIDITIES, 
                        price_simulations = price_simulations_A, 
                        initial_eth_vol = INITIAL_ETH_VOL, 
                        returns_distribution = returns_distribution, correlation = str(correlation), 
                        fastest_crash_sim_index=fastest_crash_sim_index)

correlation = 0.1
returns_distribution = 'normal'
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = returns_distribution, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = correlation, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
plots.plot_crash_sims(debt_levels = DEBTS, 
                        liquidity_levels = LIQUIDITIES, 
                        price_simulations = price_simulations_B, 
                        initial_eth_vol = INITIAL_ETH_VOL, 
                        returns_distribution = returns_distribution, 
                        correlation = str(correlation), 
                        fastest_crash_sim_index=fastest_crash_sim_index)

correlation = -0.9
returns_distribution = 'normal'
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, returns_distribution = returns_distribution, num_simulations = NUM_SIMULATIONS, T = DAYS_AHEAD, dt = TIME_INCREMENT, correlation = correlation, res_vol = 0.5, collateral_asset = COLLATERAL_ASSET)
plots.plot_crash_sims(debt_levels = DEBTS, 
                        liquidity_levels = LIQUIDITIES, 
                        price_simulations = price_simulations_C, 
                        initial_eth_vol = INITIAL_ETH_VOL, 
                        returns_distribution = returns_distribution, 
                        correlation = str(correlation), 
                        fastest_crash_sim_index=fastest_crash_sim_index)

#7. Plot heatmap for debt vs liquidity dry-up
# price_simulations = simulation.multivariate_monte_carlo(close_prices = close_prices, 
#                                                         returns_distribution = returns_distribution, 
#                                                         num_simulations = NUM_SIMULATIONS, 
#                                                         T = DAYS_AHEAD, 
#                                                         dt = TIME_INCREMENT, 
#                                                         correlation =0.9,  
#                                                         res_vol = 0.5, 
#                                                         collateral_asset = COLLATERAL_ASSET)

debts = []
for x in range(400000000, 800000000, 50000000):
    debts.append(x)

liquidities = []
for x in np.arange(0, 0.03, 0.005):
    liquidities.append(x)

plots.plot_heatmap_liquidities(debt_levels = debts, 
                                liquidity_params = liquidities, 
                                price_simulations = price_simulations_A, 
                                initial_eth_vol = INITIAL_ETH_VOL, 
                                fastest_crash_sim_index=fastest_crash_sim_index)

#8. Plot heatmap for debt vs initial eth liquidity
eth_vols = []
for x in range(0, 50000, 5000):
    eth_vols.append(x)

plots.plot_heatmap_initial_volumes(debt_levels = debts, 
                                    liquidity_param = 0.01, 
                                    price_simulations = price_simulations, 
                                    initial_eth_vols = eth_vols, 
                                    fastest_crash_sim_index=fastest_crash_sim_index)

# #9. Find the debt outstanding when market crashes
# debts_outstanding = simulation.crash_debts(debt_levels = debts, liquidity_levels = liquidities, price_simulations = price_simulations, initial_eth_vol = INITIAL_ETH_VOL, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, point_evaluate_eth_price = POINT_EVALUATE_ETH_PRICE)

# #10. Examine what the worst case is from #10 (for each economy size)
# plots.plot_protocol_universe_default(max_number_of_protocols = 30, crash_debts_df = debts_outstanding, oc_levels = OC_LEVELS, debt_size = 400000000, liquidity_param = 0.01)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker

sns.set_style("white")

from engine import get_data
from engine import simulation
from constants import DAYS_AHEAD, COLLATERALIZATION_RATIO, NUM_SIMULATIONS, QUANTITY_RESERVE_ASSET, COLLATERAL_ASSET

def plot_close_prices(close_prices):
    '''
    Plot prices of ETH and MKR on two separate y axes. 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ETH data
    ax.plot(close_prices.index, close_prices, label = 'ETH/USD', color = 'k', rasterized = True)
    ax.set_ylabel('ETH/USD price', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax.set_xlabel('')

    plt.title('ETH/USD close price', fontsize = 14)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/tokens_usd_close_price.pdf', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/tokens_usd_close_price.pdf', bbox_inches = 'tight', dpi = 300)

def plot_log_returns(log_returns):
    '''
    Plot log returns for the collateral asset (ETH).
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(log_returns, label = 'ETH/USD', color = 'k', rasterized = True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax.set_xlabel('')

    plt.title('ETH/USD log returns', fontsize = 14)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/tokens_usd_log_returns.pdf', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/tokens_usd_log_returns.pdf', bbox_inches = 'tight', dpi = 300)

def plot_histogram_log_returns(log_returns):
    '''
    Plot histogram of log returns. 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
   
    sns.distplot(log_returns, bins = 100, label = 'ETH/USD', color = 'k')
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax.set_xlabel('')

    plt.title('ETH/USD log returns', fontsize = 14)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/tokens_usd_histogram_log_returns.pdf', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/tokens_usd_histogram_log_returns.pdf', bbox_inches = 'tight', dpi = 300)

def plot_monte_carlo_simulations(price_simulations, 
                                    correlation: float, 
                                    returns_distribution: str):
    '''
    Plot the simulated prices.
    :price_simulations: input from the Monte Carlo simulator
    '''
    assets = list(price_simulations['1'].keys())
    for asset in assets:
        sims = simulation.asset_extractor_from_sims(price_simulations, asset)
        df = pd.DataFrame(sims)
        fig, ax = plt.subplots()
        df.plot(ax=ax, rasterized = True)
        ax.set_ylabel(asset + '/USD', fontsize = 14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.title(asset + '/USD: ' + str(NUM_SIMULATIONS) + ' Monte Carlo Simulations', fontsize = 14)
        ax.set_xlabel('Time steps (days)', fontsize = 14)
        ax.get_legend().remove()

        fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/' + asset + str(correlation) + str(returns_distribution) + '_monte_carlo.png', bbox_inches = 'tight', dpi = 300)
        fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/' + asset + str(correlation) + str(returns_distribution) + '_monte_carlo.png', bbox_inches = 'tight', dpi = 300)


def plot_worst_simulation(price_simulations, 
                            returns_distribution: str, 
                            fastest_crash_sim_index, 
                            correlation):
    '''
    Plot the behaviour of the collateral asset price and the other token prices for the worst outcome from Monte Carlo.
    '''
    #Find the worst collateral asset outcome
    sims_eth = simulation.asset_extractor_from_sims(price_simulations, COLLATERAL_ASSET)
    df_eth = pd.DataFrame(sims_eth)
    print('Index of sim featuring worst ' + str(COLLATERAL_ASSET) + ' outcome at ' + str(fastest_crash_sim_index))
    worst_eth = df_eth.loc[:, fastest_crash_sim_index]
    master_df = worst_eth.rename('ETH')

    #Find the corresponding simulation for the 'RES' asset
    sims_other = simulation.asset_extractor_from_sims(price_simulations, 'RES')
    df_other = pd.DataFrame(sims_other)
    corresponding_other_sims = df_other.loc[:, fastest_crash_sim_index]
    corresponding_other_sims = corresponding_other_sims.rename('RES')
    #Join and plot to see correlated movements
    master_df = pd.concat([master_df, corresponding_other_sims], axis = 1)
    
    fig, ax = plt.subplots()
    master_df.plot(ax=ax, rasterized = True)
    ax.set_ylabel('Price', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_rasterized(rasterized = True)
    plt.title('The co-evolution of the ETH and reserve token prices', fontsize = 14)
    ax.set_xlabel('Time steps (days)', fontsize = 14)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/' + str(correlation)+ str(returns_distribution) + 'co-evolution.pdf', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/'  + str(correlation)+ str(returns_distribution) + 'co-evolution.pdf', bbox_inches = 'tight', dpi = 300)

def plot_crash_sims(debt_levels, 
                    liquidity_levels, 
                    price_simulations, 
                    initial_eth_vol, 
                    fastest_crash_sim_index, 
                    returns_distribution, 
                    correlation):
    '''
    Plot system simulation.
    '''
    #Create 1 x 4 plot
    markers = ['s', 'p', 'v',]
    colors = ['g', 'k', 'r']
    fig, ax = plt.subplots(1, 4, figsize=(18,8))
    print('Using worst ' + str(COLLATERAL_ASSET) + ' outcome, ' + str(fastest_crash_sim_index))
    for i, debt in enumerate(debt_levels):
        debt_master_df_margin = pd.DataFrame(index = range(DAYS_AHEAD))
        debt_master_df_debt = pd.DataFrame(index = range(DAYS_AHEAD))
        for liquidity in liquidity_levels:
            system_simulations = simulation.crash_simulator(price_simulations = price_simulations, initial_debt = debt, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = liquidity)
            
            #Reconstruct dictionaries for margin and dai liability separately
            margin_evolutions = {}
            debt_evolutions = {}
            for x in range(1, NUM_SIMULATIONS+1):
                margin_evolutions[str(x)] = system_simulations[str(x)]['total_margin']
                debt_evolutions[str(x)] = system_simulations[str(x)]['debt']
            
            #Total margins
            df_margins = pd.DataFrame(margin_evolutions)
            worst_path_margin = df_margins.loc[:, fastest_crash_sim_index]
            debt_master_df_margin['Margin, '+ 'liq. ' + str(liquidity)] = worst_path_margin

            #Remaining debt
            df_debt = pd.DataFrame(debt_evolutions)
            worst_path_debt = df_debt.loc[:, fastest_crash_sim_index]
            debt_master_df_debt['Initial debt, ' + 'liq. ' + str(liquidity)] = worst_path_debt

        #Margin
        debt_master_df_margin.plot(ax = ax[i], markevery = 30, rasterized = True)
        for k, line in enumerate(ax[i].get_lines()):
            line.set_marker(markers[k])
            line.set_color(colors[k])

        #Debt
        debt_master_df_debt.plot(ax = ax[i], style = '--', markevery = 30)
        for k, line in enumerate(ax[i].get_lines()[len(debt_master_df_margin.columns):]):
            line.set_marker(markers[k])
            line.set_color(colors[k])

        #Graph polish
        debt_scale = debt / 100000000
        ax[i].set_title('Initial debt: ' + str(f'{debt:,}'), fontsize = 10.5)
        ax[i].tick_params(axis='both', which='major', labelsize=14)

        #Shading
        ax_lims = ax[i].get_ylim()
        ax[i].axhspan(ax_lims[0], 0, facecolor='red', alpha=0.5)

        if i<len(debt_levels)-1:
            ax[i].get_legend().remove()
        else:
            handles, labels = ax[i].get_legend_handles_labels()
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            fig.legend([extra, handles[0], handles[3], extra, handles[1], handles[4], extra, handles[2], handles[5]], ['Constant liquidity', 'Collateral margin', 'Remaining debt', 'Illiquidity: '+ str(liquidity_levels[1]), 'Collateral margin', 'Remaining debt', 'Illiquidity: '+ str(liquidity_levels[2]), 'Collateral margin', 'Remaining debt'], loc = 'lower center', ncol=3, borderaxespad=0., fontsize = 14)#, bbox_to_anchor=(0.5,-0.0005))
            ax[i].get_legend().remove()

    fig.subplots_adjust(bottom=0.2) 
    ax[0].set_ylabel('USD', fontsize = 14)
    ax[0].set_xlabel('Time steps (days)', fontsize = 14)
    fig.suptitle('A Decentralized Financial Crisis: liquidity and illiquidity causing negative margins', fontsize = 18)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/total_margin_debt' + str(returns_distribution) + str(correlation) + str(fastest_crash_sim_index) + '.pdf', bbox_inches='tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/total_margin_debt' + str(returns_distribution) + str(correlation) + str(fastest_crash_sim_index) + '.pdf', bbox_inches='tight', dpi = 300)


def plot_heatmap_liquidities(debt_levels, 
                            liquidity_params, 
                            price_simulations, 
                            initial_eth_vol, 
                            fastest_crash_sim_index):
    '''
    Plot heatmap of days until negative margin for debt and liquidity levels. 
    '''
    df = pd.DataFrame(index = debt_levels, columns = liquidity_params)
    
    for i in debt_levels:

        for j in liquidity_params:

            crash_sims = simulation.crash_simulator(price_simulations = price_simulations, 
                                                    initial_debt = i, 
                                                    initial_eth_vol = initial_eth_vol, 
                                                    collateralization_ratio = COLLATERALIZATION_RATIO, 
                                                    quantity_reserve_asset = QUANTITY_RESERVE_ASSET, 
                                                    liquidity_dryup = j)

            fastest_default = simulation.extract_sim_fastest_default(crash_sims)
            
            df.loc[int(i)][float(j)] = fastest_default[1]

    print(df)
    mask = df.isnull()
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.heatmap(df.astype(float), mask = mask, ax=ax, cmap='YlOrRd_r', yticklabels = [f'{x:,}' for x in debt_levels], xticklabels = [f'{x:,}' for x in liquidity_params], rasterized = True)
    ax.set_ylabel('Debt (USD)', fontsize = 18)
    ax.set_xlabel('Liquidity parameter', fontsize = 18)
    fig.suptitle('Number of days before Crisis', fontsize = 20, x=0.4)
    ax.tick_params(axis='both', which='major', labelsize=18)
    #ax.set_rasterized(rasterized = True)
    plt.xticks(rotation=90)
    ax.figure.axes[-1].yaxis.label.set_size(18)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/first_negative_params.pdf', bbox_inches='tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/first_negative_params.pdf', bbox_inches='tight', dpi = 300)


def plot_heatmap_initial_volumes(debt_levels, liquidity_param, price_simulations, initial_eth_vols, fastest_crash_sim_index):
    '''
    Plot heatmap of days until negative margin for debt and liquidity levels. 
    '''
    df = pd.DataFrame(index = debt_levels, columns = initial_eth_vols)
    
    for i in debt_levels:
        for j in initial_eth_vols:
            crash_sims = simulation.crash_simulator(price_simulations = price_simulations, 
                                                    initial_debt = i, 
                                                    initial_eth_vol = j, 
                                                    collateralization_ratio = COLLATERALIZATION_RATIO, 
                                                    quantity_reserve_asset = QUANTITY_RESERVE_ASSET, 
                                                    liquidity_dryup = liquidity_param)
            
            fastest_default = simulation.extract_sim_fastest_default(crash_sims)
            
            df.loc[int(i)][float(j)] = fastest_default[1]

    print(df)

    mask = df.isnull()
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.heatmap(df.astype(float), mask = mask, ax=ax, cmap='YlOrRd_r', yticklabels = [f'{x:,}' for x in debt_levels], xticklabels = [f'{x:,}' for x in initial_eth_vols], rasterized = True)
    ax.set_ylabel('Debt (USD)', fontsize = 18)
    #ax.set_rasterized(rasterized = True)
    ax.set_xlabel('Initial ETH liquidity', fontsize = 18)
    fig.suptitle('Number of days before Crisis', fontsize = 20, x=0.4)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks(rotation=90)
    ax.figure.axes[-1].yaxis.label.set_size(18)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/first_negative_vols.pdf', bbox_inches='tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/first_negative_vols.pdf', bbox_inches='tight', dpi = 300)

# def plot_worst_economy_outcomes(df, collateralization_ratio):
#     '''
#     Plot the worst case economy outcomes when protocols are composed. 
#     '''
    
#     df.plot(ax = ax, rasterized = True)
#     ax.get_legend().remove()
#     ax.set_ylabel('Loss (USD)', fontsize = 18)
#     ax.set_xlabel('Number of additional protocols', fontsize = 18)
#     fig.suptitle('Financial losses with composable protocols', fontsize = 20)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/worst_case_plot_'+str(collateralization_ratio)+'.pdf', bbox_inches='tight', dpi = 300)


# def plot_protocol_universe_default(max_number_of_protocols, crash_debts_df, oc_levels, debt_size, liquidity_param):
#     '''
#     For a list of overcollateralization levels, plot the worst case outcomes for each economy size.
#     '''
    
#     fig, ax = plt.subplots(1,3, figsize=(10,5))
#     for index, oc_level in enumerate(oc_levels):
#         sims = simulation.protocol_composer(max_number_of_protocols = max_number_of_protocols, crash_debts_df = crash_debts_df, max_oc_requirement = oc_level)
#         worst_outcomes = simulation.worst_case_per_protocol_number(sims, debt_size = debt_size, liquidity_param = liquidity_param)
#         worst_outcomes.plot(ax = ax[index])
#         ax[index].get_legend().remove()
#         ax[index].set_title('O/C: ' + str(oc_level), fontsize = 12)
#         ax[index].tick_params(axis='both', which='major', labelsize=14)
#     ax[0].set_ylabel('Total loss (USD)', fontsize = 14)
#     ax[0].set_xlabel('Number of additional protocols', fontsize = 14)
#     fig.suptitle('Financial losses with composable protocols', fontsize = 18)
#     fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
#     fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/protocol_defaults.pdf', bbox_inches='tight', dpi = 300)

def plot_tvl_defi(df):
    '''
    Plot prices of ETH and MKR on two separate y axes. 
    :df: dataframe of tvl from DeFi pulse. 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(df.index, df['tvlUSD'], label = 'ETH/USD', color = 'k', rasterized = True)
    ax.set_ylabel('USD', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax.set_xlabel('')

    plt.title('Total value locked in DeFi projects.', fontsize = 14)
    fig.savefig('../overleaf/5e8da3bb9abc6a0001c6d632/images/tvldefi.pdf', bbox_inches = 'tight', dpi = 300)
    fig.savefig('../overleaf/5ebe41169dc1fe00017c8460/figures/tvldefi.pdf', bbox_inches = 'tight', dpi = 300)

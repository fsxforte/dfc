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

from engine import get_data
from engine import simulation
from constants import TOKEN_BASKET, DAYS_AHEAD, COLLATERALIZATION_RATIO, NUM_SIMULATIONS, QUANTITY_RESERVE_ASSET

sns.set(style="darkgrid")

def plot_close_prices(start_date: dt.datetime, end_date: dt.datetime):
    '''
    Plot prices of ETH and MKR on two separate y axes. 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ETH data
    df = get_data.create_df('ETH', 'USD')
    df = df[start_date:end_date]
    ax.plot(df.index, df['close'], label = 'ETH/USD', color = 'k', rasterized = True)
    ax.set_ylabel('ETH/USD price', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax.set_xlabel('')

    #MKR data
    df = get_data.create_df('MKR', 'USD')
    df = df[start_date:end_date]
    ax2 = ax.twinx()
    ax2.plot(df.index, df['close'], label = 'MKR/USD', color = 'r', rasterized = True)
    ax2.set_ylabel('MKR/USD price', fontsize = 14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax2.set_xlabel('')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.title('ETH/USD and MKR/USD close prices', fontsize = 14)
    fig.savefig('../5d8dd7887374be0001c94b71/images/tokens_usd_close_price.pdf', bbox_inches = 'tight')

def plot_monte_carlo_simulations(price_simulations):
	'''
	Plot the simulated prices.
	:price_simulations: input from the Monte Carlo simulator
	'''
	for token in TOKEN_BASKET:
		index = TOKEN_BASKET.index(token)
		sims = simulation.asset_extractor_from_sims(price_simulations, index)
		df = pd.DataFrame(sims)
		fig, ax = plt.subplots()
		df.plot(ax=ax, rasterized = True)
		ax.set_ylabel(token + '/USD', fontsize = 14)
		ax.tick_params(axis='both', which='major', labelsize=14)
		plt.title(token + '/USD: ' + str(NUM_SIMULATIONS) + ' Monte Carlo Simulations', fontsize = 14)
		ax.set_xlabel('Time steps (days)', fontsize = 14)
		ax.get_legend().remove()
		fig.savefig('../5d8dd7887374be0001c94b71/images/' + token + '_monte_carlo.pdf', bbox_inches = 'tight')

def plot_worst_simulation(price_simulations, point_evaluate_eth_price):
	'''
	Plot the behaviour of the ETH price and the other token prices for the worst outcome from Monte Carlo.
	'''
	#Find the worst ETH outcome
	index = TOKEN_BASKET.index('ETH')
	sims_eth = simulation.asset_extractor_from_sims(price_simulations, index)
	df_eth = pd.DataFrame(sims_eth)
	worst_eth_outcome = get_data.extract_index_of_worst_eth_sim(price_simulations, point_evaluate_eth_price)
	print(worst_eth_outcome)
	worst_eth = df_eth.loc[:, worst_eth_outcome]
	column_name = worst_eth.columns.values[0]
	master_df = worst_eth.rename(columns = {column_name: "ETH"})

	for token in TOKEN_BASKET:
		if token != 'ETH':
			#Find the corresponding simulation for the other tokens
			index = TOKEN_BASKET.index(token)
			sims_other = simulation.asset_extractor_from_sims(price_simulations, index)
			df_other = pd.DataFrame(sims_other)
			corresponding_other_sims = df_other.loc[:, worst_eth_outcome]
			corresponding_other_sims = corresponding_other_sims.rename(columns = {column_name: str(token)})
			#Join and plot to see correlated movements
			master_df = pd.concat([master_df, corresponding_other_sims], axis = 1)
	
	df_normalized = master_df/master_df.loc[0]
	fig, ax = plt.subplots()
	df_normalized.plot(ax=ax, rasterized = True)
	ax.set_ylabel('Price evolution, normalized to 1', fontsize = 14)
	ax.tick_params(axis='both', which='major', labelsize=14)
	plt.title('The co-evolution of the ETH other token prices', fontsize = 14)
	ax.set_xlabel('Time steps (days)', fontsize = 14)
	fig.savefig('../5d8dd7887374be0001c94b71/images/co-evolution.pdf', bbox_inches = 'tight')

def plot_crash_sims(debt_levels, liquidity_levels, price_simulations, initial_eth_vol, point_evaluate_eth_price):
    '''
    Plot system simulation.
    '''
    #Create 1 x 4 plot
    markers = ['s', 'p', 'v',]
    colors = ['g', 'k', 'r']
    fig, ax = plt.subplots(1, 4, figsize=(18,8))
    worst_eth_outcome = get_data.extract_index_of_worst_eth_sim(price_simulations, point_evaluate_eth_price = point_evaluate_eth_price)
    print(worst_eth_outcome)
    for i, debt in enumerate(debt_levels):
        debt_master_df_margin = pd.DataFrame(index = range(DAYS_AHEAD))
        debt_master_df_debt = pd.DataFrame(index = range(DAYS_AHEAD))
        for liquidity in liquidity_levels:
            system_simulations = simulation.crash_simulator(simulations = price_simulations, initial_debt = debt, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = liquidity, token_basket = TOKEN_BASKET)
            
            #Reconstruct dictionaries for margin and dai liability separately
            margin_simulations = {}
            dai_simulations = {}
            for x in range(1, NUM_SIMULATIONS+1):
                margin_simulations[str(x)] = system_simulations[str(x)][0]
                dai_simulations[str(x)] = system_simulations[str(x)][1]
            
            #Total margins
            df_margins = pd.DataFrame(margin_simulations)
            worst_path_margin = df_margins.loc[:, worst_eth_outcome]
            worst_path_margin = worst_path_margin.rename(columns={'13': str(liquidity)})
            debt_master_df_margin['Margin, '+ 'liq. ' + str(liquidity)] = worst_path_margin

            #Remaining debt
            df_debt = pd.DataFrame(dai_simulations)
            worst_path_debt = df_debt.loc[:, worst_eth_outcome]
            worst_path_debt = worst_path_debt.rename(columns={'13': str(liquidity)})
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
            fig.legend([extra, handles[0], handles[3], extra, handles[1], handles[4], extra, handles[2], handles[5]], ['Constant liquidity', 'Margin', 'Remaining debt', 'Illiquidity: '+ str(liquidity_levels[1]), 'Margin', 'Remaining debt', 'Illiquidity: '+ str(liquidity_levels[2]), 'Margin', 'Remaining debt'], loc = 'lower center', ncol=3, borderaxespad=0., fontsize = 14)#, bbox_to_anchor=(0.5,-0.0005))
            ax[i].get_legend().remove()

    fig.subplots_adjust(bottom=0.2) 
    ax[0].set_ylabel('USD', fontsize = 14)
    ax[0].set_xlabel('Time steps (days)', fontsize = 14)
    fig.suptitle('A Decentralized Financial Crisis: liquidity and illiquidity causing negative margins', fontsize = 18)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.savefig('../5d8dd7887374be0001c94b71/images/total_margin_debt.pdf', bbox_inches='tight')

def plot_heatmap_liquidities(debt_levels, liquidity_params, price_simulations, initial_eth_vol, point_evaluate_eth_price):
	'''
	Plot heatmap of days until negative margin for debt and liquidity levels. 
	'''
	worst_eth_outcome = get_data.extract_index_of_worst_eth_sim(price_simulations, point_evaluate_eth_price = point_evaluate_eth_price)
	df_pairs = pd.DataFrame(index = debt_levels, columns = liquidity_params)
	
	for i in debt_levels:
		for j in liquidity_params:
			all_sims = simulation.crash_simulator(simulations = price_simulations, initial_debt = i, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = j, token_basket = TOKEN_BASKET)
			#Reconstruct dictionaries for margin and dai liability separately
			margin_simulations = {}
			for x in range(1, NUM_SIMULATIONS+1):
				margin_simulations[str(x)] = all_sims[str(x)][0]

			worst_margin_path = margin_simulations[worst_eth_outcome.values[0]]

			#Find the first day gone negative
			negative_days = []
			for index, margin in enumerate(worst_margin_path):
				if margin < 0:
					negative_days.append(index)
					first_negative_day = negative_days[0]
					df_pairs.loc[int(i)][float(j)] = first_negative_day
				#else:
				#	df_pairs.loc[int(i)][float(j)] = 0

	print(df_pairs)
	mask = df_pairs.isnull()
	sns.set(font_scale=1.4)
	fig, ax = plt.subplots(1,1, figsize=(10,8))
	sns.heatmap(df_pairs.astype(float), mask = mask, ax=ax, cmap='YlOrRd_r', yticklabels = [f'{x:,}' for x in debt_levels], xticklabels = [f'{x:,}' for x in liquidity_params], rasterized = True)
	ax.set_ylabel('Debt (USD)', fontsize = 18)
	ax.set_xlabel('Liquidity parameter', fontsize = 18)
	fig.suptitle('Number of days before Crisis', fontsize = 20, x=0.4)
	ax.tick_params(axis='both', which='major', labelsize=18)
	plt.xticks(rotation=90)
	ax.figure.axes[-1].yaxis.label.set_size(18)
	fig.savefig('../5d8dd7887374be0001c94b71/images/first_negative_params.pdf', bbox_inches='tight')

def plot_heatmap_initial_volumes(debt_levels, liquidity_param, price_simulations, initial_eth_vols, point_evaluate_eth_price):
	'''
	Plot heatmap of days until negative margin for debt and liquidity levels. 
	'''
	worst_eth_outcome = get_data.extract_index_of_worst_eth_sim(price_simulations, point_evaluate_eth_price = point_evaluate_eth_price)
	df_pairs = pd.DataFrame(index = debt_levels, columns = initial_eth_vols)
	
	for i in debt_levels:
		for j in initial_eth_vols:
			all_sims = simulation.crash_simulator(simulations = price_simulations, initial_debt = i, initial_eth_vol = j, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = liquidity_param, token_basket = TOKEN_BASKET)
			#Reconstruct dictionaries for margin and dai liability separately
			margin_simulations = {}
			for x in range(1, NUM_SIMULATIONS+1):
				margin_simulations[str(x)] = all_sims[str(x)][0]

			worst_margin_path = margin_simulations[worst_eth_outcome.values[0]]

			#Find the first day gone negative
			negative_days = []
			for index, margin in enumerate(worst_margin_path):
				if margin < 0:
					negative_days.append(index)
					first_negative_day = negative_days[0]
					df_pairs.loc[int(i)][float(j)] = first_negative_day
				#else:
				#	df_pairs.loc[int(i)][float(j)] = 0

	print(df_pairs)

	mask = df_pairs.isnull()
	sns.set(font_scale=1.4)
	fig, ax = plt.subplots(1,1, figsize=(10,8))
	sns.heatmap(df_pairs.astype(float), mask = mask, ax=ax, cmap='YlOrRd_r', yticklabels = [f'{x:,}' for x in debt_levels], xticklabels = [f'{x:,}' for x in initial_eth_vols], rasterized = True)
	ax.set_ylabel('Debt (USD)', fontsize = 18)
	ax.set_xlabel('Initial ETH/DAI liquidity', fontsize = 18)
	fig.suptitle('Number of days before Crisis', fontsize = 20, x=0.4)
	ax.tick_params(axis='both', which='major', labelsize=18)
	plt.xticks(rotation=90)
	ax.figure.axes[-1].yaxis.label.set_size(18)
	fig.savefig('../5d8dd7887374be0001c94b71/images/first_negative_vols.pdf', bbox_inches='tight')

def plot_worst_economy_outcomes(df, collateralization_ratio):
	'''
	Plot the worst case economy outcomes when protocols are composed. 
	'''
	
	df.plot(ax = ax, rasterized = True)
	ax.get_legend().remove()
	ax.set_ylabel('Loss (USD)', fontsize = 18)
	ax.set_xlabel('Number of additional protocols', fontsize = 18)
	fig.suptitle('Financial losses with composable protocols', fontsize = 20)
	ax.tick_params(axis='both', which='major', labelsize=18)
	fig.savefig('../5d8dd7887374be0001c94b71/images/worst_case_plot_'+str(collateralization_ratio)+'.df', bbox_inches='tight')


def plot_protocol_universe_default(max_number_of_protocols, crash_debts_df, oc_levels, debt_size, liquidity_param):
	'''
	For a list of overcollateralization levels, plot the worst case outcomes for each economy size.
	'''
	fig, ax = plt.subplots(1,3, figsize=(10,5))
	for index, oc_level in enumerate(oc_levels):
		sims = simulation.protocol_composer(max_number_of_protocols = max_number_of_protocols, crash_debts_df = crash_debts_df, max_oc_requirement = oc_level)
		worst_outcomes = simulation.worst_case_per_protocol_number(sims, debt_size = debt_size, liquidity_param = liquidity_param)
		worst_outcomes.plot(ax = ax[index])
		ax[index].get_legend().remove()
		ax[index].set_title('O/C: ' + str(oc_level), fontsize = 12)
		ax[index].tick_params(axis='both', which='major', labelsize=14)
	ax[0].set_ylabel('Total loss (USD)', fontsize = 14)
	ax[0].set_xlabel('Number of additional protocols', fontsize = 14)
	fig.suptitle('Financial losses with composable protocols', fontsize = 18)
	fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
	fig.savefig('../5d8dd7887374be0001c94b71/images/protocol_defaults.pdf', bbox_inches='tight')
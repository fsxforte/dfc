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
from scripts import analysis
from constants import TOKEN_BASKET, DAYS_AHEAD, COLLATERALIZATION_RATIO, NUM_SIMULATIONS, QUANTITY_RESERVE_ASSET

sns.set(style="darkgrid")

def plot_prices(df):
	'''
	Function to plot the monte carlo prices. 
	'''
	df = df
	SIM_NO = 1000
	PERIOD_NO = 1000
	sim = simulation.brownian_motion(df, SIM_NO, PERIOD_NO)
	fig, ax = plt.subplots()
	sim.plot(ax = ax)
	ax.get_legend().remove()
	ax.set_ylabel('Price Grin/USDC', fontsize = 16)
	ax.set_xlabel('Time (5 minute blocks)', fontsize = 16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.title(str(SIM_NO)+ ' Brownian motion simulations of the Grin/USDC price', fontsize = 16)
	plt.show()

def dist_prices(df):
	'''
	Plot distribution of final prices.
	'''
	df = df
	SIM_NO = 1000
	PERIOD_NO = 1000
	sim = simulation.brownian_motion(df, SIM_NO, PERIOD_NO)
	final_prices = list(sim.iloc[-1])
	first_price = list(sim.iloc[0])[0]
	liquidation_price = first_price*(2/3)
	fig, ax = plt.subplots()
	sns.distplot(final_prices, ax = ax, bins = 50)
	plt.axvline(liquidation_price)
	ax.set_ylabel('Price Grin/USDC', fontsize = 16)
	ax.set_xlabel('Price at end of prediction period', fontsize = 16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.title('Distribution of resultant prices', fontsize = 16)
	plt.show()

#Plot a panel of all the prices for the 5 assets
def plot_close_prices(start_date: dt.datetime, end_date: dt.datetime):
	df = analysis.make_close_matrix()
	df = df[start_date:end_date]
	fig, ax = plt.subplots(5, 1, sharex = 'col')
	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	ax[0].plot(df['ETH'])
	ax[0].set_title('ETH', fontsize=16)
	ax[1].plot(df['DAI'])
	ax[1].set_title('DAI', fontsize=16)
	ax[2].plot(df['REP'])
	ax[2].set_title('Augur', fontsize=16)
	ax[3].plot(df['ZRX'])
	ax[3].set_title('0x', fontsize=16)
	ax[4].plot(df['BAT'])
	ax[4].set_title('BAT', fontsize=16)
	plt.grid(False)
	fig.autofmt_xdate()
	plt.ylabel('Close price', fontsize=16)
	plt.show()


def plot_log_returns(start_date: dt.datetime, end_date: dt.datetime):
	df = analysis.make_logreturns_matrix()
	df = df[start_date:end_date]
	fig, ax = plt.subplots(5, 1, sharex = 'col')
	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	ax[0].plot(df['ETH'])
	ax[0].set_title('ETH', fontsize=16)
	ax[1].plot(df['DAI'])
	ax[1].set_title('DAI', fontsize=16)
	ax[2].plot(df['REP'])
	ax[2].set_title('Augur', fontsize=16)
	ax[3].plot(df['ZRX'])
	ax[3].set_title('0x', fontsize=16)
	ax[4].plot(df['BAT'])
	ax[4].set_title('BAT', fontsize=16)
	plt.grid(False)
	fig.autofmt_xdate()
	plt.ylabel('Close price', fontsize=16)
	plt.show()

def correlation_heatmap(corr_df):
	mask = np.zeros_like(corr_df)
	mask[np.triu_indices_from(mask)] = True
	#generate plot
	sns.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
	plt.yticks(rotation=0) 
	plt.xticks(rotation=90) 
	plt.show()

def plot_prices(start_date: dt.datetime, end_date: dt.datetime):
    '''
    Plot prices of ETH and MKR on two separate y axes. 
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ETH data
    df = get_data.create_df(TOKEN_BASKET[0], 'USD')
    df = df[start_date:end_date]
    ax.plot(df.index, df['close'], label = 'ETH/USD', color = 'k')
    ax.set_ylabel(TOKEN_BASKET[0] + '/USD price', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax.set_xlabel('')

    #MKR data
    df = get_data.create_df(TOKEN_BASKET[1], 'USD')
    df = df[start_date:end_date]
    ax2 = ax.twinx()
    ax2.plot(df.index, df['close'], label = 'MKR/USD', color = 'r')
    ax2.set_ylabel(TOKEN_BASKET[1] + '/USD price', fontsize = 14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    fig.autofmt_xdate()
    ax2.set_xlabel('')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.title('ETH/USD and MKR/USD close prices', fontsize = 14)
    fig.savefig('../5d8dd7887374be0001c94b71/images/tokens_usd_close_price.png', bbox_inches = 'tight', dpi = 300)

def plot_simulations(price_simulations):
    '''
    Plot the ETH and MKR simulated outcomes
    '''
    ##ETH
    sims = simulation.asset_extractor_from_sims(price_simulations, 0)
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
    sims = simulation.asset_extractor_from_sims(price_simulations, 1)
    df = pd.DataFrame(sims)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_ylabel('MKR/USD', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.title('MKR/USD: 1000 Monte Carlo Simulations', fontsize = 14)
    ax.set_xlabel('Time steps (days)', fontsize = 14)
    ax.get_legend().remove()
    fig.savefig('../5d8dd7887374be0001c94b71/images/mkr_monte_carlo.png', bbox_inches = 'tight', dpi = 600)

def plot_worst_simulation(price_simulations):
	'''
	Plot the behaviour of the ETH price and the MKR price for the worst outcome from monte carlo.
	'''
	sims_eth = simulation.asset_extractor_from_sims(price_simulations, 0)
	df_eth = pd.DataFrame(sims_eth)
	worst_eth_outcome = get_data.extract_index_of_worst_sim(price_simulations)
	worst_eth = df_eth.loc[:, worst_eth_outcome]
	worst_eth = worst_eth.rename(columns = {"13": "ETH"})
    #Find corresponding bad MKR outcomes
	sims_mkr = simulation.asset_extractor_from_sims(price_simulations, 1)
	df_mkr = pd.DataFrame(sims_mkr)
	corresponding_mkr_sims = df_mkr.loc[:, worst_eth_outcome]
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

def plot_sims(debt_levels, liquidity_levels, price_simulations, initial_eth_vol):
    '''
    Plot system simulation.
    '''
    #Create 1 x 4 plot
    markers = ['s', 'p', 'v',]
    colors = ['g', 'k', 'r']
    fig, ax = plt.subplots(1, 4, figsize=(18,8))
    worst_eth_outcome = get_data.extract_index_of_worst_sim(price_simulations)
    for i, debt in enumerate(debt_levels):
        debt_master_df_margin = pd.DataFrame(index = range(DAYS_AHEAD))
        debt_master_df_debt = pd.DataFrame(index = range(DAYS_AHEAD))
        for liquidity in liquidity_levels:
            system_simulations = simulation.crash_simulator(simulations = price_simulations, initial_debt = debt, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = liquidity)
            
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

        if i<len(debt_levels)-1:
            ax[i].get_legend().remove()
        else:
            handles, labels = ax[i].get_legend_handles_labels()
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            fig.legend([extra, handles[0], handles[3], extra, handles[1], handles[4], extra, handles[2], handles[5]], ['Liquidity: '+ str(liquidity_levels[0]), 'Margin', 'Remaining debt', 'Liquidity: '+ str(liquidity_levels[1]), 'Margin', 'Remaining debt', 'Liquidity: '+ str(liquidity_levels[2]), 'Margin', 'Remaining debt'], loc = 'lower center', ncol=3, borderaxespad=0., fontsize = 14)#, bbox_to_anchor=(0.5,-0.0005))
            ax[i].get_legend().remove()

    fig.subplots_adjust(bottom=0.2) 
    ax[0].set_ylabel('USD', fontsize = 14)
    ax[0].set_xlabel('Time steps (days)', fontsize = 14)
    fig.suptitle('A Decentralized Financial Crisis: liquidity and illiquidity causing negative margins', fontsize = 18)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.savefig('../5d8dd7887374be0001c94b71/images/total_margin_debt.png', dpi = 600, bbox_inches='tight')

def plot_heatmap(debt_levels, liquidity_levels, price_simulations, initial_eth_vol):
	'''
	Plot heatmap of days until negative margin for debt and liquidity levels. 
	'''
	worst_eth_outcome = get_data.extract_index_of_worst_sim(price_simulations)
	df_pairs = pd.DataFrame(index = debt_levels, columns = liquidity_levels)
	
	for i in debt_levels:
		for j in liquidity_levels:
			all_sims = simulation.crash_simulator(simulations = price_simulations, initial_debt = i, initial_eth_vol = initial_eth_vol, collateralization_ratio = COLLATERALIZATION_RATIO, quantity_reserve_asset = QUANTITY_RESERVE_ASSET, liquidity_dryup = j)
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

	df_pairs_clean = df_pairs.dropna()
	sns.set(font_scale=1.4)
	fig, ax = plt.subplots(1,1, figsize=(10,8))
	sns.heatmap(df_pairs_clean.astype(float), ax=ax, cmap='YlOrRd_r')
	ax.set_ylabel('Debt (USD)', fontsize = 18)
	ax.set_xlabel('Liquidity parameter', fontsize = 18)
	fig.suptitle('Number of days before Crisis', fontsize = 20, x=0.4)
	ax.tick_params(axis='both', which='major', labelsize=18)
	plt.xticks(rotation=90)
	ax.figure.axes[-1].yaxis.label.set_size(18)
	fig.savefig('../5d8dd7887374be0001c94b71/images/first_negative.png', dpi = 600, bbox_inches='tight')
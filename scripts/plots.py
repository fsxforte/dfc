import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np

from engine import get_data
from engine import monte_carlo
from scripts import analysis

def plot_prices(df):
	'''
	Function to plot the monte carlo prices. 
	'''
	df = df
	SIM_NO = 1000
	PERIOD_NO = 1000
	sim = monte_carlo.brownian_motion(df, SIM_NO, PERIOD_NO)
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
	sim = monte_carlo.brownian_motion(df, SIM_NO, PERIOD_NO)
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
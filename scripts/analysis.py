import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from engine import get_data
import constants

#Extract current PETH/ETH ratio
#ETH_PETH_RATIO = 

def make_correlations_matrix(input_matrix):
	corr_df = input_matrix.corr(method = 'pearson')
	corr_df.head().reset_index()
	del corr_df.index.name
	return corr_df

def build_cdp_dataframe():
	'''
	Build CDP DataFrame, including helper columns for later calculations.
	'''
	#Get CDP data
	data = get_data.get_CDP_data()
	df = pd.DataFrame.from_dict(data['data']['allCups']['nodes'])
	df.set_index(df['id'], inplace=True)
	df['art'] = df['art'].astype(float)
	df['ink'] = df['ink'].astype(float)

	#Add a column for the price at which the CDP becomes liable to liquidation at 150% liquidation ratio
	###NEED TO RETRIEVE PETH/ETH, not HARD CODE!!!!!####
	df['liquidation_price'] = df['art'] / (1.045 * df['ink'] *2/3)

	#Add a column for the price at which the CDP becomes undercollateralized (less than 100%)
	df['underwater_price'] = df['art'] / (1.045 * df['ink'])

	#Extract subset of liquidation prices
	#Subset constraints: (1) more than 5USD in DAI
	df = df[df['art']>5]

	return df

def plot_liquidations_for_price_drops(df, threshold: str):
	'''
	Plot liquidation price vs. ETH quantity.
	:df: input DataFrame
	:threshold: 'liquidation_price' (150% collateral) or 'underwater_price' (100% collateral)
	'''
	#Extract current ETH price
	CURRENT_ETH_PRICE = float(df['pip'].iloc[0])

	#How much volume would be sold for different prices?
	price_ethvol = {}
	for perc_fall in range(101):
		new_price = CURRENT_ETH_PRICE*(100-perc_fall)/100
		df['eth_liability'] = 1.13 * df['art']/df[threshold]
		total_eth_liquidated = df['eth_liability'][df[threshold]>new_price].sum()
		price_ethvol[new_price] = total_eth_liquidated

	df_liquidations = pd.DataFrame.from_dict(price_ethvol, orient = 'index', columns = {'eth_liquidated'})
	df_liquidations['eth_price'] = df_liquidations.index
	df_liquidations = df_liquidations[df_liquidations['eth_price']>0]

	fig, ax = plt.subplots()
	df_liquidations.plot(x = 'eth_liquidated', y = 'eth_price', ax = ax)
	ax.set_ylabel('ETH/USD price', fontsize = 16)
	ax.set_xlabel('ETH quantity (cumulative)', fontsize = 16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.title('Price vs. ETH volume for liquidation', fontsize = 16)
	ax.get_legend().remove()
	fig.savefig('pricevseth.png', bbox_inches = 'tight', dpi = 600)
	plt.show()

def plot_liquidations_for_perc_price_drops(df, threshold: str):
	'''
	Plot correspondence between percentage changes in price and ETH volumes.
	:df: input DataFrame
	:threshold: 'liquidation_price' (150% collateral) or 'underwater_price' (100% collateral)
	'''
	#Extract current ETH price
	CURRENT_ETH_PRICE = float(df['pip'].iloc[0])
	perc_ethvol = {}
	for perc_fall in range(101):
		new_price = CURRENT_ETH_PRICE*(100-perc_fall)/100
		df['eth_liability'] = 1.13 * df['art']/df[threshold]
		total_eth_liquidated = df['eth_liability'][df[threshold]>new_price].sum()
		perc_ethvol[perc_fall] = total_eth_liquidated

	df_liquidations = pd.DataFrame.from_dict(perc_ethvol, orient = 'index', columns = {'eth_liquidated'})
	df_liquidations['perc_fall'] = df_liquidations.index

	fig, ax = plt.subplots()
	df_liquidations.plot(x = 'eth_liquidated', y = 'perc_fall', ax = ax)
	ax.set_ylabel('Percentage fall in ETH/USD price', fontsize = 16)
	ax.set_xlabel('ETH quantity (cumulative)', fontsize = 16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.title('Impact of percentage fall in ETH price on liquidations', fontsize = 16)
	ax.get_legend().remove()
	fig.savefig('ethpricepercentages.png', bbox_inches = 'tight', dpi = 600)
	plt.show()


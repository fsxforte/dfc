import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from engine import get_data
import constants

def make_correlations_matrix(input_matrix):
	corr_df = input_matrix.corr(method = 'pearson')
	corr_df.head().reset_index()
	del corr_df.index.name
	return corr_df

#Get CDP data
data = get_data.get_CDP_data()
df = pd.DataFrame.from_dict(data['data']['allCups']['nodes'])
df.set_index(df['id'], inplace=True)
df['art'] = df['art'].astype(float)
df['ink'] = df['ink'].astype(float)

#Add a column for the price at which the CDP becomes liable to liquidation at 150% liquidation ratio
df['liquidation_price'] = df['art'] * 1.5 / df['ink']

#Add a column for the price at which the CDP becomes undercollateralized (less than 100%)
df['underwater_price'] = df['art'] / df['ink']

#Extract subset of liquidation prices
#Subset constraints: (1) more than 5USD in DAI
df = df[df['art']>5]

sns.distplot(df['liquidation_price'])
sns.distplot(df['underwater_price'])

#Want to see how cumulative eth volume increases with liquidation price
df.sort_values(by = 'liquidation_price', inplace=True, ascending = False)

#Extract current ETH price
current_eth_price = float(df['pip'].iloc[0])

def plot_liquidation_vols(df):
	'''
	Plot liquidation price vs. ETH quantity.
	'''
	#How much volume would be sold for different prices?
	price_ethvol = {}
	for perc_fall in range(101):
		new_price = current_eth_price*(100-perc_fall)/100
		###THE LINE BELOW IS WRONG - IN SOME CASES 113% of debt is more than the INK
		df['eth_liability'] = 1.13 * df['art']/new_price
		total_eth_liquidated = df['eth_liability'][df['liquidation_price']>new_price].sum()
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

def plot_liquidation_price_drops(df_liquidations):
	'''
	Plot correspondence between percentage changes in price and ETH volumes.
	'''
	fig, ax = plt.subplots()
	df_liquidations.plot(ax = ax)
	ax.set_ylabel('ETH volume', fontsize = 16)
	ax.set_xlabel('Percentage drop in price from $130 18 Dec 2019', fontsize = 16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.title('ETH vols and percentage price drops', fontsize = 16)
	fig.savefig('ethpricepercentages.png', bbox_inches = 'tight', dpi = 600)
	plt.show()


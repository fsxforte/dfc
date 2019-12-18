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

#Get series of liquidation prices for CDPs
data = get_data.get_CDP_data()
df = pd.DataFrame.from_dict(data['data']['allCups']['nodes'])
df.set_index(df['id'], inplace=True)
df['art'] = df['art'].astype(float)
df['ink'] = df['ink'].astype(float)

#Calculate liquidation price
df['liquidation_price'] = df['art'].astype(float) * 1.5 / df['ink'].astype(float)
#df[df['id']==147710]

#Extract subset of liquidation prices
#Subset constraints: (1) more than 5USD in DAI
df = df[df['art']>5]

sns.distplot(df['liquidation_price'])

#Want to see how cumulative eth volume increases with liquidation price
df.sort_values(by = 'liquidation_price', inplace=True, ascending = False)
df['cumulative_eth']=df['ink'].cumsum()

#Plot of the liquidation price
df.plot(x = 'cumulative_eth', y = 'liquidation_price')

def plot_liquidation_price_vol(df):
	'''
	Plot liquidation price vs. ETH quantity.
	'''
	SIM_NO = 1000
	PERIOD_NO = 1000
	fig, ax = plt.subplots()
	df.plot(x = 'cumulative_eth', y = 'liquidation_price', ax = ax)
	ax.set_ylabel('ETH/USD price', fontsize = 16)
	ax.set_xlabel('ETH quantity (cumulative)', fontsize = 16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	plt.title('Price vs. ETH volume for liquidation', fontsize = 16)
	ax.get_legend().remove()
	fig.savefig('pricevseth.png', bbox_inches = 'tight', dpi = 600)
	plt.show()
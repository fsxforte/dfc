import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from engine import get_data
from engine import monte_carlo


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





#How many of these prices correspond to a loss of more than the 100-150% margin?


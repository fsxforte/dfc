import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy import stats
from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns
from engine import kernel_estimation

from engine import get_data

sns.set(style="darkgrid")

TOKEN_BASKET = ['ETH', 'MKR']

#Select subset for 2018 until now
start_date = dt.datetime(2018,1,1) # First data for MKE token
end_date = dt.datetime(2020,1,20)

fig = plt.figure()
ax = fig.add_subplot(111)

#0. Get the data and plot the prices
df = get_data.create_df(TOKEN_BASKET[0], 'USD')
df = df[start_date:end_date]
ax.plot(df.index, df['close'], label = 'ETH/USD', color = 'k')
ax.set_ylabel(TOKEN_BASKET[0] + '/USD price', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
fig.autofmt_xdate()
ax.set_xlabel('')

df = get_data.create_df(TOKEN_BASKET[1], 'USD')
start_date = dt.datetime(2018,1,1)
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
fig.savefig('../5d8dd7887374be0001c94b71/images/tokens_usd_close_price.png', bbox_inches = 'tight', dpi = 600)


#1. Plot the returns
tokens_df = get_data.create_logrets_df()
eth_usd_logrets = tokens_df['ETH']
eth_usd_logrets = eth_usd_logrets[start_date:end_date]
fig, ax = plt.subplots()
sns.lineplot(data = eth_usd_logrets, ax = ax)
ax.set_ylabel('ETH/USD % change', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('ETH/USD daily returns', fontsize = 14)
fig.autofmt_xdate()
ax.set_xlabel('')
fig.savefig('../5d8dd7887374be0001c94b71/images/eth_usd_returns.png', bbox_inches = 'tight', dpi = 600)


#2. Plot the distribution of returns
tokens_df = get_data.create_logrets_df()
eth_usd_logrets = tokens_df['ETH']
eth_usd_logrets = eth_usd_logrets[start_date:end_date]
fig, ax = plt.subplots()
sns.distplot(eth_usd_logrets, bins = 100, kde = True, ax = ax)
ax.set_ylabel('')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('Distribution of ETH/USD daily returns', fontsize = 14)
ax.set_xlabel('Percentage change')
ax.set(yticks=[])
fig.savefig('../5d8dd7887374be0001c94b71/images/eth_usd_returns_distribution.png', bbox_inches = 'tight', dpi = 600)

#3. What are the worst shocks at different p values?
worst_shock = np.percentile(eth_usd_logrets, 1)
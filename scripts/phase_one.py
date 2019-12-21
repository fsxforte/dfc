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

#Select subset for 2016 until now
start_date = dt.datetime(2016,1,1)
end_date = dt.datetime(2019,12,20)

#0. Get the data and plot the price
eth_usd_df = get_data.create_df('ETH', 'USD')
eth_usd_df = eth_usd_df[start_date:end_date]
fig, ax = plt.subplots()
sns.lineplot(x = eth_usd_df.index, y = 'close', data = eth_usd_df, ax = ax)
ax.set_ylabel('ETH/USD price', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('ETH/USD close price', fontsize = 14)
fig.autofmt_xdate()
ax.set_xlabel('')
fig.savefig('../5d8dd7887374be0001c94b71/images/eth_usd_close_price.png', bbox_inches = 'tight', dpi = 600)


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





# #2. Inspect ACF and PACF
# plot_acf(eth_usd_logrets, lags = 100) # First 3 lags are significant
# plot_pacf(eth_usd_logrets, lags = 100) # First 4 lags are significant

# #Test for Normality
# stats.jarque_bera(eth_usd_logrets) #Reject normality

# #Fit standard GARCH model with constant mean
# am = arch_model(100*eth_usd_logrets)
# res_normal = am.fit(update_freq=5)
# print(res_normal.summary())
# fig = res_normal.plot(annualize='D') # Evidence of considerable persistence

# # #Asymmetric GARCH (GJR GARCH)
# # am = arch_model(100*log_returns, p=1, o=1, q=1)
# # res = am.fit(update_freq=5, disp='off')
# # print(res.summary())
# # fig = res.plot(annualize='D') # Evidence of considerable persistence

# # #TARCH
# # am = arch_model(log_returns, p=1, o=1, q=1, power=1.0)
# # res = am.fit(update_freq=5)
# # print(res.summary())
# # fig = res.plot(annualize='D') # Evidence of considerable persistence
# # plt.show()

# #Standardize the residuals
# std_resid = res_normal.resid / res_normal.conditional_volatility
# unit_var_resid = res_normal.resid / res_normal.resid.std()
# df = pd.concat([std_resid, unit_var_resid], 1)
# df.columns = ['Std Resids', 'Unit Variance Resids']
# subplot = df.plot(kind='kde', xlim=(-4, 4))

# #Multiply the standardized returns by the current volatility estimate (scaling all returns by the current conditional volatility estimate)
# scaled_resids = res_normal.conditional_volatility[-1] * unit_var_resid



import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy import stats
from sklearn.neighbors import KernelDensity
import numpy as np


from engine import get_data


#0. Get the data and plot the log returns
df = get_data.fetch_data()
log_returns = df['log_returns']
log_returns.plot()
#Reveals that there is a significant outlier early on in the series --> restrict data to April 2019 onwards

df_subset = df[dt.datetime(2019, 4, 1):]
log_returns = df_subset['log_returns']
log_returns.plot()
#Plot superficially shows evidence of correlated residuals

#1. Inspect ACF and PACF
plot_acf(log_returns, lags = 100) # First 3 lags are significant
plot_pacf(log_returns, lags = 100) # First 4 lags are significant

#Test for Normality
stats.jarque_bera(log_returns) #Reject normality

#Fit standard GARCH model with constant mean
am = arch_model(100*log_returns)
res_normal = am.fit(update_freq=5)
print(res.summary())
fig = res.plot(annualize='D') # Evidence of considerable persistence

# #Asymmetric GARCH (GJR GARCH)
# am = arch_model(100*log_returns, p=1, o=1, q=1)
# res = am.fit(update_freq=5, disp='off')
# print(res.summary())
# fig = res.plot(annualize='D') # Evidence of considerable persistence

# #TARCH
# am = arch_model(log_returns, p=1, o=1, q=1, power=1.0)
# res = am.fit(update_freq=5)
# print(res.summary())
# fig = res.plot(annualize='D') # Evidence of considerable persistence
# plt.show()

#Standardize the residuals
std_resid = res_normal.resid / res_normal.conditional_volatility
unit_var_resid = res_normal.resid / res_normal.resid.std()
df = pd.concat([std_resid, unit_var_resid], 1)
df.columns = ['Std Resids', 'Unit Variance Resids']
subplot = df.plot(kind='kde', xlim=(-4, 4))

#Multiply the standardized returns by the current volatility estimate (scaling all returns by the current conditional volatility estimate)
scaled_resids = res_normal.conditional_volatility[-1] * unit_var_resid



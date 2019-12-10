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

#####################################
##### Kernel Density Estimation #####
#####################################

#Convert to the right format for the Kernel Density estimation
X = 100*log_returns.to_numpy().reshape(-1,1)

X_plot = np.linspace((array.min()-1), array.max()+1, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')
colors = ['navy', 'cornflowerblue', 'darkorange']
kernels = ['gaussian', 'tophat', 'epanechnikov']
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
            linestyle='-', label="kernel = '{0}'".format(kernel))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

plt.show()

#Can only sample from 'gaussian', 'tophat' --> Go with tophat as closer to epanechnikov
kde = KernelDensity(kernel='tophat', bandwidth=0.5).fit(X)
#Sample from the data
new_data = kde.sample(1000)
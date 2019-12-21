import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

from arch.univariate import GARCH, ConstantMean, Normal

from engine import get_data

rets = 100 * get_data.create_logrets_series('ETH', 'USD')[dt.datetime(2016,1,1):dt.datetime(2019,12,20)]

# Build components to set the state for the distribution
random_state = np.random.RandomState(1)
dist = Normal(random_state=random_state)
volatility = GARCH(1, 1, 1)

mod = ConstantMean(rets, volatility=volatility, distribution=dist)

res = mod.fit(disp='off')
res

forecasts = res.forecast(start='1-1-2016', horizon=10)
print(forecasts.residual_variance.dropna().head())

sim_forecasts = res.forecast(start='1-1-2016', method='simulation', horizon=10)
print(sim_forecasts.residual_variance.dropna().head())

### Custom Random Generators
random_state = np.random.RandomState(1)

def scenario_rng(size):
    shocks = random_state.standard_normal(size)
    shocks[:, :5] *= np.sqrt(2)
    return shocks

scenario_forecasts = res.forecast(
    start='1-1-2016', method='simulation', horizon=10, rng=scenario_rng)
print(scenario_forecasts.residual_variance.dropna().head())

###Bootstrap scenarios
class ScenarioBootstrapRNG(object):
    def __init__(self, shocks, random_state):
        self._shocks = np.asarray(shocks)  # 1d
        self._rs = random_state
        self.n = shocks.shape[0]

    def rng(self, size):
        idx = self._rs.randint(0, self.n, size=size)
        return self._shocks[idx]


random_state = np.random.RandomState(1)
std_shocks = res.resid / res.conditional_volatility
shocks = std_shocks['2018-01-14':'2018-04-01']
scenario_bootstrap = ScenarioBootstrapRNG(shocks, random_state)
bs_forecasts = res.forecast(
    start='1-1-2016',
    method='simulation',
    horizon=10,
    rng=scenario_bootstrap.rng)
print(bs_forecasts.residual_variance.dropna().head())

###Visualize differences
df = pd.concat([
    forecasts.residual_variance.iloc[-1],
    sim_forecasts.residual_variance.iloc[-1],
    scenario_forecasts.residual_variance.iloc[-1],
    bs_forecasts.residual_variance.iloc[-1]
], 1)
df.columns = ['Analytic', 'Simulation', 'Scenario Sim', 'Bootstrp Scenario']
# Plot annualized vol
subplot = np.sqrt(365.25*df).plot(legend=False)
legend = subplot.legend(frameon=False)


###Comparing paths
fig, axes = plt.subplots(1, 2)
colors = sns.color_palette('dark')
# The paths for the final observation
sim_paths = sim_forecasts.simulations.residual_variances[-1].T
bs_paths = bs_forecasts.simulations.residual_variances[-1].T

x = np.arange(1, 11)
# Plot the paths and the mean, set the axis to have the same limit
axes[0].plot(x, np.sqrt(365.25 * sim_paths), color=colors[1], alpha=0.05)
axes[0].plot(
    x,
    np.sqrt(365.25 * sim_forecasts.residual_variance.iloc[-1]),
    color='k',
    alpha=1)
axes[0].set_title('Model-based Simulation')
axes[0].set_xticks(np.arange(1, 11))
axes[0].set_xlim(1, 10)
axes[0].set_ylim(60, 250)

axes[1].plot(x, np.sqrt(365.25 * bs_paths), color=colors[2], alpha=0.05)
axes[1].plot(
    x,
    np.sqrt(365.25 * bs_forecasts.residual_variance.iloc[-1]),
    color='k',
    alpha=1)
axes[1].set_xticks(np.arange(1, 11))
axes[1].set_xlim(1, 10)
axes[1].set_ylim(60, 250)
title = axes[1].set_title('Bootstrap Scenario')

###Hedghog plot
analytic = forecasts.residual_variance.dropna()
bs = bs_forecasts.residual_variance.dropna()
fig, ax = plt.subplots(1, 1)
vol = res.conditional_volatility['2016-1-1':'2019-12-20']
idx = vol.index
ax.plot(np.sqrt(365.25) * vol, alpha=0.5)
for i in range(0, len(vol), 22):
    a = analytic.iloc[i]
    b = bs.iloc[i]
    loc = idx.get_loc(a.name)
    new_idx = idx[loc + 1:loc + 11]
    a.index = new_idx
    b.index = new_idx
    ax.plot(np.sqrt(365.25 * a), color=colors[1])
    ax.plot(np.sqrt(365.25 * b), color=colors[2])
labels = [
    'Annualized Vol.', 'Analytic Forecast', 'Bootstrap Scenario Forecast'
]
legend = ax.legend(labels, frameon=False)
xlim = ax.set_xlim(vol.index[0], vol.index[-1])
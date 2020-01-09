from engine import get_data
import datetime as dt
import seaborn as sns
from engine import liquidations
from scripts import phase_one
from scipy.stats import pearsonr
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

sns.set(style="darkgrid")

######################################
#1 Using 1h data, plot probability of 33% price drop for different time horizons (based on historical data)
######################################

#Import data
df = get_data.get_hourly_cryptocompare_data('ETH', 'USD', start_date = dt.datetime(2018,1,1), end_date = dt.datetime(2018,4,20))

timeframes = []
for i in range(1,169):
    timeframe = str(i) + 'h'
    timeframes.append(timeframe)
    
probabilities = {}    
for i in timeframes:
    df_resampled = df
    df_resampled.replace(0.0, np.nan, inplace=True)
    df_resampled['log_returns'] = np.log(df_resampled['close']) - np.log(df_resampled['close'].shift(1))
    df_resampled = df_resampled[df_resampled['log_returns'].notna()]
    df_resampled = df_resampled.set_index(df_resampled.index).rolling(i).sum()
    rets = df_resampled['log_returns']
    probability = len(rets[rets<-0.50]) / len(rets)
    stripped_i = int(i[:-1])
    probabilities[stripped_i] = 100 * probability

#Create dataframe
prob_df = pd.DataFrame.from_dict(probabilities, orient = 'index')
fig, ax = plt.subplots()
prob_df.plot(ax = ax)
ax.set_ylabel('% chance of negative price drop > 50%', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('Probability of price drops', fontsize = 14)
ax.set_xlabel('Hours', fontsize = 14)
ax.get_legend().remove()
fig.savefig('../5d8dd7887374be0001c94b71/images/probabilityvshours.png', bbox_inches = 'tight', dpi = 600)

######################################
#2 Plot the amount of ETH that would be liquidated for an x% change in price, and the amount of ETH that would be underwater for an x% change in price
######################################

#0. Get CDP data
df = liquidations.build_cdp_dataframe()

#1. Examine the sensitivity of CDPs to percentage price drops
liquidations.plot_liquidations_for_perc_price_drops(df, threshold = 'liquidation_price')
liquidations.plot_liquidations_for_perc_price_drops(df, threshold = 'underwater_price')

#Consider evolution of probability of a 50% price drop (as above)
probability_50percdrop_48hours = prob_df.iloc[47]

#How much ETH would be liquidated?
shock = 50
CURRENT_ETH_PRICE = float(df['pip'].iloc[0])
new_price = CURRENT_ETH_PRICE*(100-abs(shock))/100
df['eth_liability'] = 1.13 * df['art']/df['underwater_price']

total_eth_liquidated = df['eth_liability'][df['underwater_price']>new_price].sum()

#################################
#Liquidity plot - is the total_eth_liquidated volume possible?
#################################

#As the price falls, a certain amount of ETH has to be sold off in order to avoid going underwater
#Next steps -->
#Look at historical volumes on the ETH DAI pair
# Look at market depth in bps on the ETH/DAI pair
#In both cases: see what the market coudl support

######################################
#3 Plot the amount of DAI that would be liquidated for an x% change in price, and the amount of DAI that would be underwater for an x% change in price
######################################

#0. Get CDP data
df = liquidations.build_cdp_dataframe()

#1. Examine the sensitivity of CDPs to percentage price drops
liquidations.plot_liquidations_for_perc_price_drops_dai(df, threshold = 'underwater_price')

######################################
#4 Compute the historical correlation between ETH and MKR
######################################

TOKEN_BASKET = ['ETH', 'MKR']

#Select subset for 2016 until now
start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2020,1,6)

#Get the data
df_eth = get_data.create_df(TOKEN_BASKET[0], 'USD')[start_date:end_date]
df_mkr = get_data.create_df(TOKEN_BASKET[1], 'USD')[start_date:end_date]

eth_prices = df_eth['close']
mkr_prices = df_mkr['close']
corr, _ = pearsonr(eth_prices, mkr_prices) # Value is 0.85

######################################
#5 Look at the impact of price changes in ETH on the MKR market cap
######################################

#Import and clean MKR token market cap data
df_mkr_mktcap = pd.read_csv('MKR_token.csv')
df_mkr_mktcap['new_date']=pd.to_datetime(df_mkr_mktcap['Date'])
df_mkr_mktcap.sort_values(by = 'new_date')
df_mkr_mktcap.set_index('new_date', inplace = True)
df_mkr_mktcap = df_mkr_mktcap['Market Cap']
df_mkr_mktcap = df_mkr_mktcap.str.replace(',', '').astype(float)

#Merge together on same DataFrame
df = pd.concat([df_eth, df_mkr_mktcap], axis = 1)
df = df[['close', 'Market Cap']].dropna()

#Inspect levels for stationarity (needed for VAR model)
fig = plot_acf(df['Market Cap']) # Highly correlated
fig.savefig('../5d8dd7887374be0001c94b71/images/acf_eth_close.png', bbox_inches = 'tight', dpi = 600)

fig = plot_acf(df['close']) # Highly correlated
fig.savefig('../5d8dd7887374be0001c94b71/images/acf_mkr_cap.png', bbox_inches = 'tight', dpi = 600)

#Inspect differences for stationarity (needed for VAR model)
fig = plot_acf(df['close'].diff().dropna()) # Does better, though not perfect
fig.savefig('../5d8dd7887374be0001c94b71/images/acf_eth_close_diff.png', bbox_inches = 'tight', dpi = 600)

fig = plot_acf(df['Market Cap'].diff().dropna()) # Does better, though not perfect
fig.savefig('../5d8dd7887374be0001c94b71/images/acf_mkr_cap_diff.png', bbox_inches = 'tight', dpi = 600)

#So create dataframe in differences

df_diff = df.diff().dropna()

#Fit VAR model
model = VAR(df_diff)
results = model.fit(maxlags=15, ic='aic')
results.summary()
results.plot()
results.plot_acorr()

#Forecasting
lag_order = results.k_ar
results.forecast(df.values[-lag_order:], 5)

results.plot_forecast(10)

#Plot IRFs
irf = results.irf(20)
irf.plot(orth=False)

fig = irf.plot_cum_effects(orth=False)
fig.savefig('../5d8dd7887374be0001c94b71/images/irfs_cumulative.png', bbox_inches = 'tight', dpi = 600)

#Feed this into the simulator


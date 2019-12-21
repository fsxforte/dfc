from engine import liquidations
from scripts import phase_one

#0. Get CDP data
df = liquidations.build_cdp_dataframe()

#1. Examine the sensitivity of CDPs to percentage price drops
liquidations.plot_liquidations_for_perc_price_drops(df)

#Import percentage fall at the 1% level
worst_shock = phase_one.worst_shock*100
CURRENT_ETH_PRICE = float(df['pip'].iloc[0])
new_price = CURRENT_ETH_PRICE*(100-abs(worst_shock))/100
df['eth_liability'] = 1.13 * df['art']/df['liquidation_price']

##############################
total_eth_liquidated = df['eth_liability'][df['liquidation_price']>new_price].sum()
##############################
#--> this is a huge amount of ETH to dump. How much stress would this cause?


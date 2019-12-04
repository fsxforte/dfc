import pandas as pd

from engine import get_data
from engine import monte_carlo

df = get_data.fetch_data()

#focus just on the crash period
df = df[4896:]

#5941 to the end --> 1045 steps

simulation_results = monte_carlo.brownian_motion(df, 1045, 1000)


def crash_proportions(number_of_steps: int):
	#Inspect last row of simulation results
	resultant_prices = simulation_results.iloc[[number_of_steps]].values
	resultant_prices = list(resultant_prices[0])
	crashes = []
	for price in resultant_prices:
		if price < simulation_results.iloc[[0]][0].values[0]*2/3:
			crashes.append(price)
	proportion_of_cases = len(crashes)/len(resultant_prices)
	return proportion_of_cases

proportion_defaults = []
for i in list(range(1046)):
	proportion = crash_proportions(i)
	timestamp = df.iloc[[i]].timestamp.values[0]
	proportion_tuple = (timestamp, proportion)
	proportion_defaults.append(proportion_tuple)


df2 = pd.DataFrame(proportion_defaults, columns=['timestamp', 'proportion'])
df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='s')

df2 = df2.set_index('timestamp')
